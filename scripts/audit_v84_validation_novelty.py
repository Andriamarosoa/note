"""Validate the onset novelty signal on frozen V8.4 predictions.

This is an oracle diagnostic, not a runtime filter and not a training script.
It runs frozen V8.4 models on the deterministic 12-track locked validation
subset, labels each emitted onset as TP/FP using the official +/-50 ms matching,
and measures whether spectral change is preferentially explained by the
already-active harmonic state.

Crucially, train-derived enrichment thresholds are transferred without fitting
them on validation.  We compare the best-global V8.4 control checkpoint
(control epoch 1) with the strongest solo-replay checkpoint (replay epoch 3).
The offset stream is never trained or modified here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_note.guitarset import ALLOWED_PLAYERS, SAMPLE_RATE, index_guitarset
from causal_note.guitarset_acoustics import load_rich_annotations
from causal_note.v84_predictor import V84KerasPredictor
from causal_note.v8_runtime import BoundaryKind, V8BoundaryDecoder
from scripts.audit_v81_train_fp_harmonics import (
    FFT_WINDOW,
    _active_notes,
    _contour_indices,
    _quantiles,
    _spectral_features,
)
from scripts.audit_v84_solo_comp_onsets import _auc
from scripts.evaluate_boundaries import _count_metrics, match_boundaries, milliseconds_to_samples
from scripts.evaluate_v8_boundaries import DEFAULT_SEED, DEFAULT_VALIDATION_FRACTION, _arrangement, _reference_positions
from scripts.train_boundaries import decode_pcm16_mono_wav, split_tracks_by_group


DEFAULT_THRESHOLD = 0.40
DEFAULT_LIMIT_TRACKS = 12
DEFAULT_TOLERANCE_MS = 50.0
DEFAULT_CHUNK_SIZE = 512
DEFAULT_RECEPTIVE_FIELD = 4093
CONTOUR_MAX_DISTANCE_SAMPLES = round(0.050 * SAMPLE_RATE)

# Frozen operating points discovered on the train-only audit.  These values are
# intentionally not optimized on the locked validation subset.
TRANSFER_RULES = (
    {
        "name": "fixed_enrichment_train_comp95",
        "feature": "fixed_positive_flux_enrichment",
        "reject_if_ge": 1.0929943,
        "train_comp_true_retention": 0.951909,
        "train_solo_fp_rejection": 0.822194,
    },
    {
        "name": "fixed_enrichment_train_comp99",
        "feature": "fixed_positive_flux_enrichment",
        "reject_if_ge": 1.4007305,
        "train_comp_true_retention": 0.990673,
        "train_solo_fp_rejection": 0.761980,
    },
    {
        "name": "contour_enrichment_train_comp95",
        "feature": "contour_positive_flux_enrichment",
        "reject_if_ge": 1.1013554,
        "train_comp_true_retention": 0.951618,
        "train_solo_fp_rejection": 0.835120,
    },
    {
        "name": "contour_enrichment_train_comp99",
        "feature": "contour_positive_flux_enrichment",
        "reject_if_ge": 1.3943705,
        "train_comp_true_retention": 0.990382,
        "train_solo_fp_rejection": 0.779004,
    },
)

AUDIT_FEATURES = (
    "active_note_count",
    "rms_delta_db",
    "positive_flux_over_pre_energy",
    "fixed_mask_coverage",
    "fixed_positive_flux_fraction",
    "fixed_positive_flux_enrichment",
    "contour_mask_coverage",
    "contour_positive_flux_fraction",
    "contour_positive_flux_enrichment",
)


class AuditError(RuntimeError):
    pass


def _finite(record: dict, key: str) -> Optional[float]:
    value = record.get(key)
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _predict_onsets(model_path: Path, validation_tracks, *, chunk_size: int, receptive_field: int, threshold: float):
    predictor = V84KerasPredictor.from_path(str(model_path), receptive_field=receptive_field)
    predictor.warm_up(chunk_size)
    result = {}
    for ordinal, track in enumerate(validation_tracks, start=1):
        audio = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        predictor.reset()
        decoder = V8BoundaryDecoder(onset_threshold=threshold, offset_threshold=threshold)
        predictions: List[int] = []
        position = 0
        while position < audio.frame_count:
            end = min(audio.frame_count, position + chunk_size)
            values = tuple(sample / 32768.0 for sample in audio.samples[position:end])
            scores = predictor.predict_chunk(values, start_sample=position)
            for boundary in decoder.process_chunk(scores):
                if boundary.kind is BoundaryKind.ONSET:
                    predictions.extend([boundary.sample] * boundary.count)
            position = end
        result[track.annotation_member] = tuple(predictions)
        print(f"predicted {ordinal}/{len(validation_tracks)} {model_path.name}: {track.annotation_member} onsets={len(predictions)}")
    return result


def _summarize(records: Sequence[dict]) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "count": len(records),
        "tracks": len({record["member"] for record in records}),
        "with_active_state": sum(int(record.get("active_note_count", 0)) > 0 for record in records),
    }
    for feature in AUDIT_FEATURES:
        values = [value for record in records if (value := _finite(record, feature)) is not None]
        summary[feature] = {
            "defined": len(values),
            **_quantiles(values),
        }
    return summary


def _separation(fp_records: Sequence[dict], tp_records: Sequence[dict]) -> Dict[str, object]:
    result = {}
    for feature in AUDIT_FEATURES:
        fp = [value for record in fp_records if (value := _finite(record, feature)) is not None]
        tp = [value for record in tp_records if (value := _finite(record, feature)) is not None]
        auc = _auc(fp, tp)
        if auc is None:
            continue
        result[feature] = {
            "auc_fp_high": auc,
            "auc_fp_low": 1.0 - auc,
            "separation": max(auc, 1.0 - auc),
            "preferred_fp_direction": "high" if auc >= 0.5 else "low",
            "fp_defined": len(fp),
            "tp_defined": len(tp),
        }
    return dict(sorted(result.items(), key=lambda item: item[1]["separation"], reverse=True))


def _aggregate_exact(track_records: Sequence[dict], retained_by_member: Dict[str, Tuple[int, ...]], arrangement: Optional[str]) -> dict:
    refs = preds = tp = 0
    for track in track_records:
        if arrangement is not None and track["arrangement"] != arrangement:
            continue
        references = track["references"]
        predictions = retained_by_member.get(track["member"], ())
        pairs = match_boundaries(references, predictions, track["tolerance_samples"])
        refs += len(references)
        preds += len(predictions)
        tp += len(pairs)
    metrics = _count_metrics(refs, preds, tp)
    data = asdict(metrics)
    data["prediction_reference_ratio"] = preds / refs if refs else None
    return data


def _apply_rule(records: Sequence[dict], rule: dict) -> Tuple[int, ...]:
    feature = rule["feature"]
    threshold = float(rule["reject_if_ge"])
    kept = []
    for record in records:
        value = _finite(record, feature)
        # Undefined enrichment means there was no active harmonic state capable
        # of explaining the change, so the state-explained rejector does not fire.
        if value is not None and value >= threshold:
            continue
        kept.append(int(record["sample"]))
    return tuple(kept)


def _rule_report(all_records: Sequence[dict], track_records: Sequence[dict], rule: dict) -> dict:
    retained_by_member: Dict[str, Tuple[int, ...]] = {}
    by_member: Dict[str, List[dict]] = defaultdict(list)
    for record in all_records:
        by_member[record["member"]].append(record)
    for member, records in by_member.items():
        retained_by_member[member] = _apply_rule(records, rule)

    def descriptive(arrangement: str) -> dict:
        population = [r for r in all_records if r["arrangement"] == arrangement]
        fp = [r for r in population if not r["matched"]]
        tp = [r for r in population if r["matched"]]
        feature = rule["feature"]
        threshold = float(rule["reject_if_ge"])
        rejected_fp = sum((_finite(r, feature) is not None and _finite(r, feature) >= threshold) for r in fp)
        rejected_tp = sum((_finite(r, feature) is not None and _finite(r, feature) >= threshold) for r in tp)
        return {
            "baseline_fp": len(fp),
            "baseline_tp": len(tp),
            "fp_rejection": rejected_fp / len(fp) if fp else None,
            "matched_tp_retention": 1.0 - rejected_tp / len(tp) if tp else None,
        }

    return {
        **rule,
        "descriptive": {
            "solo": descriptive("solo"),
            "comp": descriptive("comp"),
        },
        "exact_postfilter_metrics": {
            "global": _aggregate_exact(track_records, retained_by_member, None),
            "solo": _aggregate_exact(track_records, retained_by_member, "solo"),
            "comp": _aggregate_exact(track_records, retained_by_member, "comp"),
        },
    }


def _audit_arm(name: str, model_path: Path, validation_tracks, predictions_by_member, *, tolerance_samples: int) -> dict:
    records: List[dict] = []
    track_records: List[dict] = []
    for ordinal, track in enumerate(validation_tracks, start=1):
        member = track.annotation_member
        audio = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        references, _ = _reference_positions(track, audio.frame_count)
        predictions = predictions_by_member[member]
        pairs = match_boundaries(references, predictions, tolerance_samples)
        matched_predictions = Counter(prediction for _, prediction in pairs)

        rich = load_rich_annotations(track.annotation_zip, member)
        notes = tuple(sorted(rich.notes, key=lambda note: (note.onset_sample, note.slot, note.offset_sample)))
        contour_index = _contour_indices(rich.contours_by_slot)
        normalized = np.asarray(audio.samples, dtype=np.float64) / 32768.0
        cache: Dict[int, dict] = {}
        arrangement = _arrangement(member)

        for prediction in predictions:
            if prediction not in cache:
                if prediction < FFT_WINDOW or prediction + FFT_WINDOW >= audio.frame_count:
                    features = {
                        "active_note_count": len(_active_notes(notes, prediction)),
                        **{feature: None for feature in AUDIT_FEATURES if feature != "active_note_count"},
                    }
                else:
                    active = _active_notes(notes, prediction)
                    features = _spectral_features(
                        normalized,
                        prediction,
                        active,
                        contour_index,
                        CONTOUR_MAX_DISTANCE_SAMPLES,
                    )
                cache[prediction] = features
            matched = matched_predictions[prediction] > 0
            if matched:
                matched_predictions[prediction] -= 1
            records.append({
                "member": member,
                "arrangement": arrangement,
                "sample": int(prediction),
                "matched": matched,
                **cache[prediction],
            })

        track_records.append({
            "member": member,
            "arrangement": arrangement,
            "references": tuple(references),
            "predictions": tuple(predictions),
            "tolerance_samples": tolerance_samples,
        })
        print(f"audited {name} {ordinal}/{len(validation_tracks)}: {member}")

    populations = {}
    separation = {}
    for arrangement in ("global", "solo", "comp"):
        selected = records if arrangement == "global" else [r for r in records if r["arrangement"] == arrangement]
        fp = [r for r in selected if not r["matched"]]
        tp = [r for r in selected if r["matched"]]
        populations[arrangement] = {
            "false_positive": _summarize(fp),
            "true_positive_prediction": _summarize(tp),
        }
        separation[arrangement] = _separation(fp, tp)

    baseline_by_member = {track["member"]: track["predictions"] for track in track_records}
    baseline = {
        "global": _aggregate_exact(track_records, baseline_by_member, None),
        "solo": _aggregate_exact(track_records, baseline_by_member, "solo"),
        "comp": _aggregate_exact(track_records, baseline_by_member, "comp"),
    }
    transferred = [_rule_report(records, track_records, rule) for rule in TRANSFER_RULES]

    return {
        "name": name,
        "model": str(model_path.resolve()),
        "baseline": baseline,
        "populations": populations,
        "rank_separation": separation,
        "transferred_train_rules": transferred,
    }


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit V8.4 onset novelty on locked validation predictions.")
    parser.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    parser.add_argument("--control-model", type=Path, required=True)
    parser.add_argument("--replay-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--limit-tracks", type=int, default=DEFAULT_LIMIT_TRACKS)
    parser.add_argument("--tolerance-ms", type=float, default=DEFAULT_TOLERANCE_MS)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--receptive-field", type=int, default=DEFAULT_RECEPTIVE_FIELD)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def run(args) -> Dict[str, object]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.seed != DEFAULT_SEED:
        raise AuditError(f"audit is locked to split seed {DEFAULT_SEED}")
    if args.limit_tracks != DEFAULT_LIMIT_TRACKS:
        raise AuditError(f"audit is locked to {DEFAULT_LIMIT_TRACKS} validation tracks")
    if abs(float(args.threshold) - DEFAULT_THRESHOLD) > 1e-12:
        raise AuditError(f"audit is locked to onset threshold {DEFAULT_THRESHOLD}")

    indexed = tuple(track for track in index_guitarset(args.dataset_dir) if track.player_id in ALLOWED_PLAYERS)
    _, locked_validation_tracks = split_tracks_by_group(
        indexed,
        validation_fraction=DEFAULT_VALIDATION_FRACTION,
        seed=args.seed,
    )
    validation_tracks = locked_validation_tracks[: args.limit_tracks]
    tolerance_samples = milliseconds_to_samples(args.tolerance_ms)

    arms = {}
    for name, model in (("control_epoch_01", args.control_model), ("replay_epoch_03", args.replay_model)):
        predictions = _predict_onsets(
            model,
            validation_tracks,
            chunk_size=args.chunk_size,
            receptive_field=args.receptive_field,
            threshold=args.threshold,
        )
        arms[name] = _audit_arm(
            name,
            model,
            validation_tracks,
            predictions,
            tolerance_samples=tolerance_samples,
        )

    result = {
        "schema_version": 1,
        "purpose": "validation-only generalization test of V8.4 onset novelty; no training and no validation fitting",
        "split": {
            "seed": args.seed,
            "validation_fraction": DEFAULT_VALIDATION_FRACTION,
            "locked_validation_tracks_total": len(locked_validation_tracks),
            "evaluated_tracks": len(validation_tracks),
            "validation_members": [track.annotation_member for track in validation_tracks],
        },
        "configuration": {
            "threshold": args.threshold,
            "tolerance_ms": args.tolerance_ms,
            "tolerance_samples": tolerance_samples,
            "chunk_size": args.chunk_size,
            "receptive_field": args.receptive_field,
            "transfer_rules_frozen_from_train": True,
            "validation_threshold_refit": False,
            "offset_training_or_mutation": False,
        },
        "arms": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = create_argument_parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
