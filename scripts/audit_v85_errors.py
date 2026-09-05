"""Audit residual V8.5 onset errors on the frozen locked12 validation set.

This is diagnostic only: no training, no threshold fitting, no model mutation.
It separates false negatives into proposal misses (V8.4 never proposed the
reference) and verifier rejects (V8.4 proposed it but V8.5 removed it), then
characterizes residual FP/TP/FN populations with causal temporal and acoustic
state features.
"""
from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict
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
from scripts.audit_v81_train_fp_harmonics import FFT_WINDOW, _contour_indices, _quantiles, _spectral_features
from scripts.audit_v84_solo_comp_onsets import _auc
from scripts.evaluate_boundaries import match_boundaries
from scripts.evaluate_v8_boundaries import DEFAULT_SEED, DEFAULT_VALIDATION_FRACTION, _arrangement, _reference_positions
from scripts.train_boundaries import decode_pcm16_mono_wav, split_tracks_by_group
from scripts.train_v85_transition_verifier import (
    TOLERANCE_SAMPLES,
    _build_verifier,
    _candidate_records,
    _predict_tracks,
    _score_records,
)


class AuditError(RuntimeError):
    pass


FEATURE_KEYS = (
    "verifier_score",
    "active_note_count",
    "rms_delta_db",
    "positive_flux_over_pre_energy",
    "negative_flux_over_pre_energy",
    "fixed_mask_coverage",
    "fixed_positive_flux_fraction",
    "fixed_positive_flux_enrichment",
    "contour_mask_coverage",
    "contour_positive_flux_fraction",
    "contour_positive_flux_enrichment",
    "outside_active_positive_flux_fraction",
    "previous_onset_age_ms",
    "next_onset_distance_ms",
)


def _finite(record: dict, key: str) -> Optional[float]:
    value = record.get(key)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _ms(samples: Optional[int]) -> Optional[float]:
    return None if samples is None else float(samples) * 1000.0 / SAMPLE_RATE


def _previous_distance(sample: int, values: Sequence[int]) -> Optional[int]:
    index = bisect.bisect_right(values, sample) - 1
    return None if index < 0 else sample - values[index]


def _next_distance(sample: int, values: Sequence[int]) -> Optional[int]:
    index = bisect.bisect_left(values, sample)
    return None if index >= len(values) else values[index] - sample


def _temporal_bucket(record: dict) -> str:
    previous = _finite(record, "previous_onset_age_ms")
    if previous is None:
        return "no_previous_onset"
    if previous <= 10.0:
        return "after_onset_0_10ms"
    if previous <= 20.0:
        return "after_onset_10_20ms"
    if previous <= 50.0:
        return "after_onset_20_50ms"
    if previous <= 100.0:
        return "after_onset_50_100ms"
    return "after_onset_gt_100ms"


def _spacing_bucket(record: dict) -> str:
    values = [v for key in ("previous_onset_age_ms", "next_onset_distance_ms") if (v := _finite(record, key)) is not None]
    if not values:
        return "isolated_unknown"
    nearest = min(values)
    if nearest <= 10.0:
        return "neighbor_0_10ms"
    if nearest <= 20.0:
        return "neighbor_10_20ms"
    if nearest <= 50.0:
        return "neighbor_20_50ms"
    if nearest <= 100.0:
        return "neighbor_50_100ms"
    return "neighbor_gt_100ms"


def _summary(records: Sequence[dict], *, true_reference: bool = False) -> dict:
    result = {
        "count": len(records),
        "tracks": len({r["member"] for r in records}),
        "arrangement": dict(sorted(Counter(r["arrangement"] for r in records).items())),
        "active_polyphony": dict(sorted(Counter(int(r.get("active_note_count", 0)) for r in records).items())),
    }
    buckets = Counter((_spacing_bucket(r) if true_reference else _temporal_bucket(r)) for r in records)
    result["spacing_buckets" if true_reference else "temporal_buckets"] = dict(sorted(buckets.items()))
    for key in FEATURE_KEYS:
        values = [value for r in records if (value := _finite(r, key)) is not None]
        result[key] = _quantiles(values)
    return result


def _rank_separation(high: Sequence[dict], low: Sequence[dict]) -> dict:
    result = {}
    for key in FEATURE_KEYS:
        high_values = [value for r in high if (value := _finite(r, key)) is not None]
        low_values = [value for r in low if (value := _finite(r, key)) is not None]
        auc = _auc(high_values, low_values)
        if auc is None:
            continue
        result[key] = {
            "auc_high": auc,
            "auc_low": 1.0 - auc,
            "preferred_high_direction": "high" if auc >= 0.5 else "low",
            "separation": max(auc, 1.0 - auc),
            "high_count": len(high_values),
            "low_count": len(low_values),
        }
    return dict(sorted(result.items(), key=lambda item: item[1]["separation"], reverse=True))


def _enrich_position(track, normalized: np.ndarray, rich, contour_index, sample: int, *, true_reference: bool) -> dict:
    notes = tuple(sorted(rich.notes, key=lambda note: (note.onset_sample, note.slot, note.offset_sample)))
    unique_onsets = tuple(sorted(set(note.onset_sample for note in notes)))
    active_before = tuple(note for note in notes if note.onset_sample < sample < note.offset_sample)
    if true_reference:
        previous = _previous_distance(sample - 1, unique_onsets)
        following = _next_distance(sample + 1, unique_onsets)
    else:
        previous = _previous_distance(sample, unique_onsets)
        following = _next_distance(sample, unique_onsets)
    record = {
        "member": track.annotation_member,
        "arrangement": _arrangement(track.annotation_member),
        "sample": int(sample),
        "active_note_count": len(active_before),
        "previous_onset_age_ms": _ms(previous),
        "next_onset_distance_ms": _ms(following),
    }
    if sample < FFT_WINDOW or sample + FFT_WINDOW >= len(normalized):
        return record
    features = _spectral_features(normalized, sample, active_before, contour_index, round(0.050 * SAMPLE_RATE))
    fraction = features.get("fixed_positive_flux_fraction")
    record.update(features)
    record["outside_active_positive_flux_fraction"] = None if fraction is None else 1.0 - float(fraction)
    return record


def _occurrence_labels(values: Sequence[int], matched_values: Sequence[int]) -> List[bool]:
    remaining = Counter(matched_values)
    labels = []
    for value in values:
        label = remaining[value] > 0
        labels.append(label)
        if label:
            remaining[value] -= 1
    return labels


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit residual V8.5 onset errors on locked12.")
    parser.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--verifier-weights", type=Path, required=True)
    parser.add_argument("--v85-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args) -> dict:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    source_report = json.loads(args.v85_report.read_text(encoding="utf-8"))
    threshold = float(source_report["holdout_calibration"]["best"]["threshold"])
    if source_report["configuration"].get("gate_threshold_selected_on_locked_validation"):
        raise AuditError("refusing a verifier threshold selected on locked validation")

    indexed = tuple(track for track in index_guitarset(args.dataset_dir) if track.player_id in ALLOWED_PLAYERS)
    _, locked_validation = split_tracks_by_group(indexed, validation_fraction=DEFAULT_VALIDATION_FRACTION, seed=DEFAULT_SEED)
    locked12 = tuple(locked_validation[:12])
    expected_members = tuple(source_report["data"]["locked_validation_members"])
    actual_members = tuple(track.annotation_member for track in locked12)
    if actual_members != expected_members:
        raise AuditError("locked12 membership differs from frozen V8.5 report")

    print("replaying frozen V8.4 control onset proposals")
    baseline_predictions = _predict_tracks(args.base_model, locked12)
    candidate_records = _candidate_records(locked12, baseline_predictions)
    verifier = _build_verifier()
    verifier.load_weights(args.verifier_weights)
    scores = _score_records(verifier, candidate_records)
    for record, score in zip(candidate_records, scores):
        record["verifier_score"] = float(score)

    candidates_by_member: Dict[str, List[dict]] = defaultdict(list)
    for record in candidate_records:
        candidates_by_member[record["member"]].append(record)

    populations: Dict[str, List[dict]] = defaultdict(list)
    track_rows = []
    for ordinal, track in enumerate(locked12, start=1):
        member = track.annotation_member
        audio = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        normalized = np.asarray(audio.samples, dtype=np.float64) / 32768.0
        references, _ = _reference_positions(track, audio.frame_count)
        baseline = list(baseline_predictions[member])
        member_candidates = candidates_by_member[member]
        gated_candidates = [r for r in member_candidates if float(r["verifier_score"]) >= threshold]
        gated = [int(r["sample"]) for r in gated_candidates]

        baseline_pairs = match_boundaries(references, baseline, TOLERANCE_SAMPLES)
        gated_pairs = match_boundaries(references, gated, TOLERANCE_SAMPLES)
        baseline_matched_refs = [ref for ref, _ in baseline_pairs]
        gated_matched_refs = [ref for ref, _ in gated_pairs]
        gated_matched_preds = [pred for _, pred in gated_pairs]
        baseline_ref_labels = _occurrence_labels(references, baseline_matched_refs)
        gated_ref_labels = _occurrence_labels(references, gated_matched_refs)
        gated_pred_labels = _occurrence_labels(gated, gated_matched_preds)

        rich = load_rich_annotations(track.annotation_zip, member)
        contour_index = _contour_indices(rich.contours_by_slot)

        # Candidate-level retained TP/FP populations.
        for candidate, is_tp in zip(gated_candidates, gated_pred_labels):
            enriched = _enrich_position(track, normalized, rich, contour_index, int(candidate["sample"]), true_reference=False)
            enriched["verifier_score"] = float(candidate["verifier_score"])
            enriched["error_stage"] = "retained_tp" if is_tp else "retained_fp"
            populations[enriched["error_stage"]].append(enriched)

        # Reference-level FN populations, explicitly separating proposal vs verifier stage.
        baseline_ref_remaining = Counter(baseline_matched_refs)
        gated_ref_remaining = Counter(gated_matched_refs)
        candidate_scores_by_sample: Dict[int, List[float]] = defaultdict(list)
        for candidate in member_candidates:
            candidate_scores_by_sample[int(candidate["sample"])].append(float(candidate["verifier_score"]))
        baseline_preds_for_ref: Dict[int, List[int]] = defaultdict(list)
        for ref, pred in baseline_pairs:
            baseline_preds_for_ref[int(ref)].append(int(pred))

        for ref, baseline_hit, gated_hit in zip(references, baseline_ref_labels, gated_ref_labels):
            if gated_hit:
                enriched = _enrich_position(track, normalized, rich, contour_index, int(ref), true_reference=True)
                enriched["error_stage"] = "detected_reference"
                populations["detected_reference"].append(enriched)
                continue
            stage = "verifier_reject" if baseline_hit else "proposal_miss"
            enriched = _enrich_position(track, normalized, rich, contour_index, int(ref), true_reference=True)
            enriched["error_stage"] = stage
            if baseline_hit:
                paired_predictions = baseline_preds_for_ref.get(int(ref), [])
                paired_scores = [score for pred in paired_predictions for score in candidate_scores_by_sample.get(pred, [])]
                enriched["verifier_score"] = max(paired_scores) if paired_scores else None
            populations[stage].append(enriched)

        track_rows.append({
            "member": member,
            "arrangement": _arrangement(member),
            "references": len(references),
            "baseline_predictions": len(baseline),
            "baseline_tp": len(baseline_pairs),
            "gated_predictions": len(gated),
            "gated_tp": len(gated_pairs),
            "gated_fp": len(gated) - len(gated_pairs),
            "gated_fn": len(references) - len(gated_pairs),
        })
        print(f"audited {ordinal}/{len(locked12)}: {member}")

    retained_fp = populations["retained_fp"]
    retained_tp = populations["retained_tp"]
    proposal_miss = populations["proposal_miss"]
    verifier_reject = populations["verifier_reject"]
    detected_reference = populations["detected_reference"]

    result = {
        "schema_version": 1,
        "protocol": {
            "diagnostic_only": True,
            "training_performed": False,
            "threshold_refit": False,
            "gate_threshold": threshold,
            "locked_tracks": len(locked12),
            "base": "frozen V8.4 control epoch 01 onset stream",
            "verifier": "frozen V8.5 transition verifier",
        },
        "counts": {
            "references": sum(row["references"] for row in track_rows),
            "predictions": sum(row["gated_predictions"] for row in track_rows),
            "true_positive": sum(row["gated_tp"] for row in track_rows),
            "false_positive": sum(row["gated_fp"] for row in track_rows),
            "false_negative": sum(row["gated_fn"] for row in track_rows),
            "fn_by_stage": {
                "proposal_miss": len(proposal_miss),
                "verifier_reject": len(verifier_reject),
            },
        },
        "populations": {
            "retained_fp": _summary(retained_fp),
            "retained_tp": _summary(retained_tp),
            "proposal_miss_fn": _summary(proposal_miss, true_reference=True),
            "verifier_reject_fn": _summary(verifier_reject, true_reference=True),
            "detected_reference": _summary(detected_reference, true_reference=True),
        },
        "remaining_fp_vs_retained_tp_separation": _rank_separation(retained_fp, retained_tp),
        "verifier_reject_fn_vs_detected_reference_separation": _rank_separation(verifier_reject, detected_reference),
        "tracks": track_rows,
    }

    expected = source_report["locked12"]["gated"]["global"]
    counts = result["counts"]
    if counts["references"] != int(expected["reference_count"]):
        raise AuditError(f"reference count mismatch: {counts['references']} vs {expected['reference_count']}")
    if counts["predictions"] != int(expected["prediction_count"]):
        raise AuditError(f"prediction count mismatch: {counts['predictions']} vs {expected['prediction_count']}")
    if counts["true_positive"] != int(expected["true_positive"]):
        raise AuditError(f"TP mismatch: {counts['true_positive']} vs {expected['true_positive']}")
    if counts["false_positive"] != int(expected["false_positive"]):
        raise AuditError(f"FP mismatch: {counts['false_positive']} vs {expected['false_positive']}")
    if counts["false_negative"] != int(expected["false_negative"]):
        raise AuditError(f"FN mismatch: {counts['false_negative']} vs {expected['false_negative']}")
    if counts["fn_by_stage"]["proposal_miss"] + counts["fn_by_stage"]["verifier_reject"] != counts["false_negative"]:
        raise AuditError("FN stage decomposition does not sum to total FN")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "counts": result["counts"],
        "retained_fp_temporal": result["populations"]["retained_fp"]["temporal_buckets"],
        "proposal_miss_spacing": result["populations"]["proposal_miss_fn"]["spacing_buckets"],
        "verifier_reject_spacing": result["populations"]["verifier_reject_fn"]["spacing_buckets"],
        "fp_top_separators": list(result["remaining_fp_vs_retained_tp_separation"].items())[:8],
        "fn_top_separators": list(result["verifier_reject_fn_vs_detected_reference_separation"].items())[:8],
    }, indent=2, sort_keys=True))
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = create_argument_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
