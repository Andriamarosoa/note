"""Audit why true-FP onset replay helps solo but hurts comp.

This diagnostic performs no training.  It consumes the frozen train-FP acoustic
audit and recomputes acoustically comparable true-onset features on exactly the
same train members.  Populations are split by arrangement (solo/comp).

The central question is operational: can a causal scalar or simple two-signal
rule reject many solo retriggers while retaining comp true onsets?  We therefore
report population summaries, signed distance of FP predictions to nearby true
onsets, rank-separation AUCs, and threshold operating points chosen only from the
observed scalar distributions.
"""
from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_note.guitarset import SAMPLE_RATE, index_guitarset
from causal_note.guitarset_acoustics import RichNote, load_rich_annotations
from scripts.audit_v81_train_fp_harmonics import (
    FFT_WINDOW,
    _active_notes,
    _contour_indices,
    _quantiles,
    _spectral_features,
)
from scripts.evaluate_v8_boundaries import _arrangement
from scripts.train_boundaries import decode_pcm16_mono_wav


class AuditError(RuntimeError):
    pass


SCALAR_METRICS = (
    "model_onset_score",
    "active_note_count",
    "pre_rms_dbfs",
    "post_rms_dbfs",
    "rms_delta_db",
    "positive_flux_over_pre_energy",
    "negative_flux_over_pre_energy",
    "fixed_mask_coverage",
    "fixed_positive_flux_fraction",
    "fixed_positive_flux_enrichment",
    "fixed_negative_flux_fraction",
    "fixed_negative_flux_enrichment",
    "contour_mask_coverage",
    "contour_positive_flux_fraction",
    "contour_positive_flux_enrichment",
    "outside_active_positive_flux_fraction",
)


def _finite(record: dict, key: str) -> Optional[float]:
    value = record.get(key)
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _signed_nearest(sample: int, values: Sequence[int]) -> Optional[int]:
    if not values:
        return None
    index = bisect.bisect_left(values, sample)
    candidates = []
    if index < len(values):
        candidates.append(values[index] - sample)
    if index:
        candidates.append(values[index - 1] - sample)
    if not candidates:
        return None
    return min(candidates, key=lambda value: (abs(value), value))


def _previous_distance(sample: int, values: Sequence[int]) -> Optional[int]:
    index = bisect.bisect_right(values, sample) - 1
    if index < 0:
        return None
    return sample - values[index]


def _next_distance(sample: int, values: Sequence[int]) -> Optional[int]:
    index = bisect.bisect_left(values, sample)
    if index >= len(values):
        return None
    return values[index] - sample


def _ms(samples: Optional[int]) -> Optional[float]:
    return None if samples is None else float(samples) * 1000.0 / SAMPLE_RATE


def _summarize(records: Sequence[dict]) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "positions": len(records),
        "tracks": len({record["member"] for record in records}),
        "with_active_notes": sum(int(record.get("active_note_count", 0)) > 0 for record in records),
        "polyphony": dict(sorted(Counter(int(record.get("active_note_count", 0)) for record in records).items())),
    }
    for key in SCALAR_METRICS:
        values = [value for record in records if (value := _finite(record, key)) is not None]
        summary[key] = _quantiles(values)
    for key in (
        "signed_nearest_onset_ms",
        "previous_onset_age_ms",
        "next_onset_distance_ms",
    ):
        values = [value for record in records if (value := _finite(record, key)) is not None]
        summary[key] = _quantiles(values)
    for window in (5, 10, 20, 30, 50, 100):
        summary[f"within_{window}ms_of_any_onset"] = sum(
            record.get("signed_nearest_onset_ms") is not None
            and abs(float(record["signed_nearest_onset_ms"])) <= window
            for record in records
        )
        summary[f"within_{window}ms_after_previous_onset"] = sum(
            record.get("previous_onset_age_ms") is not None
            and 0.0 <= float(record["previous_onset_age_ms"]) <= window
            for record in records
        )
    return summary


def _auc(high: Sequence[float], low: Sequence[float]) -> Optional[float]:
    """Probability a random high-class value exceeds a random low-class value."""
    if not high or not low:
        return None
    ordered = sorted((float(v), 1) for v in high) + sorted((float(v), 0) for v in low)
    ordered.sort(key=lambda item: item[0])
    rank_sum = 0.0
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][0] == ordered[cursor][0]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        rank_sum += average_rank * sum(label for _, label in ordered[cursor:end])
        cursor = end
    n1 = len(high)
    n0 = len(low)
    u = rank_sum - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * n0))


def _rank_separation(fp: Sequence[dict], true: Sequence[dict]) -> Dict[str, object]:
    result = {}
    for key in SCALAR_METRICS:
        fp_values = [value for record in fp if (value := _finite(record, key)) is not None]
        true_values = [value for record in true if (value := _finite(record, key)) is not None]
        auc = _auc(fp_values, true_values)
        if auc is None:
            continue
        result[key] = {
            "auc_fp_high": auc,
            "auc_fp_low": 1.0 - auc,
            "preferred_fp_direction": "high" if auc >= 0.5 else "low",
            "separation": max(auc, 1.0 - auc),
            "fp_count": len(fp_values),
            "true_count": len(true_values),
        }
    return dict(sorted(result.items(), key=lambda item: item[1]["separation"], reverse=True))


def _operating_points(
    solo_fp: Sequence[dict],
    comp_true: Sequence[dict],
    solo_true: Sequence[dict],
) -> Dict[str, object]:
    """Search one-dimensional rejectors and report best FP rejection at comp retention floors."""
    result = {}
    retention_floors = (0.99, 0.95, 0.90, 0.80)
    for key in SCALAR_METRICS:
        fp_values = [(record, _finite(record, key)) for record in solo_fp]
        comp_values = [(record, _finite(record, key)) for record in comp_true]
        solo_values = [(record, _finite(record, key)) for record in solo_true]
        fp_values = [(r, v) for r, v in fp_values if v is not None]
        comp_values = [(r, v) for r, v in comp_values if v is not None]
        solo_values = [(r, v) for r, v in solo_values if v is not None]
        if not fp_values or not comp_values:
            continue
        candidates = sorted({v for _, v in fp_values + comp_values + solo_values})
        if len(candidates) > 512:
            indices = np.linspace(0, len(candidates) - 1, 512).round().astype(int)
            candidates = [candidates[index] for index in sorted(set(indices.tolist()))]
        rows = []
        for direction in ("high", "low"):
            for threshold in candidates:
                reject = (lambda v: v >= threshold) if direction == "high" else (lambda v: v <= threshold)
                fp_rejection = sum(reject(v) for _, v in fp_values) / len(fp_values)
                comp_retention = 1.0 - sum(reject(v) for _, v in comp_values) / len(comp_values)
                solo_retention = None if not solo_values else 1.0 - sum(reject(v) for _, v in solo_values) / len(solo_values)
                rows.append({
                    "direction": direction,
                    "threshold": float(threshold),
                    "solo_fp_rejection": float(fp_rejection),
                    "comp_true_retention": float(comp_retention),
                    "solo_true_retention": None if solo_retention is None else float(solo_retention),
                })
        best = {}
        for floor in retention_floors:
            eligible = [row for row in rows if row["comp_true_retention"] >= floor]
            if eligible:
                chosen = max(eligible, key=lambda row: (row["solo_fp_rejection"], row["solo_true_retention"] or 0.0))
                best[f"comp_retention_ge_{int(floor*100)}"] = chosen
        result[key] = best
    return result


def _combined_rules(solo_fp: Sequence[dict], comp_true: Sequence[dict], solo_true: Sequence[dict]) -> List[dict]:
    """Small interpretable grid: recent-onset age AND active-family explanation."""
    rows = []
    for age_ms in (10.0, 20.0, 30.0, 50.0, 80.0, 100.0):
        for active_fraction in (0.50, 0.60, 0.70, 0.80, 0.90):
            def reject(record):
                age = record.get("previous_onset_age_ms")
                fraction = record.get("fixed_positive_flux_fraction")
                return (
                    age is not None
                    and 0.0 <= float(age) <= age_ms
                    and fraction is not None
                    and float(fraction) >= active_fraction
                )
            fp_rejection = sum(reject(record) for record in solo_fp) / len(solo_fp) if solo_fp else 0.0
            comp_retention = 1.0 - sum(reject(record) for record in comp_true) / len(comp_true) if comp_true else 0.0
            solo_retention = 1.0 - sum(reject(record) for record in solo_true) / len(solo_true) if solo_true else 0.0
            rows.append({
                "previous_onset_age_max_ms": age_ms,
                "active_flux_fraction_min": active_fraction,
                "solo_fp_rejection": fp_rejection,
                "comp_true_retention": comp_retention,
                "solo_true_retention": solo_retention,
            })
    rows.sort(key=lambda row: (row["comp_true_retention"] >= 0.95, row["solo_fp_rejection"], row["comp_true_retention"]), reverse=True)
    return rows[:20]


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit V8.4 solo/comp onset populations.")
    parser.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    parser.add_argument("--fp-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args) -> Dict[str, object]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    data = json.loads(args.fp_audit.read_text(encoding="utf-8"))
    fp_records = data.get("false_positive_records") or data.get("false_positive_onset_records")
    if not isinstance(fp_records, list) or not fp_records:
        raise AuditError("FP audit contains no false-positive records")
    control_records = data.get("control_records") or []
    members = tuple(data["scope"]["members"])
    tracks = {track.annotation_member: track for track in index_guitarset(args.dataset_dir)}

    fp_by_member: Dict[str, List[dict]] = defaultdict(list)
    for source in fp_records:
        record = dict(source)
        record["arrangement"] = record.get("arrangement") or _arrangement(record["member"])
        fp_by_member[record["member"]].append(record)

    true_records: List[dict] = []
    enriched_fp: List[dict] = []
    for ordinal, member in enumerate(members, start=1):
        track = tracks.get(member)
        if track is None:
            raise AuditError(f"missing GuitarSet member {member}")
        rich = load_rich_annotations(track.annotation_zip, member)
        notes = tuple(sorted(rich.notes, key=lambda note: (note.onset_sample, note.slot, note.offset_sample)))
        contour_index = _contour_indices(rich.contours_by_slot)
        onset_positions = tuple(sorted(note.onset_sample for note in notes))
        unique_onsets = tuple(sorted(set(onset_positions)))
        audio = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        normalized = np.asarray(audio.samples, dtype=np.float64) / 32768.0
        arrangement = _arrangement(member)

        for source in fp_by_member.get(member, ()):
            record = dict(source)
            sample = int(record["sample"])
            signed = _signed_nearest(sample, unique_onsets)
            record["signed_nearest_onset_ms"] = _ms(signed)
            record["previous_onset_age_ms"] = _ms(_previous_distance(sample, unique_onsets))
            record["next_onset_distance_ms"] = _ms(_next_distance(sample, unique_onsets))
            fraction = _finite(record, "fixed_positive_flux_fraction")
            record["outside_active_positive_flux_fraction"] = None if fraction is None else 1.0 - fraction
            enriched_fp.append(record)

        for sample in unique_onsets:
            if sample < FFT_WINDOW or sample + FFT_WINDOW >= audio.frame_count:
                continue
            # Exclude notes starting exactly here from the active-state explanation.
            active_before = tuple(note for note in notes if note.onset_sample < sample < note.offset_sample)
            features = _spectral_features(normalized, sample, active_before, contour_index, round(0.050 * SAMPLE_RATE))
            fraction = features.get("fixed_positive_flux_fraction")
            record = {
                "member": member,
                "arrangement": arrangement,
                "sample": int(sample),
                "model_onset_score": None,
                "signed_nearest_onset_ms": 0.0,
                "previous_onset_age_ms": _ms(_previous_distance(sample - 1, unique_onsets)),
                "next_onset_distance_ms": _ms(_next_distance(sample + 1, unique_onsets)),
                "outside_active_positive_flux_fraction": None if fraction is None else 1.0 - float(fraction),
                **features,
            }
            true_records.append(record)
        print(f"audited {ordinal}/{len(members)}: {member}")

    controls = []
    for source in control_records:
        record = dict(source)
        record["arrangement"] = record.get("arrangement") or _arrangement(record["member"])
        fraction = _finite(record, "fixed_positive_flux_fraction")
        record["outside_active_positive_flux_fraction"] = None if fraction is None else 1.0 - fraction
        controls.append(record)

    populations = {}
    for name, records in (("false_positive", enriched_fp), ("true_onset", true_records), ("matched_control", controls)):
        populations[name] = {
            "global": _summarize(records),
            "solo": _summarize([record for record in records if record["arrangement"] == "solo"]),
            "comp": _summarize([record for record in records if record["arrangement"] == "comp"]),
        }

    solo_fp = [record for record in enriched_fp if record["arrangement"] == "solo"]
    comp_fp = [record for record in enriched_fp if record["arrangement"] == "comp"]
    solo_true = [record for record in true_records if record["arrangement"] == "solo"]
    comp_true = [record for record in true_records if record["arrangement"] == "comp"]

    result = {
        "schema_version": 1,
        "source_fp_audit": str(args.fp_audit),
        "members": list(members),
        "configuration": {
            "sample_rate": SAMPLE_RATE,
            "fft_window_samples": FFT_WINDOW,
            "true_onset_active_state_excludes_notes_starting_at_query": True,
        },
        "populations": populations,
        "rank_separation": {
            "solo_fp_vs_solo_true": _rank_separation(solo_fp, solo_true),
            "solo_fp_vs_comp_true": _rank_separation(solo_fp, comp_true),
            "comp_fp_vs_comp_true": _rank_separation(comp_fp, comp_true),
        },
        "single_metric_operating_points": _operating_points(solo_fp, comp_true, solo_true),
        "recent_onset_plus_active_family_rules": _combined_rules(solo_fp, comp_true, solo_true),
        "interpretation_guard": (
            "This audit identifies discriminative causal evidence; it does not relabel GuitarSet. "
            "True-onset features deliberately describe only the acoustic state already active before the onset."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = create_argument_parser().parse_args(argv)
    result = run(args)
    compact = {
        "populations": result["populations"],
        "rank_separation": result["rank_separation"],
        "single_metric_operating_points": result["single_metric_operating_points"],
        "recent_onset_plus_active_family_rules": result["recent_onset_plus_active_family_rules"][:10],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
