"""Measure how much of the V9 local-cluster oracle is realizable by frozen V8.8.

This is a train-only diagnostic. It does not train or tune a model and never
executes locked validation. The V8.8 local-cardinality probabilities are
aggregated over runtime-only candidate clusters using several fixed rules.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
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
from scripts.evaluate_boundaries import milliseconds_to_samples
from scripts.evaluate_v8_boundaries import DEFAULT_SEED, DEFAULT_VALIDATION_FRACTION
from scripts.train_boundaries import split_tracks_by_group
from scripts.train_v86_state_transition_proposals import _candidate_ceiling
from scripts.train_v87_causal_candidate_memory import _load_v86_encoder
from scripts.train_v88_regime_moe import (
    _fused_scores,
    _load_v87_encoder,
    _regime_targets,
    _retained_predictions,
)
from scripts.evaluate_v89_hard_regime_routing import _load_v88, _represent
from scripts.evaluate_v90_cluster_oracles import (
    ANALYSIS_TRACKS,
    LOCAL_RADIUS_MS,
    V90OracleError,
    _aggregate,
    _assign_refs,
    _metrics,
    _references,
    _slots,
    _topk,
)

WINDOWS_MS = (20.0, 30.0, 40.0)
CARDINALITY_CLASSES = 4


def _clusters_window(records: Sequence[dict], window_ms: float) -> List[dict]:
    window_samples = milliseconds_to_samples(window_ms)
    by_member: Dict[str, List[int]] = defaultdict(list)
    for i, record in enumerate(records):
        by_member[record["member"]].append(i)
    result = []
    for member in sorted(by_member):
        indices = sorted(by_member[member], key=lambda i: (int(records[i]["sample"]), i))
        current: List[int] = []
        first: Optional[int] = None
        for i in indices:
            sample = int(records[i]["sample"])
            if first is None or sample - first <= window_samples:
                if first is None:
                    first = sample
                current.append(i)
                continue
            result.append({
                "member": member,
                "arrangement": records[current[0]]["arrangement"],
                "indices": tuple(current),
            })
            current = [i]
            first = sample
        if current:
            result.append({
                "member": member,
                "arrangement": records[current[0]]["arrangement"],
                "indices": tuple(current),
            })
    return result


def _truth_classes(clusters, assigned) -> np.ndarray:
    return np.asarray([min(3, len(assigned.get(cid, ()))) for cid in range(len(clusters))], dtype=np.int32)


def _aggregate_cardinality_probabilities(
    cluster: dict,
    probabilities: np.ndarray,
    fused_scores: np.ndarray,
    method: str,
) -> np.ndarray:
    indices = np.asarray(cluster["indices"], dtype=np.int64)
    p = np.asarray(probabilities[indices], dtype=np.float64)
    scores = np.asarray(fused_scores[indices], dtype=np.float64)
    if method == "mean_prob":
        agg = np.mean(p, axis=0)
    elif method == "score_weighted_prob":
        weights = np.maximum(scores, 1e-6)
        agg = np.sum(p * weights[:, None], axis=0) / np.sum(weights)
    elif method == "peak_fused_candidate":
        local = int(np.argmax(scores))
        agg = p[local]
    elif method == "peak_nonzero_candidate":
        nonzero = 1.0 - p[:, 0]
        local = int(np.argmax(nonzero))
        agg = p[local]
    else:
        raise ValueError(f"unknown cardinality aggregation method {method}")
    total = float(np.sum(agg))
    return (agg / total if total > 0.0 else np.asarray([1.0, 0.0, 0.0, 0.0])).astype(np.float64)


def _predicted_classes(clusters, probabilities, fused_scores, method: str) -> Tuple[np.ndarray, np.ndarray]:
    aggregated = np.stack([
        _aggregate_cardinality_probabilities(cluster, probabilities, fused_scores, method)
        for cluster in clusters
    ]) if clusters else np.zeros((0, CARDINALITY_CLASSES), dtype=np.float64)
    return np.argmax(aggregated, axis=1).astype(np.int32), aggregated


def _prediction_map_from_k(clusters, records, scores, k_values: np.ndarray):
    retained: Dict[str, List[int]] = defaultdict(list)
    for cluster, k in zip(clusters, k_values):
        slots = _slots(cluster, records, scores)
        selected = _topk(slots, int(k))
        retained[cluster["member"]].extend(slot["sample"] for slot in selected)
    return {member: tuple(sorted(values)) for member, values in retained.items()}


def _cardinality_diagnostics(truth: np.ndarray, pred: np.ndarray) -> dict:
    confusion = np.zeros((CARDINALITY_CLASSES, CARDINALITY_CLASSES), dtype=np.int64)
    for y, p in zip(truth, pred):
        confusion[int(y), int(p)] += 1
    per_class = {}
    for k in range(CARDINALITY_CLASSES):
        total = int(np.sum(confusion[k]))
        per_class[str(k)] = {
            "count": total,
            "accuracy": float(confusion[k, k] / total) if total else None,
            "mean_predicted_class": float(np.average(np.arange(CARDINALITY_CLASSES), weights=confusion[k])) if total else None,
        }
    mask_birth = truth > 0
    mask_poly = truth >= 2
    return {
        "accuracy": float(np.mean(pred == truth)) if len(truth) else 0.0,
        "birth_cluster_accuracy": float(np.mean(pred[mask_birth] == truth[mask_birth])) if np.any(mask_birth) else None,
        "poly_cluster_accuracy": float(np.mean(pred[mask_poly] == truth[mask_poly])) if np.any(mask_poly) else None,
        "mean_absolute_class_error": float(np.mean(np.abs(pred.astype(np.int64) - truth.astype(np.int64)))) if len(truth) else 0.0,
        "predicted_zero_fraction": float(np.mean(pred == 0)) if len(pred) else 0.0,
        "confusion_true_rows_pred_columns": confusion.tolist(),
        "per_true_class": per_class,
    }


def _cluster_summary(clusters, assigned, records) -> dict:
    truth = _truth_classes(clusters, assigned)
    sizes = [len(cluster["indices"]) for cluster in clusters]
    widths = []
    for cluster in clusters:
        samples = [int(records[i]["sample"]) for i in cluster["indices"]]
        widths.append(max(samples) - min(samples))
    counts = Counter(int(x) for x in truth)
    return {
        "cluster_count": len(clusters),
        "clusters_with_births": int(np.sum(truth > 0)),
        "true_cardinality_class_counts": {str(k): int(counts[k]) for k in range(CARDINALITY_CLASSES)},
        "candidate_records_per_cluster_mean": float(np.mean(sizes)) if sizes else 0.0,
        "candidate_records_per_cluster_p90": float(np.percentile(sizes, 90)) if sizes else 0.0,
        "cluster_width_ms_mean": float(np.mean(widths) * 1000.0 / SAMPLE_RATE) if widths else 0.0,
        "cluster_width_ms_p90": float(np.percentile(widths, 90) * 1000.0 / SAMPLE_RATE) if widths else 0.0,
    }


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    p.add_argument("--base-model", type=Path, required=True)
    p.add_argument("--v86-weights", type=Path, required=True)
    p.add_argument("--v86-report", type=Path, required=True)
    p.add_argument("--v87-weights", type=Path, required=True)
    p.add_argument("--v87-report", type=Path, required=True)
    p.add_argument("--v88-weights", type=Path, required=True)
    p.add_argument("--v88-report", type=Path, required=True)
    p.add_argument("--train-audit", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def run(args):
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    indexed = tuple(t for t in index_guitarset(args.dataset_dir) if t.player_id in ALLOWED_PLAYERS)
    train_split, locked_validation = split_tracks_by_group(
        indexed, validation_fraction=DEFAULT_VALIDATION_FRACTION, seed=DEFAULT_SEED
    )
    locked_members = {t.annotation_member for t in locked_validation}
    audit = json.loads(args.train_audit.read_text())
    source_members = list(audit["scope"]["members"])[:30]
    source_set = set(source_members)
    fresh = sorted((t for t in train_split if t.annotation_member not in source_set), key=lambda t: t.annotation_member)
    tracks = tuple(fresh[:ANALYSIS_TRACKS])
    if len(tracks) != ANALYSIS_TRACKS:
        raise V90OracleError("not enough fresh analysis tracks")
    analysis_members = {t.annotation_member for t in tracks}
    if analysis_members & (source_set | locked_members):
        raise V90OracleError("analysis leakage")

    r86 = json.loads(args.v86_report.read_text())
    r87 = json.loads(args.v87_report.read_text())
    r88 = json.loads(args.v88_report.read_text())
    floor = float(r86["configuration"]["candidate_floor"])
    threshold = float(r88["configuration"]["retain_threshold"])
    if r86["configuration"].get("candidate_floor_selected_on_locked_validation") is not False:
        raise V90OracleError("V8.6 leakage")
    if r87["configuration"].get("retain_threshold_selected_on_locked_validation") is not False:
        raise V90OracleError("V8.7 leakage")
    if r88["configuration"].get("retain_threshold_selected_on_locked_validation") is not False:
        raise V90OracleError("V8.8 leakage")

    _, enc86 = _load_v86_encoder(args.v86_weights)
    _, enc87 = _load_v87_encoder(args.v87_weights)
    model88 = _load_v88(args.v88_weights)
    score_streams, records, _, out88 = _represent(
        tracks, args.base_model, floor, enc86, enc87, model88
    )
    scores = _fused_scores(out88)
    card_prob = np.asarray(out88["local_cardinality"], dtype=np.float64)
    refs = _references(tracks)
    per_candidate_route, per_candidate_truth = _regime_targets(records, tracks)
    per_candidate_pred = np.argmax(card_prob, axis=1).astype(np.int32)

    baseline = _metrics(tracks, _retained_predictions(records, scores, threshold))
    windows = {}
    methods = (
        "mean_prob",
        "score_weighted_prob",
        "peak_fused_candidate",
        "peak_nonzero_candidate",
    )
    for window_ms in WINDOWS_MS:
        clusters = _clusters_window(records, window_ms)
        assigned, assignment = _assign_refs(clusters, records, refs)
        truth = _truth_classes(clusters, assigned)
        oracle_exact = _metrics(
            tracks,
            _prediction_map_from_k(
                clusters, records, scores,
                np.asarray([len(assigned.get(cid, ())) for cid in range(len(clusters))], dtype=np.int32),
            ),
        )
        oracle_3plus = _metrics(
            tracks,
            _prediction_map_from_k(clusters, records, scores, truth),
        )
        realized = {}
        for method in methods:
            pred, aggregated = _predicted_classes(clusters, card_prob, scores, method)
            realized[method] = {
                "cardinality": _cardinality_diagnostics(truth, pred),
                "metrics": _metrics(tracks, _prediction_map_from_k(clusters, records, scores, pred)),
                "mean_confidence": float(np.mean(np.max(aggregated, axis=1))) if len(aggregated) else 0.0,
            }
        windows[str(int(window_ms))] = {
            "reference_assignment": assignment,
            "cluster_summary": _cluster_summary(clusters, assigned, records),
            "oracle_exact_k_current_ranking": oracle_exact,
            "oracle_3plus_current_ranking": oracle_3plus,
            "frozen_v88_cardinality_realization": realized,
        }

    report = {
        "schema_version": 1,
        "architecture": {
            "name": "V9.0 frozen-V8.8 cluster-cardinality realization study",
            "new_trainable_parameters": 0,
            "all_models_frozen": True,
            "runtime_inputs_use_annotations": False,
            "annotations_used_only_for_diagnostic_truth_and_metrics": True,
            "locked_validation_executed": False,
            "offset_stream_executed": False,
            "future_candidate_context": False,
        },
        "configuration": {
            "candidate_floor": floor,
            "frozen_v88_retain_threshold": threshold,
            "analysis_track_count": ANALYSIS_TRACKS,
            "windows_ms": list(WINDOWS_MS),
            "reference_assignment_radius_ms": LOCAL_RADIUS_MS,
            "aggregation_methods_fixed_before_run": list(methods),
        },
        "data": {
            "analysis_members": [t.annotation_member for t in tracks],
            "analysis_members_disjoint_source": not bool(analysis_members & source_set),
            "analysis_members_disjoint_locked_validation": not bool(analysis_members & locked_members),
        },
        "baseline_v88_frozen_threshold": baseline,
        "candidate_ceiling": _candidate_ceiling(tracks, score_streams, floor),
        "per_candidate_v88_cardinality": {
            "accuracy": float(np.mean(per_candidate_pred == per_candidate_truth)) if len(records) else 0.0,
            "count": len(records),
            "router_target_positive_fraction": float(np.mean(per_candidate_route[:, 0])) if len(records) else 0.0,
        },
        "windows": windows,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: Optional[Sequence[str]] = None):
    args = parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
