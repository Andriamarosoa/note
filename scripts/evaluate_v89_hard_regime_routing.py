"""V8.9 explicit hard regime routing over frozen V8.7/V8.8 experts.

Scientific goal: test whether V8.8's strong isolated-vs-cluster router should be
used as an actual routing decision rather than a soft mixture. No new acoustic
model is trained. The complete V8.4 -> V8.6 -> V8.7 -> V8.8 stack is frozen.

For each candidate:
  router < 0.5  -> V8.7 causal-memory birth score (isolated/retrigger expert)
  router >= 0.5 -> V8.8 cluster_birth expert (cluster/strum expert)

The router cutoff is the fixed binary classifier decision boundary 0.5; it is
not tuned. A single final onset-retention threshold is calibrated on six fresh
train-only tracks that were not among the 30 members used to fit V8.6/V8.7/V8.8.
Locked12 is evaluated once. Runtime inputs are audio-derived only. Offset is
never executed or modified, and no future candidate context is added.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
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
from scripts.audit_v84_solo_comp_onsets import _auc
from scripts.evaluate_boundaries import _count_metrics, match_boundaries
from scripts.evaluate_v8_boundaries import DEFAULT_SEED, DEFAULT_VALIDATION_FRACTION, _arrangement, _reference_positions
from scripts.train_boundaries import decode_pcm16_mono_wav, split_tracks_by_group
from scripts.train_v86_state_transition_proposals import (
    BIRTH_CLASSES,
    MAX_HORIZON,
    TOLERANCE_MS,
    TOLERANCE_SAMPLES,
    _candidate_ceiling,
    _predict_score_tracks,
    _records,
)
from scripts.train_v87_causal_candidate_memory import (
    _encode_records,
    _load_v86_encoder,
    _sequence_arrays,
)
from scripts.train_v88_regime_moe import (
    _aggregate,
    _build_moe_head,
    _encode_v87,
    _feature_matrix,
    _load_v87_encoder,
    _predictions,
    _regime_targets,
    _retained_predictions,
)

ROUTER_CUTOFF = 0.5
CALIBRATION_TRACKS = 6


class V89Error(RuntimeError):
    pass


def _load_v88(weights_path: Path):
    model = _build_moe_head()
    model.load_weights(weights_path)
    model.trainable = False
    return model


def _represent(tracks, base_model: Path, floor: float, v86_encoder, v87_encoder, v88_model):
    score_by_member = _predict_score_tracks(base_model, tracks)
    records = _records(tracks, score_by_member, floor)
    emb86, prob86 = _encode_records(v86_encoder, records)
    sequences, _ = _sequence_arrays(records, emb86, prob86)
    hidden87, prob87 = _encode_v87(v87_encoder, sequences)
    x88 = _feature_matrix(records, emb86, prob86, hidden87, prob87)
    out88 = _predictions(v88_model, x88)
    return score_by_member, records, prob87, out88


def _hard_routed_scores(prob87: np.ndarray, out88: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    isolated = np.asarray(prob87[:, BIRTH_CLASSES[0]] + prob87[:, BIRTH_CLASSES[1]], dtype=np.float32)
    cluster = np.asarray(out88["cluster_birth"], dtype=np.float32).reshape(-1)
    router = np.asarray(out88["cluster_router"], dtype=np.float32).reshape(-1)
    choose_cluster = router >= ROUTER_CUTOFF
    routed = np.where(choose_cluster, cluster, isolated).astype(np.float32)
    return routed, router, isolated, cluster


def _evaluate(records, scores, tracks, threshold: float) -> dict:
    retained = _retained_predictions(records, scores, threshold)
    return {
        key: _aggregate(tracks, retained, arrangement)
        for key, arrangement in (("global", None), ("solo", "solo"), ("comp", "comp"))
    }


def _calibrate(records, scores, tracks) -> dict:
    sweep = []
    for threshold in np.arange(0.10, 0.901, 0.025):
        metrics = _evaluate(records, scores, tracks, float(threshold))
        macro = (metrics["solo"]["f1"] + metrics["comp"]["f1"]) / 2.0
        sweep.append({"threshold": float(round(threshold, 6)), "macro_f1": float(macro), "metrics": metrics})
    best = max(
        sweep,
        key=lambda row: (
            row["macro_f1"],
            row["metrics"]["global"]["f1"],
            -abs(row["threshold"] - 0.5),
        ),
    )
    return {
        "selection_rule": "single global onset threshold maximizing macro(solo,comp) F1 on six fresh train-only calibration tracks",
        "best": best,
        "sweep": sweep,
    }


def _candidate_auc(records, scores, arrangement: Optional[str]) -> Optional[float]:
    selected = [
        (float(score), int(record["birth"]))
        for record, score in zip(records, scores)
        if arrangement is None or record["arrangement"] == arrangement
    ]
    positives = [score for score, label in selected if label]
    negatives = [score for score, label in selected if not label]
    return _auc(positives, negatives)


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    positives = [float(score) for score, label in zip(scores, labels) if int(label)]
    negatives = [float(score) for score, label in zip(scores, labels) if not int(label)]
    return _auc(positives, negatives)


def _route_summary(records, router: np.ndarray) -> dict:
    result = {}
    for arrangement in ("global", "solo", "comp"):
        indices = [
            i for i, record in enumerate(records)
            if arrangement == "global" or record["arrangement"] == arrangement
        ]
        if not indices:
            result[arrangement] = {"count": 0, "cluster_route_fraction": None}
            continue
        cluster = sum(float(router[i]) >= ROUTER_CUTOFF for i in indices)
        result[arrangement] = {
            "count": len(indices),
            "cluster_routed": int(cluster),
            "isolated_routed": int(len(indices) - cluster),
            "cluster_route_fraction": float(cluster / len(indices)),
        }
    return result


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate V8.9 explicit hard regime routing.")
    parser.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--v86-weights", type=Path, required=True)
    parser.add_argument("--v86-report", type=Path, required=True)
    parser.add_argument("--v87-weights", type=Path, required=True)
    parser.add_argument("--v87-report", type=Path, required=True)
    parser.add_argument("--v88-weights", type=Path, required=True)
    parser.add_argument("--v88-report", type=Path, required=True)
    parser.add_argument("--train-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def run(args) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    indexed = tuple(track for track in index_guitarset(args.dataset_dir) if track.player_id in ALLOWED_PLAYERS)
    by_member = {track.annotation_member: track for track in indexed}
    train_split, locked_validation = split_tracks_by_group(
        indexed, validation_fraction=DEFAULT_VALIDATION_FRACTION, seed=DEFAULT_SEED
    )
    locked12 = tuple(locked_validation[:12])
    locked_members = {track.annotation_member for track in locked_validation}

    audit = json.loads(args.train_audit.read_text(encoding="utf-8"))
    source_members = list(audit["scope"]["members"])[:30]
    source_set = set(source_members)
    if source_set & locked_members:
        raise V89Error("frozen source members overlap locked validation")

    # Fresh train-only calibration tracks: never used by the V8.6/V8.7/V8.8
    # experimental heads. Deterministic order avoids any locked12-driven choice.
    calibration_candidates = sorted(
        (track for track in train_split if track.annotation_member not in source_set),
        key=lambda track: track.annotation_member,
    )
    calibration_tracks = tuple(calibration_candidates[:CALIBRATION_TRACKS])
    if len(calibration_tracks) != CALIBRATION_TRACKS:
        raise V89Error("not enough fresh train-only calibration tracks")
    if {t.annotation_member for t in calibration_tracks} & (source_set | locked_members):
        raise V89Error("calibration leakage")

    v86_report = json.loads(args.v86_report.read_text(encoding="utf-8"))
    v87_report = json.loads(args.v87_report.read_text(encoding="utf-8"))
    v88_report = json.loads(args.v88_report.read_text(encoding="utf-8"))
    floor = float(v86_report["configuration"]["candidate_floor"])
    if v86_report["configuration"].get("candidate_floor_selected_on_locked_validation") is not False:
        raise V89Error("V8.6 floor leakage")
    if v87_report["configuration"].get("retain_threshold_selected_on_locked_validation") is not False:
        raise V89Error("V8.7 threshold leakage")
    if v88_report["configuration"].get("retain_threshold_selected_on_locked_validation") is not False:
        raise V89Error("V8.8 threshold leakage")

    _v86_model, v86_encoder = _load_v86_encoder(args.v86_weights)
    _v87_model, v87_encoder = _load_v87_encoder(args.v87_weights)
    v88_model = _load_v88(args.v88_weights)

    print("calibrating one global V8.9 threshold on fresh train-only tracks")
    _, calibration_records, calibration_prob87, calibration_out88 = _represent(
        calibration_tracks, args.base_model, floor, v86_encoder, v87_encoder, v88_model
    )
    calibration_scores, calibration_router, _, _ = _hard_routed_scores(
        calibration_prob87, calibration_out88
    )
    calibration = _calibrate(calibration_records, calibration_scores, calibration_tracks)
    threshold = float(calibration["best"]["threshold"])

    print("evaluating once on frozen locked12")
    locked_score_streams, locked_records, locked_prob87, locked_out88 = _represent(
        locked12, args.base_model, floor, v86_encoder, v87_encoder, v88_model
    )
    locked_scores, locked_router, locked_isolated, locked_cluster = _hard_routed_scores(
        locked_prob87, locked_out88
    )
    locked_ceiling = _candidate_ceiling(locked12, locked_score_streams, floor)
    locked_metrics = _evaluate(locked_records, locked_scores, locked12, threshold)
    candidate_aucs = {
        key: _candidate_auc(locked_records, locked_scores, arrangement)
        for key, arrangement in (("global", None), ("solo", "solo"), ("comp", "comp"))
    }
    route_targets, _ = _regime_targets(locked_records, locked12)
    router_auc = _binary_auc(locked_router, route_targets[:, 0])

    report = {
        "schema_version": 1,
        "architecture": {
            "name": "V8.9 explicit hard regime-routed onset composition",
            "base_trainable": False,
            "v86_encoder_trainable": False,
            "v87_expert_trainable": False,
            "v88_router_cluster_expert_trainable": False,
            "new_trainable_parameters": 0,
            "offset_stream_executed": False,
            "offset_weights_modified": False,
            "runtime_inputs_use_annotations": False,
            "future_candidate_context": False,
            "router_cutoff": ROUTER_CUTOFF,
            "router_cutoff_calibrated": False,
            "isolated_expert": "frozen V8.7 birth probability",
            "cluster_expert": "frozen V8.8 cluster_birth probability",
            "routing": "hard: router<0.5 isolated expert, router>=0.5 cluster expert",
            "maximum_verification_delay_samples": MAX_HORIZON,
            "maximum_verification_delay_ms": MAX_HORIZON * 1000.0 / SAMPLE_RATE,
        },
        "configuration": {
            "candidate_floor": floor,
            "candidate_floor_reused_from_train_only_v86": True,
            "retain_threshold": threshold,
            "retain_threshold_selected_on_locked_validation": False,
            "single_global_runtime_threshold": True,
            "matching_tolerance_ms": TOLERANCE_MS,
        },
        "data": {
            "frozen_head_source_members": source_members,
            "calibration_members": [track.annotation_member for track in calibration_tracks],
            "calibration_members_disjoint_source": True,
            "calibration_members_disjoint_locked_validation": True,
            "locked_validation_members": [track.annotation_member for track in locked12],
        },
        "calibration": calibration,
        "locked12": {
            "candidate_ceiling": locked_ceiling,
            "candidate_auc_birth_high": candidate_aucs,
            "router_cluster_auc": router_auc,
            "route_summary": _route_summary(locked_records, locked_router),
            "metrics": locked_metrics,
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "candidate_floor": floor,
        "candidate_ceiling_recall": locked_ceiling["recall"],
        "retain_threshold": threshold,
        "router_cluster_auc": router_auc,
        "candidate_auc": candidate_aucs,
        "route_summary": report["locked12"]["route_summary"],
        "locked_metrics": locked_metrics,
    }, indent=2, sort_keys=True))
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = create_argument_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
