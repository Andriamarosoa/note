"""Train-only oracle study for a structured local-cluster V9 decoder.

No model is trained. The complete V8.4 -> V8.6 -> V8.7 -> V8.8 stack is
frozen. Candidate clusters are built from runtime candidates only; annotations
are used only to provide oracle local cardinality / oracle subset selection and
final metrics. Locked validation is never evaluated by this script.
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
from scripts.evaluate_boundaries import match_boundaries, milliseconds_to_samples
from scripts.evaluate_v8_boundaries import DEFAULT_SEED, DEFAULT_VALIDATION_FRACTION, _reference_positions
from scripts.train_boundaries import decode_pcm16_mono_wav, split_tracks_by_group
from scripts.train_v86_state_transition_proposals import TOLERANCE_MS, TOLERANCE_SAMPLES, _candidate_ceiling
from scripts.train_v87_causal_candidate_memory import _load_v86_encoder
from scripts.train_v88_regime_moe import _aggregate, _fused_scores, _load_v87_encoder, _retained_predictions
from scripts.evaluate_v89_hard_regime_routing import _load_v88, _represent

LOCAL_RADIUS_MS = 20.0
CLUSTER_WINDOW_MS = 40.0
LOCAL_RADIUS_SAMPLES = milliseconds_to_samples(LOCAL_RADIUS_MS)
CLUSTER_WINDOW_SAMPLES = milliseconds_to_samples(CLUSTER_WINDOW_MS)
ANALYSIS_TRACKS = 12


class V90OracleError(RuntimeError):
    pass


def _references(tracks) -> Dict[str, Tuple[int, ...]]:
    out = {}
    for track in tracks:
        audio = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        refs, _ = _reference_positions(track, audio.frame_count)
        out[track.annotation_member] = tuple(int(x) for x in refs)
    return out


def _clusters(records: Sequence[dict]) -> List[dict]:
    by_member: Dict[str, List[int]] = defaultdict(list)
    for i, record in enumerate(records):
        by_member[record["member"]].append(i)
    result = []
    for member in sorted(by_member):
        indices = sorted(by_member[member], key=lambda i: (int(records[i]["sample"]), i))
        current: List[int] = []
        first = None
        for i in indices:
            sample = int(records[i]["sample"])
            if first is None or sample - first <= CLUSTER_WINDOW_SAMPLES:
                if first is None:
                    first = sample
                current.append(i)
                continue
            result.append({"member": member, "arrangement": records[current[0]]["arrangement"], "indices": tuple(current)})
            current = [i]
            first = sample
        if current:
            result.append({"member": member, "arrangement": records[current[0]]["arrangement"], "indices": tuple(current)})
    return result


def _distance(reference: int, cluster: dict, records: Sequence[dict]) -> int:
    return min(abs(int(records[i]["sample"]) - int(reference)) for i in cluster["indices"])


def _assign_refs(clusters, records, refs_by_member):
    ids_by_member: Dict[str, List[int]] = defaultdict(list)
    for cid, cluster in enumerate(clusters):
        ids_by_member[cluster["member"]].append(cid)
    assigned: Dict[int, List[int]] = defaultdict(list)
    total = used = 0
    for member, refs in refs_by_member.items():
        for ref in refs:
            total += 1
            choices = []
            for cid in ids_by_member.get(member, ()): 
                dist = _distance(ref, clusters[cid], records)
                if dist <= LOCAL_RADIUS_SAMPLES:
                    choices.append((dist, cid))
            if choices:
                _, cid = min(choices)
                assigned[cid].append(int(ref))
                used += 1
    return {cid: tuple(sorted(values)) for cid, values in assigned.items()}, {
        "reference_count": total,
        "assigned_reference_count": used,
        "unassigned_reference_count": total - used,
        "assigned_fraction": used / total if total else None,
    }


def _slots(cluster, records, scores):
    out = []
    ordinal = 0
    for i in cluster["indices"]:
        for copy in range(max(1, int(records[i]["count"]))):
            out.append({"sample": int(records[i]["sample"]), "score": float(scores[i]), "ordinal": ordinal})
            ordinal += 1
    return out


def _topk(slots, k):
    return sorted(slots, key=lambda s: (-s["score"], s["sample"], s["ordinal"]))[: min(k, len(slots))]


def _oracle_subset(slots, refs, k):
    if k <= 0 or not slots:
        return []
    pairs = list(match_boundaries(tuple(refs), tuple(s["sample"] for s in slots), TOLERANCE_SAMPLES))
    pairs.sort(key=lambda p: (abs(p[1] - p[0]), p[0], p[1]))
    by_sample: Dict[int, List[int]] = defaultdict(list)
    for i, slot in enumerate(slots):
        by_sample[slot["sample"]].append(i)
    for values in by_sample.values():
        values.sort(key=lambda i: (-slots[i]["score"], slots[i]["ordinal"]))
    chosen: List[int] = []
    for _, pred in pairs:
        if len(chosen) >= k:
            break
        candidates = by_sample[pred]
        while candidates and candidates[0] in chosen:
            candidates.pop(0)
        if candidates:
            chosen.append(candidates.pop(0))
    if len(chosen) < min(k, len(slots)):
        chosen_set = set(chosen)
        rest = [i for i in range(len(slots)) if i not in chosen_set]
        rest.sort(key=lambda i: (-slots[i]["score"], slots[i]["sample"], slots[i]["ordinal"]))
        chosen.extend(rest[: min(k, len(slots)) - len(chosen)])
    return [slots[i] for i in chosen]


def _prediction_map(clusters, records, scores, assigned, *, cap3: bool, oracle_subset: bool):
    retained: Dict[str, List[int]] = defaultdict(list)
    for cid, cluster in enumerate(clusters):
        refs = assigned.get(cid, ())
        k = min(3, len(refs)) if cap3 else len(refs)
        slots = _slots(cluster, records, scores)
        selected = _oracle_subset(slots, refs, k) if oracle_subset else _topk(slots, k)
        retained[cluster["member"]].extend(slot["sample"] for slot in selected)
    return {member: tuple(sorted(values)) for member, values in retained.items()}


def _metrics(tracks, predictions):
    return {key: _aggregate(tracks, predictions, arrangement) for key, arrangement in (("global", None), ("solo", "solo"), ("comp", "comp"))}


def _cluster_summary(clusters, assigned, records):
    counts = Counter(min(3, len(assigned.get(cid, ()))) for cid in range(len(clusters)))
    widths = []
    sizes = []
    for cluster in clusters:
        samples = [int(records[i]["sample"]) for i in cluster["indices"]]
        widths.append(max(samples) - min(samples))
        sizes.append(len(samples))
    return {
        "cluster_count": len(clusters),
        "clusters_with_births": sum(bool(assigned.get(cid)) for cid in range(len(clusters))),
        "true_cardinality_class_counts": {str(i): int(counts[i]) for i in range(4)},
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
    train_split, locked_validation = split_tracks_by_group(indexed, validation_fraction=DEFAULT_VALIDATION_FRACTION, seed=DEFAULT_SEED)
    locked_members = {t.annotation_member for t in locked_validation}
    audit = json.loads(args.train_audit.read_text())
    source_members = list(audit["scope"]["members"])[:30]
    source_set = set(source_members)
    if source_set & locked_members:
        raise V90OracleError("source/locked overlap")
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
    score_streams, records, _, out88 = _represent(tracks, args.base_model, floor, enc86, enc87, model88)
    scores = _fused_scores(out88)
    clusters = _clusters(records)
    refs = _references(tracks)
    assigned, assignment = _assign_refs(clusters, records, refs)

    baseline = _metrics(tracks, _retained_predictions(records, scores, threshold))
    variants = {}
    for name, cap3, oracle in (
        ("oracle_exact_k_current_ranking", False, False),
        ("oracle_3plus_current_ranking", True, False),
        ("oracle_exact_k_best_subset", False, True),
        ("oracle_3plus_best_subset", True, True),
    ):
        variants[name] = _metrics(tracks, _prediction_map(clusters, records, scores, assigned, cap3=cap3, oracle_subset=oracle))

    report = {
        "schema_version": 1,
        "architecture": {
            "name": "V9.0 pre-training structured local-cluster oracle study",
            "base_trainable": False,
            "v86_encoder_trainable": False,
            "v87_encoder_trainable": False,
            "v88_trainable": False,
            "new_trainable_parameters": 0,
            "runtime_inputs_use_annotations": False,
            "oracle_annotations_used_only_for_analysis": True,
            "offset_stream_executed": False,
            "offset_weights_modified": False,
            "future_candidate_context": False,
        },
        "configuration": {
            "candidate_floor": floor,
            "frozen_v88_retain_threshold": threshold,
            "cluster_window_ms": CLUSTER_WINDOW_MS,
            "oracle_reference_assignment_radius_ms": LOCAL_RADIUS_MS,
            "matching_tolerance_ms": TOLERANCE_MS,
            "analysis_track_count": ANALYSIS_TRACKS,
            "analysis_split_selected_without_locked_validation": True,
            "locked_validation_executed": False,
            "candidate_units_expand_existing_multiplicity": True,
        },
        "data": {
            "analysis_members": [t.annotation_member for t in tracks],
            "frozen_head_source_members": source_members,
            "analysis_members_disjoint_source": not bool(analysis_members & source_set),
            "analysis_members_disjoint_locked_validation": not bool(analysis_members & locked_members),
        },
        "baseline_v88_frozen_threshold": baseline,
        "candidate_ceiling": _candidate_ceiling(tracks, score_streams, floor),
        "reference_assignment": assignment,
        "cluster_summary": _cluster_summary(clusters, assigned, records),
        "variants": variants,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: Optional[Sequence[str]] = None):
    args = parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
