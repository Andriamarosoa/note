"""Audit why V10.1 exact polyphonic cardinality remains low.

This is a diagnostic-only script. It does not train, tune, or select a model on
locked12. It reuses frozen V10 caches and the already-trained V10.1 weights to
separate data/label limitations from representation and count-decoder errors.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from causal_note.guitarset import ALLOWED_PLAYERS, SAMPLE_RATE, SLOT_COUNT, index_guitarset
from scripts.train_boundaries import group_stem
from scripts.train_v90_structured_cluster_cardinality import _cluster_data, _prediction_map, _represent_full
from scripts.train_v91_ordinal_cardinality import _dataset_split, _load_frozen_stack
from scripts.train_v92_string_factorized_cardinality import LOCAL_RADIUS_SAMPLES, _reconstruct_candidates, _slot_targets_for_runtime_clusters
from scripts.train_v100_spectral_string_slots import _load_spectral_caches, _spectral_maps_for_runtime
from scripts import train_v101_string_query_attention as v101


class AuditError(RuntimeError):
    pass


def _pattern(row) -> str:
    return "".join("1" if float(x) > 0.5 else "0" for x in row)


def _regime(member: str) -> str:
    return "comp" if "_comp" in member else "solo" if "_solo" in member else "unknown"


def _player(member: str) -> str:
    return member.split("_", 1)[0]


def _q(values, q):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(np.quantile(values, q)) if values else None


def _mean(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(np.mean(values)) if values else None


def _pair_features(midis: Sequence[float]):
    midis = [float(x) for x in midis]
    if len(midis) < 2:
        return {"pitch_spread": 0.0, "min_pitch_gap": None, "close_pair": False, "octave_related_pair": False}
    diffs = [abs(midis[i] - midis[j]) for i in range(len(midis)) for j in range(i + 1, len(midis))]
    close = any(d <= 2.0 for d in diffs)
    octave = any(abs(d - 12.0 * round(d / 12.0)) <= 0.5 for d in diffs)
    return {
        "pitch_spread": float(max(midis) - min(midis)),
        "min_pitch_gap": float(min(diffs)),
        "close_pair": bool(close),
        "octave_related_pair": bool(octave),
    }


def _assign_events(dataset_dir: Path, members: Sequence[str], candidate_samples: Sequence[np.ndarray]):
    indexed = tuple(t for t in index_guitarset(dataset_dir) if t.player_id in ALLOWED_PLAYERS)
    by_member_track = {t.annotation_member: t for t in indexed}
    by_member_rows: Dict[str, List[int]] = defaultdict(list)
    for cid, member in enumerate(members):
        by_member_rows[str(member)].append(cid)
    events: List[List[Tuple[int, int, float, int]]] = [[] for _ in members]
    assigned = 0
    unassigned = 0
    same_slot_collisions = 0
    for member, ids in by_member_rows.items():
        track = by_member_track.get(member)
        if track is None:
            raise AuditError(f"missing track {member}")
        occupied = set()
        for slot, onset, midi in v101._pitch_events(track):
            best = None
            for cid in ids:
                samples = candidate_samples[cid]
                if not len(samples):
                    continue
                dist = int(np.min(np.abs(samples - onset)))
                if dist <= LOCAL_RADIUS_SAMPLES and (best is None or dist < best[0]):
                    best = (dist, cid)
            if best is None:
                unassigned += 1
                continue
            dist, cid = best
            if (cid, slot) in occupied:
                same_slot_collisions += 1
            occupied.add((cid, slot))
            events[cid].append((int(slot), int(onset), float(midi), int(dist)))
            assigned += 1
    return events, {
        "assigned": assigned,
        "unassigned": unassigned,
        "assigned_fraction": assigned / (assigned + unassigned) if assigned + unassigned else None,
        "same_slot_collisions": same_slot_collisions,
    }


def _cluster_features(members, candidate_samples, events, slot_truth, group_map):
    rows = []
    for cid, (member, samples, ev, slots) in enumerate(zip(members, candidate_samples, events, slot_truth)):
        member = str(member)
        onsets = [e[1] for e in ev]
        midis = [e[2] for e in ev]
        dists = [e[3] for e in ev]
        pair = _pair_features(midis)
        rows.append({
            "cid": cid,
            "member": member,
            "player": _player(member),
            "regime": _regime(member),
            "group": group_map.get(member, member),
            "k": int(np.sum(np.asarray(slots) > 0.5)),
            "pattern": _pattern(slots),
            "candidate_count": int(len(samples)),
            "candidate_width_ms": float((int(np.max(samples)) - int(np.min(samples))) * 1000.0 / SAMPLE_RATE) if len(samples) else 0.0,
            "event_count": len(ev),
            "event_span_ms": float((max(onsets) - min(onsets)) * 1000.0 / SAMPLE_RATE) if len(onsets) >= 2 else 0.0,
            "nearest_candidate_dist_mean_ms": float(np.mean(dists) * 1000.0 / SAMPLE_RATE) if dists else None,
            "nearest_candidate_dist_max_ms": float(np.max(dists) * 1000.0 / SAMPLE_RATE) if dists else None,
            **pair,
        })
    return rows


def _concentration(rows, k):
    subset = [r for r in rows if r["k"] == k]
    counts = Counter(r["member"] for r in subset)
    group_counts = Counter(r["group"] for r in subset)
    total = len(subset)
    shares = sorted((c / total for c in counts.values()), reverse=True) if total else []
    hhi = sum(s * s for s in shares)
    patterns = Counter(r["pattern"] for r in subset)
    return {
        "clusters": total,
        "unique_tracks": len(counts),
        "unique_groups": len(group_counts),
        "unique_players": len({r["player"] for r in subset}),
        "comp_fraction": _mean([r["regime"] == "comp" for r in subset]),
        "top1_track_share": shares[0] if shares else None,
        "top5_track_share": sum(shares[:5]) if shares else None,
        "track_hhi": hhi if shares else None,
        "string_pattern_count": len(patterns),
        "top_string_pattern_share": max(patterns.values()) / total if total and patterns else None,
        "median_candidates": _q([r["candidate_count"] for r in subset], 0.5),
        "median_candidate_width_ms": _q([r["candidate_width_ms"] for r in subset], 0.5),
        "median_event_span_ms": _q([r["event_span_ms"] for r in subset], 0.5),
        "p90_event_span_ms": _q([r["event_span_ms"] for r in subset], 0.9),
        "event_span_gt_10ms_fraction": _mean([r["event_span_ms"] > 10.0 for r in subset]),
        "event_span_gt_20ms_fraction": _mean([r["event_span_ms"] > 20.0 for r in subset]),
        "event_span_gt_30ms_fraction": _mean([r["event_span_ms"] > 30.0 for r in subset]),
        "median_nearest_candidate_dist_ms": _q([r["nearest_candidate_dist_mean_ms"] for r in subset], 0.5),
        "candidate_count_lt_k_fraction": _mean([r["candidate_count"] < k for r in subset]),
        "close_pitch_pair_fraction": _mean([r["close_pair"] for r in subset]),
        "octave_related_pair_fraction": _mean([r["octave_related_pair"] for r in subset]),
        "median_pitch_spread": _q([r["pitch_spread"] for r in subset], 0.5),
    }


def _distribution(rows):
    return {str(k): _concentration(rows, k) for k in range(SLOT_COUNT + 1)}


def _model_error_report(rows, truth_k, slot_truth, slot_probs, conditionals):
    modes = {
        "ordinal_cumulative_050": v101._decode(slot_probs, conditionals, "ordinal_cumulative_050"),
        "slot_threshold_050": v101._decode(slot_probs, conditionals, "slot_threshold_050"),
        "slot_expected_round": v101._decode(slot_probs, conditionals, "slot_expected_round"),
        "slot_poisson_binomial_argmax": v101._decode(slot_probs, conditionals, "slot_poisson_binomial_argmax"),
    }
    slot_vec = slot_probs >= 0.5
    truth_vec = np.asarray(slot_truth) > 0.5
    slot_vec_exact = np.all(slot_vec == truth_vec, axis=1)
    slot_count_exact = np.sum(slot_vec, axis=1) == truth_k
    poly = truth_k >= 2

    by_k = {}
    for k in range(2, SLOT_COUNT + 1):
        m = truth_k == k
        if not np.any(m):
            continue
        active_probs = slot_probs[m][truth_vec[m]]
        inactive_probs = slot_probs[m][~truth_vec[m]]
        by_k[str(k)] = {
            "clusters": int(np.sum(m)),
            "string_vector_exact_accuracy": float(np.mean(slot_vec_exact[m])),
            "slot_threshold_count_accuracy": float(np.mean(slot_count_exact[m])),
            "mean_true_active_slot_probability": float(np.mean(active_probs)) if len(active_probs) else None,
            "mean_true_inactive_slot_probability": float(np.mean(inactive_probs)) if len(inactive_probs) else None,
        }
        for name, pred in modes.items():
            by_k[str(k)][f"{name}_accuracy"] = float(np.mean(pred[m] == truth_k[m]))
            by_k[str(k)][f"{name}_mae"] = float(np.mean(np.abs(pred[m] - truth_k[m])))
            by_k[str(k)][f"{name}_undercount_fraction"] = float(np.mean(pred[m] < truth_k[m]))
            by_k[str(k)][f"{name}_overcount_fraction"] = float(np.mean(pred[m] > truth_k[m]))

    ordinal = modes["ordinal_cumulative_050"]
    poly_rows = [rows[i] for i in np.flatnonzero(poly)]
    poly_correct = ordinal[poly] == truth_k[poly]

    def feature_slice(mask):
        rr = [r for r, keep in zip(poly_rows, mask) if keep]
        return {
            "clusters": len(rr),
            "median_candidate_count": _q([r["candidate_count"] for r in rr], 0.5),
            "median_candidate_width_ms": _q([r["candidate_width_ms"] for r in rr], 0.5),
            "median_event_span_ms": _q([r["event_span_ms"] for r in rr], 0.5),
            "p90_event_span_ms": _q([r["event_span_ms"] for r in rr], 0.9),
            "median_nearest_candidate_dist_ms": _q([r["nearest_candidate_dist_mean_ms"] for r in rr], 0.5),
            "close_pitch_pair_fraction": _mean([r["close_pair"] for r in rr]),
            "octave_related_pair_fraction": _mean([r["octave_related_pair"] for r in rr]),
        }

    return {
        "poly_cluster_count": int(np.sum(poly)),
        "poly_string_vector_exact_accuracy": float(np.mean(slot_vec_exact[poly])),
        "poly_slot_threshold_count_accuracy": float(np.mean(slot_count_exact[poly])),
        "poly_representation_exact_but_ordinal_wrong_fraction": float(np.mean(slot_vec_exact[poly] & (ordinal[poly] != truth_k[poly]))),
        "poly_slot_count_correct_but_ordinal_wrong_fraction": float(np.mean(slot_count_exact[poly] & (ordinal[poly] != truth_k[poly]))),
        "poly_ordinal_correct_but_slot_count_wrong_fraction": float(np.mean((ordinal[poly] == truth_k[poly]) & ~slot_count_exact[poly])),
        "by_k": by_k,
        "poly_correct_feature_profile": feature_slice(poly_correct),
        "poly_wrong_feature_profile": feature_slice(~poly_correct),
        "predictions": {name: pred.tolist() for name, pred in modes.items()},
    }


def _pattern_novelty(train_rows, locked_rows, truth_k, ordinal_pred):
    train_patterns = Counter(r["pattern"] for r in train_rows)
    buckets = {"unseen": [], "rare_1_5": [], "medium_6_25": [], "common_gt25": []}
    for i, r in enumerate(locked_rows):
        if truth_k[i] < 2:
            continue
        freq = train_patterns[r["pattern"]]
        if freq == 0:
            key = "unseen"
        elif freq <= 5:
            key = "rare_1_5"
        elif freq <= 25:
            key = "medium_6_25"
        else:
            key = "common_gt25"
        buckets[key].append(i)
    out = {}
    for key, ids in buckets.items():
        out[key] = {
            "clusters": len(ids),
            "ordinal_accuracy": float(np.mean(ordinal_pred[ids] == truth_k[ids])) if ids else None,
            "train_pattern_frequency_median": _q([train_patterns[locked_rows[i]["pattern"]] for i in ids], 0.5),
        }
    return out


def audit(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cache = _load_spectral_caches(args.cache_dir)
    indexed, train_split, validation = _dataset_split(args.dataset_dir)
    group_map = {t.annotation_member: group_stem(t) for t in indexed}

    train_candidates, reconstruction = _reconstruct_candidates(cache)
    train_members = [str(x) for x in cache["members"]]
    train_events, train_assignment = _assign_events(args.dataset_dir, train_members, train_candidates)
    train_slots = np.asarray(cache["slot_targets"], dtype=np.float32)
    train_rows = _cluster_features(train_members, train_candidates, train_events, train_slots, group_map)
    train_exact = np.minimum(np.asarray(cache["exact"], dtype=np.int32), SLOT_COUNT)
    train_slot_k = np.sum(train_slots > 0.5, axis=1).astype(np.int32)

    locked12 = tuple(validation[:12])
    floor, v88_threshold, enc86, enc87, model88 = _load_frozen_stack(args)
    print("reconstructing locked12 for diagnostic audit")
    score_streams, records, x88, out88 = _represent_full(locked12, args.base_model, floor, enc86, enc87, model88)
    clusters, fused, assignment, sequence, mask, stats, target, exact, truncated = _cluster_data(locked12, records, x88, out88)
    locked_candidates = [np.asarray([int(records[i]["sample"]) for i in c["indices"]], dtype=np.int32) for c in clusters]
    locked_members = [str(c["member"]) for c in clusters]
    locked_slots, locked_slot_assignment = _slot_targets_for_runtime_clusters(locked12, clusters, records)
    locked_events, locked_pitch_assignment = _assign_events(args.dataset_dir, locked_members, locked_candidates)
    locked_rows = _cluster_features(locked_members, locked_candidates, locked_events, locked_slots, group_map)
    truth_k = np.minimum(np.asarray(exact, dtype=np.int32), SLOT_COUNT)

    model, _, _ = v101._build_model()
    model.load_weights(args.v101_weights)
    spectral = _spectral_maps_for_runtime(locked12, clusters, records)
    locked_cache = {"sequence": sequence, "mask": mask, "stats": stats, "spectral": spectral}
    slot_probs, pitch_probs, conditionals = v101._predict(model, v101._inputs(locked_cache))
    error = _model_error_report(locked_rows, truth_k, locked_slots, slot_probs, conditionals)
    ordinal_pred = np.asarray(error["predictions"]["ordinal_cumulative_050"], dtype=np.int32)

    train_distribution = _distribution(train_rows)
    locked_distribution = _distribution(locked_rows)
    player0_rows = [r for r in train_rows if r["player"] == "00"]

    report = {
        "schema_version": 1,
        "purpose": "diagnostic-only audit; no locked12 tuning or model selection",
        "data_integrity": {
            "train_clusters": len(train_rows),
            "locked_clusters": len(locked_rows),
            "train_exact_vs_slot_count_match": float(np.mean(train_exact == train_slot_k)),
            "train_reconstruction": reconstruction,
            "train_pitch_assignment": train_assignment,
            "locked_slot_assignment": locked_slot_assignment,
            "locked_pitch_assignment": locked_pitch_assignment,
            "locked_exact_vs_slot_count_match": float(np.mean(truth_k == np.sum(locked_slots > 0.5, axis=1))),
            "locked_truncated_clusters": int(np.sum(np.asarray(truncated) > 0)),
        },
        "train_distribution_by_k": train_distribution,
        "train_player00_distribution_by_k": _distribution(player0_rows),
        "locked12_distribution_by_k": locked_distribution,
        "locked12_model_error_decomposition": error,
        "locked12_pattern_novelty_vs_train": _pattern_novelty(train_rows, locked_rows, truth_k, ordinal_pred),
        "locked12_true_histogram": {str(k): int(np.sum(truth_k == k)) for k in range(SLOT_COUNT + 1)},
        "train_true_histogram": {str(k): int(np.sum(train_exact == k)) for k in range(SLOT_COUNT + 1)},
        "notes": {
            "close_pitch_pair": "at least one pair within 2 semitones",
            "octave_related_pair": "at least one pair within 0.5 semitone of an integer number of octaves, including unison",
            "event_span_ms": "span between earliest and latest assigned true onsets inside one candidate cluster",
            "pattern_frequency": "frequency of the exact six-bit physical-string occupancy vector in full train caches",
        },
    }
    path = args.output_dir / "audit.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    compact = {
        "integrity": report["data_integrity"],
        "train_poly": {k: train_distribution[k] for k in ("2", "3", "4", "5", "6")},
        "locked_poly": {k: locked_distribution[k] for k in ("2", "3", "4", "5", "6")},
        "error": {k: v for k, v in error.items() if k != "predictions"},
        "pattern_novelty": report["locked12_pattern_novelty_vs_train"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return report


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--v101-weights", type=Path, required=True)
    p.add_argument("--base-model", type=Path, required=True)
    p.add_argument("--v86-weights", type=Path, required=True)
    p.add_argument("--v86-report", type=Path, required=True)
    p.add_argument("--v87-weights", type=Path, required=True)
    p.add_argument("--v87-report", type=Path, required=True)
    p.add_argument("--v88-weights", type=Path, required=True)
    p.add_argument("--v88-report", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    audit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
