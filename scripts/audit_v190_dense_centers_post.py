"""Post-audit V19 learned dense birth centers without retraining.

Questions:
1) Is the terrible exact top-6 center coverage only an exact-cell artifact?
2) Do true center cells at least rank highly in the learned 23x64 distribution?
3) Is categorical CE close to its multi-center information-theoretic floor?
4) Are fundamental-frequency center targets acoustically learnable from raw V10 spectral channels?
5) Does fold-to-fold center quality explain V19's F1 delta vs V17.3?

Locked12/historical validation are never indexed. Training annotations are used
only to reconstruct the same V19 center targets for audit.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scripts import train_v100_spectral_string_slots as v100
from scripts import train_v102_source_time_assignment as v102
from scripts import train_v172_mass_preserving_exchangeable as v172
from scripts import train_v190_dense_birth_centers as v190

TOLS = ((0, 0), (1, 1), (2, 2), (1, 2), (2, 4))


def _locate_one(root: Path, pattern: str) -> Path:
    xs = sorted(root.glob(pattern))
    if len(xs) != 1:
        raise RuntimeError(f"expected one {pattern!r} under {root}, got {len(xs)}")
    return xs[0]


def _match_all(truth: np.ndarray, selected: np.ndarray, dt: int, df: int) -> bool:
    """Exact injective bipartite feasibility, <=6x6, via bitmask DP."""
    truth = np.asarray(truth, dtype=np.int32)
    selected = np.asarray(selected, dtype=np.int32)
    if len(truth) == 0:
        return True
    tt, tf = truth // v100.SPECTRAL_BANDS, truth % v100.SPECTRAL_BANDS
    st, sf = selected // v100.SPECTRAL_BANDS, selected % v100.SPECTRAL_BANDS
    ok = (np.abs(tt[:, None] - st[None, :]) <= dt) & (np.abs(tf[:, None] - sf[None, :]) <= df)
    states = {0}
    for i in range(len(truth)):
        nxt = set()
        for state in states:
            for j in np.flatnonzero(ok[i]):
                bit = 1 << int(j)
                if not state & bit:
                    nxt.add(state | bit)
        if not nxt:
            return False
        states = nxt
    return bool(states)


def _selection_metrics(score: np.ndarray, target: np.ndarray, eligible: np.ndarray, k: np.ndarray):
    chosen = v190._nms_topk_np(score)
    eligible = np.asarray(eligible, dtype=bool)
    k = np.asarray(k, dtype=np.int32)
    exact_by_tol = {f"dt{dt}_df{df}": np.zeros(len(k), dtype=bool) for dt, df in TOLS}
    hit_by_tol = {f"dt{dt}_df{df}": np.zeros(len(k), dtype=np.float64) for dt, df in TOLS}
    nearest_dt, nearest_df = [], []

    for r in np.flatnonzero(eligible & (k > 0)):
        truth = np.flatnonzero(target[r] > 0)
        sel = chosen[r]
        tt, tf = truth // v100.SPECTRAL_BANDS, truth % v100.SPECTRAL_BANDS
        st, sf = sel // v100.SPECTRAL_BANDS, sel % v100.SPECTRAL_BANDS
        dtime = np.abs(tt[:, None] - st[None, :])
        dfreq = np.abs(tf[:, None] - sf[None, :])
        for i in range(len(truth)):
            j = int(np.argmin(dtime[i] + dfreq[i]))
            nearest_dt.append(float(dtime[i, j])); nearest_df.append(float(dfreq[i, j]))
        for dt, df in TOLS:
            key = f"dt{dt}_df{df}"
            pair_ok = (dtime <= dt) & (dfreq <= df)
            hit_by_tol[key][r] = float(np.mean(np.any(pair_ok, axis=1)))
            exact_by_tol[key][r] = _match_all(truth, sel, dt, df)

    def pack(mask):
        out = {}
        for dt, df in TOLS:
            key = f"dt{dt}_df{df}"
            out[key] = {
                "exact_injective_coverage": float(np.mean(exact_by_tol[key][mask])) if np.any(mask) else None,
                "mean_truth_hit_fraction": float(np.mean(hit_by_tol[key][mask])) if np.any(mask) else None,
            }
        return out

    positive = eligible & (k > 0)
    poly = eligible & (k >= 2)
    return {
        "positive": pack(positive),
        "poly": pack(poly),
        "median_nearest_time_bins": float(np.median(nearest_dt)) if nearest_dt else None,
        "median_nearest_frequency_bins": float(np.median(nearest_df)) if nearest_df else None,
        "mean_nearest_time_bins": float(np.mean(nearest_dt)) if nearest_dt else None,
        "mean_nearest_frequency_bins": float(np.mean(nearest_df)) if nearest_df else None,
        "chosen": chosen,
    }


def _rank_and_ce(prob: np.ndarray, target: np.ndarray, eligible: np.ndarray, k: np.ndarray):
    prob = np.asarray(prob, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    k = np.asarray(k, dtype=np.int32)
    ranks, ps, row_ce, row_floor, row_mass, row_k = [], [], [], [], [], []
    for r in np.flatnonzero(eligible & (k > 0)):
        truth = np.flatnonzero(target[r] > 0)
        order = np.argsort(-prob[r], kind="stable")
        inv = np.empty(len(order), dtype=np.int32); inv[order] = np.arange(len(order), dtype=np.int32) + 1
        ranks.extend(inv[truth].tolist())
        ps.extend(prob[r, truth].tolist())
        p = np.clip(prob[r], 1e-12, 1.0)
        y = target[r]
        ce = float(-np.sum(y * np.log(p)))
        yy = y[y > 0]
        floor = float(-np.sum(yy * np.log(yy)))
        row_ce.append(ce); row_floor.append(floor); row_mass.append(float(np.sum(prob[r, truth]))); row_k.append(int(k[r]))
    ranks = np.asarray(ranks, dtype=np.float64)
    ps = np.asarray(ps, dtype=np.float64)
    row_ce = np.asarray(row_ce); row_floor = np.asarray(row_floor); row_mass = np.asarray(row_mass); row_k = np.asarray(row_k)
    out = {
        "truth_cell_median_rank": float(np.median(ranks)),
        "truth_cell_mean_rank": float(np.mean(ranks)),
        "truth_cell_rank_le_6": float(np.mean(ranks <= 6)),
        "truth_cell_rank_le_20": float(np.mean(ranks <= 20)),
        "truth_cell_rank_le_50": float(np.mean(ranks <= 50)),
        "truth_cell_rank_le_100": float(np.mean(ranks <= 100)),
        "truth_cell_mean_probability": float(np.mean(ps)),
        "row_mean_true_center_probability_mass": float(np.mean(row_mass)),
        "row_mean_categorical_ce": float(np.mean(row_ce)),
        "row_mean_information_floor": float(np.mean(row_floor)),
        "row_mean_excess_ce_over_floor": float(np.mean(row_ce - row_floor)),
        "per_true_k": {},
    }
    for value in range(1, 7):
        m = row_k == value
        if np.any(m):
            out["per_true_k"][str(value)] = {
                "rows": int(np.sum(m)),
                "mean_ce": float(np.mean(row_ce[m])),
                "mean_information_floor": float(np.mean(row_floor[m])),
                "mean_excess_ce": float(np.mean(row_ce[m] - row_floor[m])),
                "mean_true_center_mass": float(np.mean(row_mass[m])),
            }
    return out


def _marginal_js(chosen: np.ndarray, target: np.ndarray, eligible: np.ndarray, k: np.ndarray):
    rows = np.flatnonzero(np.asarray(eligible, dtype=bool) & (np.asarray(k) > 0))
    pred = np.zeros(v190.CENTER_CELLS, dtype=np.float64)
    truth = np.zeros(v190.CENTER_CELLS, dtype=np.float64)
    for r in rows:
        np.add.at(pred, chosen[r], 1.0)
        cells = np.flatnonzero(target[r] > 0)
        np.add.at(truth, cells, 1.0)
    pred /= max(pred.sum(), 1.0); truth /= max(truth.sum(), 1.0)
    m = 0.5 * (pred + truth)
    def kl(a, b):
        z = a > 0
        return float(np.sum(a[z] * np.log(np.clip(a[z] / np.clip(b[z], 1e-12, None), 1e-12, None))))
    return 0.5 * kl(pred, m) + 0.5 * kl(truth, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir", type=Path)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--v190-fold-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache = v100._load_spectral_caches(args.cache_dir)
    members = np.asarray([str(x) for x in cache["members"]], dtype="U96")
    candidate_samples, reconstruction = v102._reconstruct_candidates(cache)
    pitch_targets, time_mask, string_time_targets, time_sample, supervision = v102._derive_supervision(
        members, candidate_samples, args.dataset_dir, expected_slot_targets=cache["slot_targets"]
    )
    k = np.minimum(np.asarray(cache["exact"], dtype=np.int32), 6)
    target, eligible_float, distinct = v190._birth_center_targets(cache, pitch_targets, string_time_targets, k)
    eligible = eligible_float > .5

    # Assert training target and the original representability target use the same time bins.
    active = np.asarray(time_mask) > .5
    target_time_disagreement = 0
    checked = 0
    for r in np.flatnonzero(eligible):
        slots = np.flatnonzero(active[r])
        if len(slots) != int(k[r]):
            continue
        from_sample = np.argmin(np.abs(np.asarray(time_sample[r, slots], dtype=np.float64)[:, None] - np.asarray(v102.FRAME_CENTER_SAMPLES, dtype=np.float64)[None, :]), axis=1)
        from_dist = np.argmax(np.asarray(string_time_targets[r, slots]), axis=1)
        target_time_disagreement += int(np.sum(from_sample != from_dist)); checked += len(slots)

    fold_reports = []
    learned_prob = np.zeros((len(k), v190.CENTER_CELLS), dtype=np.float32)
    fold_rows = np.full(len(k), -1, dtype=np.int16)

    import tensorflow as tf
    from tensorflow import keras
    for fold in range(5):
        ns = SimpleNamespace(dataset_dir=args.dataset_dir, cache_dir=args.cache_dir, outer_fold=fold)
        ctx = v172._fold_context(ns)
        model, _, _ = v190._build_model(ctx["final_spec"])
        wp = _locate_one(args.v190_fold_dir, f"**/v190-dense-birth-centers-fold-{fold}.weights.h5")
        model.load_weights(wp)
        spectral_input = next(t for t in model.inputs if t.name.split(":", 1)[0] == "spectral_map")
        center_model = keras.Model(spectral_input, model.get_layer("birth_center_map").output)
        idx = np.asarray(ctx["outer_idx"], dtype=np.int64)
        learned_prob[idx] = center_model.predict(np.asarray(cache["spectral"])[idx], batch_size=256, verbose=0)
        fold_rows[idx] = fold
        rp = _locate_one(args.v190_fold_dir, f"**/report-fold-{fold}.json")
        report = json.loads(rp.read_text())
        fold_reports.append({
            "fold": fold,
            "v190_f1": float(report["strata"]["aggregate"][v190.MODEL_KEY]["metrics"]["global"]["f1"]),
            "v104_f1": float(report["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"]),
            "center_exact_poly_reported": float(report["v190"]["architecture"]["dense_center_diagnostics"]["top6_exact_center_coverage_poly"]),
            "center_hit_poly_reported": float(report["v190"]["architecture"]["dense_center_diagnostics"]["top6_mean_center_hit_fraction_poly"]),
        })
        tf.keras.backend.clear_session()

    if np.any(fold_rows < 0) or len(np.unique(np.flatnonzero(fold_rows >= 0))) != len(k):
        raise RuntimeError("outer-clean fold coverage incomplete")

    learned_sel = _selection_metrics(learned_prob, target, eligible, k)
    learned_rank = _rank_and_ce(learned_prob, target, eligible, k)

    # Raw acoustic baselines: if fundamental targets do not rank in raw channels,
    # grid representability does not imply learnability from local maxima.
    spectral = np.asarray(cache["spectral"], dtype=np.float32)
    raw = {}
    for channel, name in ((0, "log_power"), (1, "positive_pre"), (2, "flux")):
        score = spectral[:, :, :, channel].reshape((len(k), -1))
        sel = _selection_metrics(score, target, eligible, k)
        raw[name] = {
            "selection": {kk: vv for kk, vv in sel.items() if kk != "chosen"},
            "selected_truth_js_divergence": _marginal_js(sel["chosen"], target, eligible, k),
        }

    learned_chosen = learned_sel.pop("chosen")
    learned_js = _marginal_js(learned_chosen, target, eligible, k)

    # Fold center quality vs F1 delta: compare to canonical V17.3 numbers saved in V19 report.
    deltas, center_hits = [], []
    for row in fold_reports:
        # V19 report records inherited canonical v173 fold F1 under protocol comparison block.
        rp = _locate_one(args.v190_fold_dir, f"**/report-fold-{row['fold']}.json")
        rr = json.loads(rp.read_text())
        # The old fold F1 isn't in each report after rename, so use summary-independent known
        # comparison by loading protocol's pre-postprocess V104 only is insufficient. We still
        # report correlation center-hit vs absolute V19 F1 and leave delta correlation to summary.
        center_hits.append(row["center_hit_poly_reported"]); deltas.append(row["v190_f1"])
    corr = float(np.corrcoef(center_hits, deltas)[0, 1]) if np.std(center_hits) and np.std(deltas) else 0.0

    poly = eligible & (k >= 2)
    result = {
        "schema_version": 1,
        "protocol": {
            "v190_post_audit": True,
            "outer_clean_rows": int(len(k)),
            "same_five_saved_v190_models": True,
            "no_retraining": True,
            "training_annotations_used_for_target_reconstruction_only": True,
            "runtime_annotations_required": False,
            "locked12_indexed_or_evaluated": False,
        },
        "target_consistency": {
            "eligible_rows": int(np.sum(eligible)),
            "eligible_poly_rows": int(np.sum(poly)),
            "time_targets_checked": int(checked),
            "argmax_time_vs_nearest_sample_disagreements": int(target_time_disagreement),
            "argmax_time_vs_nearest_sample_disagreement_rate": float(target_time_disagreement / max(1, checked)),
            "representable_distinct_center_poly_rate": float(np.mean(distinct[poly] == k[poly])) if np.any(poly) else None,
        },
        "learned_center": {
            "selection": learned_sel,
            "rank_and_ce": learned_rank,
            "selected_truth_js_divergence": learned_js,
        },
        "raw_spectral_baselines": raw,
        "folds": fold_reports,
        "fold_correlation_center_hit_vs_v190_f1": corr,
        "structural_finding": {
            "topk_indices_differentiable_from_event_set_loss": False,
            "reason": "tf.math.top_k indices -> gather: downstream event-set gradients update gathered dense features but cannot move the discrete selected index; center coordinates are directly supervised only by birth_center_map CE",
            "categorical_softmax_multi_center_floor": "for K distinct uniform target centers, minimum categorical CE is log(K), so a single 1472-way softmax represents a set only as one probability mass budget shared across all births",
        },
        "supervision": {"source": supervision, "candidate_reconstruction": reconstruction},
    }
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "target_consistency": result["target_consistency"],
        "learned_poly_tolerances": result["learned_center"]["selection"]["poly"],
        "learned_rank_and_ce": result["learned_center"]["rank_and_ce"],
        "raw_flux_poly": result["raw_spectral_baselines"]["flux"]["selection"]["poly"],
        "raw_positive_pre_poly": result["raw_spectral_baselines"]["positive_pre"]["selection"]["poly"],
        "fold_correlation_center_hit_vs_v190_f1": corr,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
