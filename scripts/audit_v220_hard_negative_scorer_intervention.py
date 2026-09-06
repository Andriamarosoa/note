"""V22 pre-audit: replace only the frozen V19 1x1 center scorer.

For each canonical V19 outer fold:
- load the saved V19 model unchanged;
- freeze the full conv latent and decoder;
- fit one deterministic linear ridge scorer on FINAL-FIT rows only, using
  true birth cells as positives and the original V19 head's own far/top-scoring
  cells as hard negatives (same construction as V21);
- analytically fold standardization into a single 96->1 linear kernel and
  replace ONLY v190_birth_center_logits;
- evaluate the untouched outer fold with the unchanged V19 decoder/count rule.

This is an intervention test, not a promoted model. No decoder retraining, no
threshold tuning, no hyperparameter sweep, no Locked12 access.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scripts.train_v91_ordinal_cardinality import _dataset_split
from scripts import train_v100_spectral_string_slots as v100
from scripts import train_v102_source_time_assignment as v102
from scripts import train_v120_integrated_birth_source_time as v120
from scripts import train_v130_causal_event_set_decoder as v130
from scripts import train_v172_mass_preserving_exchangeable as v172
from scripts import train_v190_dense_birth_centers as v190
from scripts import audit_v210_v190_latent_conv as v210


def one(root: Path, pattern: str) -> Path:
    xs = sorted(root.glob(pattern))
    if len(xs) != 1:
        raise RuntimeError(f"expected one {pattern}, got {len(xs)}")
    return xs[0]


def poly(card: dict):
    for key in ("poly_cluster_accuracy", "poly_accuracy", "poly_exact_accuracy"):
        if key in card:
            return float(card[key])
    return None


def fold_linear_scorer(model: dict):
    """Convert standardized 2-output ridge difference to raw x@w+b."""
    coef = np.asarray(model["coef"], dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    std = np.asarray(model["std"], dtype=np.float64)
    d = coef[:, 1] - coef[:, 0]
    w = d[:-1] / std
    b = float(d[-1] - np.sum((mean / std) * d[:-1]))
    return w.astype(np.float32), np.float32(b)


def per_k(k, pred):
    k = np.asarray(k, np.int32); pred = np.asarray(pred, np.int32)
    out = {}
    for value in range(7):
        m = k == value
        out[str(value)] = {
            "rows": int(np.sum(m)),
            "exact": float(np.mean(pred[m] == value)) if np.any(m) else None,
            "under": float(np.mean(pred[m] < value)) if np.any(m) else None,
            "over": float(np.mean(pred[m] > value)) if np.any(m) else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir", type=Path)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--v190-fold-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)

    cache = v100._load_spectral_caches(args.cache_dir)
    spectral = np.asarray(cache["spectral"])
    members = np.asarray([str(x) for x in cache["members"]], dtype="U96")
    _, train_split, validation = _dataset_split(args.dataset_dir)
    candidates, recon = v102._reconstruct_candidates(cache)
    pitch, tm, td, ts, supervision = v102._derive_supervision(
        members, candidates, args.dataset_dir, expected_slot_targets=cache["slot_targets"]
    )
    k = np.minimum(np.asarray(cache["exact"], dtype=np.int32), 6)
    target, eligf, distinct = v190._birth_center_targets(cache, pitch, td, k)
    eligible = np.asarray(eligf) > 0.5

    old_all = np.full(len(k), -1, np.int32)
    new_all = np.full(len(k), -1, np.int32)
    fold_rows = []

    import tensorflow as tf
    from tensorflow import keras

    for fold in range(5):
        ctx = v172._fold_context(SimpleNamespace(dataset_dir=args.dataset_dir, cache_dir=args.cache_dir, outer_fold=fold))
        outer = np.asarray(ctx["outer_idx"], np.int64)
        train_idx = v210.balanced_train_sample(ctx["final_fit_idx"], k, fold)
        model, _, _ = v190._build_model(ctx["final_spec"])
        model.load_weights(one(args.v190_fold_dir, f"**/v190-dense-birth-centers-fold-{fold}.weights.h5"))

        pred_npz = one(args.v190_fold_dir, f"**/predictions-fold-{fold}.npz")
        with np.load(pred_npz, allow_pickle=False) as z:
            gi = np.asarray(z["global_index"], np.int64)
            old_pred = np.asarray(z[v190.PRED_KEY], np.int32)
        if not np.array_equal(gi, outer):
            raise RuntimeError(f"fold {fold}: V19 prediction coverage mismatch")
        old_all[outer] = old_pred

        spectral_input = next(t for t in model.inputs if t.name.split(":", 1)[0] == "spectral_map")
        latent = keras.Model(
            spectral_input,
            [model.get_layer("v190_dense_conv3").output, model.get_layer("v190_birth_center_logits_flat").output],
        )
        center_model = keras.Model(spectral_input, model.get_layer("birth_center_map").output)

        # Fit scorer on final-fit rows only. V21's sampling is fixed and contains no outer rows.
        train_cells = v210.extract(latent, spectral, train_idx, target, eligible, k)
        train_hard = train_cells["cell_tag"] != 2
        scorer = v210.ridge_binary_fit(train_cells["cell_conv"][train_hard], train_cells["cell_y"][train_hard])

        # Outer cells are extracted BEFORE intervention so "hard" means the same original-V19
        # alternatives used by V21, making AUC before/after directly comparable.
        outer_cells = v210.extract(latent, spectral, outer, target, eligible, k)
        hard = outer_cells["cell_tag"] != 2
        old_hard_auc = v210.auc_binary(outer_cells["cell_y"][hard], outer_cells["cell_head"][hard])
        new_cell_score = v210.ridge_binary_score(scorer, outer_cells["cell_conv"])
        new_hard_auc = v210.auc_binary(outer_cells["cell_y"][hard], new_cell_score[hard])

        old_center = center_model.predict(np.asarray(spectral[outer], np.float32), batch_size=256, verbose=0)
        old_anchor = v190._nms_topk_np(old_center)

        w, b = fold_linear_scorer(scorer)
        head = model.get_layer("v190_birth_center_logits")
        weights = head.get_weights()
        if len(weights) != 2 or weights[0].shape != (1, 1, v190.QUERY_DIM, 1) or weights[1].shape != (1,):
            raise RuntimeError(f"fold {fold}: unexpected head shapes {[x.shape for x in weights]}")
        head.set_weights([w.reshape(1, 1, -1, 1), np.asarray([b], np.float32)])

        new_pred, presence, event_time, event_candidate, prefix = v130._decode(model, v102._inputs(cache, outer))
        new_all[outer] = new_pred
        new_center = center_model.predict(np.asarray(spectral[outer], np.float32), batch_size=256, verbose=0)
        new_anchor = v190._nms_topk_np(new_center)
        cdiag = v190._center_diagnostics(new_center, target[outer], eligible[outer], k[outer])

        overlap = []
        for a, bidx in zip(old_anchor, new_anchor):
            overlap.append(len(set(map(int, a)).intersection(map(int, bidx))) / 6.0)

        old_metrics = v120._metrics(cache, train_split, outer, old_pred)
        new_metrics = v120._metrics(cache, train_split, outer, new_pred)
        old_card = v120._card(k[outer], old_pred); new_card = v120._card(k[outer], new_pred)
        fold_row = {
            "fold": fold,
            "outer_rows": int(len(outer)),
            "train_probe_rows": int(len(train_idx)),
            "train_cell_examples_hard_only": int(np.sum(train_hard)),
            "old_v19_f1": float(old_metrics["global"]["f1"]),
            "new_v22_intervention_f1": float(new_metrics["global"]["f1"]),
            "delta_f1_pp": 100.0 * float(new_metrics["global"]["f1"] - old_metrics["global"]["f1"]),
            "old_pred_ref": float(old_metrics["global"]["prediction_reference_ratio"]),
            "new_pred_ref": float(new_metrics["global"]["prediction_reference_ratio"]),
            "old_poly_exact": poly(old_card),
            "new_poly_exact": poly(new_card),
            "old_hard_negative_auc": old_hard_auc,
            "new_hard_negative_auc_same_cells": new_hard_auc,
            "mean_anchor_set_changed_fraction": float(1.0 - np.mean(overlap)),
            "new_center_exact_poly": cdiag["top6_exact_center_coverage_poly"],
            "new_center_hit_poly": cdiag["top6_mean_center_hit_fraction_poly"],
            "old_per_k": per_k(k[outer], old_pred),
            "new_per_k": per_k(k[outer], new_pred),
        }
        fold_rows.append(fold_row)
        print(json.dumps(fold_row, sort_keys=True))
        tf.keras.backend.clear_session()

    if np.any(old_all < 0) or np.any(new_all < 0):
        raise RuntimeError("incomplete five-fold coverage")
    all_idx = np.arange(len(k), dtype=np.int64)
    old_m = v120._metrics(cache, train_split, all_idx, old_all)
    new_m = v120._metrics(cache, train_split, all_idx, new_all)
    old_c = v120._card(k, old_all); new_c = v120._card(k, new_all)

    result = {
        "schema_version": 1,
        "protocol": {
            "v220_hard_negative_scorer_intervention": True,
            "outer_clean_rows": int(len(k)),
            "same_five_saved_v190_models": True,
            "v190_decoder_retrained": False,
            "conv_latent_retrained": False,
            "only_runtime_weight_replaced": "v190_birth_center_logits 1x1 linear 96->1",
            "scorer_training_rows_are_final_fit_only": True,
            "scorer_positive": "true birth center cell",
            "scorer_negative": "original V19 head top-scoring far cell",
            "scorer_hyperparameter_sweep": False,
            "ridge": float(v210.RIDGE),
            "runtime_annotations_required": False,
            "runtime_presence_threshold": float(v190.PRESENCE_THRESHOLD),
            "runtime_presence_threshold_tuned": False,
            "locked12_indexed_or_evaluated": False,
        },
        "pooled": {
            "old_v19_metrics": old_m,
            "new_v22_intervention_metrics": new_m,
            "old_v19_cardinality": old_c,
            "new_v22_intervention_cardinality": new_c,
            "delta_f1_pp": 100.0 * float(new_m["global"]["f1"] - old_m["global"]["f1"]),
            "fold_wins": int(sum(r["new_v22_intervention_f1"] > r["old_v19_f1"] for r in fold_rows)),
            "mean_old_hard_auc": float(np.mean([r["old_hard_negative_auc"] for r in fold_rows])),
            "mean_new_hard_auc": float(np.mean([r["new_hard_negative_auc_same_cells"] for r in fold_rows])),
            "mean_anchor_set_changed_fraction": float(np.mean([r["mean_anchor_set_changed_fraction"] for r in fold_rows])),
            "old_per_k": per_k(k, old_all),
            "new_per_k": per_k(k, new_all),
        },
        "folds": fold_rows,
        "supervision": {"source": supervision, "candidate_reconstruction": recon},
    }
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["pooled"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
