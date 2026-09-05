"""Train-only audit of V10.4 OOF-to-full expert transfer.

This audit deliberately does NOT index or evaluate locked12. It answers two
separate questions:

1. Does the V10.4 fusion generalize across strict OOF composition folds when its
   inputs come from fold-trained experts, or was fold 0 unusually favorable?
2. When the same 76,768 train clusters are passed through the historical
   full-training V10.1/V10.2 experts, how much do the expert distributions,
   83 fusion features, and seven class gates move relative to strict OOF inputs?

The second comparison is descriptive because the full experts were trained on
these train clusters. It can establish an input-distribution mismatch, but by
itself cannot prove that mismatch caused the locked12 performance drop.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (ROOT, SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from causal_note.guitarset import SLOT_COUNT
from scripts.train_v91_ordinal_cardinality import _dataset_split
from scripts.train_v100_spectral_string_slots import _load_spectral_caches
from scripts import train_v101_string_query_attention as v101
from scripts import train_v102_source_time_assignment as v102
from scripts import run_v102_competitive_mass as v102_mass
from scripts import train_v103_residual_soft_fusion as v103
from scripts import train_v104_class_conditional_fusion as v104

EPS = 1e-8


def _tv(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return 0.5 * np.sum(np.abs(a - b), axis=1)


def _dist_summary(a, b):
    d = _tv(a, b)
    return {
        "mean_total_variation": float(np.mean(d)),
        "median_total_variation": float(np.median(d)),
        "p90_total_variation": float(np.percentile(d, 90)),
        "p99_total_variation": float(np.percentile(d, 99)),
    }


def _agreement(a, b):
    a = np.asarray(a, dtype=np.int32)
    b = np.asarray(b, dtype=np.int32)
    return float(np.mean(a == b))


def _hist(x):
    x = np.asarray(x, dtype=np.int32)
    return {str(k): int(np.sum(x == k)) for k in range(SLOT_COUNT + 1)}


def _card(k, pred):
    return v102._cardinality_report(np.asarray(k, dtype=np.int32), np.asarray(pred, dtype=np.int32))


def _per_k_shift(k, oof, full):
    out = {}
    for kk in range(SLOT_COUNT + 1):
        m = np.asarray(k) == kk
        if not np.any(m):
            continue
        d = _tv(oof[m], full[m])
        out[str(kk)] = {
            "clusters": int(np.sum(m)),
            "mean_total_variation": float(np.mean(d)),
            "p90_total_variation": float(np.percentile(d, 90)),
        }
    return out


def _crossfit_fusion(cache, train_split, oof, selected_epochs: int, seed: int):
    import tensorflow as tf

    fold = np.asarray(oof["fold"], dtype=np.int16)
    k = np.asarray(oof["k"], dtype=np.int32)
    features = np.asarray(oof["features"], dtype=np.float32)
    anchor = np.asarray(oof["anchor"], dtype=np.float32)
    p102 = np.asarray(oof["p102"], dtype=np.float32)
    pred = np.full(len(k), -1, dtype=np.int32)
    fold_reports = {}

    for held in range(v104.FOLD_COUNT):
        tf.keras.backend.clear_session()
        random.seed(seed + held)
        np.random.seed(seed + held)
        tf.random.set_seed(seed + held)
        fit = np.flatnonzero(fold != held).astype(np.int64)
        val = np.flatnonzero(fold == held).astype(np.int64)
        x_fit, mean, std = v104._standardize_fit(features[fit])
        x_val = v104._standardize_apply(features[val], mean, std)
        model, alpha_model, _ = v104._build_fusion(features.shape[1])
        model.fit(
            v104._inputs(x_fit, anchor[fit], p102[fit]),
            np.eye(SLOT_COUNT + 1, dtype=np.float32)[k[fit]],
            sample_weight=v104._mild_count_weights(k[fit]),
            epochs=selected_epochs,
            batch_size=128,
            shuffle=True,
            verbose=0,
        )
        p = np.asarray(model.predict(v104._inputs(x_val, anchor[val], p102[val]), batch_size=256, verbose=0))
        a = np.asarray(alpha_model.predict(v104._inputs(x_val, anchor[val], p102[val]), batch_size=256, verbose=0))
        pv = np.argmax(p, axis=1).astype(np.int32)
        pred[val] = pv
        fold_reports[str(held)] = {
            "clusters": int(len(val)),
            "v101_cardinality": _card(k[val], np.asarray(oof["pred101"])[val]),
            "v102_cardinality": _card(k[val], np.asarray(oof["pred102"])[val]),
            "v104_cardinality": _card(k[val], pv),
            "v101_metrics": v104._metrics_for_indices(cache, train_split, val, np.asarray(oof["pred101"])[val]),
            "v102_metrics": v104._metrics_for_indices(cache, train_split, val, np.asarray(oof["pred102"])[val]),
            "v104_metrics": v104._metrics_for_indices(cache, train_split, val, pv),
            "class_alpha": v104._alpha_report(a),
        }
    if np.any(pred < 0):
        raise RuntimeError("cross-fitted fusion did not cover every OOF row")
    all_idx = np.arange(len(k), dtype=np.int64)
    return pred, {
        "selected_epochs_fixed_from_frozen_v104": int(selected_epochs),
        "folds": fold_reports,
        "aggregate": {
            "v101_metrics": v104._metrics_for_indices(cache, train_split, all_idx, np.asarray(oof["pred101"])),
            "v102_metrics": v104._metrics_for_indices(cache, train_split, all_idx, np.asarray(oof["pred102"])),
            "v104_metrics": v104._metrics_for_indices(cache, train_split, all_idx, pred),
            "v101_cardinality": _card(k, np.asarray(oof["pred101"])),
            "v102_cardinality": _card(k, np.asarray(oof["pred102"])),
            "v104_cardinality": _card(k, pred),
        },
    }


def audit(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    import tensorflow as tf
    tf.random.set_seed(args.seed)

    cache = _load_spectral_caches(args.cache_dir)
    _, train_split, validation = _dataset_split(args.dataset_dir)
    if set(cache["track_members"]) != {t.annotation_member for t in train_split}:
        raise RuntimeError("cache/train split mismatch")
    if {t.annotation_member for t in train_split} & {t.annotation_member for t in validation}:
        raise RuntimeError("train/validation leakage")

    k = np.minimum(np.asarray(cache["exact"], dtype=np.int32), SLOT_COUNT)
    oof, paths, group_folds = v104._load_oof(args.oof_dir, len(k))
    if not np.array_equal(k, np.asarray(oof["k"], dtype=np.int32)):
        raise RuntimeError("OOF labels differ from cache")

    frozen = json.loads(args.v104_report.read_text())
    selected_epochs = int(frozen["configuration"]["selected_epochs"])
    crossfit_pred, crossfit = _crossfit_fusion(cache, train_split, oof, selected_epochs, args.seed + 1000)

    report101 = json.loads(args.v101_report.read_text())
    report102 = json.loads(args.v102_report.read_text())
    mode101 = str(report101["configuration"]["decode_mode"])
    mode102 = str(report102["configuration"]["decode_mode"])
    model101, _, _ = v101._build_model()
    model101.load_weights(args.v101_weights)
    model102, _, _ = v102_mass._build_model_mass_aware()
    model102.load_weights(args.v102_weights)
    full = v103._prepare_fusion_inputs(model101, model102, cache, None, mode101, mode102)

    with np.load(args.v104_scaler) as z:
        mean = np.asarray(z["mean"], dtype=np.float32)
        std = np.asarray(z["std"], dtype=np.float32)
    final_model, alpha_model, residual_model = v104._build_fusion(np.asarray(oof["features"]).shape[1])
    final_model.load_weights(args.v104_weights)

    x_oof = v104._standardize_apply(np.asarray(oof["features"], dtype=np.float32), mean, std)
    x_full = v104._standardize_apply(np.asarray(full["features"], dtype=np.float32), mean, std)
    delta_x = np.asarray(x_full - x_oof, dtype=np.float64)
    mean_shift = np.mean(x_full, axis=0) - np.mean(x_oof, axis=0)
    full_std = np.std(x_full, axis=0)
    oof_std = np.std(x_oof, axis=0)

    # Counterfactual isolation with the frozen deployed fusion. A is the exact OOF
    # input, B changes only gate features, C changes only direct expert count
    # distributions, D changes both to full-expert inputs.
    variants = {
        "A_oof_all": (x_oof, np.asarray(oof["anchor"]), np.asarray(oof["p102"])),
        "B_full_features_only": (x_full, np.asarray(oof["anchor"]), np.asarray(oof["p102"])),
        "C_full_distributions_only": (x_oof, np.asarray(full["anchor"]), np.asarray(full["p102"])),
        "D_full_all": (x_full, np.asarray(full["anchor"]), np.asarray(full["p102"])),
    }
    vout = {}
    probs = {}
    preds = {}
    for name, (xx, aa, pp) in variants.items():
        inp = v104._inputs(xx, aa, pp)
        pr = np.asarray(final_model.predict(inp, batch_size=256, verbose=0))
        al = np.asarray(alpha_model.predict(inp, batch_size=256, verbose=0))
        rs = np.asarray(residual_model.predict(inp, batch_size=256, verbose=0))
        pd = np.argmax(pr, axis=1).astype(np.int32)
        probs[name], preds[name] = pr, pd
        vout[name] = {
            "cardinality": _card(k, pd),
            "predicted_histogram": _hist(pd),
            "class_alpha": v104._alpha_report(al),
            "residual_abs_mean": float(np.mean(np.abs(rs))),
        }

    alpha_oof = np.asarray(alpha_model.predict(v104._inputs(x_oof, oof["anchor"], oof["p102"]), batch_size=256, verbose=0))
    alpha_full = np.asarray(alpha_model.predict(v104._inputs(x_full, full["anchor"], full["p102"]), batch_size=256, verbose=0))

    shift = {
        "same_cluster_comparison_warning": "Full experts were trained on these train clusters; use this section to measure input mismatch, not deployment generalization.",
        "v101": {
            "probability_shift": _dist_summary(oof["p101"], full["p101"]),
            "decoded_agreement": _agreement(oof["pred101"], full["pred101"]),
            "oof_histogram": _hist(oof["pred101"]),
            "full_histogram": _hist(full["pred101"]),
            "per_true_k_probability_shift": _per_k_shift(k, np.asarray(oof["p101"]), np.asarray(full["p101"])),
        },
        "v102": {
            "probability_shift": _dist_summary(oof["p102"], full["p102"]),
            "decoded_agreement": _agreement(oof["pred102"], full["pred102"]),
            "oof_histogram": _hist(oof["pred102"]),
            "full_histogram": _hist(full["pred102"]),
            "per_true_k_probability_shift": _per_k_shift(k, np.asarray(oof["p102"]), np.asarray(full["p102"])),
        },
        "anchor_probability_shift": _dist_summary(oof["anchor"], full["anchor"]),
        "features_83d": {
            "feature_dim": int(delta_x.shape[1]),
            "row_mean_abs_delta_mean": float(np.mean(np.mean(np.abs(delta_x), axis=1))),
            "row_mean_abs_delta_median": float(np.median(np.mean(np.abs(delta_x), axis=1))),
            "row_mean_abs_delta_p90": float(np.percentile(np.mean(np.abs(delta_x), axis=1), 90)),
            "mean_abs_feature_mean_shift": float(np.mean(np.abs(mean_shift))),
            "max_abs_feature_mean_shift": float(np.max(np.abs(mean_shift))),
            "features_abs_mean_shift_gt_0_25sd": int(np.sum(np.abs(mean_shift) > 0.25)),
            "features_abs_mean_shift_gt_0_50sd": int(np.sum(np.abs(mean_shift) > 0.50)),
            "features_abs_mean_shift_gt_1_00sd": int(np.sum(np.abs(mean_shift) > 1.00)),
            "median_full_to_oof_std_ratio": float(np.median(full_std / np.maximum(oof_std, 1e-6))),
            "rows_with_any_abs_z_gt_4_full": float(np.mean(np.any(np.abs(x_full) > 4.0, axis=1))),
            "rows_with_any_abs_z_gt_4_oof": float(np.mean(np.any(np.abs(x_oof) > 4.0, axis=1))),
        },
        "class_gate": {
            "mean_abs_alpha_delta": float(np.mean(np.abs(alpha_full - alpha_oof))),
            "median_row_mean_abs_alpha_delta": float(np.median(np.mean(np.abs(alpha_full - alpha_oof), axis=1))),
            "p90_row_mean_abs_alpha_delta": float(np.percentile(np.mean(np.abs(alpha_full - alpha_oof), axis=1), 90)),
            "oof_alpha": v104._alpha_report(alpha_oof),
            "full_alpha": v104._alpha_report(alpha_full),
        },
        "frozen_fusion_counterfactuals": {
            "variants": vout,
            "A_vs_B_features_only": {
                "output_probability_shift": _dist_summary(probs["A_oof_all"], probs["B_full_features_only"]),
                "decoded_agreement": _agreement(preds["A_oof_all"], preds["B_full_features_only"]),
            },
            "A_vs_C_distributions_only": {
                "output_probability_shift": _dist_summary(probs["A_oof_all"], probs["C_full_distributions_only"]),
                "decoded_agreement": _agreement(preds["A_oof_all"], preds["C_full_distributions_only"]),
            },
            "A_vs_D_all_full": {
                "output_probability_shift": _dist_summary(probs["A_oof_all"], probs["D_full_all"]),
                "decoded_agreement": _agreement(preds["A_oof_all"], preds["D_full_all"]),
            },
        },
    }

    result = {
        "schema_version": 1,
        "protocol": {
            "locked12_indexed_or_evaluated": False,
            "train_only_audit": True,
            "oof_fold_count": v104.FOLD_COUNT,
            "oof_exact_one_time_coverage": True,
            "same_cluster_full_expert_comparison_is_descriptive_not_causal": True,
        },
        "data": {
            "clusters": int(len(k)),
            "composition_groups": int(len(group_folds)),
            "oof_shards": [p.name for p in paths],
        },
        "cross_fitted_fusion_generalization": crossfit,
        "oof_to_full_input_shift": shift,
        "interpretation_rules": {
            "input_shift_exists": "Supported only if same-cluster probability/features/gate changes are materially non-zero.",
            "input_shift_caused_locked_drop": "Not proven by this train-only audit; causal attribution would require a deployment-matched clean evaluation, which locked12 must not be reused for tuning.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    summary = {
        "crossfit_global_f1": crossfit["aggregate"]["v104_metrics"]["global"]["f1"],
        "crossfit_v101_global_f1": crossfit["aggregate"]["v101_metrics"]["global"]["f1"],
        "crossfit_v102_global_f1": crossfit["aggregate"]["v102_metrics"]["global"]["f1"],
        "v101_decoded_agreement_oof_full": shift["v101"]["decoded_agreement"],
        "v102_decoded_agreement_oof_full": shift["v102"]["decoded_agreement"],
        "v101_mean_tv": shift["v101"]["probability_shift"]["mean_total_variation"],
        "v102_mean_tv": shift["v102"]["probability_shift"]["mean_total_variation"],
        "feature_row_mean_abs_delta": shift["features_83d"]["row_mean_abs_delta_mean"],
        "features_mean_shift_gt_0_5sd": shift["features_83d"]["features_abs_mean_shift_gt_0_50sd"],
        "mean_abs_alpha_delta": shift["class_gate"]["mean_abs_alpha_delta"],
        "A_vs_B_agreement": shift["frozen_fusion_counterfactuals"]["A_vs_B_features_only"]["decoded_agreement"],
        "A_vs_C_agreement": shift["frozen_fusion_counterfactuals"]["A_vs_C_distributions_only"]["decoded_agreement"],
        "A_vs_D_agreement": shift["frozen_fusion_counterfactuals"]["A_vs_D_all_full"]["decoded_agreement"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return result


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--oof-dir", type=Path, required=True)
    p.add_argument("--v101-weights", type=Path, required=True)
    p.add_argument("--v101-report", type=Path, required=True)
    p.add_argument("--v102-weights", type=Path, required=True)
    p.add_argument("--v102-report", type=Path, required=True)
    p.add_argument("--v104-weights", type=Path, required=True)
    p.add_argument("--v104-scaler", type=Path, required=True)
    p.add_argument("--v104-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=10491)
    return p


def main(argv: Optional[Sequence[str]] = None):
    args = parser().parse_args(argv)
    audit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
