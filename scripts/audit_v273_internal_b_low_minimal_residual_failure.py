"""Audit the fixed two-residual guard and export its inputs for independent replay.

The neural checkpoints are only used for inference on internal folds 0/1/2/4.
Register slices reuse the GLOBAL probabilities; optional register refits are
explicitly separate diagnostics. No threshold selection or model promotion.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np
from scripts.v273_residual_audit import (
    EXPERIMENT, FEATURES, FOLDS, REGISTERS, SCHEMA_VERSION, analyze,
    array_identity, check_partitions, extracted_matrix, require,
    sha256_file, verify_export, write_json,
)


def export_rows(path, arrays, sample_rate):
    with path.open("w", encoding="utf-8") as stream:
        for split in ("fit", "val"):
            action = arrays[split + "_b_low"] & (arrays[split + "_base_k"] == 3)
            positions = np.flatnonzero(action)
            valid = arrays[split + "_valid"]
            j = 0
            for i, pos in enumerate(positions):
                row = {
                    "split": split, "row_id": int(arrays[split + "_ids"][pos]),
                    "recording_id": str(arrays[split + "_recording"][pos]),
                    "fold": int(arrays[split + "_fold"][pos]),
                    "start_sample": int(arrays[split + "_start_sample"][pos]),
                    "start_seconds": float(arrays[split + "_start_sample"][pos] / sample_rate),
                    "true_K": int(arrays[split + "_true_k"][pos]),
                    "base_prediction": int(arrays[split + "_base_k"][pos]),
                    "valid": bool(valid[i]),
                    "excluded_reason": str(arrays[split + "_excluded_reason"][i]) or None,
                }
                for k, value in zip((*FEATURES, "median_triplet_f0"),
                                    (*arrays[split + "_X"][i], arrays[split + "_f0"][i])):
                    row[k] = float(value) if np.isfinite(value) else None
                row.update(probability_K2_weighted=None, global_score=None,
                           register=None, action_applied=False, prediction_after=3)
                if valid[i]:
                    p = float(arrays[split + "_probability"][j])
                    row.update(probability_K2_weighted=p,
                               global_score=float(arrays[split + "_score"][j]),
                               register=str(arrays[split + "_register"][j]),
                               action_applied=p >= .5, prediction_after=2 if p >= .5 else 3)
                    j += 1
                row["decision_role"] = "observed_validation" if split == "val" else "in_sample_diagnostic"
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-dir", "bundle", "config", "fold-root", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--val-fold", type=int, choices=FOLDS, required=True)
    ap.add_argument("--include-register-refit", action="store_true")
    a = ap.parse_args()
    require(not a.output.exists(), "refusing overwrite")
    # Import audio/neural dependencies only after checking CLI/fold constraints.
    from causal_note.guitarset import SAMPLE_RATE
    from scripts.audit_v273_internal_b_like_boundary_corrector import (
        NEURONS, SEED, discover_reports, nested_base, predict, fold_ids, structural_features,
    )
    from scripts.audit_v273_internal_b_low_minimal_residual_exact_guard import audio_rows
    from scripts.audit_v273_internal_b_low_harmonic_strata import build_blow
    from scripts.train_v273_group_gate_ab import build_model
    from scripts.v273_native_protocol import load_config
    from scripts.v273_window_experiment import load_bundle

    reports = discover_reports(a.fold_root)
    fd = reports[a.val_fold][0].parent
    cfg = load_config(a.config)
    cache, _, bundle_report = load_bundle(a.bundle, a.config)
    fit, val = fold_ids(cache, cfg, a.val_fold)
    arrays = {}
    for split, ids in (("fit", fit), ("val", val)):
        members = np.asarray(cache["members"][ids]).astype(str)
        arrays.update({
            split + "_ids": ids,
            split + "_recording": members,
            split + "_fold": np.asarray([cfg["member_folds"][m] for m in members], np.int32),
            split + "_true_k": np.minimum(cache["exact"][ids].astype(np.int32), 6),
            split + "_start_sample": np.asarray(cache["cluster_start_samples"][ids], np.int64),
        })
    check_partitions(arrays, a.val_fold)

    mu = build_model("learned_gate", SEED)
    mu.load_weights(fd / "uniform.weights.h5")
    mg = build_model("learned_gate", SEED)
    mg.load_weights(fd / "freeze_local_combo.weights.h5")
    bu, bg = nested_base(mu), nested_base(mg)
    ku, b1 = [np.asarray(x).copy() for x in bu.get_layer("candidate_hidden1").get_weights()]
    kg, b2 = [np.asarray(x).copy() for x in bg.get_layer("candidate_hidden1").get_weights()]
    ii = np.asarray(NEURONS, np.int64)
    kg[:, ii], b2[ii] = ku[:, ii], b1[ii]
    bg.get_layer("candidate_hidden1").set_weights([kg, b2])

    for split, ids in (("fit", fit), ("val", val)):
        P, G = predict(mg, cache, ids)
        arrays[split + "_base_probability"] = P
        arrays[split + "_base_k"] = G
        arrays[split + "_structural"] = structural_features(cache, ids, P)
    fb, vb = build_blow(arrays["fit_structural"], arrays["val_structural"],
                       arrays["fit_true_k"], arrays["fit_base_k"], a.val_fold)
    arrays.update(fit_b_low=fb, val_b_low=vb)
    inputs = []
    exclusions = {}
    for split in ("fit", "val"):
        action = arrays[split + "_b_low"] & (arrays[split + "_base_k"] == 3)
        ids = arrays[split + "_ids"][action]
        rows = audio_rows(cache, ids, a.dataset_dir)
        X, f0, valid, reasons = extracted_matrix(rows)
        arrays.update({split + "_action_ids": ids, split + "_X": X, split + "_f0": f0,
                       split + "_valid": valid, split + "_excluded_reason": reasons})
        inputs.extend((X[valid], arrays[split + "_true_k"][action][valid], f0[valid]))
        exclusions[split] = {"action_candidates": len(ids), "valid": int(valid.sum()),
                             "excluded": int((~valid).sum()),
                             "by_reason": {str(r): int(np.sum(reasons == r))
                                           for r in np.unique(reasons[~valid])}}
    detail, state, predictions = analyze(*inputs, include_register_refit=a.include_register_refit)
    arrays.update(predictions)
    report = {
        "status": "completed", "schema_version": SCHEMA_VERSION,
        "training": False, "neural_training": False, "diagnostic_lr_fit": True,
        "protocol": {
            "experiment": EXPERIMENT, "validation_fold": a.val_fold,
            "fit_folds": [x for x in FOLDS if x != a.val_fold], "outer_fold_3_used": False,
            "action_population": "all valid B_low + base K3, including other true K",
            "training_population": "valid action rows with true K in {2,3}",
            "features": list(FEATURES), "classifier": "logistic C=1 class_weight=balanced",
            "threshold": .5, "action": "3->2 only", "register_gating": False,
            "register_stratification": "global predictions, no refit",
            "automatic_promotion": False, "exploratory": True,
        },
        "exclusions": exclusions, **detail,
    }
    a.output.mkdir(parents=True)
    write_json(a.output / "report.json", report)
    write_json(a.output / "model.json", state)
    np.savez_compressed(a.output / "replay.npz", **arrays)
    export_rows(a.output / "rows.jsonl", arrays, SAMPLE_RATE)
    q = report["global_action"]
    def ff(value):
        return "n/a" if value is None else f"{value:.6f}"
    lines = [f"# Residual audit v2 — fold {a.val_fold}", "",
             f"Global K2/K3 AUC: **{ff(report['joint']['val_auc'])}**.",
             f"Exact-K net: **{q['global_net']:+d}**; corrections/regressions: **{q['corrections']}/{q['regressions']}**.",
             f"Actions: **{q['applied']}**, including **{q['other_k_actions']}** on other K.", "",
             "## Same global model, stratified by estimated register", "",
             "| Register | Valid rows | K2/K3 rows | AUC | Actions | Corrections | Regressions | Other K | Net |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in REGISTERS:
        r = report["global_stratified"][name]
        lines.append(f"| {name} | {r['val_rows']} | {r['val_k23_rows']} | {ff(r['joint_auc'])} | {r['applied']} | {r['corrections']} | {r['regressions']} | {r['other_k_actions']} | {r['global_net']:+d} |")
    lines += ["", "Counts add exactly; AUCs are not additive. Register cuts use FIT K2/K3 only.",
              "Optional specialized models are separate in `register_refit`; their nets must not be added as global slices.",
              "", "`rows.jsonl`, `replay.npz`, `model.json`, and `manifest.json` support independent replay.",
              "Correlation and mean inversions are descriptive, not demonstrated causes. No outer-fold evaluation or promotion."]
    (a.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    repo = Path(__file__).resolve().parents[1]
    source_paths = ["scripts/v273_residual_audit.py", "scripts/audit_v273_internal_b_low_minimal_residual_failure.py",
                    "scripts/audit_v273_internal_b_low_minimal_residual_exact_guard.py",
                    "scripts/audit_v273_internal_b_low_harmonic_strata.py",
                    "scripts/audit_v273_internal_b_like_boundary_corrector.py"]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "source_sha256": {p: sha256_file(repo / p) for p in source_paths},
        "config_sha256": sha256_file(a.config), "bundle_identity": bundle_report,
        "checkpoint_sha256": {p: sha256_file(fd / p) for p in ("uniform.weights.h5", "freeze_local_combo.weights.h5")},
        "runtime": {"python": platform.python_version(),
                    **{p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn", "tensorflow")}},
        "thread_environment": {p: os.environ.get(p) for p in ("OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")},
        "arrays": {name: array_identity(value) for name, value in arrays.items()},
        "files": {p: sha256_file(a.output / p) for p in ("rows.jsonl", "replay.npz", "model.json", "report.json", "report.md")},
    }
    write_json(a.output / "manifest.json", manifest)
    write_json(a.output / "replay_check.json", verify_export(a.output))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
