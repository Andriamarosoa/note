"""Aggregate only verified v2 residual audits; refuse mixed/legacy populations."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

FOLDS = (0, 1, 2, 4)
REGISTERS = ("low", "mid", "high", "unassigned")
COUNT_KEYS = ("val_rows", "val_k23_rows", "applied", "corrections", "regressions",
              "other_k_actions", "global_net", "k23_exact_net")
EXPERIMENT = "v273_internal_B_low_minimal_residual_failure_audit"


def discover(root):
    out, identities = {}, []
    for p in sorted(root.rglob("report.json")):
        r = json.loads(p.read_text())
        if r.get("protocol", {}).get("experiment") != EXPERIMENT:
            continue
        if r.get("schema_version") != 2:
            raise RuntimeError("legacy report: rerun with audit schema v2")
        f = r["protocol"]["validation_fold"]
        if f not in FOLDS or r["protocol"]["outer_fold_3_used"]:
            raise RuntimeError("forbidden fold")
        if f in out:
            raise RuntimeError(f"duplicate fold {f}")
        manifest = json.loads((p.parent / "manifest.json").read_text())
        if hashlib.sha256(p.read_bytes()).hexdigest() != manifest["files"]["report.json"]:
            raise RuntimeError("report checksum mismatch")
        replay = json.loads((p.parent / "replay_check.json").read_text())
        if replay.get("status") != "verified" or replay.get("validation_fold") != f or replay.get("global_net") != r["global_action"]["global_net"]:
            raise RuntimeError("replay not verified")
        for key in COUNT_KEYS:
            if sum(r["global_stratified"][reg][key] for reg in REGISTERS) != r["global_action"][key]:
                raise RuntimeError("non-additive register counts: " + key)
        identities.append(json.dumps({k: manifest[k] for k in
                          ("commit", "source_sha256", "config_sha256", "runtime")}, sort_keys=True))
        out[f] = r
    if set(out) != set(FOLDS):
        raise RuntimeError(f"missing folds {set(FOLDS) - set(out)}")
    if len(set(identities)) != 1:
        raise RuntimeError("mixed source/config/runtime identities")
    return out


def summarize(reports):
    detail = []
    for f in FOLDS:
        q = reports[f]
        detail.append({"fold": f, "joint_auc": q["joint"]["val_auc"],
                       "global_action": q["global_action"],
                       "global_stratified": q["global_stratified"],
                       "register_refit": q["register_refit"],
                       "feature_stats": q["feature_stats"], "univariate": q["univariate"],
                       "correlation": q["correlation"],
                       "standardized_fit_condition_number": q["standardized_fit_condition_number"],
                       "register_cuts_hz": q["register_cuts_hz"], "exclusions": q["exclusions"]})
    totals = {key: sum(r["global_action"][key] for r in detail) for key in COUNT_KEYS}
    aucs = [r["joint_auc"] for r in detail if r["joint_auc"] is not None]
    return {"status": "completed", "schema_version": 2,
            "outer_fold_3_used": False, "automatic_promotion": False, "folds": list(FOLDS),
            "folds_detail": detail, "global_totals": totals,
            "mean_joint_auc": statistics.mean(aucs) if aucs else None,
            "auc_folds_available": len(aucs),
            "positive_folds": sum(r["global_action"]["global_net"] > 0 for r in detail),
            "note": "Descriptive audit; correlations and mean shifts do not establish a causal explanation."}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise RuntimeError("refusing overwrite")
    result = summarize(discover(a.input_root))
    a.output.mkdir(parents=True)
    (a.output / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    t = result["global_totals"]
    lines = ["# Residual audit v2 — internal folds", "",
             f"Exact-K net: **{t['global_net']:+d}**; corrections/regressions: **{t['corrections']}/{t['regressions']}**.",
             f"**{t['applied']}** actions, including **{t['other_k_actions']}** on other K.", "",
             "| Fold | K2/K3 AUC | Actions | Corrections | Regressions | Other K | Net |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in result["folds_detail"]:
        q = r["global_action"]
        auc = "n/a" if r["joint_auc"] is None else f"{r['joint_auc']:.6f}"
        lines.append(f"| {r['fold']} | {auc} | {q['applied']} | {q['corrections']} | {q['regressions']} | {q['other_k_actions']} | {q['global_net']:+d} |")
    lines += ["", "## Global model by estimated register", "",
              "| Fold | Low net | Mid net | High net | Unassigned net | Global net |",
              "|---|---:|---:|---:|---:|---:|"]
    for r in result["folds_detail"]:
        nets = " | ".join(f"{r['global_stratified'][name]['global_net']:+d}" for name in REGISTERS)
        lines.append(f"| {r['fold']} | {nets} | {r['global_action']['global_net']:+d} |")
    lines += ["", "These slices reuse the global model; counts add exactly. AUCs are not additive.",
              "Register refits, when enabled, remain separate in the JSON and are not combined into these totals.",
              "Frozen-score replay verified for every fold. See each manifest for input, source, and runtime identities.",
              "Descriptive audit only; no demonstrated causal diagnosis, outer evaluation, threshold tuning, or promotion."]
    (a.output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
