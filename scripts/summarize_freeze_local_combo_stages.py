"""Aggregate the four held-out freeze_local_combo stage-ablation folds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.audit_v273_candidate_hidden1_multifold_worker import INTERNAL_FOLDS
from scripts.yourmt3_exactk_common import metrics, paired, require, write_json


def main(a):
    reports = {}
    arrays = []
    for fold in INTERNAL_FOLDS:
        hits = list(a.input.rglob(f"fold-{fold}/report.json"))
        if not hits:
            hits = [p for p in a.input.rglob("report.json")
                    if json.loads(p.read_text()).get("protocol", {}).get("validation_fold") == fold]
        require(len(hits) == 1, f"fold {fold} report count {len(hits)}")
        report = json.loads(hits[0].read_text())
        require(report["status"] == "completed", f"fold {fold} incomplete")
        reports[str(fold)] = report
        pred_path = hits[0].with_name("predictions.npz")
        require(pred_path.exists(), f"missing predictions fold {fold}")
        with np.load(pred_path, allow_pickle=False) as z:
            arrays.append({k: z[k] for k in z.files})

    ids = np.concatenate([x["global_index"].astype(np.int64) for x in arrays])
    y = np.concatenate([x["k"].astype(np.int32) for x in arrays])
    full = np.concatenate([x["freeze_local_combo"].astype(np.int32) for x in arrays])
    enc = np.concatenate([x["encoder_linear_probe"].astype(np.int32) for x in arrays])
    pre = np.concatenate([x["predecoder_linear_probe"].astype(np.int32) for x in arrays])

    require(len(ids) == 59309 and len(np.unique(ids)) == 59309, "aggregate cohort drift")
    require(np.all((0 <= y) & (y <= 6)), "bad targets")

    agg = {
        "freeze_local_combo": metrics(y, full),
        "encoder_linear_probe": {
            "metrics": metrics(y, enc),
            "paired_vs_full": paired(y, full, enc),
        },
        "predecoder_linear_probe": {
            "metrics": metrics(y, pre),
            "paired_vs_full": paired(y, full, pre),
        },
    }
    require(
        agg["freeze_local_combo"]["correct"] == 48454
        and agg["freeze_local_combo"]["poly"]["correct"] == 2530,
        "freeze reference aggregate drift",
    )

    dims = {
        "encoder": sorted({int(r["dimensions"]["encoder"]) for r in reports.values()}),
        "predecoder": sorted({int(r["dimensions"]["predecoder"]) for r in reports.values()}),
    }
    require(len(dims["encoder"]) == 1 and dims["predecoder"] == [96], "representation dimension drift")

    out = {
        "status": "completed",
        "protocol": {
            "experiment": "freeze_local_combo_stage_ablation",
            "folds": list(INTERNAL_FOLDS),
            "fold3_used": False,
            "player05_used": False,
            "same_probe_as_yourmt3_ablation": True,
            "encoder_cut": "v240_cardinality_context",
            "predecoder_cut": "v240_cardinality_hidden2",
        },
        "dimensions": {"encoder": dims["encoder"][0], "predecoder": 96},
        "aggregate": agg,
        "by_fold": reports,
    }

    a.output.mkdir(parents=True, exist_ok=False)
    write_json(a.output / "report.json", out)
    np.savez_compressed(
        a.output / "predictions.npz",
        global_index=ids,
        k=y,
        freeze_local_combo=full,
        encoder_linear_probe=enc,
        predecoder_linear_probe=pre,
    )

    lines = [
        "# freeze_local_combo encoder / pre-decoder ablation",
        "",
        "Same fixed cross-fold linear probe protocol as the YourMT3+ representation audit.",
        "",
        "| Stage | Global Exact-K | Poly K2-K6 | Poly under | Poly over |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, rec in [
        ("encoder linear probe", agg["encoder_linear_probe"]["metrics"]),
        ("pre-decoder linear probe", agg["predecoder_linear_probe"]["metrics"]),
        ("freeze_local_combo full", agg["freeze_local_combo"]),
    ]:
        p = rec["poly"]
        lines.append(
            f"| {name} | {100*rec['exact']:.4f}% | {100*p['exact']:.4f}% | {p['under']} | {p['over']} |"
        )

    lines += [
        "",
        "## Per-K Exact-K",
        "",
        "| K | encoder probe | pre-decoder probe | full freeze_local_combo |",
        "|---:|---:|---:|---:|",
    ]
    for k in range(7):
        vals = [
            agg["encoder_linear_probe"]["metrics"]["by_k"][str(k)]["exact"],
            agg["predecoder_linear_probe"]["metrics"]["by_k"][str(k)]["exact"],
            agg["freeze_local_combo"]["by_k"][str(k)]["exact"],
        ]
        lines.append(f"| {k} | {100*vals[0]:.3f}% | {100*vals[1]:.3f}% | {100*vals[2]:.3f}% |")

    lines += [
        "",
        f"Encoder representation dimension: {dims['encoder'][0]}.",
        "Pre-decoder representation dimension: 96.",
    ]
    (a.output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
