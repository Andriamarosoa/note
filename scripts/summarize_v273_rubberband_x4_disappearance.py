"""Aggregate Librosa-vs-Rubber-Band x4 disappearance comparison across folds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FOLDS = (0, 1, 2, 4)
EXPERIMENT = "v273_rubberband_x4_disappearance_fold"


def require(c, m):
    if not c:
        raise RuntimeError(m)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    require(not a.output.exists(), "refusing overwrite")

    reports = {}
    arrays = []
    for rp in a.input_root.rglob("report.json"):
        try:
            r = json.loads(rp.read_text())
        except Exception:
            continue
        if r.get("experiment") != EXPERIMENT:
            continue
        f = int(r["validation_fold"])
        require(f in FOLDS and f not in reports, "duplicate/bad fold")
        cp = rp.parent / "comparison.npz"
        require(cp.exists(), "missing comparison npz")
        reports[f] = r
        with np.load(cp, allow_pickle=False) as z:
            arrays.append({
                "fold": f,
                "row_id": np.asarray(z["row_id"], np.int64),
                "true_k": np.asarray(z["true_k"], np.int32),
                "lib": np.asarray(z["librosa_x4_pred"], np.int32),
                "rub": np.asarray(z["rubberband_x4_pred"], np.int32),
            })

    require(set(reports) == set(FOLDS), f"missing folds: {set(FOLDS)-set(reports)}")

    row_id = np.concatenate([x["row_id"] for x in arrays])
    y = np.concatenate([x["true_k"] for x in arrays])
    lib = np.concatenate([x["lib"] for x in arrays])
    rub = np.concatenate([x["rub"] for x in arrays])

    require(len(row_id) == 488 and len(np.unique(row_id)) == 488, "frozen cohort drift")
    require(np.all(np.isin(y, (2, 3))), "bad truth")
    lib_take = lib == 2
    rub_take = rub == 2

    def acct(mask):
        corr = int(np.sum(mask & (y == 2)))
        reg = int(np.sum(mask & (y == 3)))
        return {
            "actions": int(mask.sum()),
            "corrections": corr,
            "regressions": reg,
            "net": corr - reg,
        }

    both = lib_take & rub_take
    lib_only = lib_take & ~rub_take
    rub_only = rub_take & ~lib_take
    neither = ~lib_take & ~rub_take

    def cell(mask):
        return {
            "rows": int(mask.sum()),
            "K2": int(np.sum(mask & (y == 2))),
            "K3": int(np.sum(mask & (y == 3))),
        }

    lib_acct = acct(lib_take)
    rub_acct = acct(rub_take)
    persistence = float(both.sum() / lib_take.sum()) if lib_take.any() else None

    report = {
        "status": "completed",
        "experiment": "v273_rubberband_x4_disappearance_summary",
        "rows": int(len(y)),
        "K2_rows": int(np.sum(y == 2)),
        "K3_rows": int(np.sum(y == 3)),
        "folds": list(FOLDS),
        "outer_fold_3_used": False,
        "librosa_x4": lib_acct,
        "rubberband_x4": rub_acct,
        "net_delta_rubber_vs_librosa": int(rub_acct["net"] - lib_acct["net"]),
        "overlap": {
            "both": cell(both),
            "librosa_only": cell(lib_only),
            "rubberband_only": cell(rub_only),
            "neither": cell(neither),
            "librosa_disappearances": int(lib_take.sum()),
            "persist_with_rubberband": int(both.sum()),
            "persistence_rate": persistence,
            "librosa_disappearances_lost_with_rubberband": int(lib_only.sum()),
            "new_rubberband_disappearances": int(rub_only.sum()),
        },
        "by_fold": {str(f): reports[f] for f in FOLDS},
        "automatic_promotion": False,
    }

    a.output.mkdir(parents=True)
    (a.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    def pct(v):
        return "n/a" if v is None else f"{100*v:.1f}%"

    lines = [
        "# Rubber Band x4 disappearance — multi-fold summary",
        "",
        f"Frozen cohort: **{len(y)} rows** (K2={report['K2_rows']}, K3={report['K3_rows']}).",
        "",
        "| engine | K3→K2 signals | true-K2 corrections | true-K3 regressions | net |",
        "|---|---:|---:|---:|---:|",
        f"| Librosa x4 | {lib_acct['actions']} | {lib_acct['corrections']} | {lib_acct['regressions']} | {lib_acct['net']:+d} |",
        f"| Rubber Band x4 | {rub_acct['actions']} | {rub_acct['corrections']} | {rub_acct['regressions']} | {rub_acct['net']:+d} |",
        "",
        f"Rubber Band retains **{report['overlap']['persist_with_rubberband']}/{report['overlap']['librosa_disappearances']} = {pct(persistence)}** of Librosa's x4 K3→K2 disappearances.",
        f"Librosa-only disappearances: **{report['overlap']['librosa_disappearances_lost_with_rubberband']}**.",
        f"Rubber-Band-only disappearances: **{report['overlap']['new_rubberband_disappearances']}**.",
        "",
        "| overlap class | rows | true K2 | true K3 |",
        "|---|---:|---:|---:|",
        f"| both engines K2 | {report['overlap']['both']['rows']} | {report['overlap']['both']['K2']} | {report['overlap']['both']['K3']} |",
        f"| Librosa only | {report['overlap']['librosa_only']['rows']} | {report['overlap']['librosa_only']['K2']} | {report['overlap']['librosa_only']['K3']} |",
        f"| Rubber Band only | {report['overlap']['rubberband_only']['rows']} | {report['overlap']['rubberband_only']['K2']} | {report['overlap']['rubberband_only']['K3']} |",
        f"| neither | {report['overlap']['neither']['rows']} | {report['overlap']['neither']['K2']} | {report['overlap']['neither']['K3']} |",
        "",
        f"Net delta Rubber Band vs Librosa: **{report['net_delta_rubber_vs_librosa']:+d}**.",
        "Fold 3 excluded. Diagnostic only; no automatic promotion.",
    ]
    (a.output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
