"""Recount archived native raw-spectral A/B predictions; never select a model."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.yourmt3_exactk_common import metrics, digest, write_json, require

ARMS = ("normalized_duplicate", "raw")
FOLDS = (0, 1, 2, 4)


def paired(y, before, after):
    out = {}
    for name, mask in [("global", np.ones(len(y), bool)), ("low", y <= 1), ("poly", y >= 2)]:
        out[name] = {"corrected": int(np.sum(mask & (before != y) & (after == y))),
                     "regressed": int(np.sum(mask & (before == y) & (after != y))),
                     "net": int(np.sum(mask & (after == y)) - np.sum(mask & (before == y)))}
    return out


def summarize(a):
    require(not a.output.exists(), "refusing to overwrite")
    records = {}
    arrays = []
    for path in sorted(a.input_root.rglob("report.json")):
        r = json.loads(path.read_text())
        if r.get("protocol", {}).get("experiment") != "v273_raw_spectral_native_matched_ab":
            continue
        f = r["protocol"]["validation_fold"]
        require(f in FOLDS and f not in records and r["status"] == "completed", "fold incomplete/duplicate")
        require(r["protocol"]["arms"] == list(ARMS), "arm drift")
        with np.load(path.parent / "predictions.npz", allow_pickle=False) as z:
            d = {key: np.asarray(z[key]) for key in z.files}
        require(np.all(d["fold"] == f), "fold row mismatch")
        require(metrics(d["k"], d["baseline"]) == r["reference"], "reference recount mismatch")
        for arm in ARMS:
            require(metrics(d["k"], d[arm + "_predicted"]) == r["arms"][arm]["metrics"], "arm recount mismatch")
        require(r["arms"][ARMS[0]]["parameters"] == r["arms"][ARMS[1]]["parameters"], "capacity mismatch")
        require(r["arms"][ARMS[0]]["initial_hash"] == r["arms"][ARMS[1]]["initial_hash"], "anchor mismatch")
        records[f] = r
        arrays.append(d)
    require(set(records) == set(FOLDS), "four completed folds required")
    merged = {key: np.concatenate([d[key] for d in arrays]) for key in arrays[0]}
    order = np.argsort(merged["global_index"])
    merged = {key: value[order] for key, value in merged.items()}
    require(len(np.unique(merged["global_index"])) == 59309, "population drift")
    y, baseline = merged["k"], merged["baseline"]
    ref = metrics(y, baseline)
    require(ref["correct"] == 48454 and ref["poly"]["rows"] == 7385 and ref["poly"]["correct"] == 2530,
            "historical reference drift")
    report = {"status": "completed", "experiment": "v273_raw_spectral_native_matched_ab",
              "reference": ref, "arms": {}, "folds": records,
              "automatic_promotion": False, "fold3_used": False, "player05_used": False,
              "scope": "repeated internal cross-validation; independent confirmation remains necessary"}
    for arm in ARMS:
        pred = merged[arm + "_predicted"]
        report["arms"][arm] = {"metrics": metrics(y, pred), "vs_reference": paired(y, baseline, pred)}
    raw, control = merged["raw_predicted"], merged["normalized_duplicate_predicted"]
    report["raw_vs_capacity_control"] = paired(y, control, raw)
    raw_folds = [records[f]["arms"]["raw"]["vs_reference"] for f in FOLDS]
    report["raw_positive_poly_all_folds"] = all(r["poly"]["net"] > 0 for r in raw_folds)
    report["raw_low_k_nonnegative_all_folds"] = all(r["low"]["net"] >= 0 for r in raw_folds)
    a.output.mkdir(parents=True)
    write_json(a.output / "report.json", report)
    np.savez_compressed(a.output / "predictions.npz", **merged)
    lines = ["# Native Exact-K: normalized plus raw spectral evidence", "",
             "59,309 groups, 7,385 polyphonic; folds 0,1,2,4; same 92.9 ms cached audio.",
             "Two matched 864-parameter extensions of the archived uniform count networks.",
             "Eight weighted fine-tuning epochs per arm. Targets are only K0–K6.", "",
             "| Model | Global Exact-K | Poly Exact-K | Poly under | Poly over |",
             "|---|---:|---:|---:|---:|"]
    models = {"freeze_local_combo": ref, **{arm: report["arms"][arm]["metrics"] for arm in ARMS}}
    for name, m in models.items():
        lines.append(f"| {name} | {m['exact']*100:.4f}% | {m['poly']['exact']*100:.4f}% | {m['poly']['under']} | {m['poly']['over']} |")
    lines += ["", "| True K | Rows | Reference | Raw | Net exacts |", "|---:|---:|---:|---:|---:|"]
    for cls in range(7):
        b, r = ref["by_k"][str(cls)], models["raw"]["by_k"][str(cls)]
        lines.append(f"| {cls} | {b['rows']} | {b['exact']*100:.3f}% | {r['exact']*100:.3f}% | {r['correct']-b['correct']:+d} |")
    lines += ["", "| Fold | Raw vs reference: global net | Low-K net | Poly net | Raw vs capacity control: poly net |",
              "|---:|---:|---:|---:|---:|"]
    for f in FOLDS:
        r = records[f]["arms"]["raw"]
        delta = r["vs_reference"]
        lines.append(f"| {f} | {delta['global']['net']:+d} | {delta['low']['net']:+d} | {delta['poly']['net']:+d} | {r['vs_normalized_duplicate']['poly']['net']:+d} |")
    lines += ["", "No automatic promotion. No epoch, threshold or arm is selected on these evaluation labels.",
              "These composition folds have been used in earlier experiments; this is an exploratory comparison."]
    (a.output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    summarize(p.parse_args())
