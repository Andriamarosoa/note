"""Aggregate all four folds; reject partial, overlapping, or incompatible runs."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.yourmt3_exactk_common import (FOLDS, CHECKPOINT_SHA256, SPACE_REVISION,
    metrics, paired, require, report_markdown, write_json)


def main(a):
    reports, arrays = {}, []
    for p in a.input_root.rglob("report.json"):
        r = json.loads(p.read_text())
        require(r["status"] == "completed" and not r["smoke_subset"], "partial fold")
        f = r["fold"]
        require(f in FOLDS and f not in reports, "duplicate or excluded fold")
        require(r["protocol"]["checkpoint_sha256"] == CHECKPOINT_SHA256 and
                r["protocol"]["space_revision"] == SPACE_REVISION, "model revision drift")
        with np.load(p.with_name("predictions.npz"), allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        require(metrics(d["k"], d["predicted"]) == r["yourmt3_plus"], "fold metrics drift")
        require(metrics(d["k"], d["baseline"]) == r["freeze_local_combo"], "baseline metrics drift")
        d["fold"] = np.full(len(d["k"]), f, np.int8)
        arrays.append(d); reports[f] = r
    require(set(reports) == set(FOLDS), "missing folds")
    fields = arrays[0].keys()
    all_rows = {k: np.concatenate([d[k] for d in arrays]) for k in fields}
    require(len(np.unique(all_rows["global_index"])) == 59309 == len(all_rows["k"]), "row inventory drift")
    y, b, p = all_rows["k"], all_rows["baseline"], all_rows["predicted"]
    ref = metrics(y, b)
    require(ref["correct"] == 48454 and ref["poly"]["correct"] == 2530, "reference score drift")
    report = {"status": "completed", "folds": list(FOLDS), "freeze_local_combo": ref,
              "yourmt3_plus": metrics(y, p), "paired": paired(y, b, p),
              "by_fold": {str(f): reports[f] for f in FOLDS},
              "automatic_promotion": False, "interpretation": "exploratory_offline_pretrained_comparison",
              "unmatched_predicted_events": sum(r["unmatched_predicted_events"] for r in reports.values()),
              "rows_above_six_raw": sum(r["rows_above_six_raw"] for r in reports.values())}
    a.output.mkdir(parents=True, exist_ok=False)
    write_json(a.output / "report.json", report)
    (a.output / "report.md").write_text(report_markdown(report))
    np.savez_compressed(a.output / "predictions.npz", **all_rows)
    print(report_markdown(report))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
