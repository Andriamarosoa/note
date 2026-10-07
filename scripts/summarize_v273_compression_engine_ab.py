"""Aggregate Librosa vs Rubber Band x2/x4 compression quality across folds."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

FOLDS = (0, 1, 2, 4)
ENGINES = ("librosa", "rubberband")
VIEWS = ("x2", "x4")


def avg(vals):
    vals = [x for x in vals if x is not None]
    return statistics.mean(vals) if vals else None


def amin(vals):
    vals = [x for x in vals if x is not None]
    return min(vals) if vals else None


def discover(root):
    out = {}
    for p in root.rglob("report.json"):
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        if r.get("protocol", {}).get("experiment") != "v273_compression_engine_ab":
            continue
        f = r["protocol"].get("validation_fold")
        if f in FOLDS:
            if f in out:
                raise RuntimeError(f"duplicate fold {f}")
            out[f] = r
    if set(out) != set(FOLDS):
        raise RuntimeError(f"missing folds: {set(FOLDS) - set(out)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    if a.output.exists():
        raise RuntimeError("refusing overwrite")
    a.output.mkdir(parents=True)
    reports = discover(a.input_root)

    result = {
        "status": "completed",
        "experiment": "v273_compression_engine_ab",
        "folds": list(FOLDS),
        "outer_fold_3_used": False,
        "exact_k_model_evaluated": False,
        "engines": {},
    }

    for engine in ENGINES:
        result["engines"][engine] = {}
        for view in VIEWS:
            rows = [reports[f]["results"][engine][view] for f in FOLDS]
            result["engines"][engine][view] = {
                "same_auc_by_fold": {str(f): rows[i]["same_representation_auc"] for i, f in enumerate(FOLDS)},
                "same_auc_mean": avg([x["same_representation_auc"] for x in rows]),
                "same_auc_min": amin([x["same_representation_auc"] for x in rows]),
                "natural_high_transfer_by_fold": {str(f): rows[i]["natural_high_model_transfer_auc"] for i, f in enumerate(FOLDS)},
                "natural_high_transfer_mean": avg([x["natural_high_model_transfer_auc"] for x in rows]),
                "natural_high_transfer_min": amin([x["natural_high_model_transfer_auc"] for x in rows]),
                "attack_corr_by_fold": {str(f): rows[i]["attack_envelope_corr_mean"] for i, f in enumerate(FOLDS)},
                "attack_corr_mean": avg([x["attack_envelope_corr_mean"] for x in rows]),
                "above_high_cut_mean": avg([x["fraction_shifted_low_above_original_high_cut"] for x in rows]),
            }

    comparisons = {}
    for view in VIEWS:
        lib = result["engines"]["librosa"][view]
        rub = result["engines"]["rubberband"][view]
        comparisons[view] = {
            "rubber_minus_librosa_same_auc": (
                None if lib["same_auc_mean"] is None or rub["same_auc_mean"] is None
                else rub["same_auc_mean"] - lib["same_auc_mean"]
            ),
            "rubber_minus_librosa_natural_high_transfer": (
                None if lib["natural_high_transfer_mean"] is None or rub["natural_high_transfer_mean"] is None
                else rub["natural_high_transfer_mean"] - lib["natural_high_transfer_mean"]
            ),
            "rubber_minus_librosa_attack_corr": (
                None if lib["attack_corr_mean"] is None or rub["attack_corr_mean"] is None
                else rub["attack_corr_mean"] - lib["attack_corr_mean"]
            ),
        }
    result["comparisons"] = comparisons

    (a.output / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    def ff(v):
        return "n/a" if v is None else f"{v:.3f}"

    lines = [
        "# Compression engine A/B — multi-fold summary",
        "",
        "This isolates x2 (+12 semitones) and x4 (+24 semitones) waveform compression quality.",
        "No Exact-K correction is applied.",
        "",
        "| engine | view | same-low AUC | natural-high transfer AUC | attack envelope corr | above old high cut |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for engine in ENGINES:
        for view in VIEWS:
            x = result["engines"][engine][view]
            lines.append(
                f"| {engine} | {view} | {ff(x['same_auc_mean'])} | "
                f"{ff(x['natural_high_transfer_mean'])} | {ff(x['attack_corr_mean'])} | "
                f"{ff(x['above_high_cut_mean'])} |"
            )

    lines += ["", "## Rubber Band minus Librosa", ""]
    for view in VIEWS:
        x = comparisons[view]
        lines.append(
            f"- {view}: same-low AUC {ff(x['rubber_minus_librosa_same_auc'])}; "
            f"natural-high transfer {ff(x['rubber_minus_librosa_natural_high_transfer'])}; "
            f"attack correlation {ff(x['rubber_minus_librosa_attack_corr'])}."
        )

    lines += [
        "",
        "Interpretation: for attack correlation and natural-high transfer, positive Rubber-Band deltas are favorable.",
        "The same-low AUC is a downstream separability measure rather than a pure audio-quality score.",
    ]
    (a.output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
