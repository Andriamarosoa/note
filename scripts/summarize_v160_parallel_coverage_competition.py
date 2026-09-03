"""Aggregate V16 outer-clean folds and compare against frozen V13/V15 references."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Optional, Sequence

import numpy as np

from scripts import summarize_v130_causal_event_set_decoder as v130sum


def _rename_model_in_report(report: dict, src: str, dst: str) -> dict:
    for row in report.get("strata", {}).values():
        if row and src in row:
            row[dst] = row.pop(src)
    for row in report.get("per_true_k", {}).values():
        if row and src in row:
            row[dst] = row.pop(src)
    for row in report.get("folds", {}).values():
        key = f"{src}_f1"
        if key in row:
            row[f"{dst}_f1"] = row.pop(key)
    if "comparison" in report:
        report["comparison"] = {
            k.replace(src, dst): v for k, v in report["comparison"].items()
        }
    return report


def _metric(report: dict, model: str, stratum: str = "aggregate") -> float:
    return float(report["strata"][stratum][model]["metrics"]["global"]["f1"])


def _poly(report: dict, model: str) -> float:
    card = report["strata"]["aggregate"][model]["cardinality"]
    if "poly_accuracy" in card:
        return float(card["poly_accuracy"])
    return float(card["poly_exact_accuracy"])


def _k(report: dict, model: str, value: int) -> float:
    return float(report["per_true_k"][str(value)][model]["exact"])


def _prepare_v130_compatible(input_dir: Path, tmp: Path) -> None:
    for fold in range(5):
        rp = sorted(input_dir.glob(f"**/report-fold-{fold}.json"))
        npz = sorted(input_dir.glob(f"**/predictions-fold-{fold}.npz"))
        if len(rp) != 1 or len(npz) != 1:
            raise RuntimeError(f"fold {fold}: expected exactly one V16 report and prediction shard")
        out = tmp / f"fold-{fold}"
        out.mkdir(parents=True, exist_ok=True)

        report = json.loads(rp[0].read_text())
        _rename_model_in_report(report, "v160", "v130")
        (out / f"report-fold-{fold}.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )

        with np.load(npz[0], allow_pickle=False) as z:
            data = {k: np.asarray(z[k]) for k in z.files}
        data["pred130"] = data.pop("pred160")
        np.savez_compressed(out / f"predictions-fold-{fold}.npz", **data)


def summarize(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v160-summary-") as td:
        compat = Path(td)
        _prepare_v130_compatible(args.input_dir, compat)
        result = v130sum.summarize(
            SimpleNamespace(input_dir=compat, output_dir=args.output_dir)
        )

    _rename_model_in_report(result, "v130", "v160")
    result["protocol"].update({
        "parallel_event_decoder": True,
        "sequential_stop_decoder": False,
        "global_coverage_competition": True,
        "leave_one_out_coverage": True,
        "global_query_reconciliation": True,
    })
    result["architecture"] = {
        "name": "V16.0 parallel birth proposals + global coverage competition",
        "parallel_proposals": True,
        "leave_one_out_tf_and_candidate_coverage": True,
        "global_self_attention_reconciliation": True,
        "sequential_state_or_stop": False,
    }

    v160_global = _metric(result, "v160")
    v160_rock = _metric(result, "v160", "player00_rock_comp")
    v160_poly = _poly(result, "v160")
    v160_k5 = _k(result, "v160", 5)
    v160_k6 = _k(result, "v160", 6)

    references = {}
    if args.v130_report is not None:
        r130 = json.loads(args.v130_report.read_text())
        references["v130"] = {
            "global_f1": _metric(r130, "v130"),
            "player00_rock_comp_f1": _metric(r130, "v130", "player00_rock_comp"),
            "poly_exact": _poly(r130, "v130"),
            "k5_exact": _k(r130, "v130", 5),
            "k6_exact": _k(r130, "v130", 6),
        }
        result["comparison"].update({
            "v160_minus_v130_global_f1": v160_global - references["v130"]["global_f1"],
            "v160_minus_v130_player00_rock_comp_f1": v160_rock - references["v130"]["player00_rock_comp_f1"],
            "v160_minus_v130_poly_exact": v160_poly - references["v130"]["poly_exact"],
            "v160_minus_v130_k5_exact": v160_k5 - references["v130"]["k5_exact"],
            "v160_minus_v130_k6_exact": v160_k6 - references["v130"]["k6_exact"],
        })

    if args.v150_report is not None:
        r150 = json.loads(args.v150_report.read_text())
        references["v150"] = {
            "global_f1": _metric(r150, "v150"),
            "player00_rock_comp_f1": _metric(r150, "v150", "player00_rock_comp"),
            "poly_exact": _poly(r150, "v150"),
            "k5_exact": _k(r150, "v150", 5),
            "k6_exact": _k(r150, "v150", 6),
        }
        result["comparison"].update({
            "v160_minus_v150_global_f1": v160_global - references["v150"]["global_f1"],
            "v160_minus_v150_player00_rock_comp_f1": v160_rock - references["v150"]["player00_rock_comp_f1"],
            "v160_minus_v150_poly_exact": v160_poly - references["v150"]["poly_exact"],
            "v160_minus_v150_k5_exact": v160_k5 - references["v150"]["k5_exact"],
            "v160_minus_v150_k6_exact": v160_k6 - references["v150"]["k6_exact"],
        })

    result["references"] = references
    if "v150" in references:
        result["comparison"]["promotion_candidate"] = bool(
            result["comparison"]["promotion_candidate"]
            and v160_global > references["v150"]["global_f1"]
        )

    (args.output_dir / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    pred_path = args.output_dir / "predictions.npz"
    with np.load(pred_path, allow_pickle=False) as z:
        merged = {k: np.asarray(z[k]) for k in z.files}
    merged["pred160"] = merged.pop("pred130")
    np.savez_compressed(pred_path, **merged)

    print(json.dumps({
        "global": {
            "v104": _metric(result, "v104"),
            "v160": v160_global,
            **({"v130_reference": references["v130"]["global_f1"]} if "v130" in references else {}),
            **({"v150_reference": references["v150"]["global_f1"]} if "v150" in references else {}),
        },
        "poly_exact_v160": v160_poly,
        "k5_exact_v160": v160_k5,
        "k6_exact_v160": v160_k6,
        "player00_rock_comp_v160": v160_rock,
        "mean_predicted_count": result["event_presence"]["mean_predicted_count"],
        "mean_true_count": result["event_presence"]["mean_true_count"],
        "comparison": result["comparison"],
    }, indent=2, sort_keys=True))
    return result


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--v130-report", type=Path)
    p.add_argument("--v150-report", type=Path)
    return p


def main(argv: Optional[Sequence[str]] = None):
    summarize(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
