"""Aggregate corrected V17.7 and explicitly audit candidate-supply infeasibility."""
from __future__ import annotations

import json
from typing import Optional, Sequence

from scripts import summarize_v177_candidate_centric as base


def summarize(args):
    result = base.summarize(args)

    reports = []
    for fold in range(5):
        candidates = []
        for path in args.input_dir.glob(f"**/report-fold-{fold}.json"):
            report = json.loads(path.read_text())
            if report.get("protocol", {}).get("v177_candidate_feasibility_correction") is True:
                candidates.append(report)
        if len(candidates) != 1:
            raise RuntimeError(f"corrected V17.7 fold={fold}: expected one report, got {len(candidates)}")
        reports.append(candidates[0])

    outer_rows = 0
    infeasible_rows = 0
    by_k = {str(k): {"rows": 0, "feasible_rows": 0, "infeasible_rows": 0} for k in range(7)}
    shortfall = {}
    for report in reports:
        f = report["v177"]["architecture"]["candidate_feasibility"]
        outer_rows += int(f["outer_rows"])
        infeasible_rows += int(f["infeasible_rows"])
        for s, n in f["shortfall_histogram"].items():
            shortfall[s] = shortfall.get(s, 0) + int(n)
        for k, row in f["by_true_k"].items():
            by_k[k]["rows"] += int(row["rows"])
            by_k[k]["feasible_rows"] += int(row["feasible_rows"])
            by_k[k]["infeasible_rows"] += int(row["infeasible_rows"])
            if float(row["max_abs_mass_error_on_feasible_rows"]) >= 1e-6:
                raise RuntimeError(f"mass preservation failed on feasible K={k}: {row}")

    if outer_rows != 76768:
        raise RuntimeError(f"candidate feasibility outer coverage {outer_rows} != 76768")
    for row in by_k.values():
        row["infeasible_rate"] = row["infeasible_rows"] / row["rows"] if row["rows"] else 0.0

    feasibility = {
        "outer_rows": outer_rows,
        "feasible_rows": outer_rows - infeasible_rows,
        "infeasible_rows": infeasible_rows,
        "infeasible_rate": infeasible_rows / outer_rows,
        "candidate_supply_ceiling_rate": (outer_rows - infeasible_rows) / outer_rows,
        "shortfall_histogram": dict(sorted(shortfall.items(), key=lambda kv: int(kv[0]))),
        "by_true_k": by_k,
        "event_set_loss_masked_only_on_infeasible_rows": True,
        "outer_evaluation_includes_all_rows": True,
    }
    result["candidate_feasibility"] = feasibility
    result["protocol"].update({
        "v177_candidate_feasibility_correction": True,
        "candidate_set_loss_scope": "C>=K only",
        "candidate_infeasible_rows_kept_in_outer_evaluation": True,
        "candidate_infeasible_rows_removed_from_dataset": False,
        "v173_mass_coefficient_preservation_scope": "all event-set-supervised feasible rows",
        "architecture_changed_by_feasibility_correction": False,
        "parameters_added_by_feasibility_correction": 0,
    })

    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"candidate_feasibility": feasibility}, indent=2, sort_keys=True))
    return result


def main(argv: Optional[Sequence[str]] = None):
    summarize(base.parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
