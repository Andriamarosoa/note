"""Run V16 while tolerating only the stale V13/V16 final-report print key.

The inherited V13 harness writes report/prediction/weight files before its final
console summary.  Current cardinality reports expose ``poly_accuracy`` while that
legacy print still asks for ``poly_exact_accuracy``.  V16's own final print has
the same stale key.  This wrapper catches only that exact KeyError after verifying
that the expected report exists; all other exceptions still fail the job.
"""
from __future__ import annotations

import json

from scripts import train_v130_causal_event_set_decoder as v130
from scripts import train_v160_parallel_coverage_competition as v160


LEGACY_KEY = "poly_exact_accuracy"
_original_v130_train_fold = v130.train_fold


def _report_path(args):
    return args.output_dir / f"report-fold-{args.outer_fold}.json"


def _load_written_report(args, *, require_v160: bool):
    path = _report_path(args)
    if not path.exists():
        raise RuntimeError(f"expected completed fold report not found after legacy print failure: {path}")
    report = json.loads(path.read_text())
    model_key = "v160" if require_v160 else "v130"
    if model_key not in report.get("strata", {}).get("aggregate", {}):
        raise RuntimeError(f"report exists but expected model key {model_key!r} is missing")
    return report


def _v130_train_fold_compat(args):
    try:
        return _original_v130_train_fold(args)
    except KeyError as exc:
        if exc.args != (LEGACY_KEY,):
            raise
        report = _load_written_report(args, require_v160=False)
        card = report["strata"]["aggregate"]["v130"]["cardinality"]
        if "poly_accuracy" not in card:
            raise
        print(json.dumps({
            "compat": "recovered_after_v130_legacy_summary_key",
            "outer": args.outer_fold,
            "v104_f1": report["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"],
            "v130_compat_f1": report["strata"]["aggregate"]["v130"]["metrics"]["global"]["f1"],
            "poly_accuracy": card["poly_accuracy"],
        }, indent=2, sort_keys=True))
        return report


def main(argv=None):
    args = v160.parser().parse_args(argv)
    v130.train_fold = _v130_train_fold_compat
    try:
        try:
            report = v160.train_fold(args)
        except KeyError as exc:
            if exc.args != (LEGACY_KEY,):
                raise
            report = _load_written_report(args, require_v160=True)
            card = report["strata"]["aggregate"]["v160"]["cardinality"]
            if "poly_accuracy" not in card:
                raise
            print(json.dumps({
                "compat": "recovered_after_v160_legacy_summary_key",
                "outer": args.outer_fold,
                "v104_f1": report["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"],
                "v160_f1": report["strata"]["aggregate"]["v160"]["metrics"]["global"]["f1"],
                "poly_accuracy_v160": card["poly_accuracy"],
                "k5_exact_v160": report["per_true_k"]["5"]["v160"]["exact"],
                "k6_exact_v160": report["per_true_k"]["6"]["v160"]["exact"],
            }, indent=2, sort_keys=True))
        return report
    finally:
        v130.train_fold = _original_v130_train_fold


if __name__ == "__main__":
    main()
