"""Run V19 while tolerating only the inherited stale V13 final-print key."""
from __future__ import annotations
import json
from scripts import train_v130_causal_event_set_decoder as v130
from scripts import train_v190_dense_birth_centers as v190

LEGACY_KEY = "poly_exact_accuracy"
POLY_KEYS = ("poly_cluster_accuracy", "poly_accuracy", "poly_exact_accuracy")
_original = v130.train_fold


def _poly(card):
    for key in POLY_KEYS:
        if key in card:
            return float(card[key])
    raise RuntimeError(f"no supported poly accuracy key in {sorted(card)}")


def _compat(args):
    try:
        return _original(args)
    except KeyError as exc:
        if exc.args != (LEGACY_KEY,):
            raise
        path = args.output_dir / f"report-fold-{args.outer_fold}.json"
        if not path.exists():
            raise RuntimeError(f"completed fold report missing: {path}")
        report = json.loads(path.read_text())
        if "v130" not in report.get("strata", {}).get("aggregate", {}):
            raise RuntimeError("written report missing pre-postprocess v130 key")
        print(json.dumps({"compat": "recovered_after_v130_legacy_summary_key", "outer": args.outer_fold, "v104_f1": report["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"], "v130_compat_f1": report["strata"]["aggregate"]["v130"]["metrics"]["global"]["f1"], "poly_accuracy": _poly(report["strata"]["aggregate"]["v130"]["cardinality"])}, indent=2, sort_keys=True))
        return report


def main(argv=None):
    args = v190.parser().parse_args(argv)
    v130.train_fold = _compat
    try:
        return v190.train_fold(args)
    finally:
        v130.train_fold = _original


if __name__ == "__main__":
    main()
