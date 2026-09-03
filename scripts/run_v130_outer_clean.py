"""Run V13 outer-clean with two static-audit guards.

1) The initial V13 event-time diagnostic call passes supervision arrays instead
   of local outer prediction arrays.  Headline training/cardinality/F1 are not
   affected, but that diagnostic would be invalid, so it is disabled explicitly.
2) V13's final console summary asks for the legacy alias
   ``poly_exact_accuracy`` while the shared cardinality report exposes
   ``poly_cluster_accuracy``.  Add the alias in the runner so a successful fold
   cannot fail after training merely while printing its summary.
"""
from __future__ import annotations

from scripts import train_v130_causal_event_set_decoder as v130


def _disabled_event_diagnostics(*_args, **_kwargs):
    return {
        "status": "disabled_before_run_after_static_audit",
        "reason": "outer prediction/target indexing alias in initial V13 diagnostic call",
        "headline_metrics_affected": False,
    }


_original_card = v130.v120._card


def _card_with_legacy_alias(k, pred):
    report = _original_card(k, pred)
    if "poly_exact_accuracy" not in report and "poly_cluster_accuracy" in report:
        report = dict(report)
        report["poly_exact_accuracy"] = report["poly_cluster_accuracy"]
    return report


v130._event_diagnostics = _disabled_event_diagnostics
v130.v120._card = _card_with_legacy_alias

if __name__ == "__main__":
    raise SystemExit(v130.main())
