"""Run V13 outer-clean while disabling one known non-headline diagnostic.

Static audit found that train_v130_causal_event_set_decoder currently calls its
outer event-time diagnostic with full supervision arrays rather than the local
outer prediction arrays.  This does not affect training, event-presence decode,
cardinality, onset F1, per-K results, or saved predictions, but would make the
time/candidate diagnostic invalid.  Disable that diagnostic explicitly so no
misleading value can enter the report.  A post-hoc diagnostic can be computed
from the saved outer predictions if V13 is promising.
"""
from __future__ import annotations

from scripts import train_v130_causal_event_set_decoder as v130


def _disabled_event_diagnostics(*_args, **_kwargs):
    return {
        "status": "disabled_before_run_after_static_audit",
        "reason": "outer prediction/target indexing alias in initial V13 diagnostic call",
        "headline_metrics_affected": False,
    }


v130._event_diagnostics = _disabled_event_diagnostics

if __name__ == "__main__":
    raise SystemExit(v130.main())
