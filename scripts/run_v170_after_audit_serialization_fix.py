"""Reporting-only wrapper for the completed V17 after-audit.

The original audit computes all diagnostics correctly but accidentally places the
full NumPy event-candidate target tensor into the JSON report, making json.dumps
fail. This wrapper removes only that non-reporting tensor from the supervision
metadata before serialization. No model, prediction, threshold, ranking, data,
or evaluation logic changes.
"""
from __future__ import annotations

from scripts import audit_v170_cardinality_after as base


_original_recoverability = base._recoverability


def _recoverability_without_tensor(*args, **kwargs):
    recoverable, complete, supervision = _original_recoverability(*args, **kwargs)
    supervision = dict(supervision)
    supervision.pop("event_candidate_target", None)
    return recoverable, complete, supervision


def main(argv=None):
    base._recoverability = _recoverability_without_tensor
    try:
        return base.main(argv)
    finally:
        base._recoverability = _original_recoverability


if __name__ == "__main__":
    raise SystemExit(main())
