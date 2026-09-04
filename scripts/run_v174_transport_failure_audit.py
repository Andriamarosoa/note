"""Compatibility runner for the post-V17.4 analysis-only audit.

Normalizes one historical V17.3 summary key name; no scientific computation,
model output, threshold, row set, or metric definition is changed.
"""
from __future__ import annotations

from pathlib import Path

from scripts import audit_v174_transport_failure as audit

_original_load_json = audit._load_json


def _load_json_compat(path: Path):
    data = _original_load_json(path)
    comp = data.get("comparison") if isinstance(data, dict) else None
    if isinstance(comp, dict) and "global_f1_v173" not in comp and "global_f1_v173_poibin" in comp:
        comp["global_f1_v173"] = comp["global_f1_v173_poibin"]
    return data


def main(argv=None):
    audit._load_json = _load_json_compat
    try:
        return audit.main(argv)
    finally:
        audit._load_json = _original_load_json


if __name__ == "__main__":
    raise SystemExit(main())
