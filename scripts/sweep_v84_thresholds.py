"""Threshold sweep for V8.4 using its dual independent stateful predictor."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_note.v84_predictor import V84KerasPredictor
from scripts import sweep_v8_thresholds as _base

_base.V8KerasPredictor = V84KerasPredictor


def main(argv=None):
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
