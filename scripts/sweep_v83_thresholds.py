"""Threshold sweep for V8.3 using its exact split-task stateful predictor."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_note.v83_predictor import V83KerasPredictor
from scripts import sweep_v8_thresholds as _base

# Reuse the locked split, matching, decoder, aggregation and output schema.  The
# only replacement is the causal predictor implementation for V8.3 topology.
_base.V8KerasPredictor = V83KerasPredictor


def main(argv=None):
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
