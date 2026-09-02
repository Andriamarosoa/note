"""Run the V8.4 validation novelty audit without evaluating the offset stream."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_v84_validation_novelty as _base
from causal_note.v84_onset_predictor import V84OnsetOnlyKerasPredictor

_base.V84KerasPredictor = V84OnsetOnlyKerasPredictor


if __name__ == "__main__":
    raise SystemExit(_base.main())
