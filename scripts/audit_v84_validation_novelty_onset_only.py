"""Run the V8.4 validation novelty audit without evaluating the offset stream."""
from scripts import audit_v84_validation_novelty as _base
from causal_note.v84_onset_predictor import V84OnsetOnlyKerasPredictor

_base.V84KerasPredictor = V84OnsetOnlyKerasPredictor


if __name__ == "__main__":
    raise SystemExit(_base.main())
