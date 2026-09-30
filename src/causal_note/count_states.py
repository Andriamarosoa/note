"""Explicit silence state for onset counts; -1 is a state, not a negative count."""
from dataclasses import asdict, dataclass
import numpy as np

STATES = tuple(range(-1, 7))
UNLABELLED = -2


def _integers(values, low, high):
    a = np.asarray(values)
    if a.dtype.kind not in 'iu' or np.any(a < low) or np.any(a > high):
        raise ValueError(f'integer values in [{low}, {high}] required')
    return a.astype(np.int64)


def encode_states(states):
    """K=-1,0,...,6 -> nonnegative sparse-cross-entropy class IDs 0,...,7."""
    return _integers(states, -1, 6) + 1


def decode_classes(classes):
    return _integers(classes, 0, 7) - 1


def decode_probabilities(probability):
    p = np.asarray(probability)
    if (p.ndim != 2 or p.shape[1] != 8 or not np.isfinite(p).all()
            or np.any(p < 0) or np.any(p > 1)
            or not np.allclose(p.sum(1), 1, atol=1e-6, rtol=0)):
        raise ValueError('eight normalized probabilities in state order -1..6 required')
    return decode_classes(p.argmax(1))


@dataclass(frozen=True)
class SilencePolicy:
    """Conservative operational silence on ORIGINAL PCM, not compressed views.

    The thresholds are explicit experimental choices, not a claim about human
    audibility. All samples in the feature support must have been observed.
    Annotation overlap vetoes silence during offline target construction.
    """
    max_frame_rms_dbfs: float
    max_peak_dbfs: float
    before_samples: int = 3100
    after_samples: int = 2788
    frame_samples: int = 256

    def __post_init__(self):
        if not (np.isfinite(self.max_frame_rms_dbfs)
                and np.isfinite(self.max_peak_dbfs)
                and self.max_frame_rms_dbfs <= self.max_peak_dbfs <= 0):
            raise ValueError('finite RMS <= peak <= 0 dBFS limits required')
        for name in ('before_samples', 'after_samples', 'frame_samples'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')

    def serializable(self):
        return asdict(self)


def acoustic_evidence(samples, origin, policy):
    """No zero padding and no reads outside the declared source support."""
    x = np.asarray(samples)
    if x.ndim != 1 or isinstance(origin, (bool, np.bool_)) or not isinstance(origin, (int, np.integer)) or origin < 0:
        raise ValueError('mono PCM and nonnegative integer origin required')
    begin, end = int(origin)-policy.before_samples, int(origin)+policy.after_samples
    left, right = max(begin, 0), min(end, len(x))
    segment = x[left:right].astype(np.float64)
    if not len(segment) or not np.isfinite(segment).all() or np.any(np.abs(segment) > 1):
        raise ValueError('observed, finite PCM normalized to [-1,1] required')
    peak = float(np.max(np.abs(segment)))
    # Include the last short block; never dilute it with artificial zeros.
    frame_rms = max(float(np.sqrt(np.mean(segment[i:i+policy.frame_samples]**2)))
                    for i in range(0, len(segment), policy.frame_samples))
    quiet = (frame_rms <= 10**(policy.max_frame_rms_dbfs/20)
             and peak <= 10**(policy.max_peak_dbfs/20))
    return dict(quiet=bool(quiet), fully_observed=begin >= 0 and end <= len(x),
                max_frame_rms=frame_rms, peak=peak, begin=begin, end=end,
                observed_samples=len(segment))


def target_state(assigned_count, evidence, annotated_note_overlap):
    """Positive attacks take priority; sounding/foreign notes veto -1.

    Return (state, trainable). Unobserved quiet support without annotation
    evidence is unknown, never an invented silence example.
    """
    k = _integers([assigned_count], 0, 6)[0]
    overlap = _integers([annotated_note_overlap], 0, np.iinfo(np.int64).max)[0]
    if k > 0:
        return int(k), True
    if overlap > 0 or not evidence['quiet']:
        return 0, True
    if evidence['fully_observed']:
        return -1, True
    return UNLABELLED, False


def state_metrics(truth, predicted):
    truth = _integers(truth, -1, 6)
    predicted = _integers(predicted, -1, 6)
    if truth.ndim != 1 or predicted.shape != truth.shape or not len(truth):
        raise ValueError('nonempty aligned one-dimensional states required')
    cm = np.bincount((truth+1)*8 + predicted+1, minlength=64).reshape(8, 8)
    def subset(mask):
        n = int(mask.sum())
        return dict(rows=n, correct=int(np.sum(mask & (truth == predicted))),
                    exact=float(np.mean(truth[mask] == predicted[mask])) if n else None)
    # Both -1 and 0 emit zero onsets. Confusing them is a state error, not a
    # physically negative event count. Keep the two metrics separate.
    true_count, pred_count = np.maximum(truth, 0), np.maximum(predicted, 0)
    return dict(rows=len(truth), state_order=list(STATES),
                state_exact=float(np.mean(truth == predicted)),
                onset_count_exact=float(np.mean(true_count == pred_count)),
                polyphonic=subset(truth >= 2),
                by_state={str(k): subset(truth == k) for k in STATES},
                false_births=int(np.sum((truth <= 0) & (predicted > 0))),
                overcount=int(np.sum(pred_count > true_count)),
                undercount=int(np.sum(pred_count < true_count)),
                sounding_zero_as_silence=int(np.sum((truth == 0) & (predicted == -1))),
                silence_as_sounding_zero=int(np.sum((truth == -1) & (predicted == 0))),
                silence_precision=(float(np.mean(truth[predicted == -1] == -1))
                                   if np.any(predicted == -1) else None),
                confusion_true_by_predicted=cm.tolist())
