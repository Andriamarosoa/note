"""Past-only, Schur-stable lagged Burg prediction; never a note counter.

One AR polynomial is estimated jointly over all lag interleavings. The forward
and backward errors never join two different interleavings. Reflection
coefficients are constrained during fitting, not by clipping predicted audio.
Stability is asymptotic; it does NOT bound finite-horizon amplitude relative to
a quiet recent history or make residual energy identify musical note counts.
"""
from numbers import Integral
import numpy as np

REFLECTION_MARGIN = 1e-6
RELATIVE_ENERGY_FLOOR = 1e-12


def _validate(past, order, lag):
    x = np.asarray(past, dtype=np.float64)
    if (x.ndim != 1 or not np.isfinite(x).all() or
            any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1
                for v in (order, lag)) or len(x) <= order * lag + order):
        raise ValueError('invalid history or recurrence')
    return x


def fit_burg(past, *, order=32, lag=64):
    """Return [1, a1, ...] for x[t] + sum(a[j]*x[t-j*lag]) = 0.

    At each order, pool the forward/backward squared errors and cross-product
    over all available within-interleaving pairs. A max-absolute-value scaling
    prevents overflow and cancels in reflection-coefficient ratios.
    """
    x = _validate(past, order, lag)
    peak = float(np.max(np.abs(x)))
    if peak == 0:
        return np.array([1.]), np.empty(0)
    forward = x / peak
    backward = forward.copy()
    energy_reference = 2 * float(forward @ forward)
    polynomial = np.array([1.])
    reflections = []
    for _ in range(order):
        f = forward[lag:]
        b = backward[:-lag]
        denominator = float(f @ f + b @ b)
        if denominator <= RELATIVE_ENERGY_FLOOR * energy_reference:
            break
        reflection = -2 * float(f @ b) / denominator
        reflection = float(np.clip(reflection, -1 + REFLECTION_MARGIN,
                                   1 - REFLECTION_MARGIN))
        polynomial = (np.r_[polynomial, 0.] +
                      reflection * np.r_[0., polynomial[::-1]])
        forward = f + reflection * b
        backward = b + reflection * f
        reflections.append(reflection)
    return polynomial, np.asarray(reflections)


def predict_polynomial(past, length, polynomial, *, lag):
    """Autonomous block recurrence; all dependencies precede each lag block."""
    x = np.asarray(past, dtype=np.float64)
    p = np.asarray(polynomial, dtype=np.float64)
    if (isinstance(length, bool) or not isinstance(length, Integral) or length < 1
            or p.ndim != 1 or len(p) < 1 or p[0] != 1
            or not np.isfinite(p).all()):
        raise ValueError('invalid forecast length or polynomial')
    wave = np.r_[x, np.zeros(length)]
    if len(p) > 1:
        lags = lag * np.arange(1, len(p))
        with np.errstate(over='raise', invalid='raise'):
            for start in range(len(x), len(wave), lag):
                rows = np.arange(start, min(start + lag, len(wave)))
                wave[rows] = wave[rows[:, None] - lags] @ (-p[1:])
    if not np.isfinite(wave).all():
        raise FloatingPointError('nonfinite prediction')
    return wave[len(x):]


def forecast_stable(past, length, *, order=32, lag=64):
    x = _validate(past, order, lag)
    polynomial, reflections = fit_burg(x, order=order, lag=lag)
    predicted = predict_polynomial(x, length, polynomial, lag=lag)
    radius = float(np.max(np.abs(np.roots(polynomial)))) if len(polynomial) > 1 else 0.
    if radius > 1 + 1e-7:
        # A numerical implementation failure is not silently reported as stable.
        raise FloatingPointError('numerically unstable Burg polynomial')
    return predicted, dict(effective_order=len(polynomial) - 1,
        max_pole_radius_per_lag=radius,
        max_abs_reflection=float(np.max(np.abs(reflections))) if len(reflections) else 0.,
        coefficient_l2=float(np.linalg.norm(polynomial[1:])))
