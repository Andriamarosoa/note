"""Recovered S4-owned/S5 527D audio features.

The original source file was not found. This implementation was reconstructed
from the S4 protocol and the frozen S5 input bin signatures, then verified
against every saved S5 probability on all 59,309 historical events. It uses
only acoustic inputs and the frozen baseline K, never annotations or cached
teacher predictions. See verify_v273_s5_owned527_recovery.py.
"""
from __future__ import annotations

import numpy as np

LAGS = (1, 2, 4, 8)
ZONE_NAMES = ("native", "dilated2", "before", "after")
EPSILON = 1e-12


def _zones(native: np.ndarray) -> list[np.ndarray]:
    own = native.astype(bool)
    dilated = own.copy()
    for distance in (1, 2):
        dilated[:, distance:] |= own[:, :-distance]
        dilated[:, :-distance] |= own[:, distance:]
    nonempty = own.any(axis=1)
    first = own.argmax(axis=1)
    last = 41 - own[:, ::-1].argmax(axis=1)
    frames = np.arange(42)[None, :]
    return [own, dilated, (frames < first[:, None]) & nonempty[:, None],
            (frames > last[:, None]) & nonempty[:, None]]


def _descriptors(masked_amplitude: np.ndarray) -> np.ndarray:
    # Masked-out frames stay in the 42-frame grid as zeros. Removing them
    # changes temporal means, deviations, and concentration.
    pitch_max = masked_amplitude.max(axis=1)
    ranked = np.sort(pitch_max, axis=1)[:, ::-1]
    mass = pitch_max.sum(axis=1)
    energy = np.square(pitch_max).sum(axis=1)
    concentration = energy / (mass**2 + EPSILON)
    effective = mass**2 / (energy + EPSILON)
    counts = [(pitch_max > ranked[:, :1] * threshold).sum(axis=1)
              for threshold in (.02, .05, .10, .20)]
    flux = masked_amplitude.sum(axis=2)
    flux_mass = flux.sum(axis=1)
    temporal = [flux_mass, flux.mean(axis=1), flux.std(axis=1), flux.max(axis=1),
                np.square(flux).sum(axis=1) / (flux_mass**2 + EPSILON)]
    return np.column_stack([ranked[:, :8], mass, concentration, effective,
                            *counts, *temporal]).astype(np.float32)


def owned416(sequence42x49: np.ndarray, geometry46: np.ndarray) -> np.ndarray:
    """Construct 4 zones × (20 activation + 4 lags × 21 flow) features."""
    sequence = np.asarray(sequence42x49)
    geometry = np.asarray(geometry46)
    n = len(sequence)
    if sequence.shape != (n, 42, 49) or geometry.shape != (n, 46):
        raise ValueError("expected sequence (N,42,49) and geometry (N,46)")
    if not np.isfinite(sequence).all() or not np.isfinite(geometry).all():
        raise ValueError("nonfinite acoustic input")
    if (sequence < 0).any() or not np.isin(geometry[:, :42], (0, 1)).all():
        raise ValueError("expected log1p activations and binary ownership mask")
    if n == 0:
        return np.empty((0, 416), dtype=np.float32)
    # Float64 before expm1, differencing and reductions is necessary to
    # reproduce the original bin thresholds. Cast only the final descriptors.
    amplitude = np.expm1(sequence.astype(np.float64))
    if not np.isfinite(amplitude).all():
        raise ValueError("activation inversion overflow")
    blocks = []
    for mask in _zones(geometry[:, :42]):
        blocks.append(_descriptors(amplitude * mask[:, :, None]))
        for lag in LAGS:
            derivative = np.zeros_like(amplitude)
            derivative[:, lag:] = (amplitude[:, lag:] - amplitude[:, :-lag]) / lag
            derivative *= mask[:, :, None]
            blocks.append(_descriptors(np.maximum(derivative, 0)))
            blocks.append(np.maximum(-derivative, 0).sum(axis=(1, 2))[:, None])
    result = np.column_stack(blocks).astype(np.float32)
    if result.shape != (n, 416) or not np.isfinite(result).all():
        raise ValueError("invalid recovered 416D output")
    return result


def build_owned527(summary58: np.ndarray, geometry46: np.ndarray,
                   sequence42x49: np.ndarray, freeze_k: np.ndarray,
                   chunk_size: int = 512) -> np.ndarray:
    """Pure acoustic constructor; freeze_k is an existing model prediction."""
    summary, geometry, sequence, baseline = map(
        np.asarray, (summary58, geometry46, sequence42x49, freeze_k))
    n = len(summary)
    if (summary.shape != (n, 58) or geometry.shape != (n, 46)
            or sequence.shape != (n, 42, 49) or baseline.shape != (n,)):
        raise ValueError("incompatible original acoustic shapes")
    if chunk_size < 1 or not np.isfinite(summary).all():
        raise ValueError("invalid chunk size or summary")
    if not np.isin(baseline, np.arange(7)).all():
        raise ValueError("freeze_k must contain predicted K0..K6")
    design = np.empty((n, 527), dtype=np.float32)
    design[:, :58] = summary
    design[:, 58:104] = geometry
    design[:, 104:111] = np.eye(7, dtype=np.float32)[baseline.astype(np.intp)]
    for start in range(0, n, chunk_size):
        end = start + chunk_size
        design[start:end, 111:] = owned416(sequence[start:end], geometry[start:end])
    if not np.isfinite(design).all():
        raise ValueError("nonfinite recovered design")
    return design
