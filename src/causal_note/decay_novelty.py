"""Past-only exponential-envelope evidence for an already causal CQT crop.

This is an acoustic feature, not a note counter. A separate, paired diagnostic
measures whether it adds useful information to frozen V27.3 predictions.
"""
from __future__ import annotations

import numpy as np

FRAMES = 24
PRE_FRAMES = 17
HOP_SECONDS = 256 / 44100
LOG_GAIN = 100.0
AMPLITUDE_FLOOR = 1e-6
RELATIVE_FLOOR = 0.02
MIN_VALID_FRAMES = 8
MIN_DECAY_PER_SECOND = 0.25
MAX_DECAY_PER_SECOND = 60.0
MAX_LOG_RMSE = 0.30


def estimate_decay(log_magnitude):
    """Fit ln(amplitude) using only the first 17 of the 24 cached frames.

    The crop ends <= cluster_start + 1764 samples. Frame 16 ends at least
    7 * 256 samples earlier, hence strictly before cluster_start, including
    a short end-of-track crop. No new future samples or lookahead are used.
    Magnitudes are reconstructed from the cache's ln(1 + 100 * amplitude).
    """
    x = np.asarray(log_magnitude, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] != FRAMES or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('expected finite nonnegative [batch,24,frequency] log magnitudes')
    if (x > 30).any():
        raise ValueError('magnitude outside supported numerical range')
    amplitude = np.expm1(x[:, :PRE_FRAMES]) / LOG_GAIN
    floor = np.maximum(AMPLITUDE_FLOOR, RELATIVE_FLOOR * amplitude.max(axis=1, keepdims=True))
    valid = amplitude > floor
    y = np.log(np.maximum(amplitude, AMPLITUDE_FLOOR))
    t = (np.arange(PRE_FRAMES) - PRE_FRAMES + 1)[None, :, None] * HOP_SECONDS
    count = valid.sum(axis=1)
    denominator = np.maximum(count, 1)
    mean_t = (t * valid).sum(axis=1) / denominator
    mean_y = (y * valid).sum(axis=1) / denominator
    centered_t = t - mean_t[:, None, :]
    variance = (centered_t**2 * valid).sum(axis=1)
    slope = (centered_t * (y - mean_y[:, None, :]) * valid).sum(axis=1) / np.maximum(variance, 1e-12)
    intercept = mean_y - slope * mean_t
    fitted = intercept[:, None, :] + slope[:, None, :] * t
    rmse = np.sqrt(((y - fitted)**2 * valid).sum(axis=1) / denominator)
    reliable = ((count >= MIN_VALID_FRAMES) & (slope <= -MIN_DECAY_PER_SECOND)
                & (slope >= -MAX_DECAY_PER_SECOND) & (rmse <= MAX_LOG_RMSE))
    future_t = np.arange(1, FRAMES - PRE_FRAMES + 1)[None, :, None] * HOP_SECONDS
    predicted_ln = intercept[:, None, :] + np.clip(slope, -MAX_DECAY_PER_SECOND, 0)[:, None, :] * future_t
    predicted_log = np.log1p(LOG_GAIN * np.exp(np.clip(predicted_ln, -40, 20)))
    novelty = np.maximum(x[:, PRE_FRAMES:] - predicted_log, 0) * reliable[:, None, :]
    return novelty.astype(np.float32), {
        'slope': slope, 'intercept': intercept, 'log_rmse': rmse,
        'reliable': reliable, 'predicted_log': predicted_log,
    }


def pitch_pool(values):
    """Average adjacent 3-bin semitone groups; keep the final short group."""
    x = np.asarray(values)
    return np.stack([x[..., i:i+3].mean(axis=-1) for i in range(0, x.shape[-1], 3)], axis=-1)


def summarize_crop(features):
    """Identical ordinary acoustic evidence, plus two decay-novelty summaries."""
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 4 or x.shape[1:] != (FRAMES, 238, 3) or not np.isfinite(x).all():
        raise ValueError('expected finite [batch,24,238,3] causal features')
    if (x < 0).any():
        raise ValueError('causal magnitude and positive novelty channels cannot be negative')
    log = x[..., 0]
    ordinary = [log[:, :PRE_FRAMES].mean(axis=1), log[:, PRE_FRAMES-1],
                log[:, PRE_FRAMES:].mean(axis=1), log[:, PRE_FRAMES:].max(axis=1),
                x[:, PRE_FRAMES:, :, 1].mean(axis=1), x[:, PRE_FRAMES:, :, 2].max(axis=1)]
    novelty, audit = estimate_decay(log)
    added = [novelty.mean(axis=1), novelty.max(axis=1)]
    static_novelty = np.maximum(log[:, PRE_FRAMES:] - log[:, PRE_FRAMES-1:PRE_FRAMES], 0)
    audit = {
        'reliable_bin_fraction': audit['reliable'].mean(axis=1).astype(np.float32),
        'decay_only_max': np.max(np.where(static_novelty <= 1e-6, novelty, 0), axis=(1, 2)),
    }
    return (np.concatenate([pitch_pool(a) for a in ordinary], axis=1).astype(np.float32),
            np.concatenate([pitch_pool(a) for a in added], axis=1).astype(np.float32), audit)
