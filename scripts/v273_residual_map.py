"""Causal, multiresolution residual maps; no labels and no K prediction.

Predict consecutive 128-sample blocks from the preceding 8820 PCM samples.
Observed samples update the history only after that block prediction is fixed.
Short/long spectra share explicit right endpoints and fixed physical scaling.
Exact proposal ownership is supplied separately, never used as an audio mask.
"""
import numpy as np

from scripts.v273_stable_prediction import forecast_stable
from scripts.v273_ownership_experiment import sorted_proposals, lookup, frame_fraction
from scripts.train_v100_spectral_string_slots import _pcm_window

HISTORY = 8820
BLOCK = 128
FRAME_ENDS = -1052 + 128*np.arange(31, dtype=np.int64)
FREQUENCIES = np.geomspace(55., 6000., 64)
WINDOWS = (256, 2048)
CHANNELS = ('observed_256', 'residual_256', 'observed_2048', 'residual_2048')
POWER_UNIT = 1e-6
FFT_SIZE = 2048
SAMPLE_RATE = 44100
CONFIG = dict(history=HISTORY, block=BLOCK, order=32, lag=64,
    frame_ends=FRAME_ENDS.tolist(), windows=list(WINDOWS),
    frequencies_hz=FREQUENCIES.tolist(), fft_size=FFT_SIZE,
    channels=list(CHANNELS), power_unit=POWER_UNIT, clip_applied=False,
    map_shape=[31, 64, 4], ownership_samples=4096,
    forecast_kind='past-only Burg; block length 128; fixed absolute sample grid')


class ResidualStream:
    """Chunk sizes and unseen future samples cannot affect emitted residuals."""
    def __init__(self):
        self.past = np.zeros(HISTORY, np.float64)
        self.seen = 0
        self.prediction = None
        self.offset = 0
        self.blocks = []

    def process(self, samples):
        x = np.asarray(samples, np.float64)
        if x.ndim != 1 or not np.isfinite(x).all():
            raise ValueError('expected finite normalized mono PCM')
        residual = np.empty(len(x), np.float32)
        cursor = 0
        while cursor < len(x):
            if self.prediction is None:
                self.prediction, info = forecast_stable(self.past, BLOCK)
                self.blocks.append(dict(start=self.seen, **info))
                self.offset = 0
            size = min(len(x)-cursor, BLOCK-self.offset)
            observed = x[cursor:cursor+size]
            residual[cursor:cursor+size] = observed-self.prediction[self.offset:self.offset+size]
            self.past = np.r_[self.past, observed][-HISTORY:]
            self.seen += size; cursor += size; self.offset += size
            if self.offset == BLOCK:
                self.prediction = None
        if not np.isfinite(residual).all():
            raise FloatingPointError('residual is nonfinite after float32 storage')
        return residual


def innovation(samples):
    stream = ResidualStream()
    residual = stream.process(samples)
    return residual, stream.blocks


def time_frequency_map(observed, residual, origin):
    """Read no samples at or after origin+2788; negative/file-edge support is zero.

    Channel pairs differ only by observed versus residual PCM. Window division
    by sum(Hann) and constant POWER_UNIT never inspect labels or future frames.
    Frequency interpolation/zero padding do not increase physical resolution.
    """
    if (isinstance(origin, bool) or not isinstance(origin, (int, np.integer)) or origin < 0
            or np.asarray(observed).ndim != 1 or np.asarray(residual).ndim != 1
            or len(observed) != len(residual)):
        raise ValueError('invalid audio, residual or group origin')
    # The longest first window starts at origin-3100. All 31 endpoints are fixed.
    begin = int(origin)+int(FRAME_ENDS[0])-max(WINDOWS)
    span = int(FRAME_ENDS[-1]-FRAME_ENDS[0])+max(WINDOWS)
    segments = [_pcm_window(x, begin, span).astype(np.float64) for x in (observed, residual)]
    if any(not np.isfinite(x).all() for x in segments):
        raise ValueError('nonfinite observed map support')
    endpoints = max(WINDOWS)+128*np.arange(31)
    grid = np.fft.rfftfreq(FFT_SIZE, 1/SAMPLE_RATE)
    right = np.searchsorted(grid, FREQUENCIES)
    weight = (FREQUENCIES-grid[right-1])/(grid[right]-grid[right-1])
    features = []
    for length in WINDOWS:
        taper = np.hanning(length)
        indices = endpoints[:, None]-length+np.arange(length)
        for segment in segments:
            spectra = np.fft.rfft(segment[indices]*taper, n=FFT_SIZE, axis=1)/taper.sum()
            power = np.abs(spectra)**2
            band_power = power[:, right-1]*(1-weight)+power[:, right]*weight
            features.append(np.log1p(band_power/POWER_UNIT))
    result = np.stack(features, axis=-1).astype(np.float32)
    if result.shape != (31, 64, 4) or not np.isfinite(result).all():
        raise FloatingPointError('invalid map')
    return result


def ownership_input(groups, group_id, *, complete_through, decision_sample, proposals=None):
    """Exact sample mask plus lossy frame fractions; no annotated onsets.

    A timestamp watermark certifies proposal completeness, not just read audio.
    The +5 bound follows the frozen peak/merge producer's sample support; it is
    not a measured live latency guarantee.
    """
    if not isinstance(group_id, (int, np.integer)) or not 0 <= group_id < len(groups):
        raise ValueError('invalid group')
    current = np.asarray(groups[group_id])
    if (current.ndim != 1 or current.dtype.kind not in 'iu' or len(current) == 0 or
            current[0] < 0 or np.any(np.diff(current) < 0) or current[-1]-current[0] > 1764):
        raise ValueError('invalid full group')
    origin = int(current[0]); needed = int(current[-1])+1764
    if complete_through < needed or decision_sample < max(origin+2788, complete_through+5):
        raise ValueError('incomplete proposal or audio decision context')
    positions, ids = sorted_proposals(groups) if proposals is None else proposals
    end = np.searchsorted(positions, complete_through, side='right')
    queries = origin-1308+np.arange(4096, dtype=np.int64)
    owners, _ = lookup(positions[:end], ids[:end], queries)
    owned = (queries >= 0) & (owners == group_id)
    # All locally eligible onsets precede current[-1]+882, hence earlier than
    # the completeness bound. Later queries must never be declared owned.
    if np.any(owned & (queries > int(current[-1])+882)):
        raise RuntimeError('ownership outside the admissible interval')
    short_fraction = frame_fraction(owned)
    extended = np.r_[np.zeros(1792, bool), owned]
    cumulative = np.r_[0, np.cumsum(extended, dtype=np.int32)]
    ends = 2048+128*np.arange(31)
    long_fraction = ((cumulative[ends]-cumulative[ends-2048])/2048.).astype(np.float32)
    return np.packbits(owned, bitorder='little'), np.stack([short_fraction, long_fraction], axis=-1)


def onset_coverage(relative_onsets):
    """Audit-only positive-Hann support, not a predictor input or audibility claim."""
    q = np.asarray(relative_onsets, np.int64)
    return (q[:, None] > FRAME_ENDS[None, :]-256) & (q[:, None] < FRAME_ENDS[None, :]-1)
