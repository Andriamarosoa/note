"""Normal and compressed-context views of causal guitar audio.

HPR-v2 follows the dual-stream idea used in Andriamarosoa/midi:
the rescue branch sees twice as much real past audio, compresses it by pairwise
averaging, and keeps physical pitch labels unchanged by evaluating the
compressed sequence at the effective half sample rate.

The normal branch is unchanged. No future samples are added.
"""
import numpy as np

from scripts.train_v100_spectral_string_slots import _pcm_window

SAMPLE_RATE = 44100
COMPRESSION_FACTOR = 2
COMPRESSED_SAMPLE_RATE = SAMPLE_RATE // COMPRESSION_FACTOR
FRAME_ENDS = -1052 + 128*np.arange(31, dtype=np.int64)
FREQUENCIES = np.geomspace(55., 6000., 64)
WINDOWS = (256, 2048)
FFT_SIZE = 2048
POWER_UNIT = 1e-6

CONFIG = dict(
    sample_rate=SAMPLE_RATE,
    rescue_effective_sample_rate=COMPRESSED_SAMPLE_RATE,
    semitones=0,
    compression_factor=COMPRESSION_FACTOR,
    compression="pairwise_average_pooling",
    pitch_preserved=True,
    original_clock_preserved=True,
    normal_source_window_samples=list(WINDOWS),
    rescue_source_window_samples=[COMPRESSION_FACTOR*n for n in WINDOWS],
    compressed_window_samples=list(WINDOWS),
    frame_ends=FRAME_ENDS.tolist(),
    frequencies_hz=FREQUENCIES.tolist(),
    fft_size=FFT_SIZE,
    power_unit=POWER_UNIT,
    map_shape=[31,64,4],
    channels=[
        "normal_256",
        "compressed_context_512_to_256",
        "normal_2048",
        "compressed_context_4096_to_2048",
    ],
    added_audio_lookahead_samples=0,
    added_source_history_samples=max(WINDOWS),
    filter_boundary="none; pairwise averaging only",
)


def compress_context(frames):
    """Average adjacent real samples, exactly like AveragePooling1D(2, 2)."""
    x = np.asarray(frames, np.float64)
    if x.ndim < 1 or x.shape[-1] < 2 or x.shape[-1] % 2 or not np.isfinite(x).all():
        raise ValueError("finite even-length audio frames required")
    return .5*(x[..., 0::2] + x[..., 1::2])


def _band_power(frames, *, sample_rate=SAMPLE_RATE):
    frames = np.asarray(frames, np.float64)
    taper = np.hanning(frames.shape[-1])
    spectra = np.fft.rfft(frames*taper, n=FFT_SIZE, axis=-1)/taper.sum()
    power = np.abs(spectra)**2
    grid = np.fft.rfftfreq(FFT_SIZE, 1/float(sample_rate))
    if FREQUENCIES[-1] >= grid[-1]:
        raise ValueError("frequency grid exceeds Nyquist")
    right = np.searchsorted(grid, FREQUENCIES)
    weight = (FREQUENCIES-grid[right-1])/(grid[right]-grid[right-1])
    return np.log1p(
        (power[..., right-1]*(1-weight) + power[..., right]*weight)/POWER_UNIT
    )


def time_frequency_map(samples, origin, *, include_pitch=True):
    """Read source [origin-5148, origin+2788); rescue adds past, never future."""
    x = np.asarray(samples)
    if isinstance(origin, bool) or not isinstance(origin, (int, np.integer)) or origin < 0 or x.ndim != 1:
        raise ValueError("invalid mono audio or group origin")

    longest = COMPRESSION_FACTOR*max(WINDOWS)
    begin = int(origin) + int(FRAME_ENDS[0]) - longest
    span = int(FRAME_ENDS[-1]-FRAME_ENDS[0]) + longest
    segment = _pcm_window(x, begin, span).astype(np.float64)
    if not np.isfinite(segment).all():
        raise ValueError("nonfinite audio in authorized support")

    endpoints = longest + 128*np.arange(31)
    features = []
    for length in WINDOWS:
        normal = segment[endpoints[:, None]-length+np.arange(length)]
        normal_power = _band_power(normal)

        if include_pitch:
            source_length = COMPRESSION_FACTOR*length
            rescue_source = segment[
                endpoints[:, None]-source_length+np.arange(source_length)
            ]
            rescue = compress_context(rescue_source)
            rescue_power = _band_power(
                rescue, sample_rate=COMPRESSED_SAMPLE_RATE
            )
        else:
            rescue_power = normal_power

        features.extend([normal_power, rescue_power])

    result = np.stack(features, axis=-1).astype(np.float32)
    if result.shape != (31,64,4) or not np.isfinite(result).all():
        raise FloatingPointError("invalid high-pitch-v2 feature map")
    return result
