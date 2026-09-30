"""Normal and octave-up views of exactly the same past audio samples.

Each frame is filtered and decimated independently. Zero extension at its edges
uses no actual sample after its right endpoint. The original sample clock,
proposal ownership, and source-window lengths remain unchanged.
"""
from functools import lru_cache
import numpy as np

from scripts.train_v100_spectral_string_slots import _pcm_window

SAMPLE_RATE = 44100
FRAME_ENDS = -1052 + 128*np.arange(31, dtype=np.int64)
FREQUENCIES = np.geomspace(55., 6000., 64)
WINDOWS = (256, 2048)
FFT_SIZE = 2048
POWER_UNIT = 1e-6
FILTER_TAPS = 127
FILTER_CUTOFF_CYCLES = .225
FILTER_KAISER_BETA = 8.6
_n = np.arange(FILTER_TAPS) - (FILTER_TAPS-1)//2
FIR = 2*FILTER_CUTOFF_CYCLES*np.sinc(2*FILTER_CUTOFF_CYCLES*_n)*np.kaiser(FILTER_TAPS, FILTER_KAISER_BETA)
FIR /= FIR.sum()
CONFIG = dict(sample_rate=SAMPLE_RATE, semitones=12, compression_factor=2,
    original_clock_preserved=True, source_window_samples=list(WINDOWS),
    compressed_window_samples=[n//2 for n in WINDOWS],
    frame_ends=FRAME_ENDS.tolist(), frequencies_hz=FREQUENCIES.tolist(),
    fft_size=FFT_SIZE, power_unit=POWER_UNIT, map_shape=[31,64,4],
    channels=['normal_256','octave_up_from_256','normal_2048','octave_up_from_2048'],
    filter_taps=FILTER_TAPS, filter_cutoff_cycles_per_sample=FILTER_CUTOFF_CYCLES,
    filter_kaiser_beta=FILTER_KAISER_BETA, filter_boundary='zero outside each past frame',
    decimation_phase=1, added_audio_lookahead_samples=0, added_source_history_samples=0)


@lru_cache(maxsize=8)
def _filter_spectrum(length):
    size = 1 << (int(length)+FILTER_TAPS-2).bit_length()
    return size, np.fft.rfft(FIR, n=size)


def octave_up(frames):
    """Low-pass and keep samples 1,3,...,N-1, interpreted at the original fs.

This is a real factor-two time compression, not ordinary sample-rate conversion:
the output is interpreted at 44.1 kHz, so a tone f becomes 2f. Centered filtering
uses only the supplied past frame and synthetic zero extension at both edges.
"""
    x = np.asarray(frames, np.float64)
    if x.ndim < 1 or x.shape[-1] < 2 or x.shape[-1] % 2 or not np.isfinite(x).all():
        raise ValueError('finite even-length audio frames required')
    length = x.shape[-1]
    size, response = _filter_spectrum(length)
    full = np.fft.irfft(np.fft.rfft(x,n=size,axis=-1)*response,n=size,axis=-1)
    center = (FILTER_TAPS-1)//2
    return full[...,center:center+length][...,1::2]


def _band_power(frames):
    taper = np.hanning(frames.shape[-1])
    spectra = np.fft.rfft(frames*taper,n=FFT_SIZE,axis=-1)/taper.sum()
    power = np.abs(spectra)**2
    grid = np.fft.rfftfreq(FFT_SIZE,1/SAMPLE_RATE)
    right = np.searchsorted(grid,FREQUENCIES)
    weight = (FREQUENCIES-grid[right-1])/(grid[right]-grid[right-1])
    return np.log1p((power[...,right-1]*(1-weight)+power[...,right]*weight)/POWER_UNIT)


def time_frequency_map(samples, origin, *, include_pitch=True):
    """Read source [origin-3100, origin+2788); all frames stay on source time."""
    x = np.asarray(samples)
    if isinstance(origin,bool) or not isinstance(origin,(int,np.integer)) or origin < 0 or x.ndim != 1:
        raise ValueError('invalid mono audio or group origin')
    begin = int(origin)+int(FRAME_ENDS[0])-max(WINDOWS)
    span = int(FRAME_ENDS[-1]-FRAME_ENDS[0])+max(WINDOWS)
    segment = _pcm_window(x,begin,span).astype(np.float64)
    if not np.isfinite(segment).all():
        raise ValueError('nonfinite audio in authorized support')
    endpoints = max(WINDOWS)+128*np.arange(31)
    features=[]
    for length in WINDOWS:
        frames=segment[endpoints[:,None]-length+np.arange(length)]
        normal=_band_power(frames)
        features.extend([normal,_band_power(octave_up(frames)) if include_pitch else normal])
    result=np.stack(features,axis=-1).astype(np.float32)
    if result.shape != (31,64,4) or not np.isfinite(result).all():
        raise FloatingPointError('invalid high-pitch feature map')
    return result
