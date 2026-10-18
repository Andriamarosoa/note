"""S706 extra native-acoustic inputs: narrow interpolation and phase change.

Unlike the original +/-1-bin maximum, each candidate is sampled at its own
frequency. Hann windows remain entirely BEFORE the current decision sample.
No pitch labels, note durations, future PCM or pretrained foreign engine.
"""
import numpy as np
from causal_note.strict_live import SR,PITCHES,WINDOWS,HOP


def sharp_track(audio, ends):
    ends=np.asarray(ends,np.int64)
    if audio.ndim!=1 or not np.isfinite(audio).all():
        raise ValueError('Finite mono PCM required')
    if np.any(ends>len(audio)) or np.any(ends<1) or np.any(np.diff(ends)<=0):
        raise ValueError('Unavailable or non-monotonic decision sample')
    hz=(440*2.**((PITCHES-69)/12.))[:,None]*np.arange(1,9)[None,:]
    centers=[];windows=[];phase=[];grid=[]
    for n in WINDOWS:
        centers.append(hz*n/SR)
        windows.append(np.hanning(n))
        grid.append(np.arange(n//2+1))
        # Align FFT-bin phases to the common center of the PAST window
        # before interpolation; interpolation of raw opposing phases cancels.
        phase.append(np.exp(1j*2*np.pi*np.arange(n//2+1)*(n-1)/(2*n)))
    result=[];previous=None;previous_end=None
    for end in ends:
        logs=[];contrasts=[];complex_values=[]
        for n,window,center,rot,bins in zip(WINDOWS,windows,centers,phase,grid):
            pcm=np.asarray(audio[max(0,int(end)-n):int(end)],np.float64)
            if len(pcm)<n:pcm=np.pad(pcm,(n-len(pcm),0))
            spectrum=np.fft.rfft(pcm*window)*(2/window.sum())
            amplitude=np.abs(spectrum)
            def interp(values,freq):
                return np.interp(freq.ravel(),bins,values).reshape(49,8)
            own=interp(amplitude,center)
            sides=(interp(amplitude,center*2**(-.5/12))+
                   interp(amplitude,center*2**(.5/12)))*.5
            logs.append(np.log1p(600*own))
            contrasts.append(np.log1p(600*own)-np.log1p(600*sides))
            aligned=spectrum*rot
            value=interp(aligned.real,center)+1j*interp(aligned.imag,center)
            complex_values.append(value)
        value=np.stack(complex_values,1)
        cos=np.zeros((49,3,8));sin=np.zeros_like(cos)
        if previous is not None:
            advance=np.exp(-1j*2*np.pi*hz[:,None,:]*(int(end)-previous_end)/SR)
            prod=value*np.conj(previous)*advance
            norm=np.abs(value)*np.abs(previous)
            valid=norm>1e-12
            coherence=np.divide(prod,np.maximum(norm,1e-12),out=np.zeros_like(prod),where=valid)
            cos=coherence.real;sin=coherence.imag
        previous=value;previous_end=int(end)
        result.append(np.concatenate((np.stack(logs,1).reshape(49,24),
            np.stack(contrasts,1).reshape(49,24),cos.reshape(49,24),sin.reshape(49,24)),1))
    out=np.asarray(result,np.float32)
    if out.shape!=(len(ends),49,96) or not np.isfinite(out).all():
        raise AssertionError('Sharp harmonic feature ABI failure')
    return out
