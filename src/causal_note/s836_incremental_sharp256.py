"""S836 optional causal sharp spectra cache for S834 PCM256 SHADOW only.

This replaces redundant recomputation of the previous 256-hop spectrum with
a read-only cached result. No neural weights, six axis head/selector/decoder,
MIDI output, timestamps or future samples are changed.
The original sharp_track() stays untouched as scientific reference.
"""
from __future__ import annotations

import numpy as np
from causal_note.strict_live import HOP, SR, PITCHES, WINDOWS


class IncrementalSharp256:
    """Original S706 sharp_track ABI: one newly received PCM256 frame per call."""

    def __init__(self):
        hz=(440*2.**((PITCHES-69)/12.))[:,None]*np.arange(1,9)[None,:]
        self.hz=hz
        self.tables=[]
        for n in WINDOWS:
            center=hz*n/SR
            window=np.hanning(n)
            bins=np.arange(n//2+1)
            rot=np.exp(1j*2*np.pi*np.arange(n//2+1)*(n-1)/(2*n))
            self.tables.append((n,window,center,rot,bins))
        self.previous=None
        self.previous_end=None
        self.last_end=0
        self.calls=0

    def step(self, audio, *, end):
        if type(end) is not int or end != self.last_end+HOP:
            raise ValueError("S836 requires exactly one new causal 256 hop")
        a=np.asarray(audio)
        if a.ndim!=1 or not np.isfinite(a).all() or len(a)<min(end,max(WINDOWS)):
            raise ValueError("S836 requires complete finite past-only audio window")
        logs=[];contrasts=[];complex_values=[]
        for n,window,center,rot,bins in self.tables:
            pcm=np.asarray(a[max(0,len(a)-n):],np.float64)
            if len(pcm)<n:
                pcm=np.pad(pcm,(n-len(pcm),0))
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
            complex_values.append(interp(aligned.real,center)+
                                  1j*interp(aligned.imag,center))

        value=np.stack(complex_values,1)
        cos=np.zeros((49,3,8));sin=np.zeros_like(cos)
        if self.previous is not None:
            advance=np.exp(-1j*2*np.pi*self.hz[:,None,:]*
                           (end-self.previous_end)/SR)
            prod=value*np.conj(self.previous)*advance
            norm=np.abs(value)*np.abs(self.previous)
            valid=norm>1e-12
            coherence=np.divide(prod,np.maximum(norm,1e-12),
                                out=np.zeros_like(prod),where=valid)
            cos=coherence.real;sin=coherence.imag
        out=np.concatenate((np.stack(logs,1).reshape(49,24),
            np.stack(contrasts,1).reshape(49,24),
            cos.reshape(49,24),sin.reshape(49,24)),1)
        result=np.asarray(out,np.float32)
        if result.shape!=(49,96) or not np.isfinite(result).all():
            raise AssertionError("S836 sharp harmonic feature ABI mismatch")
        self.previous=value
        self.previous_end=end
        self.last_end=end
        self.calls+=1
        return result
