"""S634: three acoustic VIEWS of the same PCM: half, native, double pitch.

Reference: Andriamarosoa/midi codex/dual-stream-bass uses a separately
trained compressed bass branch (AveragePooling1D(2,2)). In NOTE v273 HPR
the x2 view was already investigated. S634 deliberately treats those as
EXPERIMENTAL diagnostic input transforms, not a new trained reference.

These views correspond to -12, 0, +12 semitones (frequency /2, x1, x2).
All three use ONLY real captured audio at or before each decision hop.
The 2x view needs 16384 original historical samples to fill an 8192
virtual window. It adds history, not future lookahead.
"""
from __future__ import annotations
import numpy as np
from causal_note.strict_live import CausalPitchFeatures,HOP,PITCHES,SR,WINDOWS

FACTOR = 2
OCTAVE = 12
EXTENDED_HISTORY = 2*WINDOWS[-1]


def remap_pitch_probabilities(probabilities, transposition: int):
    """Shift DETECTED MIDI bins back from virtual transposition.

    When sound is transposed +12, original note i is read from prediction
    i+12. When sound is transposed -12, original i from prediction i-12.
    Outside the trained [40,88] note axis, evidence is absent, not extrapolated.
    """
    p=np.asarray(probabilities,np.float32)
    if p.shape != (len(PITCHES),3) or not np.isfinite(p).all():
        raise ValueError("Expected native 49x3 finite pitch probabilities")
    if transposition not in (-12,0,12):
        raise ValueError("S634 only supports actual octave factor two")
    output=np.zeros_like(p)
    if transposition == 0:
        output[:]=p
    elif transposition > 0:
        output[:-OCTAVE]=p[OCTAVE:]
    else:
        output[OCTAVE:]=p[:-OCTAVE]
    return output


def warp_past_window(past):
    """Return two right-edge anchored 8192-sample virtual acoustic windows.

    Up x2: pair-average the last 16384 real samples, interpreted at 44.1k;
    down /2: repeat the last 4096 past samples; no future interpolation.
    Pair averaging follows midi dual-stream bass design. The down view is
    diagnostic zero-order hold and is NOT a trained pitch-shift model.
    """
    past=np.asarray(past,np.float32)
    if past.shape != (EXTENDED_HISTORY,) or not np.isfinite(past).all():
        raise ValueError("Expected 16384 finite past mono PCM samples")
    doubled=.5*(past[0::2]+past[1::2])
    halved=np.repeat(past[-WINDOWS[-1]//2:],2)
    assert doubled.shape==halved.shape==(WINDOWS[-1],)
    return doubled.astype(np.float32),halved.astype(np.float32)


class ThreeViewPitchProbe:
    """Research-only 3x 126D/native-head evaluator. Never owns MIDI events.

    The baseline feature row is produced by exactly the S613 CausalPitchFeatures
    state machine; the other two states use its internal acoustic _frame().
    Returned windows share the physical decision_sample; no audio is backdated.
    """

    def __init__(self, heads):
        self.heads=heads
        self.normal=CausalPitchFeatures(hop=HOP)
        self.up=CausalPitchFeatures(hop=HOP)
        self.down=CausalPitchFeatures(hop=HOP)
        self.history=np.zeros(EXTENDED_HISTORY,np.float32)
        self.pending=np.empty(0,np.float32)
        self.received=0
        self.processed=0
        self.records=0

    def push(self,samples,*,start_sample):
        audio=np.asarray(samples)
        if start_sample != self.received or audio.ndim!=1 or audio.dtype.kind!='f' or not np.isfinite(audio).all():
            raise ValueError("S634 requires contiguous, finite mono PCM")
        self.received+=len(audio)
        self.pending=np.concatenate((self.pending,audio.astype(np.float32,copy=False)))
        out=[]
        while len(self.pending)>=HOP:
            part=self.pending[:HOP].copy()
            self.pending=self.pending[HOP:]
            native=self.normal.push(part,start_sample=self.processed)
            self.processed+=HOP
            assert len(native)==1 and native[0][0]==self.processed
            # Every source sample in this 16384 history was physically
            # captured at or before self.processed. Extra history != future.
            self.history[:-HOP]=self.history[HOP:]
            self.history[-HOP:]=part
            up_frame,down_frame=warp_past_window(self.history)
            x=np.stack((native[0][1],self.up._frame(up_frame),
                        self.down._frame(down_frame)))
            assert x.shape==(3,49,126)
            pred,ks=self.heads.predict(x)
            if pred.shape!=(3,49,3) or ks.shape!=(3,7):
                raise ValueError("Native frozen heads must support batch of 3 windows")
            # First 16384 samples contain zero-padded history. The alternative
            # views are not trusted before the long source window is populated.
            valid=self.processed>=EXTENDED_HISTORY
            out.append(dict(
                decision_sample=self.processed,
                available_sample=self.received,
                native=pred[0].copy(),
                native_k=ks[0].copy(),
                bass_rescue=(remap_pitch_probabilities(pred[1],12)
                             if valid else np.zeros((49,3),np.float32)),
                high_rescue=(remap_pitch_probabilities(pred[2],-12)
                             if valid else np.zeros((49,3),np.float32)),
                valid_rescue=bool(valid),
            ))
            self.records+=1
        return out
