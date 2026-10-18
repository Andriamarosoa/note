"""S724: causal loudness-relative harmonic timbre and E2->E3 onset evidence.

The S708 288D features/checkpoint remain byte-for-byte untouched.
Add 32 features per pitch using ONLY the current and previous received S706
sharp-amplitude blocks. No gain normalization of note tails, no future data.

The 2nd partial of E2 == fundamental of E3. A new E3 may preferentially
excite EVEN partials of an already sounding E2, whereas E2 plucked harder
also excites odd partials. This ratio is an acoustic clue, NOT an oracle.
Ratios at low signal-to-noise are imperfect; no automatic new notes.
"""
from __future__ import annotations
import numpy as np
from causal_note.s723_polyphonic_novelty import BASE,EXTRA,TOTAL,init_from_s708

SHARP_START=192


def harmonic_timbre_features(base):
    """[frames,49,32], causal, level-aware + level-relative evidence.

    output: positive onset deltas by partial (8), level-relative differential
    harmonic shape (8), current level-relative timbre (8), lower octave
    even-vs-odd differences by FFT resolution (3), lower 19-semitone
    3rd-multiple vs nonmultiple changes (3), and neighbor-valid masks (2).
    """
    x=np.asarray(base,np.float32)
    if x.ndim!=3 or x.shape[1:]!=(49,BASE) or not np.isfinite(x).all():
        raise ValueError("S724 requires original finite [time,49,288] acoustic ABI")
    n=len(x)
    if not n:
        return np.empty((0,49,EXTRA),np.float32)
    logamp=x[:,:,SHARP_START:SHARP_START+24].reshape(n,49,3,8)
    past=np.concatenate((logamp[:1],logamp[:-1]),axis=0)
    delta=logamp-past
    # Keep absolute onset energy for sensitivity to soft true plucks.
    new_energy=np.maximum(delta,0.).mean(axis=2)
    # Subtract common per-pitch harmonic intensity changes: gain/tone
    # invariant in high-SNR logarithmic approximation, but not exact.
    shape_change=(delta-delta.mean(axis=-1,keepdims=True)).mean(axis=2)
    timbre_shape=(logamp-logamp.mean(axis=-1,keepdims=True)).mean(axis=2)
    pitch=np.arange(49)
    oct_parent=np.clip(pitch-12,0,48)
    fifth_parent=np.clip(pitch-19,0,48)
    oct_valid=(pitch>=12).astype(np.float32)
    fifth_valid=(pitch>=19).astype(np.float32)
    # On an E2->E3 onset, the even E2 harmonics may grow faster than odd:
    # at parent MIDI E2 index 0, E3 is offset 12. This distinguishes an
    # even-only change from a global change in E2's loudness.
    octave_delta=delta[:,oct_parent]
    even_odd=(octave_delta[:,:,:,[1,3,5,7]].mean(-1) -
              octave_delta[:,:,:,[0,2,4,6]].mean(-1))
    even_odd*=oct_valid[None,:,None]
    # For 19 semitones below (~3rd harmonic), distinguish growth of
    # third-multiple partials (3rd and 6th) from other parent partials.
    parent19_delta=delta[:,fifth_parent]
    threes=(parent19_delta[:,:,:,[2,5]].mean(-1) -
            parent19_delta[:,:,:,[0,1,3,4,6,7]].mean(-1))
    threes*=fifth_valid[None,:,None]
    flags=np.broadcast_to(np.stack((oct_valid,fifth_valid),-1)[None],
                          (n,49,2))
    out=np.concatenate((new_energy,shape_change,timbre_shape,
                        even_odd,threes,flags),axis=-1)
    if out.shape!=(n,49,EXTRA) or not np.isfinite(out).all():
        raise AssertionError("S724 32D timbre/loudness feature contract failed")
    return out.astype(np.float32)


def augmented_track(base):
    raw=np.asarray(base,np.float32)
    return np.concatenate((raw,harmonic_timbre_features(raw)),axis=-1).astype(np.float32)
