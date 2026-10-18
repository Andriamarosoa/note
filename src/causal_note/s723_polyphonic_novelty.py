"""S723 additional per-pitch causal novelty evidence, complementing learned Exact-K.

Append-only 32-dimensional ABI extension. NEVER edit original S708 288-D
features or checkpoint. The first 24 dimensions quantify new positive harmonic
energy since the previous *received* 4-hop decision, rather than broadcasting
one global full-spectrum flux to every pitch. The next four dimensions correct
the missing-lower-octave/fifth boundary handling in S701 without changing its
historically trained columns. Remaining four summarize aligned onset evidence.

All inputs are current/past audio-derived. No labels, audio beyond the current
frame, decoded future notes, or ground-truth Exact-K enter this transform.
"""
from __future__ import annotations
import numpy as np

BASE=288
EXTRA=32
TOTAL=320
START_SHARP=192


def causal_novelty_track(base):
    """Append-only [time,49,32] features from the exact S708 [time,49,288]."""
    x=np.asarray(base,np.float32)
    if x.ndim!=3 or x.shape[1:]!=(49,BASE) or not np.isfinite(x).all():
        raise ValueError("S723 expects original finite [frames,49,288] S708 ABI")
    n=len(x)
    if not n:
        return np.empty((0,49,EXTRA),np.float32)
    # S706 starts at 192D. Its first 24 channels are log amplitudes
    # of 3 resolutions * 8 harmonic partials.
    amp=x[:,:,START_SHARP:START_SHARP+24]
    previous=np.concatenate((amp[:1],amp[:-1]),axis=0)
    positive=np.maximum(amp-previous,0.)
    # S701 fixed column 15 is per-pitch F_rise, original columns 27/28
    # subtract f_rise at a clamped lower index even for nonexisting pitches.
    frise=x[:,:,126+15]
    pitches=np.arange(49)
    low12=np.maximum(0,pitches-12)
    low19=np.maximum(0,pitches-19)
    valid12=(pitches>=12).astype(np.float32)
    valid19=(pitches>=19).astype(np.float32)
    lower12=np.where(valid12[None,:]>0,frise[:,low12],0.)
    lower19=np.where(valid19[None,:]>0,frise[:,low19],0.)
    corrected12=frise-lower12
    corrected19=frise-lower19
    # Four acoustic summaries: positive growth of fundamental/2nd/3rd
    # harmonics, and shared (minimum) increase across all three.
    flux=positive.reshape(n,49,3,8)
    fund=flux[:,:,:,0].mean(-1)
    second=flux[:,:,:,1].mean(-1)
    third=flux[:,:,:,2].mean(-1)
    simultaneous=np.minimum(np.minimum(fund,second),third)
    out=np.concatenate((
        positive,
        corrected12[...,None],
        corrected19[...,None],
        np.broadcast_to(valid12[None,:,None],(n,49,1)),
        np.broadcast_to(valid19[None,:,None],(n,49,1)),
        fund[...,None],second[...,None],third[...,None],simultaneous[...,None]
    ),axis=-1)
    if out.shape!=(n,49,EXTRA) or not np.isfinite(out).all():
        raise AssertionError("S723 failed 32D causal extra feature ABI")
    return out.astype(np.float32)


def augmented_track(base):
    x=np.asarray(base,np.float32)
    return np.concatenate((x,causal_novelty_track(x)),axis=-1).astype(np.float32)


def init_from_s708(model,extension_mean,extension_scale):
    """Same original frozen network weights, with zero new-input weights.

    At initialization, the augmented model is exactly S708 regardless of
    novelty features (to standard FP32 tolerance). Extra channels gain
    influence only through FIT-only training.
    """
    import torch
    from causal_note.s705_temporal_heads import TemporalHeads
    mean=torch.cat((model.mean.detach().cpu(),torch.as_tensor(extension_mean,dtype=torch.float32)))
    scale=torch.cat((model.scale.detach().cpu(),torch.as_tensor(extension_scale,dtype=torch.float32)))
    if mean.shape!=(TOTAL,) or scale.shape!=(TOTAL,) or (scale<=0).any():
        raise ValueError("S723 extension normalization invalid")
    trial=TemporalHeads(mean,scale,width=model.width)
    orig=model.state_dict()
    state=trial.state_dict()
    for key,value in orig.items():
        if key in ("mean","scale","encode.0.weight"):
            continue
        if key not in state or state[key].shape!=value.shape:
            raise AssertionError("S723 network changed beyond append-only first layer")
        state[key]=value.clone()
    state["encode.0.weight"].zero_()
    state["encode.0.weight"][:,:BASE]=orig["encode.0.weight"]
    trial.load_state_dict(state,strict=True)
    return trial
