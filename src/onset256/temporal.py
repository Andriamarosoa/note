"""Exact-K frozen neural backbone and strictly causal count head.
The old S708 model contains auxiliary pitch prediction weights as internal
representation, but this branch exposes ONLY K0..K6 counts.
"""
from __future__ import annotations
from collections import deque
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .acoustics import SR,HOP

CONTEXT = 30


def pitch_shift(x, amount):
    z = torch.zeros_like(x)
    if amount < 0:
        z[:, :, -amount:, :] = x[:, :, :amount, :]
    else:
        z[:, :, :-amount, :] = x[:, :, amount:, :]
    return z


class PastBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.dilation = dilation
        self.depth = nn.Conv2d(width,width,(3,1),dilation=(dilation,1),groups=width)
        self.mix = nn.Linear(width,width)
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        z = x.permute(0,3,1,2)
        z = self.depth(F.pad(z,(0,0,2*self.dilation,0)))
        z = self.mix(F.gelu(z.permute(0,2,3,1)))
        return self.norm(x+z)


class TemporalHeads(nn.Module):
    def __init__(self, mean, scale, width=48):
        super().__init__()
        self.width = width
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('scale',torch.as_tensor(scale,dtype=torch.float32))
        self.encode = nn.Sequential(nn.Linear(len(mean),width),nn.GELU(),nn.LayerNorm(width))
        self.harmonics = nn.Sequential(nn.Linear(width*5,width),nn.GELU(),nn.LayerNorm(width))
        self.past = nn.ModuleList(PastBlock(width,d) for d in (1,2,4,8))
        self.count = nn.Sequential(nn.Linear(49*width,64),nn.GELU(),nn.Linear(64,7))
        self.pitch = nn.Sequential(nn.Linear(width+7,width),nn.GELU(),nn.Linear(width,3))

    def forward(self, x, valid=None):
        z = self.encode(torch.clamp((x-self.mean)/self.scale,-8,8))
        z = z+self.harmonics(torch.cat([z,*[pitch_shift(z,s) for s in (-12,-19,12,19)]],-1))
        mask = 1 if valid is None else valid[:,:,None,None]
        z = z*mask
        for block in self.past:
            z = block(z)*mask
        k = self.count(z.flatten(-2))
        kp = k.softmax(-1).unsqueeze(-2).expand(*z.shape[:-1],7)
        p = self.pitch(torch.cat((z,kp),-1))
        return p,k

TRAIN_CAP_NEG = 3600


def lag(x, steps):
    """Look backward only; stable for the first frame."""
    t = np.maximum(0, np.arange(len(x)) - steps)
    return x[t]

def auxiliary_acoustic_context(feat, post, k):
    """Per-note 3D feature tensor from NOTE's source 126D ABI, current/past only."""
    feat = np.asarray(feat, np.float32)
    post = np.asarray(post, np.float32)
    k = np.asarray(k, np.float32)
    n = len(feat)
    if feat.shape != (n, 49, 126) or post.shape != (n, 49, 3) or k.shape != (n, 7):
        raise ValueError("S701 frozen NOTE ABI mismatch")
    if n == 0:
        return np.empty((0, 49, 39), np.float32)
    onset, sus, release = post.transpose(2, 0, 1)
    fundamental = np.maximum(feat[:, :, [0, 8, 16]], 0).mean(axis=2)
    overtone2 = np.maximum(feat[:, :, [1, 9, 17]], 0).mean(axis=2)
    overtone3 = np.maximum(feat[:, :, [2, 10, 18]], 0).mean(axis=2)
    near = np.maximum(feat[:, :, 54:57], 0).mean(axis=2)
    distant = np.maximum(feat[:, :, 66:69], 0).mean(axis=2)
    f_rise = (.65 * near + .35 * distant) / (.2 + fundamental)
    # Own fundamental vs evidence copied from lower-octave / fifth harmonics.
    cols = np.arange(49)
    lower12 = np.maximum(cols - 12, 0)
    lower19 = np.maximum(cols - 19, 0)
    low12_ok = np.broadcast_to((cols >= 12)[None, :], (n, 49))
    low19_ok = np.broadcast_to((cols >= 19)[None, :], (n, 49))
    sus12 = np.where(low12_ok, sus[:, lower12], 0)
    ons12 = np.where(low12_ok, onset[:, lower12], 0)
    sus19 = np.where(low19_ok, sus[:, lower19], 0)
    ons19 = np.where(low19_ok, onset[:, lower19], 0)
    pitch = np.broadcast_to((cols[None, :] + 40.) / 127., (n, 49))
    # This list is deliberately fixed; every column is inferable from past audio.
    values = [
        onset, sus, release, lag(onset, 2), lag(onset, 8),
        lag(sus, 2), lag(sus, 8), onset - lag(onset, 2),
        onset - lag(onset, 8), sus - lag(sus, 8),
        fundamental, overtone2, overtone3, near, distant, f_rise,
        fundamental - lag(fundamental, 2),
        fundamental - lag(fundamental, 8),
        overtone2 - lag(overtone2, 8),
        overtone3 - lag(overtone3, 8),
        overtone2 / (.2 + fundamental),
        overtone3 / (.2 + fundamental),
        sus12, ons12, sus19, ons19,
        f_rise - f_rise[:, lower12],
        f_rise - f_rise[:, lower19],
        np.maximum(0, onset - ons12), np.maximum(0, onset - ons19),
        pitch,
        *[np.broadcast_to(k[:, i, None], (n, 49)) for i in range(7)],
        np.broadcast_to(np.max(k[:, 2:], axis=1)[:, None], (n, 49)),
    ]
    out = np.stack(values, axis=2).astype(np.float32, copy=False)
    if out.shape[-1] != 39 or not np.isfinite(out).all():
        raise AssertionError("S701 feature layout error")
    return out



class IncrementalS708TemporalHead:
    """Causal S705 net with per-layer cached states, one new frame per call.

    Mathematically equivalent to the frozen full-context TemporalHeads.forward:
    each depthwise 3-tap dilated convolution needs only t, t-d and t-2d.
    The cache records the INPUT of each past block, not its output.
    """
    def __init__(self, model):
        self.model = model.eval()
        self.histories = [deque(maxlen=2*block.dilation) for block in model.past]
        self.steps = 0

    def step(self, x):
        x = np.asarray(x)
        if x.shape != (49, 288) or not np.isfinite(x).all():
            raise ValueError("S708 temporal ABI expects finite [49,288] current frame")
        with torch.inference_mode():
            # Identical FP16->FP32 quantization to training and predict_track().
            q = torch.from_numpy(np.asarray(x, np.float16).astype(np.float32))[None,None]
            m = self.model
            z = m.encode(torch.clamp((q-m.mean)/m.scale,-8,8))
            z = z+m.harmonics(torch.cat(
                [z,*[pitch_shift(z,shift) for shift in (-12,-19,12,19)]],-1))
            for block, history in zip(m.past, self.histories):
                dilation = block.dilation
                current = z.permute(0,3,1,2)  # [1,width,1,49]
                zero = torch.zeros_like(current)
                one = history[-dilation] if len(history)>=dilation else zero
                two = history[-2*dilation] if len(history)>=2*dilation else zero
                taps = torch.cat((two,one,current),dim=2)
                conv = F.conv2d(taps,block.depth.weight,block.depth.bias,
                                groups=block.depth.groups)
                z = block.norm(z+block.mix(F.gelu(conv.permute(0,2,3,1))))
                history.append(current)
            k = m.count(z.flatten(-2)).softmax(-1)
            kp = k.unsqueeze(-2).expand(*z.shape[:-1],7)
            p = m.pitch(torch.cat((z,kp),-1)).sigmoid()
            self.steps += 1
            return p[0,0].cpu().numpy(),k[0,0].cpu().numpy()


def latest_s708_acoustic_features(feat, post, count):
    """Exactly S704 acoustic_features(...)[-1] without recomputing 65 frames.

    Only eight previous native frames are needed by the 39D S701 onset
    embedding; the three 9D frozen context taps require 8/32/64 past frames.
    Never change the 192D feature ABI or model weights.
    """
    n=len(feat)
    if n==0 or len(post)!=n or len(count)!=n:
        raise ValueError("S708 acoustic history counts must match and be nonempty")
    # S701 onset_features uses only the current frame plus lags 2 and 8.
    recent=max(0,n-9)
    short_f=np.stack([feat[i] for i in range(recent,n)])
    short_p=np.stack([post[i] for i in range(recent,n)])
    short_k=np.stack([count[i] for i in range(recent,n)])
    current_onset=auxiliary_acoustic_context(short_f,short_p,short_k)[-1]
    context=[]
    for lag in (8,32,64):
        i=max(0,n-1-lag)
        context.append(np.concatenate((feat[i][:,48:54],post[i]),axis=-1))
    out=np.concatenate((feat[-1],current_onset,*context),axis=-1).astype(np.float32)
    if out.shape!=(49,192) or not np.isfinite(out).all():
        raise AssertionError("S708 current 192D acoustic feature ABI mismatch")
    return out

def load_frozen_source(checkpoint,report):
    """Pin frozen original 1024-trained S708 weights; evaluate at 256."""
    cp, rp=Path(checkpoint),Path(report)
    data=json.loads(rp.read_text(encoding="utf-8"))
    if data.get("schema")!="S708_K_GUIDED_S706_CAL_LOOPS" or data.get("source_family")!=706 or data.get("promoted") is not False:
        raise ValueError("Unexpected original S708 model provenance")
    sha=hashlib.sha256(cp.read_bytes()).hexdigest()
    if sha!=data.get("checkpoint_sha256") or cp.name!="s708_selected.pt" or data["frozen_candidate"]["choice"]["model_file"]!="s706_epoch10.pt":
        raise ValueError("Frozen S708 SHA/model mismatch")
    bundle=torch.load(cp,map_location="cpu",weights_only=True)
    model=TemporalHeads(bundle["model"]["mean"],bundle["model"]["scale"],width=int(bundle["width"]))
    model.load_state_dict(bundle["model"],strict=True)
    model.eval()
    torch.set_num_threads(1)
    if len(model.mean)!=288:
        raise ValueError("S708 source 288D feature ABI mismatch")
    return model,sha
