"""S705 bounded-past temporal/harmonic heads. Experimental, never installed LIVE.

Frequency neighbours at the CURRENT frame are allowed. All temporal kernels
are left-padded; no bidirectional recurrence or whole-clip normalization.
"""
from collections import deque
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

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


def predict_track(model, x, chunk=256):
    """Chunked inference exactly preserves the finite causal receptive field."""
    model.eval()
    ps,ks=[],[]
    with torch.inference_mode():
        for start in range(0,len(x),chunk):
            lo=max(0,start-CONTEXT)
            # Quantization is a declared cache/runtime ABI, used in training too.
            xx=np.asarray(x[lo:start+chunk],np.float16).astype(np.float32)
            p,k=model(torch.from_numpy(xx)[None])
            ps.append(p[0,start-lo:].sigmoid().numpy())
            ks.append(k[0,start-lo:].softmax(-1).numpy())
    return np.concatenate(ps),np.concatenate(ks)


class TemporalHeadStream:
    def __init__(self, model):
        self.model=model.eval()
        self.history=deque(maxlen=CONTEXT+1)

    def step(self, x):
        self.history.append(np.asarray(x,np.float16))
        p,k=predict_track(self.model,np.stack(self.history))
        return p[-1],k[-1]
