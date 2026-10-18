"""Native causal acoustic heads with six supervised monophonic string slots.

String annotations supervise TRAIN only. At inference the six distributions
are predicted from microphone acoustics, not from labels or hex pickup audio.
No tuning/fret masks are imposed. Pitch-presence and K are derived from the
six predicted distributions, while attack/release retain learned heads.
"""
import numpy as np
import torch
from torch import nn
from causal_note.s705_temporal_heads import PastBlock,pitch_shift


class StringTemporalHeads(nn.Module):
    def __init__(self,mean,scale,width=48):
        super().__init__();self.width=width
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('scale',torch.as_tensor(scale,dtype=torch.float32))
        self.encode=nn.Sequential(nn.Linear(len(mean),width),nn.GELU(),nn.LayerNorm(width))
        self.harmonics=nn.Sequential(nn.Linear(width*5,width),nn.GELU(),nn.LayerNorm(width))
        self.past=nn.ModuleList(PastBlock(width,d) for d in (1,2,4,8))
        self.context=nn.Sequential(nn.Linear(49*width,64),nn.GELU())
        self.string_context=nn.Linear(64,6*width)
        self.string_embedding=nn.Parameter(torch.randn(6,width)*.05)
        self.string_score=nn.Linear(width,1)
        self.silence=nn.Linear(64,6)
        self.boundary=nn.Sequential(nn.Linear(width+8,width),nn.GELU(),nn.Linear(width,2))

    def forward(self,x,valid=None,return_strings=False):
        z=self.encode(torch.clamp((x-self.mean)/self.scale,-8,8))
        z=z+self.harmonics(torch.cat([z,*[pitch_shift(z,s) for s in (-12,-19,12,19)]],-1))
        mask=1 if valid is None else valid[:,:,None,None]
        z=z*mask
        for block in self.past:z=block(z)*mask
        context=self.context(z.flatten(-2))
        condition=self.string_context(context).reshape(*z.shape[:2],1,6,self.width)
        scored=torch.tanh(z.unsqueeze(-2)+self.string_embedding+condition)
        pitch_logits=self.string_score(scored).squeeze(-1).transpose(-1,-2)
        strings=torch.cat((self.silence(context).unsqueeze(-1),pitch_logits),-1)
        sp=strings.softmax(-1)
        present=1-torch.prod(1-sp[...,1:],dim=-2)
        # Poisson-binomial count from the six predicted active-string odds.
        kp=torch.ones_like(sp[:,:,0,:1])
        for s in range(6):
            silent=sp[:,:,s,0:1]
            kp=nn.functional.pad(kp*silent,(0,1))+nn.functional.pad(kp*(1-silent),(1,0))
        count_logits=kp.clamp_min(1e-9).log()
        evidence=torch.cat((z,kp.unsqueeze(-2).expand(*z.shape[:-1],7),present.unsqueeze(-1)),-1)
        boundary=self.boundary(evidence)
        active=torch.logit(present.clamp(1e-6,1-1e-6))
        p=torch.stack((boundary[...,0],active,boundary[...,1]),-1)
        return (p,count_logits,strings) if return_strings else (p,count_logits)

    def warm_encoder(self,old):
        own=self.state_dict()
        for key in own:
            if key.startswith(('encode.','harmonics.','past.')):
                own[key]=old[key]
        own['context.0.weight']=old['count.0.weight']
        own['context.0.bias']=old['count.0.bias']
        own['boundary.0.weight'][:,:-1]=old['pitch.0.weight']
        own['boundary.0.weight'][:,-1]=0
        own['boundary.0.bias']=old['pitch.0.bias']
        own['boundary.2.weight']=old['pitch.2.weight'][[0,2]]
        own['boundary.2.bias']=old['pitch.2.bias'][[0,2]]
        self.load_state_dict(own)
