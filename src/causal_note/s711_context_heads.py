"""S711: causal temporal heads with current-frame cross-pitch attention.

Only the already-received frame is shared across pitches. Temporal memory is
the same 30 past blocks as S705/S706. A zero-initialized residual and expanded
boundary input preserve the frozen S706 predictions exactly at warm start.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from causal_note.s705_temporal_heads import TemporalHeads as BaseHeads,pitch_shift

class ContextTemporalHeads(BaseHeads):
    def __init__(self,mean,scale,width=48):
        super().__init__(mean,scale,width)
        self.attention_norm=nn.LayerNorm(width)
        self.pitch_position=nn.Parameter(torch.randn(49,width)*.01)
        self.cross_pitch=nn.MultiheadAttention(width,3,batch_first=True,dropout=0.)
        self.attention_residual=nn.Linear(width,width)
        nn.init.zeros_(self.attention_residual.weight)
        nn.init.zeros_(self.attention_residual.bias)
        self.pitch=nn.Sequential(nn.Linear(width+7+64,width),nn.GELU(),nn.Linear(width,3))

    def forward(self,x,valid=None):
        z=self.encode(torch.clamp((x-self.mean)/self.scale,-8,8))
        z=z+self.harmonics(torch.cat([z,*[pitch_shift(z,s) for s in (-12,-19,12,19)]],-1))
        mask=1 if valid is None else valid[:,:,None,None]
        z=z*mask
        for block in self.past:z=block(z)*mask
        flat=self.attention_norm(z).reshape(-1,49,self.width)
        qk=flat+self.pitch_position
        mixed,_=self.cross_pitch(qk,qk,flat,need_weights=False)
        z=(z+self.attention_residual(mixed).reshape_as(z))*mask
        context=F.gelu(self.count[0](z.flatten(-2)))
        k=self.count[2](context)
        kp=k.softmax(-1).unsqueeze(-2).expand(*z.shape[:-1],7)
        global_context=context.unsqueeze(-2).expand(*z.shape[:-1],64)
        return self.pitch(torch.cat((z,kp,global_context),-1)),k

    def warm_encoder(self,old):
        own=self.state_dict()
        for name,value in old.items():
            if name in own and own[name].shape==value.shape:own[name]=value
        own["pitch.0.weight"][:,:self.width+7]=old["pitch.0.weight"]
        own["pitch.0.weight"][:,self.width+7:]=0
        self.load_state_dict(own)

def transition_loss(logits,active_target):
    """One-step expected occupancy, supervised only during FIT learning."""
    previous=logits[:,:-1,:,1].sigmoid()
    onset=logits[:,1:,:,0].sigmoid()
    release=logits[:,1:,:,2].sigmoid()
    state=previous*(1-release)+(1-previous)*onset
    loss=F.binary_cross_entropy(state.clamp(1e-6,1-1e-6),active_target[:,1:],reduction="none").mean(-1)
    return F.pad(loss,(1,0))
