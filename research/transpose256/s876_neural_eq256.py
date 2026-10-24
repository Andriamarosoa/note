"""S875/S876 neural multiband correction on a single standard -12 output.
Shadow research only. No future audio; exactly 256 new PCM per call.
Full training and signed real GuitarSet data in accompanying S874-S881 archive.
Unlike S873's two differently phased standard streams, this head shapes
one standard source via sum-to-identity causal FIR bands.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.signal import firwin,lfilter
from scipy.io import wavfile
from s872_hybrid_standard_ai256 import GranularDown256
HOP=256
SR=44100
BAND_FILTERS=[firwin(129,c,fs=SR).astype(np.float32) for c in (150,650,2400)]

class NeuralEQ256(nn.Module):
    def __init__(self,limit=.4,hidden=48):
        super().__init__()
        self.limit=limit
        self.encoder=nn.Sequential(nn.Linear(12,hidden),nn.GELU(),
                 nn.LayerNorm(hidden),nn.Linear(hidden,hidden),nn.GELU())
        self.gru=nn.GRUCell(hidden,hidden)
        self.head=nn.Linear(hidden,4)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
    def forward(self,original,standard,bands,state=None,last=None):
        if original.ndim!=3 or original.shape[-1]!=HOP:
            raise ValueError("require exactly 256 PCM samples")
        if original.shape!=standard.shape or bands.shape!=(*original.shape[:2],4,HOP):
            raise ValueError("shape mismatch")
        B,T,_=original.shape
        h=original.new_zeros((B,self.gru.hidden_size)) if state is None else state
        prev=original.new_ones((B,4)) if last is None else last
        ramp=torch.linspace(0,1,HOP,device=original.device)[None,None,:]
        ys=[];weights=[]
        for t in range(T):
            x=original[:,t,:];s=standard[:,t,:];b=bands[:,t,:,:]
            features=torch.stack([
                 x.square().mean(-1).sqrt(),s.square().mean(-1).sqrt(),
                 (x[:,1:]-x[:,:-1]).square().mean(-1).sqrt(),
                 (s[:,1:]-s[:,:-1]).square().mean(-1).sqrt(),
                 b[:,0,:].square().mean(-1).sqrt(),
                 b[:,1,:].square().mean(-1).sqrt(),
                 b[:,2,:].square().mean(-1).sqrt(),
                 b[:,3,:].square().mean(-1).sqrt(),
                 x.abs().amax(-1),s.abs().amax(-1),x.mean(-1),s.mean(-1)],dim=-1)
            feat=torch.log1p(features.abs()*50)*torch.sign(features)
            h=self.gru(self.encoder(feat),h)
            gain=1+self.limit*torch.tanh(self.head(h))
            smooth=prev[:,:,None]*(1-ramp)+gain[:,:,None]*ramp
            ys.append((smooth*b).sum(dim=1))
            weights.append(gain);prev=gain
        return torch.stack(ys,1),h,prev,torch.stack(weights,1)

class BandCorrectionStream256:
    def __init__(self,model,*,grain=8192,ai_enabled=True):
        self.model=model.cpu().eval()
        self.standard=GranularDown256(grain)
        self.previous=np.zeros(128,np.float32)
        self.state=None;self.previous_gain=None
        self.ai_enabled=ai_enabled;self.count=0
    def push(self,block,start_sample):
        if not(isinstance(block,np.ndarray) and block.dtype==np.float32
           and block.shape==(HOP,) and np.isfinite(block).all()
           and type(start_sample)==int and start_sample==self.count):
            raise ValueError("256 new contiguous float32 samples")
        reference=self.standard.push(block)
        self.count+=HOP
        hist=np.concatenate((self.previous,reference))
        self.previous=hist[-128:].copy()
        low=[lfilter(f,[1],hist)[-HOP:].astype(np.float32)
             for f in BAND_FILTERS]
        bands=np.stack((low[0],low[1]-low[0],
                        low[2]-low[1],reference-low[2]))
        with torch.no_grad():
            out,self.state,self.previous_gain,_=self.model(
                torch.from_numpy(block[None,None].copy()),
                torch.from_numpy(reference[None,None].copy()),
                torch.from_numpy(bands[None,None].copy()),
                self.state,self.previous_gain)
        return reference.copy() if not self.ai_enabled else out[0,0].numpy().copy()

def load_checkpoint(path):
    item=torch.load(str(path),map_location="cpu",weights_only=True)
    if item.get("kind") not in ("S875_NEURAL_BAND_CORRECTOR",
                                "S876_NEURAL_BAND_CORRECTOR"):
        raise ValueError("not a verified S875/S876 experiment checkpoint")
    m=NeuralEQ256(limit=float(item["limit"]))
    m.load_state_dict(item["weights"],strict=True)
    return m.eval()

def render(model,input_wav,output_wav,*,ai_enabled=True):
    sr,raw=wavfile.read(str(input_wav))
    if sr!=SR or raw.dtype!=np.int16 or raw.ndim!=1:
        raise ValueError("44.1 kHz mono PCM16 input required")
    audio=raw.astype(np.float32)/32768.
    original_length=len(audio)
    audio=np.pad(audio,(0,(-len(audio))%HOP))
    stream=BandCorrectionStream256(model,ai_enabled=ai_enabled)
    out=np.concatenate([stream.push(audio[i:i+HOP],i)
                        for i in range(0,len(audio),HOP)])[:original_length]
    wavfile.write(str(output_wav),SR,np.int16(np.clip(out,-1,.9999)*32768))
    return out

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--input-wav",required=True)
    p.add_argument("--output-wav",required=True)
    p.add_argument("--no-ai",action="store_true")
    args=p.parse_args()
    render(load_checkpoint(args.checkpoint),args.input_wav,args.output_wav,
           ai_enabled=not args.no_ai)
