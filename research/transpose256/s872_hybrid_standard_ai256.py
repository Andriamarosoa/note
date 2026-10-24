"""S872/S873 standard + learned AI corrector. Research-only, not NOTE Live.
Dual delay readers pitch down -12 semitones with sine-square crossfades.
GRU only MIXES two standard outputs; cannot hallucinate notes.
256 NEW samples per call, no future access. Algorithmic acoustic delay
is strictly nonzero and phase dependent: do not claim zero-latency output.
Trained checkpoints are in conversation's reproducible archive.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.io import wavfile

SR=44100
HOP=256

class GranularDown256:
    def __init__(self,grain=4096,min_delay=64):
        if type(grain)!=int or grain%HOP!=0 or grain<512:
            raise ValueError("grain must be multiple of 256")
        self.grain=grain
        self.min_delay=min_delay
        self.ratio=.5
        self.ring=np.zeros(grain*2+HOP*4,np.float32)
        self.tick=0

    def push(self,block):
        if (not isinstance(block,np.ndarray) or block.shape!=(HOP,)
            or block.dtype!=np.float32 or not np.isfinite(block).all()):
            raise ValueError("exactly 256 new finite PCM32 samples required")
        n=np.arange(self.tick,self.tick+HOP,dtype=np.float64)
        size=len(self.ring)
        self.ring[(np.arange(HOP)+self.tick)%size]=block
        phase=(n%self.grain)/self.grain
        out=np.zeros(HOP,np.float64)
        for p in (phase,(phase+.5)%1):
            # Always t - strictly positive delay, NEVER t + future.
            pos=n-self.min_delay-(1-self.ratio)*p*self.grain
            j=np.floor(pos).astype(np.int64)
            f=pos-j
            a=self.ring[j%size]
            b=self.ring[(j+1)%size]
            w=.5-.5*np.cos(2*np.pi*p)
            out+=(a+(b-a)*f)*w
        self.tick+=HOP
        return out.astype(np.float32)

class NeuralGate256(nn.Module):
    def __init__(self,hidden=24):
        super().__init__()
        self.hidden=hidden
        self.encoder=nn.Sequential(
            nn.Conv1d(3,12,9,stride=4,padding=4),nn.GELU(),
            nn.Conv1d(12,16,9,stride=4,padding=4),nn.GELU(),
            nn.AdaptiveAvgPool1d(4),nn.Flatten(),
            nn.Linear(64,hidden),nn.GELU())
        self.gru=nn.GRUCell(hidden+3,hidden)
        self.decide=nn.Linear(hidden,2)

    def forward(self,original,fast,quality,state=None,initial_blend=None):
        for a in (original,fast,quality):
            if a.ndim!=3 or a.shape[-1]!=HOP:
                raise ValueError("256 samples per timestep")
        if original.shape!=fast.shape or original.shape!=quality.shape:
            raise ValueError("input shape mismatch")
        B,T,_=original.shape
        h=original.new_zeros((B,self.hidden)) if state is None else state
        prev=original.new_ones((B,1)) if initial_blend is None else initial_blend
        mixed=[];weights=[];gains=[]
        ramp=torch.linspace(0,1,HOP,device=original.device,dtype=original.dtype)[None,:]
        for t in range(T):
            x=original[:,t,:];a=fast[:,t,:];b=quality[:,t,:]
            f=self.encoder(torch.stack((x,a,b),dim=1))
            rms=torch.stack(
                [x.square().mean(-1).sqrt(),
                 a.square().mean(-1).sqrt(),
                 b.square().mean(-1).sqrt()],dim=-1)
            h=self.gru(torch.cat([f,rms],dim=-1),h)
            raw=self.decide(h)
            blend=torch.sigmoid(raw[:,0:1])
            gain=.9+.2*torch.sigmoid(raw[:,1:2])
            smooth=(1-ramp)*prev+ramp*blend
            out=(smooth*b+(1-smooth)*a)*gain
            mixed.append(out);weights.append(blend);gains.append(gain)
            prev=blend
        return (torch.stack(mixed,1),h,prev,
                torch.stack(weights,1),torch.stack(gains,1))

class StandardPlusAI256:
    def __init__(self,model,*,ai_enabled=True,grain_fast=2048,grain_quality=4096):
        self.fast=GranularDown256(grain_fast)
        self.quality=GranularDown256(grain_quality)
        self.model=model.cpu().eval()
        self.state=None;self.alpha=None
        self.ai_enabled=ai_enabled
        self.received=0

    def push(self,block,start_sample):
        if (not isinstance(block,np.ndarray) or block.shape!=(HOP,)
            or block.dtype!=np.float32 or not np.isfinite(block).all()
            or type(start_sample)!=int or start_sample!=self.received):
            raise ValueError("256 new contiguous finite float PCM required")
        fast=self.fast.push(block);quality=self.quality.push(block)
        if self.ai_enabled:
            with torch.inference_mode():
                y,self.state,self.alpha,_,_=self.model(
                    torch.from_numpy(block.copy()).reshape(1,1,HOP),
                    torch.from_numpy(fast.copy()).reshape(1,1,HOP),
                    torch.from_numpy(quality.copy()).reshape(1,1,HOP),
                    self.state,self.alpha)
                out=y.numpy().reshape(HOP).copy()
        else:
            # Fail-safe: exact standard output, unchanged, no AI.
            out=quality
        self.received+=HOP
        return out

def load_checkpoint(path):
    item=torch.load(path,map_location="cpu",weights_only=True)
    if item.get("kind") not in ("S872_NEURAL_STANDARD_MIXER","S873_NEURAL_STANDARD_MIXER"):
        raise ValueError("unrecognized research checkpoint")
    network=NeuralGate256()
    network.load_state_dict(item["model"],strict=True)
    return network.eval(),item["grain_fast"],item["grain_quality"]

def transform_wav(model,source,out,grain_fast,grain_quality,enable_ai):
    sr,raw=wavfile.read(str(source))
    if sr!=SR or raw.dtype!=np.int16 or raw.ndim!=1:
        raise ValueError("44.1kHz mono PCM16 input required")
    x=raw.astype(np.float32)/32768.
    size=len(x)
    pad=np.pad(x,(0,(-size)%HOP))
    stream=StandardPlusAI256(
        model,ai_enabled=enable_ai,
        grain_fast=grain_fast,grain_quality=grain_quality)
    wave=np.concatenate(
        [stream.push(pad[i:i+HOP],i)
         for i in range(0,len(pad),HOP)])[:size]
    wavfile.write(str(out),SR,
        np.int16(np.clip(wave,-1,1-1/32768)*32767))
    return wave

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--no-ai",action="store_true")
    a=p.parse_args()
    m,g0,g1=load_checkpoint(a.checkpoint)
    transform_wav(m,a.input,a.output,g0,g1,not a.no_ai)
    print("Research-only, -12 semitone hybrid PCM256",a.output)
