"""S868-S870 true -12 semitone neural audio research inference, PCM256.
Use local checkpoints supplied in the S868-S871 experiment archive.
The standard pitch shift was used OFFLINE for supervision only, never here.
NOT a functional live pitch shifter until independent musical validation.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.io import wavfile

HOP=256
HISTORY=2048
SR=44100

class CausalBlock(nn.Module):
    def __init__(self,channels,dilation):
        super().__init__()
        self.d=dilation
        self.conv=nn.Conv1d(channels,channels,3,dilation=dilation)
        self.mix=nn.Conv1d(channels,channels,1)
    def forward(self,x):
        z=self.conv(F.pad(x,(self.d*2,0)))
        return x+.25*self.mix(torch.tanh(z))

class NeuralDownOctave256(nn.Module):
    def __init__(self,variant="tcn",channels=12):
        super().__init__()
        self.variant=variant
        self.encoder=nn.Conv1d(1,channels,9)
        self.blocks=nn.ModuleList(CausalBlock(channels,d) for d in
            (1,2,4,8,16,32,64,128,256,512))
        self.head=nn.Conv1d(channels,1,1)
        if variant not in ("tcn","tcn_gate"):raise ValueError("variant")
        self.gate=nn.Sequential(
            nn.Conv1d(channels,channels,1),nn.Sigmoid()
        ) if variant=="tcn_gate" else None
    def forward(self,pcm):
        if pcm.ndim!=2 or pcm.shape[-1]%HOP:
            raise ValueError("audio must be in 256 sample blocks")
        z=self.encoder(F.pad(pcm[:,None,:],(8,0)))
        for block in self.blocks:z=block(z)
        if self.gate is not None:z=z*self.gate(z)
        return self.head(z)[:,0,:]

class DownOctaveStream256:
    def __init__(self,model):
        self.model=model.cpu().eval()
        self.past=np.zeros(HISTORY,np.float32)
        self.total=0
    def push(self,pcm,start_sample):
        if (not isinstance(pcm,np.ndarray) or pcm.dtype!=np.float32 or
            pcm.shape!=(HOP,) or not np.isfinite(pcm).all() or
            type(start_sample)!=int or start_sample!=self.total):
            raise ValueError("requires exactly 256 new contiguous PCM samples")
        x=np.concatenate((self.past,pcm))
        with torch.inference_mode():
            y=self.model(torch.from_numpy(x.copy())[None,:])[0,-HOP:].numpy().copy()
        self.past=x[-HISTORY:].copy()
        self.total+=HOP
        return y

class RecurrentDownOctave256(nn.Module):
    def __init__(self,hidden=96,kind="gru"):
        super().__init__()
        self.hidden=hidden
        self.kind=kind
        self.encoder=nn.Sequential(
            nn.Conv1d(1,24,9,stride=2,padding=4),nn.SiLU(),
            nn.Conv1d(24,48,9,stride=4,padding=4),nn.SiLU(),
            nn.AdaptiveAvgPool1d(8),nn.Flatten(),
            nn.Linear(48*8,hidden),nn.LayerNorm(hidden),nn.SiLU()
        )
        self.rnn=nn.GRUCell(hidden,hidden)
        self.out=nn.Sequential(
            nn.Linear(hidden*2,hidden*2),nn.SiLU(),nn.Linear(hidden*2,HOP)
        )
        self.gain=nn.Parameter(torch.tensor(0.5))
    def forward(self,blocks,state=None):
        if blocks.ndim!=3 or blocks.shape[-1]!=HOP:
            raise ValueError("exactly 256 samples per block required")
        h=(torch.zeros(blocks.shape[0],self.hidden,
            device=blocks.device,dtype=blocks.dtype)
            if state is None else state)
        ys=[]
        for t in range(blocks.shape[1]):
            features=self.encoder(blocks[:,t:t+1,:].reshape(-1,1,HOP))
            h=self.rnn(features,h)
            y=self.out(torch.cat((features,h),dim=1))*self.gain
            ys.append(y)
        return torch.stack(ys,dim=1),h

class RecurrentDownStream:
    def __init__(self,model):
        self.model=model.cpu().eval()
        self.state=None
        self.counter=0
    def push(self,block,start_sample):
        if not(isinstance(block,np.ndarray) and block.shape==(HOP,) and
            block.dtype==np.float32 and np.isfinite(block).all() and
            type(start_sample)==int and start_sample==self.counter):
            raise ValueError("requires exactly 256 new contiguous PCM samples")
        with torch.inference_mode():
            y,self.state=self.model(
                torch.from_numpy(block[None,None,:]),self.state
            )
            wave=y[0,0].numpy().copy()
        self.counter+=HOP
        return wave

def load_model(checkpoint):
    payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if payload.get("hop")!=HOP or payload.get("shift")!=-12:
        raise ValueError("Only research -12 PCM256 checkpoints supported")
    variant=payload.get("variant","recurrent")
    model=(RecurrentDownOctave256() if variant=="recurrent" else
        NeuralDownOctave256(variant))
    model.load_state_dict(payload["weights"],strict=True)
    return model.eval(),(RecurrentDownStream if variant=="recurrent" else DownOctaveStream256)

def process_wav(checkpoint,input_wav,output_wav):
    model,kind=load_model(checkpoint)
    rate,wav=wavfile.read(input_wav)
    if rate!=SR or wav.ndim!=1 or wav.dtype!=np.int16:
        raise ValueError("expected mono 44.1kHz PCM16 WAV")
    source=wav.astype(np.float32)/32768.
    stream=kind(model);out=[]
    for i in range(0,len(source),HOP):
        n=min(HOP,len(source)-i)
        block=np.zeros(HOP,np.float32)
        block[:n]=source[i:i+n]
        out.append(stream.push(block,i)[:n])
    data=np.concatenate(out)
    wavfile.write(output_wav,SR,
        np.int16(np.clip(data,-1,1-1/32768)*32768))
    return data

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--input-wav",type=Path,required=True)
    p.add_argument("--output-wav",type=Path,required=True)
    a=p.parse_args()
    process_wav(a.checkpoint,a.input_wav,a.output_wav)
