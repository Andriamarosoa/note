"""S863/S866 exploratory causal neural WAV processing. NOT LIVE-APPROVED.

Input: 256 fresh mono PCM samples at 44,100 Hz on every call. No pitch
calculation, reference audio, note bank or standard DSP pitch shifter at
inference. Causal FIR filters imply nonzero acoustic group delay even though
no future samples are requested: do not deploy to latency-critical NOTE.
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
SR=44100
FILTERS=7
TAPS=513

class NeuralOctave256(nn.Module):
    def __init__(self):
        super().__init__()
        self.bank=nn.Conv1d(1,FILTERS,TAPS,bias=False)
        self.gate=nn.Sequential(nn.Linear(FILTERS*2,32),nn.SiLU(),nn.Linear(32,FILTERS))
        self.scale=nn.Parameter(torch.ones(FILTERS))
        self.master_gain=nn.Parameter(torch.tensor(1.))

    def compute_last_256(self,previous_512_and_new_256):
        if previous_512_and_new_256.ndim!=2 or previous_512_and_new_256.shape[1]!=TAPS-1+HOP:
            raise ValueError("512 past samples and precisely 256 new ones required")
        z=self.bank(previous_512_and_new_256[:,None,:]).reshape(-1,FILTERS,1,HOP)
        rms=(z.square().mean(dim=-1)+1e-8).sqrt()
        ctl=torch.cat((torch.log1p(100*rms),torch.log1p(100*rms[:,0:1]).expand(-1,FILTERS,-1)),dim=1).permute(0,2,1)
        weight=F.softmax(self.gate(ctl),dim=-1).permute(0,2,1).unsqueeze(-1)
        even=z.square()-rms[:,:,:,None].square()
        return (weight*even*self.scale[None,:,None,None]).sum(dim=1).reshape(-1,HOP)*self.master_gain

class NeuralPCM256Stream:
    def __init__(self,network):
        self.network=network.cpu().eval()
        self.previous=np.zeros(TAPS-1,np.float32)
        self.received=0

    def push(self,new_pcm,start_sample):
        if (not isinstance(new_pcm,np.ndarray) or new_pcm.dtype!=np.float32 or
            new_pcm.shape!=(HOP,) or not np.isfinite(new_pcm).all() or
            start_sample!=self.received):
            raise ValueError("Exactly 256 new contiguous finite float32 PCM samples required")
        x=np.concatenate((self.previous,new_pcm))
        with torch.inference_mode():
            y=self.network.compute_last_256(torch.from_numpy(x)[None,:])[0].numpy().copy()
        self.previous=x[-(TAPS-1):].copy()
        self.received+=HOP
        return y

def load_model(path):
    payload=torch.load(str(path),map_location="cpu",weights_only=True)
    if not isinstance(payload,dict) or "weights" not in payload:
        raise ValueError("Expected S863/S866 experimental checkpoint")
    model=NeuralOctave256()
    model.load_state_dict(payload["weights"],strict=True)
    return model.eval()

def transform_file(network,source,output):
    sr,wav=wavfile.read(str(source))
    if sr!=SR or wav.dtype!=np.int16 or wav.ndim!=1:
        raise ValueError("Expect 44,100 Hz mono PCM16 WAV")
    real=wav.astype(np.float32)/32768.
    player=NeuralPCM256Stream(network)
    chunks=[]
    for start in range(0,len(real),HOP):
        n=min(HOP,len(real)-start)
        block=np.zeros(HOP,np.float32)
        block[:n]=real[start:start+n]
        chunks.append(player.push(block,start)[:n])
    audio=np.concatenate(chunks)
    wavfile.write(str(output),SR,np.int16(np.clip(audio,-1,1-1/32768)*32768))
    return audio

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--input-wav",type=Path,required=True)
    p.add_argument("--output-wav",type=Path,required=True)
    a=p.parse_args()
    transform_file(load_model(a.model),a.input_wav,a.output_wav)
    print("Wrote research-only PCM256 WAV",a.output_wav)
