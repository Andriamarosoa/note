"""S850 independent neural transposition head with actual polyphonic GT.

A separate experiment, NOT modifying S847 or NOTE's official six heads.
Exactly one NEW mono PCM256 block per decision. CNN->GRU state is causal
PAST ONLY, no audio queue, no future PCM, no onset lookahead.

Unlike S847's all-one-note collapse, output has independent multi-label
guitar pitches, an explicit real-note vs silence gate, and can be trained
against *real* GuitarSet note activity with hard-negative ranking.
No sinusoidal synthesis, frequency formula or DSP pitch shifting is used
inside this learned neural model.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SR=44100
HOP=256
LOW=40
PITCHES=49
SHIFTS=(-12,0,12)
ABI="NOTE_S850_POLYPHONIC_REAL_GUITAR_NEURAL_TRANSPOSE256_V1"


def check_audio(x):
    if (not isinstance(x,torch.Tensor) or x.ndim!=3 or
        x.shape[-1]!=HOP or not x.is_floating_point() or
        not bool(torch.isfinite(x).all())):
        raise ValueError("model requires exactly 256 new float PCM per hop")


def shifted_truth_only_offline(gt,offsets):
    if gt.ndim!=3 or gt.shape[-1]!=PITCHES:
        raise ValueError("signed TRAIN note GT shape expected")
    if (offsets.shape!=(len(gt),) or offsets.dtype!=torch.int64 or
        bool((offsets<-12).any()) or bool((offsets>12).any())):
        raise ValueError("valid -12..12 int64 shift GT")
    targets=torch.zeros_like(gt)
    for shift in torch.unique(offsets).tolist():
        mask=(offsets==shift)
        if shift>0:targets[mask,:,shift:]=gt[mask,:,:PITCHES-shift]
        elif shift<0:targets[mask,:,:PITCHES+shift]=gt[mask,:,-shift:]
        else:targets[mask]=gt[mask]
    return targets


class NeuralPolyphonicTranspose256(nn.Module):
    def __init__(self,hidden=128):
        super().__init__()
        self.hidden=hidden
        self.encoder=nn.Sequential(
          nn.Conv1d(1,32,9,stride=2,padding=4),nn.GELU(),
          nn.Conv1d(32,48,9,stride=2,padding=4),nn.GELU(),
          nn.Conv1d(48,64,9,stride=2,padding=4),nn.GELU(),
          nn.AdaptiveAvgPool1d(8),nn.Flatten(),
          nn.Linear(64*8,hidden),nn.LayerNorm(hidden),nn.GELU())
        self.gru=nn.GRUCell(hidden,hidden)
        self.state_norm=nn.LayerNorm(hidden)
        self.shift_embedding=nn.Embedding(25,32)
        self.shift_transport=nn.Sequential(
            nn.Linear(hidden+32,hidden),nn.GELU(),
            nn.Linear(hidden,hidden))
        self.original_notes=nn.Linear(hidden,PITCHES)
        self.transposed_notes=nn.Linear(hidden,PITCHES)
        self.activity=nn.Linear(hidden,1)
        self.target_activity=nn.Linear(hidden,1)

    def initial_state(self,batch,device,dtype):
        return torch.zeros((batch,self.hidden),device=device,dtype=dtype)

    def forward(self,pcm,shift,state=None):
        check_audio(pcm)
        B,T,_=pcm.shape
        if (not isinstance(shift,torch.Tensor) or shift.shape!=(B,) or
            shift.dtype!=torch.int64 or
            bool((shift<-12).any()) or bool((shift>12).any())):
            raise ValueError("need trainable shift codes -12..12")
        h=self.initial_state(B,pcm.device,pcm.dtype) if state is None else state
        if h.shape!=(B,self.hidden):raise ValueError("bad past state")
        code=self.shift_embedding(shift+12)
        orig=[];target=[];active=[];targetactive=[]
        for i in range(T):
            features=self.encoder(pcm[:,i,:].unsqueeze(1))
            h=self.gru(features,h)
            hn=self.state_norm(h)
            original=self.original_notes(hn)
            transformed=hn+self.shift_transport(torch.cat((hn,code),dim=-1))
            orig.append(original)
            target.append(self.transposed_notes(transformed))
            active.append(self.activity(hn))
            targetactive.append(self.target_activity(transformed))
        return dict(source_logits=torch.stack(orig,1),
                    target_logits=torch.stack(target,1),
                    source_activity_logits=torch.stack(active,1),
                    target_activity_logits=torch.stack(targetactive,1)),h


def train_only_weights(ground_truth):
    if ground_truth.ndim!=3 or ground_truth.shape[-1]!=PITCHES:
        raise ValueError("TRAIN data only")
    count=ground_truth.reshape(-1,PITCHES).sum(dim=0)
    frames=ground_truth.shape[0]*ground_truth.shape[1]
    # Limited weights; large weights made S847 predict positives everywhere.
    weights=torch.where(count>0,
      torch.minimum(torch.sqrt((frames-count)/count.clamp_min(1)).clamp_min(1),
                    torch.full_like(count,8.)),
      torch.ones_like(count))
    return weights,count


def hard_negative_rank(logits,truth):
    positives=truth>.5
    valid=positives.any(dim=-1)&(~positives).any(dim=-1)
    if not bool(valid.any()):
        return logits.sum()*0
    min_pos=logits.masked_fill(~positives,1e5).amin(dim=-1)
    max_neg=logits.masked_fill(positives,-1e5).amax(dim=-1)
    return F.softplus(max_neg[valid]-min_pos[valid]+.35).mean()


def real_note_loss(model,pcm,original_gt,shifts,train_weights,*,ranking=.5):
    check_audio(pcm)
    if (original_gt.shape!=(*pcm.shape[:2],PITCHES) or
        train_weights.shape!=(PITCHES,) or
        not bool(((original_gt==0)|(original_gt==1)).all())):
        raise ValueError("signed true GuitarSet GT only, not hallucinated audio")
    if ranking<0 or ranking>2:raise ValueError("bounded positive ranking")
    target=shifted_truth_only_offline(original_gt,shifts)
    logits,_=model(pcm,shifts)
    source=F.binary_cross_entropy_with_logits(
            logits["source_logits"],original_gt,pos_weight=train_weights)
    shifted=F.binary_cross_entropy_with_logits(
            logits["target_logits"],target,pos_weight=train_weights)
    source_presence=(original_gt.sum(dim=-1,keepdim=True)>0).float()
    target_presence=(target.sum(dim=-1,keepdim=True)>0).float()
    active=F.binary_cross_entropy_with_logits(
            logits["source_activity_logits"],source_presence)
    target_active=F.binary_cross_entropy_with_logits(
            logits["target_activity_logits"],target_presence)
    rank_orig=hard_negative_rank(logits["source_logits"],original_gt)
    rank_shift=hard_negative_rank(logits["target_logits"],target)
    # Train ranking explicitly against false competing pitches. A lower
    # raw BCE alone might simply learn to predict all notes everywhere.
    loss=source+shifted+.35*(active+target_active)+ranking*(rank_orig+rank_shift)
    parts=dict(source=source.detach(),target=shifted.detach(),
               activity=(active+target_active).detach(),
               ranking=(rank_orig+rank_shift).detach())
    return loss,parts


@dataclass(frozen=True)
class PolyphonicPrediction:
    sample_end:int
    midi_probabilities:np.ndarray
    real_note_presence_probability:float
    source_probabilities:np.ndarray
    future_mic_samples:int=0
    hold_samples:int=0


class CausalPolyphonicTransposeStream256:
    """Streaming RESEARCH predictions. Does not write MIDI or resynthesize."""
    def __init__(self,model,*,shift=12):
        if not isinstance(model,NeuralPolyphonicTranspose256):
            raise TypeError("must use S850 trained neural model")
        if type(shift)!=int or not -12<=shift<=12:
            raise ValueError("shift must be int -12..12")
        self.model=model.cpu().eval()
        self.shift=shift
        self.state=None
        self.received=0
        self.added_hold_samples=0
        self.future_mic_samples=0

    def push(self,block,*,start_sample):
        x=np.asarray(block)
        if (x.shape!=(HOP,) or x.dtype!=np.float32 or
            not np.isfinite(x).all() or type(start_sample)!=int or
            start_sample!=self.received):
            raise ValueError("new PCM256 only, contiguous samples")
        with torch.inference_mode():
            output,self.state=self.model(
                torch.from_numpy(np.ascontiguousarray(x))[None,None,:],
                torch.tensor([self.shift],dtype=torch.int64),
                state=self.state)
            src=torch.sigmoid(output["source_logits"][0,0]).numpy().copy()
            tgt=torch.sigmoid(output["target_logits"][0,0]).numpy().copy()
            activity=float(torch.sigmoid(output["target_activity_logits"][0,0,0]))
        self.received+=HOP
        return PolyphonicPrediction(
            sample_end=self.received,midi_probabilities=tgt,
            source_probabilities=src,real_note_presence_probability=activity)
