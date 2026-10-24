"""S847: neural target-note selection followed by genuine recorded guitar audio.

PURPOSE: an alternate 'not synthetic' transposition experiment after S845's
neural AUDIO decoder failed and output high-frequency noise. A learned neural
transposer predicts the destination pitch; the audio output is copied verbatim
from an indexed, REAL GuitarSet hex-channel recording of that pitch.

IMPORTANT LIMITATION: sample selection from REAL notes is NOT a faithful
transposition of the musician's original performance: attacks, dynamics, and
timbre can differ. It is NOT newly generated audio and should never be called
a reconstructed recording of the original performance. This experimental
sample-based instrument is not a production MIDI/Onset detector.

Input: only 256 NEW mono mic samples per decision. Recurrent state comes from
previous input hops. No DSP pitch formula, resampling, vocoder, oscillator,
FFT-based pitch shift, future microphone PCM, buffering, or added hold.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SR=44100
HOP=256
PITCH_LOW=40
PITCHES=49
MIN_SHIFT=-12
MAX_SHIFT=12
ABI="NOTE_S847_AI_REAL_RECORDED_AUDIO_LOOKUP_256_V1"


def _verify_pcm(samples):
    if (not isinstance(samples,torch.Tensor) or samples.ndim!=3 or
            samples.shape[-1]!=HOP or not samples.is_floating_point() or
            not bool(torch.isfinite(samples).all())):
        raise ValueError("input must be finite [batch, time, 256] raw mic PCM")


class NeuralShiftSelector256(nn.Module):
    """Joint source-note predictor and learned shift-conditioned output."""
    def __init__(self,hidden=96):
        super().__init__()
        self.hidden=hidden
        self.cnn=nn.Sequential(
            nn.Conv1d(1,24,9,stride=2,padding=4),nn.GELU(),
            nn.Conv1d(24,48,9,stride=2,padding=4),nn.GELU(),
            nn.Conv1d(48,64,9,stride=2,padding=4),nn.GELU(),
            nn.AdaptiveAvgPool1d(4),nn.Flatten(),nn.Linear(64*4,hidden),
            nn.LayerNorm(hidden))
        self.gru=nn.GRUCell(hidden,hidden)
        self.shift_embedding=nn.Embedding(25,24)
        self.transport=nn.Sequential(nn.Linear(hidden+24,128),nn.GELU(),
                                      nn.Linear(128,hidden))
        self.shifted_head=nn.Linear(hidden,PITCHES)
        self.source_head=nn.Linear(hidden,PITCHES)

    def forward(self,samples,shifts,state=None):
        """One output per existing input hop. No horizon beyond supplied audio."""
        _verify_pcm(samples)
        B,T,_=samples.shape
        if (not isinstance(shifts,torch.Tensor) or
                shifts.shape!=(B,) or shifts.dtype!=torch.int64 or
                bool((shifts<MIN_SHIFT).any()) or
                bool((shifts>MAX_SHIFT).any())):
            raise ValueError("shifts must be int64 semitones -12..12 per source")
        h=samples.new_zeros((B,self.hidden)) if state is None else state
        if h.shape!=(B,self.hidden):raise ValueError("recurrent state shape")
        src=[];tgt=[]
        code=self.shift_embedding(shifts-MIN_SHIFT)
        for t in range(T):
            h=self.gru(self.cnn(samples[:,t,:].unsqueeze(1)),h)
            src.append(self.source_head(h))
            z=h+self.transport(torch.cat((h,code),dim=-1))
            tgt.append(self.shifted_head(z))
        return dict(source_logits=torch.stack(src,dim=1),
                    target_logits=torch.stack(tgt,dim=1)),h


def shift_truth_for_supervision_only(source_gt,offsets):
    """Label generation OFFLINE ONLY from JAMS GT, never in audio inference."""
    if source_gt.ndim!=3 or source_gt.shape[-1]!=PITCHES:
        raise ValueError("must train against 49 true binary GT note channels")
    B,T,P=source_gt.shape
    if offsets.shape!=(B,) or bool((offsets.abs()>12).any()):
        raise ValueError("invalid GT shifts")
    target=torch.zeros_like(source_gt)
    for b in range(B):
        positions=torch.nonzero(source_gt[b]>.5,as_tuple=False)
        if len(positions):
            to=positions[:,1]+int(offsets[b])
            valid=(to>=0)&(to<PITCHES)
            target[b,positions[valid,0],to[valid]]=1.
    return target


def supervised_loss(model,pcm,source_gt,offsets,pos_weights):
    """Weighted BCE with weights determined *only* by TRAIN GT."""
    truth=shift_truth_for_supervision_only(source_gt,offsets)
    if pos_weights.shape!=(PITCHES,) or bool((pos_weights<1).any()):
        raise ValueError("positive weights must be [49] and >=1")
    output,_=model(pcm,offsets)
    ordinary=F.binary_cross_entropy_with_logits(
        output["source_logits"],source_gt,pos_weight=pos_weights)
    shifted=F.binary_cross_entropy_with_logits(
        output["target_logits"],truth,pos_weight=pos_weights)
    return ordinary+shifted,dict(source=ordinary.detach(),shifted=shifted.detach())


def make_recorded_bank(waves,notes,source_ids):
    """Build bank from actual recorded hex waveforms [N, 256*k].

    All audio is copied directly; no synthetic note, no pitch math.
    Offline GT may label each exemplar; no JAMS truth is used in Live.
    """
    if (len(waves)!=len(notes) or len(notes)!=len(source_ids)):
        raise ValueError("bad physical sample bank metadata")
    bank={}
    for wave,note,source in zip(waves,notes,source_ids):
        if note in bank:continue
        clip=np.asarray(wave)
        if (type(note) not in (int,np.int64,int) or not PITCH_LOW<=int(note)<PITCH_LOW+PITCHES or
            clip.dtype!=np.float32 or clip.ndim!=1 or
            len(clip)<HOP*4 or len(clip)%HOP or
            not np.isfinite(clip).all() or not source):
            raise ValueError("exemplar must be authentic long real float32 44100Hz")
        bank[int(note)]=np.ascontiguousarray(clip.copy())
    if not bank:raise ValueError("at least one real recorded note required")
    return bank


@dataclass(frozen=True)
class RecordedNoteOutput:
    end_sample:int
    note:int|None
    confidence:float
    physical_recorded_wave256:np.ndarray
    source_probabilities:np.ndarray
    learned_shift_probabilities:np.ndarray
    sample_bank_is_real_not_reconstructed:bool=True


class RecordedAudioNeuralTransposeStream256:
    """Record-exemplar *playback* shadow, no output MIDI, no Live future input."""
    def __init__(self,model,recorded_bank,*,shift=12,threshold=.6):
        if not isinstance(model,NeuralShiftSelector256):
            raise TypeError("trained neural selector required")
        if type(shift)!=int or not -12<=shift<=12:
            raise ValueError("learned shift must be int [-12,+12]")
        if not 0<=threshold<=1:raise ValueError("probability threshold")
        if not recorded_bank:raise ValueError("must supply genuine recorded note bank")
        for pitch,clip in recorded_bank.items():
            if (not isinstance(pitch,int) or pitch<PITCH_LOW or pitch>=PITCH_LOW+PITCHES or
                not isinstance(clip,np.ndarray) or clip.dtype!=np.float32 or
                clip.ndim!=1 or len(clip)<HOP*4 or len(clip)%HOP):
                raise ValueError("invalid real audio bank")
        self.model=model.cpu().eval()
        self.bank=recorded_bank
        self.shift=shift
        self.threshold=threshold
        self.state=None
        self.received=0
        self.previous_pitch=None
        self.cursor=0
        self.added_hold_samples=0
        self.future_microphone_samples=0

    def push(self,pcm,*,start_sample):
        block=np.asarray(pcm)
        if (block.shape!=(HOP,) or block.dtype!=np.float32 or
            not np.isfinite(block).all() or type(start_sample)!=int or
            start_sample!=self.received):
            raise ValueError("strictly one contiguous mono 256-sample block")
        with torch.inference_mode():
            out,self.state=self.model(torch.from_numpy(np.ascontiguousarray(block))[None,None,:],
                                      torch.tensor([self.shift],dtype=torch.long),
                                      state=self.state)
            probs=torch.sigmoid(out["target_logits"][0,0]).cpu().numpy()
            src=torch.sigmoid(out["source_logits"][0,0]).cpu().numpy()
        # Neural target prediction only. No rule 'pitch + shift' in inference.
        eligible=[(float(probs[p-PITCH_LOW]),p) for p in self.bank]
        confidence,pitch=max(eligible) if eligible else (0.,None)
        if confidence<self.threshold:pitch=None
        if pitch!=self.previous_pitch:
            self.previous_pitch=pitch
            self.cursor=0
        if pitch is None:
            physical=np.zeros(HOP,np.float32)  # Silence, never invented notes.
        else:
            clip=self.bank[pitch]
            physical=clip[self.cursor:self.cursor+HOP].copy()
            self.cursor=(self.cursor+HOP)%len(clip)
        self.received+=HOP
        return RecordedNoteOutput(end_sample=self.received,note=pitch,
            confidence=confidence,physical_recorded_wave256=physical,
            source_probabilities=src.copy(),
            learned_shift_probabilities=probs.copy())
