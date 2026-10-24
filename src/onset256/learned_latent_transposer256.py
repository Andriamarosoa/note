"""Isolated learned latent transposition to propose more guitar notes.

NOT a fixed-frequency pitch shifter, audio effect or MIDI transposer.
Each hypothesis sees one 256 PCM block (not an 8192-point FFT); learned
shift-conditioned neural transport is supervised by independently recorded
transposed real-note pairs. No audio future, no hold, no S708, no other head.
Research-only outputs: never send NoteOn, NoteOff, or alter official MIDI.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SR=44100
HOP=256
PITCH_LOW=40
PITCH_COUNT=49
MIN_SHIFT=-12
MAX_SHIFT=12
MODEL_ABI="ONSET_LEARNED_LATENT_TRANSPOSE_RAW256_V1"


def verify_pcm(x):
    if not isinstance(x,torch.Tensor) or x.ndim!=2 or x.shape[1]!=HOP:
        raise ValueError("exactly [batch,256] fresh raw mono PCM required")
    if not x.is_floating_point() or not bool(torch.isfinite(x).all()):
        raise ValueError("finite floating-point input required")


def verify_shifts(offsets,n,*,nonzero=False):
    if (not isinstance(offsets,torch.Tensor) or offsets.shape!=(n,) or
        offsets.dtype not in (torch.int32,torch.int64) or
        bool((offsets<MIN_SHIFT).any()) or bool((offsets>MAX_SHIFT).any()) or
        (nonzero and bool((offsets==0).any()))):
        raise ValueError("integer semitone shifts -12..+12 are required")


def verify_recorded_pairs(source,target,source_labels,target_labels,offsets):
    """Only validates GT indices, never shifts audio or its frequencies.

    This cannot prove the recordings are authentic: signed WAV provenance
    and TRAIN/CAL/OUTER split IDs must be checked by the training harness.
    """
    verify_pcm(source);verify_pcm(target)
    n=len(source)
    if (len(target)!=n or target.device!=source.device or
        source_labels.shape!=(n,PITCH_COUNT) or target_labels.shape!=(n,PITCH_COUNT) or
        source_labels.device!=source.device or target_labels.device!=source.device or
        offsets.device!=source.device):
        raise ValueError("paired clips/labels must agree")
    verify_shifts(offsets,n,nonzero=True)
    for labels in (source_labels,target_labels):
        if (not labels.is_floating_point() or not bool(torch.isfinite(labels).all()) or
            bool(((labels!=0.)&(labels!=1.)).any())):
            raise ValueError("truth must be binary GT pitch labels")
    if bool((source_labels.sum(1)<1).any()):
        raise ValueError("a real note or chord is required per pair")
    translated=torch.zeros_like(target_labels)
    for i in range(n):
        ix=torch.nonzero(source_labels[i],as_tuple=True)[0]
        to=ix+offsets[i]
        if bool(((to<0)|(to>=PITCH_COUNT)).any()):
            raise ValueError("transposed GT note outside MIDI 40..88")
        translated[i,to]=1.
    if not torch.equal(translated,target_labels):
        raise ValueError("real recorded pair does not match declared pitch shift")


class AudioEncoder256(nn.Module):
    """Learnable raw-waveform CNN; no FFT, DSP resampling or pitch equation."""
    def __init__(self,latent=64):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Conv1d(1,24,9,stride=2,padding=4),nn.GELU(),
            nn.Conv1d(24,40,9,stride=2,padding=4),nn.GELU(),
            nn.Conv1d(40,48,9,stride=2,padding=4),nn.GELU(),
            nn.AdaptiveAvgPool1d(1),nn.Flatten(),
            nn.Linear(48,latent),nn.LayerNorm(latent))
    def forward(self,pcm):
        verify_pcm(pcm)
        return self.layers(pcm.unsqueeze(1))


class NeuralLatentTransposer256(nn.Module):
    """True trainable transport in latent acoustics, not an index shift model.

    The shifted latent is trained against an independently recorded physical
    target embedding AND that target's pitch labels. A separate learned decoder
    reads ORIGINAL pitches from each transposed view without inverse bin math.
    """
    def __init__(self,latent_dim=64,shift_dim=16):
        super().__init__()
        self.encoder=AudioEncoder256(latent_dim)
        self.shift_embedding=nn.Embedding(MAX_SHIFT-MIN_SHIFT+1,shift_dim)
        self.transport=nn.Sequential(nn.Linear(latent_dim+shift_dim,96),nn.GELU(),
                                     nn.Linear(96,latent_dim))
        self.transport_gate=nn.Sequential(nn.Linear(latent_dim+shift_dim,latent_dim),nn.Sigmoid())
        self.plain_note_head=nn.Linear(latent_dim,PITCH_COUNT)
        self.shifted_note_head=nn.Linear(latent_dim,PITCH_COUNT)
        self.original_note_head=nn.Sequential(nn.Linear(2*latent_dim+shift_dim,96),
                                               nn.GELU(),nn.Linear(96,PITCH_COUNT))
        self.view_quality=nn.Sequential(nn.Linear(2*latent_dim+shift_dim,32),
                                         nn.GELU(),nn.Linear(32,1))

    def _embedding(self,offsets,n):
        verify_shifts(offsets,n)
        return self.shift_embedding(offsets.long()-MIN_SHIFT)

    def transport_latent(self,z,offsets):
        if z.ndim!=2:raise ValueError("expected two-dimensional latent")
        code=self._embedding(offsets,len(z))
        current=torch.cat((z,code),dim=-1)
        evolved=z+self.transport_gate(current)*self.transport(current)
        return torch.where((offsets==0)[:,None],z,evolved)

    def one_view(self,z,offsets):
        code=self._embedding(offsets,len(z))
        shifted=self.transport_latent(z,offsets)
        joined=torch.cat((z,shifted,code),dim=-1)
        return shifted,self.original_note_head(joined),self.view_quality(joined)

    def plain(self,x):
        return self.plain_note_head(self.encoder(x))

    def forward(self,x,shifts=(-12,-7,7,12)):
        """Produce NOTE hypotheses in the original tuning, no MIDI events."""
        z=self.encoder(x)
        base=self.plain_note_head(z)
        preds=[base]
        quality=[torch.zeros((len(z),1),device=z.device,dtype=base.dtype)]
        for offset in shifts:
            if type(offset) is not int or offset==0 or not MIN_SHIFT<=offset<=MAX_SHIFT:
                raise ValueError("requested view must be nonzero integer -12..12")
            offsets=torch.full((len(z),),offset,device=z.device,dtype=torch.long)
            _,logits,q=self.one_view(z,offsets)
            preds.append(logits);quality.append(q)
        weights=torch.softmax(torch.cat(quality,dim=1),dim=1)
        combined=(weights[:,:,None]*torch.stack(preds,dim=1)).sum(dim=1)
        return combined,base,weights


def recorded_pair_training_loss(model,source,target,source_labels,target_labels,offsets):
    """Supervise learned transposition on independently *recorded* note pairs.

    A real-pair dataset is mandatory for any science claim: using synthetic
    examples is valid only for testing code and backprop, not evaluating music.
    """
    verify_recorded_pairs(source,target,source_labels,target_labels,offsets)
    original=model.encoder(source)
    actual_target=model.encoder(target)
    mapped,original_logits,_=model.one_view(original,offsets)
    positive=torch.full((PITCH_COUNT,),6.,device=source.device)
    bce=lambda x,y:F.binary_cross_entropy_with_logits(x,y,pos_weight=positive)
    supervision=(bce(model.plain_note_head(original),source_labels)+
                 bce(model.plain_note_head(actual_target),target_labels)+
                 bce(model.shifted_note_head(mapped),target_labels)+
                 bce(original_logits,source_labels))
    alignment=(1.-F.cosine_similarity(mapped,actual_target.detach(),dim=-1)).mean()
    return supervision+0.3*alignment,dict(supervised=supervision.detach(),
                                         physical_pair_alignment=alignment.detach())


@dataclass(frozen=True)
class NoteHypothesis:
    sample:int
    note_probabilities:np.ndarray
    plain_probabilities:np.ndarray
    view_weights:np.ndarray


class CausalNeuralTransposerStream256:
    """Isolated one-hop read-only research helper, NOT an Onset/MIDI decoder."""
    hop_samples=256
    added_hold_samples=0
    future_audio_samples=0
    history_raw_samples=0

    def __init__(self,model,*,shifts=(-12,-7,7,12)):
        if not isinstance(model,NeuralLatentTransposer256):
            raise TypeError("trained neural latent transposer required")
        self.model=model.eval().cpu()
        self.shifts=tuple(shifts)
        self.received=0

    def push(self,pcm,*,start_sample):
        pcm=np.asarray(pcm)
        if (pcm.shape!=(HOP,) or pcm.dtype!=np.float32 or
            not np.isfinite(pcm).all() or
            type(start_sample) is not int or start_sample!=self.received):
            raise ValueError("must receive precisely one contiguous 256 block")
        with torch.inference_mode():
            block=torch.from_numpy(np.ascontiguousarray(pcm))[None,:]
            out,control,weight=self.model(block,shifts=self.shifts)
            out=torch.sigmoid(out)[0].numpy().copy()
            control=torch.sigmoid(control)[0].numpy().copy()
            weight=weight[0].numpy().copy()
        self.received+=HOP
        return NoteHypothesis(self.received,out,control,weight)
