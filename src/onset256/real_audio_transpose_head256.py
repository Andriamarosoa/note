"""S843: independent causal neural reconstruction + learned transposition.

Input: EXACTLY 256 new samples from a real guitar microphone at 44.1 kHz.
State: a small recurrent hidden vector from PAST blocks only; no lookahead,
no attack confirmation, no temporal output queue, no DSP pitch shift.
Targets: genuinely recorded six-channel hexaphonic guitar pickup and labels.

The decoded waveforms are NEURAL ESTIMATES of acoustic strings. They are not
the physical original recordings, guaranteed isolated notes, or a proven
sounding transposed performance; this distinction is mandatory.
No official MIDI, NoteOn/Off or any of NOTE's other six heads is modified.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SR=44100
HOP=256
STRINGS=6
PITCHES=49
LOW_MIDI=40
MIN_SHIFT=-12
MAX_SHIFT=12
DEFAULT_VIEWS=(-12,12)
ABI="NOTE_S843_CAUSAL_REAL_AUDIO_RECONSTRUCTION_TRANSPOSE_PCM256_V1"


def _pcm(x):
    if not isinstance(x,torch.Tensor) or x.ndim!=3 or x.shape[-1]!=HOP:
        raise ValueError("expected real [batch,time,256] PCM at 44.1 kHz")
    if not x.is_floating_point() or not bool(torch.isfinite(x).all()):
        raise ValueError("finite floating-point microphone PCM required")


def _offsets(offsets,batch):
    if (not isinstance(offsets,torch.Tensor) or offsets.shape!=(batch,) or
            offsets.dtype not in (torch.int32,torch.int64) or
            bool((offsets<MIN_SHIFT).any()) or bool((offsets>MAX_SHIFT).any())):
        raise ValueError("learned pitch-shift codes must be integers -12..12")


class RealGuitarEncoder256(nn.Module):
    """Entirely learned raw-audio encoder; NO FFT, oscillator or resampling."""
    def __init__(self,hidden=96):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv1d(1,24,9,stride=2,padding=4),nn.GELU(),
            nn.Conv1d(24,48,9,stride=2,padding=4),nn.GELU(),
            nn.Conv1d(48,64,9,stride=2,padding=4),nn.GELU(),
            nn.AdaptiveAvgPool1d(4),nn.Flatten(),
            nn.Linear(64*4,hidden),nn.LayerNorm(hidden))
    def forward(self,x):
        if x.ndim!=2 or x.shape[-1]!=HOP:
            raise ValueError("encoder accepts only one raw 256-sample hop")
        return self.conv(x.unsqueeze(1))


class NeuralAcousticTransposeHead256(nn.Module):
    """Audio-to-six-real-string-estimates + learned ±octave latent evidence.

    Full acoustic separation requires real mic/hex stem supervision.
    The *same* reconstruction decoder is applied to shifted latent views,
    which are trained against independent REAL (not DSP transposed) guitar
    recordings using latent matching and pitch labels.
    """
    def __init__(self,hidden=96,shift_dim=24):
        super().__init__()
        self.hidden=hidden
        self.encoder=RealGuitarEncoder256(hidden)
        self.history=nn.GRUCell(hidden,hidden)
        self.shift_embedding=nn.Embedding(MAX_SHIFT-MIN_SHIFT+1,shift_dim)
        self.transport=nn.Sequential(
            nn.Linear(hidden+shift_dim,hidden*2),nn.GELU(),
            nn.Linear(hidden*2,hidden))
        self.transport_gate=nn.Sequential(nn.Linear(hidden+shift_dim,hidden),
                                          nn.Sigmoid())
        self.stem_waveform=nn.Sequential(
            nn.Linear(hidden,hidden*2),nn.GELU(),
            nn.Linear(hidden*2,STRINGS*HOP),nn.Tanh())
        # S845: learn *local waveform-preserving* filtering of the ORIGINAL
        # recorded microphone samples. No oscillator or predefined pitch
        # shift. Recurrent latent only conditions the dynamic string gains.
        self.local_string_filters=nn.Sequential(
            nn.Conv1d(1,24,kernel_size=17,padding=8),
            nn.GELU(),
            nn.Conv1d(24,STRINGS,kernel_size=17,padding=8))
        self.string_condition=nn.Linear(hidden,STRINGS)
        self.stem_presence=nn.Linear(hidden,STRINGS)
        self.source_note=nn.Linear(hidden,PITCHES)
        self.shifted_note=nn.Linear(hidden,PITCHES)
        self.reverse_mic=nn.Sequential(
            nn.Linear(STRINGS*HOP,hidden*2),nn.GELU(),
            nn.Linear(hidden*2,HOP),nn.Tanh())
        self.original_from_view=nn.Sequential(
            nn.Linear(2*hidden+shift_dim,hidden),nn.GELU(),
            nn.Linear(hidden,PITCHES))
        self.view_reliability=nn.Sequential(
            nn.Linear(2*hidden+shift_dim,hidden//2),nn.GELU(),
            nn.Linear(hidden//2,1))

    def initial_state(self,batch,device=None,dtype=torch.float32):
        return torch.zeros((batch,self.hidden),device=device,dtype=dtype)

    def encode_hop(self,x,state):
        if x.ndim!=2 or x.shape[1]!=HOP:
            raise ValueError("exactly one 256 sample hop")
        if state.shape!=(len(x),self.hidden):
            raise ValueError("causal hidden state shape changed")
        z=self.encoder(x)
        return self.history(z,state)

    def transposed_latent(self,h,offset):
        _offsets(offset,len(h))
        code=self.shift_embedding(offset.long()-MIN_SHIFT)
        x=torch.cat((h,code),dim=1)
        shifted=h+self.transport_gate(x)*self.transport(x)
        return torch.where((offset==0)[:,None],h,shifted)

    def reconstruct(self,h,raw_pcm=None):
        # Train a waveform output against actual *simultaneously recorded*
        # hex pickup. Preserve the current real microphone's local temporal
        # structure so the network doesn't have to invent every sample from
        # the bottleneck alone. This is neural separation, NOT DSP pitch shift.
        waveform=self.stem_waveform(h).reshape(-1,STRINGS,HOP)
        presence_logits=self.stem_presence(h)
        if raw_pcm is not None:
            if raw_pcm.shape!=(len(h),HOP):
                raise ValueError("raw current PCM must remain exactly 256")
            local=self.local_string_filters(raw_pcm[:,None,:])
            gains=1.+torch.tanh(self.string_condition(h))[:,:,None]
            estimated=torch.tanh(.5*waveform + gains*local)
        else:
            # Latent transposed audio has no authentic direct PCM. These
            # shifted waveform outputs remain unverified neural estimates.
            estimated=torch.tanh(waveform)
        # Do not force the waveform toward zero with a presence sigmoid:
        # a masked quiet estimate can win BCE yet lose all real acoustics.
        mic=self.reverse_mic(estimated.flatten(1))
        return estimated,presence_logits,mic

    def from_latent(self,h,*,views=DEFAULT_VIEWS,original_pcm=None):
        stems,presence,mic=self.reconstruct(h,original_pcm)
        base=self.source_note(h)
        candidate=[base];quality=[torch.zeros((len(h),1),device=h.device,dtype=h.dtype)]
        shifted_audio=[]
        for shift in views:
            if type(shift) is not int or shift==0 or not MIN_SHIFT<=shift<=MAX_SHIFT:
                raise ValueError("only learned nonzero semitone views -12..12")
            offsets=torch.full((len(h),),shift,device=h.device,dtype=torch.long)
            code=self.shift_embedding(offsets-MIN_SHIFT)
            transformed=self.transposed_latent(h,offsets)
            joined=torch.cat((h,transformed,code),dim=1)
            candidate.append(self.original_from_view(joined))
            quality.append(self.view_reliability(joined))
            shifted_audio.append(self.reconstruct(transformed)[0])
        weights=torch.softmax(torch.cat(quality,dim=1),dim=1)
        logits=(weights[:,:,None]*torch.stack(candidate,dim=1)).sum(dim=1)
        return dict(stems=stems,stem_presence_logits=presence,
            mic_reconstructed=mic,plain_note_logits=base,
            note_logits=logits,view_weights=weights,
            transposed_acoustic_estimates=tuple(shifted_audio))

    def forward(self,blocks,state=None,*,views=DEFAULT_VIEWS):
        """Unroll ONLY over time already supplied; output at each input hop."""
        _pcm(blocks)
        B,T,N=blocks.shape
        h=self.initial_state(B,device=blocks.device,dtype=blocks.dtype) if state is None else state
        rows=[]
        for t in range(T):
            h=self.encode_hop(blocks[:,t,:],h)
            rows.append(self.from_latent(h,views=views,original_pcm=blocks[:,t,:]))
        keys=("stems","stem_presence_logits","mic_reconstructed",
              "plain_note_logits","note_logits","view_weights")
        data={k:torch.stack([row[k] for row in rows],dim=1) for k in keys}
        data["transposed_acoustic_estimates"]=tuple(
            torch.stack([row["transposed_acoustic_estimates"][k] for row in rows],dim=1)
            for k in range(len(views)))
        return data,h


def reconstruction_loss(model,pcm,real_hex,pitch_targets,string_targets,
                        *,state=None,views=DEFAULT_VIEWS,note_positive_weight=None):
    """Use only real, time-matched hex pickup stems; no simulated note audio.

    raw real_hex: [B,T,6,256] from physically recorded hex pickup;
    six strings may still contain BLEED, so source separation is an estimate.
    GT note and string activity must originate from JAMS, not model guesses.
    """
    _pcm(pcm)
    b,t,n=pcm.shape
    if (real_hex.shape!=(b,t,STRINGS,HOP) or
        pitch_targets.shape!=(b,t,PITCHES) or
        string_targets.shape!=(b,t,STRINGS)):
        raise ValueError("unaligned recorded hex or ground-truth labels")
    for x in (real_hex,pitch_targets,string_targets):
        if (x.device!=pcm.device or not torch.isfinite(x).all()):
            raise ValueError("mismatched or invalid real recording data")
    for labels in (pitch_targets,string_targets):
        if bool(((labels!=0)&(labels!=1)).any()):
            raise ValueError("GT must be binary, never pseudo labels")
    estimated,h=model(pcm,state,views=views)
    pred=estimated["stems"]
    # MSE/L1 waveform evidence and channel activity are supervised by REAL
    # recorded audio only; no sine oscillator or formulaic pitch synthesis.
    scale=real_hex.abs().mean(dim=(-1,-2,-3),keepdim=True).clamp_min(.001)
    waveform=F.smooth_l1_loss(pred/scale,real_hex/scale,beta=.5)
    mic=F.smooth_l1_loss(estimated["mic_reconstructed"],pcm)
    active=F.binary_cross_entropy_with_logits(
        estimated["stem_presence_logits"],string_targets)
    if note_positive_weight is not None:
        if (note_positive_weight.shape!=(PITCHES,) or
            bool((note_positive_weight<1).any()) or
            bool((note_positive_weight>30).any())):
            raise ValueError("note positive weights must be a bounded TRAIN-only 49D vector")
    note=F.binary_cross_entropy_with_logits(
        estimated["plain_note_logits"],pitch_targets,
        pos_weight=note_positive_weight)
    combined=F.binary_cross_entropy_with_logits(
        estimated["note_logits"],pitch_targets,
        pos_weight=note_positive_weight)
    # S844: a faint E2 must count. S843 learned the all-negative solution
    # because virtually every row/pitch was silent. Keep audio supervision
    # but give the observed rare TRUE notes substantial gradient strength.
    total=.25*waveform+.2*mic+.5*active+2.*note+2.*combined
    return total,{k:val.detach() for k,val in dict(
        waveform=waveform,mic_cycle=mic,string_activity=active,
        note_activity=note,fused_note_activity=combined).items()},h


def real_shift_pair_loss(model,source_pcm,target_pcm,
                         source_pitch_gt,target_pitch_gt,shift,
                         *,views=DEFAULT_VIEWS):
    """Learn transposition using separately *recorded* guitar notes, not DSP.

    This is LATENT alignment, not sample-perfect waveform regression across
    unrelated phases/performances. Requires semitone-grounded original GT
    pairs verified offline; target never provided to source live inference.
    """
    if (source_pcm.shape!=(len(source_pcm),1,HOP) or
        source_pcm.shape!=target_pcm.shape or
        source_pitch_gt.shape!=(len(source_pcm),PITCHES) or
        target_pitch_gt.shape!=source_pitch_gt.shape):
        raise ValueError("real independent source/target clips only")
    _pcm(source_pcm);_pcm(target_pcm)
    _offsets(shift,len(source_pcm))
    b=len(source_pcm)
    hs=model.encode_hop(source_pcm[:,0,:],model.initial_state(b,source_pcm.device,source_pcm.dtype))
    ht=model.encode_hop(target_pcm[:,0,:],model.initial_state(b,target_pcm.device,target_pcm.dtype))
    transported=model.transposed_latent(hs,shift)
    latent_loss=(1-F.cosine_similarity(transported,ht.detach(),dim=-1)).mean()
    target_class=F.binary_cross_entropy_with_logits(model.shifted_note(transported),target_pitch_gt)
    source_class=F.binary_cross_entropy_with_logits(model.source_note(hs),source_pitch_gt)
    code=model.shift_embedding(shift.long()-MIN_SHIFT)
    source_v=model.original_from_view(torch.cat((hs,transported,code),dim=1))
    return .25*latent_loss+target_class+source_class+F.binary_cross_entropy_with_logits(
        source_v,source_pitch_gt)


@dataclass(frozen=True)
class ReconstructedAcousticHop:
    sample:int
    strings:np.ndarray                  # 6 x 256 waveform ESTIMATES, not ground truth
    mic_reconstruction:np.ndarray      # 256 waveform estimate
    note_probabilities:np.ndarray     # 49 independent music-note candidates
    plain_probabilities:np.ndarray    # same model, no transposed views
    view_weights:np.ndarray          # [plain, learned shifts]
    transposed_acoustic_estimates:Tuple[np.ndarray,...]  # estimated 6x256 each


class CausalRealAudioTransposeStream256:
    """Real-time research-only streaming; no MIDI, no internal audio queue."""
    block_samples=HOP
    added_hold_samples=0
    future_samples=0

    def __init__(self,model:NeuralAcousticTransposeHead256,*,views=DEFAULT_VIEWS,
                 device="cpu"):
        if not isinstance(model,NeuralAcousticTransposeHead256):
            raise TypeError("trained neural acoustic reconstruction model required")
        self.model=model.to(device).eval()
        self.views=tuple(views)
        self.state=None
        self.received=0
        self.device=torch.device(device)
        self.last_output=None

    def reset(self):
        self.received=0
        self.state=None
        self.last_output=None

    def push(self,block,*,start_sample):
        x=np.asarray(block)
        if (x.shape!=(HOP,) or x.dtype!=np.float32 or
            not np.isfinite(x).all() or type(start_sample) is not int or
            start_sample!=self.received):
            raise ValueError("exactly one contiguous 256-sample real float32 hop required")
        with torch.inference_mode():
            wav=torch.from_numpy(np.ascontiguousarray(x))[None,None,:].to(self.device)
            output,self.state=self.model(wav,self.state,views=self.views)
            as_np=lambda tensor:tensor[0,0].detach().cpu().numpy().copy()
            record=ReconstructedAcousticHop(
                sample=self.received+HOP,
                strings=as_np(output["stems"]),
                mic_reconstruction=as_np(output["mic_reconstructed"]),
                note_probabilities=as_np(torch.sigmoid(output["note_logits"])),
                plain_probabilities=as_np(torch.sigmoid(output["plain_note_logits"])),
                view_weights=as_np(output["view_weights"]),
                transposed_acoustic_estimates=tuple(as_np(t) for t in output["transposed_acoustic_estimates"])
            )
        self.received+=HOP
        self.last_output=record
        return record
