"""S404: K-first time/pitch note event model, informed by YourMT3+ and
Andriamarosoa/midi, without importing either architecture/weights.

Architectural references:
- YourMT3+ (Chang et al., MLSP 2024): time-frequency encoder attention and
  structured event prediction (its decoder is autoregressive Multi-T5).
- Andriamarosoa/midi/src/polyphonic/model.py: independent multi-label frame
  and onset heads, masked harmonics as AUXILIARY supervision.
- S80_S397_K5_GUARDED: predicted exact K is a FIRST CLASS encoder input.
  This prototype does not modify/train any S80 checkpoint.

This is a NEW architecture PROTOTYPE, NOT a trained note recognizer.
Never cite any score for S404 until the full pipeline is trained on separate
compositions and evaluated independently. 42x49 here are harmonic templates,
not the full log-spectrogram in YourMT3+.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor,nn
from torch.nn import functional as F

NUM_MIDI=49
MIDI_MIN=40
HARMONICS=20
MAX_K=6
FRAMES=42

class KFirstMultiHeadNoteNet(nn.Module):
    """A compact initial time-pitch cross-attention network with K embedded
    BEFORE the encoder, then note queries for frame/onset/harmonic evidence.

    Reference: YourMT3+ attention concept + MIDI V2 four-head concept;
    unlike YourMT3+, NOT an autoregressive event-token MIDI decoder.
    """
    def __init__(self,dimension:int=96,heads:int=4,layers:int=2):
        super().__init__()
        if dimension%heads or dimension<32 or layers<1:
            raise ValueError("Invalid temporal attention hyperparameters")
        self.dim=dimension
        self.k_embed=nn.Embedding(MAX_K+1,dimension)
        self.acoustic_front=nn.Sequential(nn.Linear(NUM_MIDI,dimension),
                              nn.LayerNorm(dimension),nn.GELU())
        self.k_film=nn.Linear(dimension,2*dimension)
        self.time_embed=nn.Parameter(torch.randn(1,FRAMES,dimension)*.02)
        layer=nn.TransformerEncoderLayer(d_model=dimension,nhead=heads,
                 dim_feedforward=4*dimension,dropout=0.0,
                 batch_first=True,norm_first=True)
        self.time_encoder=nn.TransformerEncoder(layer,layers,
                                               enable_nested_tensor=False)
        self.pitch_queries=nn.Parameter(torch.randn(1,NUM_MIDI,dimension)*.02)
        self.k_pitch=nn.Linear(dimension,dimension)
        self.cross_attention=nn.MultiheadAttention(
             embed_dim=dimension,num_heads=heads,batch_first=True,dropout=0)
        self.query_norm=nn.LayerNorm(dimension)
        self.frame_head=nn.Linear(dimension,1)
        self.onset_head=nn.Linear(dimension,1)
        self.harmonic_head=nn.Linear(dimension,HARMONICS)
        self.harmonic_cent_head=nn.Linear(dimension,HARMONICS)

    def forward(self,harmonic_time:Tensor,predicted_k:Tensor)->dict[str,Tensor]:
        if harmonic_time.ndim!=3 or harmonic_time.shape[1:]!=(FRAMES,NUM_MIDI):
            raise ValueError("Expected original 42-frame x 49-pitch acoustic tensor")
        if predicted_k.ndim!=1 or predicted_k.shape[0]!=harmonic_time.shape[0]:
            raise ValueError("Expected per-event S80 K tensor")
        if predicted_k.dtype not in (torch.int64,torch.long):
            raise TypeError("S80 predicted K must be int64")
        if not bool(torch.all((predicted_k>=0)&(predicted_k<=MAX_K))):
            raise ValueError("Predicted K must be in 0..6")
        if not bool(torch.isfinite(harmonic_time).all()) or bool((harmonic_time<0).any()):
            raise ValueError("Original log1p acoustic harmonic features must be finite and >=0")
        context=self.k_embed(predicted_k)  # K enters BEFORE note probabilities
        x=self.acoustic_front(harmonic_time)
        gain,bias=self.k_film(context).chunk(2,dim=-1)
        # K directly conditions the entire acoustic temporal representation.
        x=x*(1+torch.tanh(gain).unsqueeze(1))+bias.unsqueeze(1)+self.time_embed
        memory=self.time_encoder(x)
        queries=self.pitch_queries.expand(x.shape[0],-1,-1)
        queries=queries+self.k_pitch(context).unsqueeze(1)
        q,_=self.cross_attention(queries,memory,memory,need_weights=False)
        q=self.query_norm(q+queries)
        frame=self.frame_head(q).squeeze(-1)
        onset=self.onset_head(q).squeeze(-1)
        harmonics=self.harmonic_head(q)
        harmonic_cents=35.0*torch.tanh(self.harmonic_cent_head(q))
        # The pitch output learns both whether a note is present and whether
        # it is newly attacked at this event (S399 lacks this distinction).
        note_logits=frame+onset
        return dict(note_logits=note_logits,frame_logits=frame,
          onset_logits=onset,harmonic_amplitude_logits=harmonics,
          harmonic_offset_cents=harmonic_cents)

@torch.no_grad()
def select_exact_k_notes(note_logits:Tensor,predicted_k:Tensor)->Tensor:
    """Return (B,6) padded with -1. Do not infer cardinality from spectral peaks."""
    if note_logits.ndim!=2 or note_logits.shape[1]!=NUM_MIDI:
        raise ValueError("Expected (B,49) logits")
    if predicted_k.shape!=(note_logits.shape[0],) or predicted_k.dtype!=torch.long:
        raise ValueError("S80 K shape/dtype must match batch")
    if not bool(torch.all((predicted_k>=0)&(predicted_k<=MAX_K))):
        raise ValueError("S80 K0..6 required")
    if not bool(torch.isfinite(note_logits).all()):
        raise ValueError("Nonfinite MIDI note logits")
    ranking=torch.argsort(note_logits,dim=1,descending=True,stable=True)[:,:MAX_K]
    midi=ranking+MIDI_MIN
    slot=torch.arange(MAX_K,device=note_logits.device)[None,:]
    result=torch.where(slot<predicted_k[:,None],midi,-1)
    if not bool(torch.equal((result>=0).sum(dim=1),predicted_k)):
        raise AssertionError("K cardinality violated")
    return result

@dataclass(frozen=True)
class Supervision:
    """Train-only multi-label targets; *never* replace predicted_k with true_K
    when calling the model. Acoustic harmonic labels unavailable in mixed audio
    must be entirely masked, not generated from mixed spectral peaks.
    """
    pitch_onset:Tensor      # [B,49] multihot of newly attacked MIDI notes
    pitch_active:Tensor     # [B,49] multihot sustained / active notes
    harmonic_amp:Tensor    # [B,49,20] measured physical partial amps
    harmonic_valid:Tensor  # [B,49,20] 1 ONLY for separable ground truth

def masked_binary_loss(logits:Tensor,target:Tensor,valid:Tensor)->Tensor:
    if logits.shape!=target.shape or logits.shape!=valid.shape:
        raise ValueError("Loss mask/target width mismatch")
    if not bool(((valid>=0)&(valid<=1)).all()):
        raise ValueError("Invalid mask")
    loss=F.binary_cross_entropy_with_logits(logits,target,reduction="none")*valid
    return loss.sum()/valid.sum().clamp(min=1)

def multitask_loss(out:dict[str,Tensor],y:Supervision)->dict[str,Tensor]:
    for x in (y.pitch_onset,y.pitch_active):
        if x.shape!=out["frame_logits"].shape:raise ValueError("Frame/onset target size")
    if y.harmonic_amp.shape!=out["harmonic_amplitude_logits"].shape:
        raise ValueError("Physical harmonic supervision size")
    nloss=F.binary_cross_entropy_with_logits(out["onset_logits"],y.pitch_onset)
    floss=F.binary_cross_entropy_with_logits(out["frame_logits"],y.pitch_active)
    # MIDI V2 insight: do NOT train harmonics on invented mixed-audio truth.
    hloss=masked_binary_loss(out["harmonic_amplitude_logits"],
                             y.harmonic_amp,y.harmonic_valid)
    total=nloss+.5*floss+.05*hloss
    return dict(total=total,onset=nloss,frame=floss,harmonic_masked=hloss)

def self_test()->None:
    torch.set_num_threads(2)
    torch.manual_seed(404)
    net=KFirstMultiHeadNoteNet(dimension=32,heads=4,layers=1)
    net.eval()
    audio=torch.rand(4,FRAMES,NUM_MIDI)*2
    predicted_k=torch.tensor([0,1,3,6],dtype=torch.long)
    with torch.no_grad():
        a=net(audio,predicted_k)
        changed=net(audio,torch.tensor([3,1,3,6]))
    assert a["note_logits"].shape==(4,49)
    assert a["harmonic_amplitude_logits"].shape==(4,49,20)
    assert (a["note_logits"][0]-changed["note_logits"][0]).abs().max().item()>1e-7,(
        "K must affect logits BEFORE top-K, not only cardinality")
    chosen=select_exact_k_notes(a["note_logits"],predicted_k)
    assert chosen.shape==(4,6)
    assert (chosen>=0).sum(1).tolist()==[0,1,3,6]
    assert (chosen[0]==-1).all()
    assert bool(((chosen[chosen>=0]>=MIDI_MIN)&(chosen[chosen>=0]<=88)).all())
    no_labels=torch.zeros(4,49,20)
    zero_mask=torch.zeros_like(no_labels)
    y=Supervision(pitch_onset=torch.zeros(4,49),
          pitch_active=torch.zeros(4,49),
          harmonic_amp=no_labels,harmonic_valid=zero_mask)
    loss=multitask_loss(a,y)
    assert abs(loss["harmonic_masked"].item())<1e-10
    assert torch.isfinite(loss["total"])
    net.train()
    trainout=net(audio,predicted_k)
    trainloss=multitask_loss(trainout,y)["total"]
    trainloss.backward()
    assert net.k_embed.weight.grad is not None
    assert torch.isfinite(net.k_embed.weight.grad).all()
    print("S404 PASS: K before encoder and note logits, K0/K1/K3/K6 slots,"
          " onset/frame/harmonic heads, zero harmonic labels -> zero harmonic loss;"
          " K embedding receives training gradients")
if __name__=="__main__":
    self_test()
