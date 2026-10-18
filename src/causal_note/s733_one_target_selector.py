"""S733 research only: sequential one-target, K-conditional neural selector.

The design preserves the architecture of the historical
ClassConditionalAuditSelector: trained attention between available heads,
per-K structural masks, learned correction/regression risks and KEEP.
There are NO trained weights here, no LIVE integration, no promotion.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import torch
from torch import nn
from torch.nn import functional as F

Target = Literal["onset", "sustain", "release"]
KEEP, ADOPT, SUPPRESS = 0, 1, 2


@dataclass(frozen=True)
class SingleTargetContract:
    target: Target
    source_sha256: str
    audit_is_train_only_oof: bool
    reference_is_frozen: bool = True

    def validate(self):
        if self.target not in ("onset", "sustain", "release"):
            raise ValueError("S733 trains exactly one target head")
        if len(self.source_sha256) != 64 or any(
                c not in "0123456789abcdef" for c in self.source_sha256.lower()):
            raise ValueError("Expected 64-hex signed S708 source SHA256")
        if not self.audit_is_train_only_oof or not self.reference_is_frozen:
            raise ValueError("Requires frozen source and genuine FIT-only OOF audits")
        return self


class OneTargetClassConditionalSelector(nn.Module):
    """One pitch/time and one trainable target per row, never simultaneous.

    Inputs: head_scores [B,H], audit [B,H,21] (train-only OOF),
    head_type [B,H,4], head_mask [B,H], head_k_mask [B,H,7],
    context [B,C], k_probs [B,7] (frozen S708), baseline [B].
    Outputs: action_logits [B,3] KEEP/ADOPT/SUPPRESS,
    class_selection_weights [B,H,7], selection_weights [B,H],
    head_risk [B,H,2], offered_probability [B].
    """
    def __init__(self, context_dim=16, width=48, n_attention_heads=3):
        super().__init__()
        if width % n_attention_heads:
            raise ValueError("width must divide attention heads")
        self.context_dim = context_dim
        self.head_proj = nn.Sequential(nn.Linear(26, width), nn.GELU())
        self.context_proj = nn.Sequential(nn.Linear(context_dim+8,width),nn.GELU())
        self.attention = nn.MultiheadAttention(
            width,n_attention_heads,batch_first=True,dropout=0.)
        self.fuse = nn.Sequential(nn.Linear(width,width),nn.GELU())
        self.k_gate = nn.Linear(width,7)
        self.risk = nn.Sequential(nn.Linear(width,2),nn.Sigmoid())
        self.action = nn.Sequential(nn.Linear(width*2+9,width),nn.GELU(),
                                    nn.Linear(width,3))

    def forward(self,head_scores,audit,head_type,head_mask,head_k_mask,
                context,k_probs,baseline):
        b,h=head_scores.shape
        if (audit.shape!=(b,h,21) or head_type.shape!=(b,h,4)
            or head_mask.shape!=(b,h) or head_k_mask.shape!=(b,h,7)
            or context.shape!=(b,self.context_dim) or k_probs.shape!=(b,7)
            or baseline.shape!=(b,)):
            raise ValueError("Incorrect S733 single-target ABI")
        if not all(bool(torch.isfinite(x).all()) for x in
                   (head_scores,audit,context,k_probs,baseline)):
            raise ValueError("Nonfinite frozen input")
        mask=head_mask.bool()
        class_mask=head_k_mask.bool() & mask[:,:,None]
        if not bool(mask[:,0].all()) or not bool(class_mask[:,0,:].all()):
            raise ValueError("Unmodified baseline h=0 required for all K")
        if bool((head_scores<0).any()) or bool((head_scores>1).any()) or bool((k_probs<0).any()):
            raise ValueError("Expected probabilities")
        if bool((k_probs.sum(-1)-1).abs().max()>1.e-3):
            raise ValueError("Frozen K posteriors must sum to one")
        features=torch.cat([head_scores[:,:,None],audit,head_type],-1)
        heads=self.head_proj(features)
        ctx=self.context_proj(torch.cat([context,k_probs,baseline[:,None]],-1))
        attended,_=self.attention(heads,heads,heads,
                    key_padding_mask=~mask,need_weights=False)
        state=self.fuse(heads+attended+ctx[:,None,:])
        logits=self.k_gate(state).masked_fill(~class_mask,-1.e9)
        by_k=torch.softmax(logits,dim=1)*class_mask
        by_k=by_k/by_k.sum(1,keepdim=True).clamp_min(1.e-9)
        selection=torch.einsum("bhk,bk->bh",by_k,k_probs)
        pooled=torch.einsum("bh,bhd->bd",selection,state)
        offered=(selection*head_scores).sum(-1)
        action_input=torch.cat([pooled,ctx,k_probs,baseline[:,None],
                                offered[:,None]],-1)
        return dict(action_logits=self.action(action_input),
                    class_selection_weights=by_k,selection_weights=selection,
                    head_risk=self.risk(state),offered_probability=offered)


def target_loss(outputs,actions,risk_labels,head_mask,risk_weight=.20):
    """FIT-only single-target supervised action and OOF head-risk loss."""
    if outputs["head_risk"].shape!=risk_labels.shape:
        raise ValueError("risk target shapes mismatch")
    if actions.ndim!=1 or actions.shape[0]!=risk_labels.shape[0]:
        raise ValueError("one action required for each row")
    active=head_mask.float()[:,:,None]
    bce=F.binary_cross_entropy(
        outputs["head_risk"].clamp(1.e-6,1-1.e-6),risk_labels,
        reduction="none")
    return (F.cross_entropy(outputs["action_logits"],actions)+
            risk_weight*(bce*active).sum()/(2*active.sum()).clamp_min(1.))


def propose_single_channel(post,pitches,outputs,target:Target):
    """Shadow-only intervention; other target channels are bit-identical."""
    target_channels={"onset":0,"sustain":1,"release":2}
    if target not in target_channels:
        raise ValueError("Unknown one-head target")
    if post.ndim!=3 or post.shape[1:]!=(49,3) or pitches.shape!=(len(post),):
        raise ValueError("Expected B x 49 x 3 posteriors")
    actions=outputs["action_logits"].argmax(-1)
    offers=outputs["offered_probability"]
    if actions.shape!=(len(post),) or offers.shape!=(len(post),):
        raise ValueError("Specialist output rows mismatch")
    result=post.clone()
    for j in range(len(post)):
        pitch=int(pitches[j])
        if not 40<=pitch<=88:
            raise ValueError("Pitch outside S708's MIDI40..88 range")
        col=target_channels[target]
        if int(actions[j])==ADOPT:
            result[j,pitch-40,col]=offers[j].clamp(0.,1.)
        elif int(actions[j])==SUPPRESS:
            result[j,pitch-40,col]=0.
    return result
