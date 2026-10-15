"""S627: jointly learned new-note attack and NEW-onset Exact-K-style head.

The new K denotes newly attacked pitches within the current short causal
window, NOT the number of sustained notes (S614 active-K), and is NOT the
historical S80 classifier. Shared pitch encoder -> both K and onset losses.
Uses ONLY the original NOTE right-aligned PCM frontend and S614 probabilities.
Training uses labels, inference never uses labels or a post-attack wait.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import numpy as np
from causal_note.s618_attack_head import PICK

PITCH_COUNT=49
INPUT_DIM=len(PICK)+6
LATENT=40
SCHEMA="S627_joint_causal_new_pitch_eventK_v1"


def frames(x,p):
    """T x 49 x 126, T x 49 x 3 -> T x 49 x (66+6). No future."""
    x=np.asarray(x,dtype=np.float32)
    p=np.asarray(p,dtype=np.float32)
    if x.ndim!=3 or x.shape[1:]!=(49,126) or p.shape!=(*x.shape[:2],3):
        raise ValueError("Original right-aligned NOTE acoustic/head ABI required")
    prev=np.concatenate((np.zeros_like(p[:1]),p[:-1]),axis=0)
    return np.concatenate((x[:,:,PICK],p,p-prev),axis=-1).astype(np.float32)


def targets(ends,notes,sr=44100,window_ms=40):
    """Training-only event K and pitch labels; one new MIDI onset counts once."""
    ends=np.asarray(ends)
    label=np.zeros((len(ends),49),np.float32)
    # Positive within [attack,+40 ms]. Event count equals the sum of *unique*
    # simultaneous note-starts, not the number of currently sounding pitches.
    for note in notes:
        pitch=int(note["pitch"])-40
        if not 0<=pitch<49:continue
        i=np.searchsorted(ends,round(float(note["time"])*sr),side="left")
        j=np.searchsorted(ends,round((float(note["time"])+window_ms/1000.)*sr),side="right")
        if j>i:label[i:j,pitch]=1.
    count=np.minimum(6,label.sum(axis=1)).astype(np.int64)
    return label,count


@dataclass(frozen=True)
class Fusion:
    pitch_weight:float=0.
    support:float=0.
    count_gate:float=0.

    def __post_init__(self):
        if any(not np.isfinite(v) or not 0.<=v<=1. for v in
               (self.pitch_weight,self.support,self.count_gate)):
            raise ValueError("Fusion weights must be in [0,1]")


def fuse(p,q,k,policy):
    """Optional causal fusion. No hard tuning from historical TEST."""
    p=np.asarray(p,dtype=np.float32)
    q=np.asarray(q,dtype=np.float32)
    k=np.asarray(k,dtype=np.float32)
    if p.shape!=(49,3) or q.shape!=(49,) or k.shape!=(7,):
        raise ValueError("Native 49-pitch/7-K dimensions required")
    if not (np.isfinite(p).all() and np.isfinite(q).all() and np.isfinite(k).all()):
        raise ValueError("Nonfinite model output")
    corrected=p.copy()
    new_onset=(1-policy.pitch_weight)*p[:,0]+policy.pitch_weight*q
    # Do not replace sustain unless BOTH onset heads corroborate the event;
    # a high attack score alone is not a new musical pitch.
    corrected[:,0]=new_onset
    agreement=np.sqrt(np.maximum(0,p[:,0]*q))
    corrected[:,1]=np.maximum(p[:,1],policy.support*agreement)
    if policy.count_gate:
        # Event K is NOT active-note K. This is a soft frame-level gate
        # rather than capping already-active notes incorrectly.
        event_likelihood=float(np.clip(1-k[0],0,1))
        corrected[:,0]*=(1-policy.count_gate*(1-event_likelihood))
    return np.clip(corrected,0,1)


class JointNPZ:
    """Minimal numpy inference; all parameters learned on train, no Torch LIVE."""
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as z:
            self.arrays={key:z[key].copy() for key in z.files}
        if str(self.arrays.pop("schema"))!=SCHEMA:
            raise ValueError("Wrong joint NOTE model schema")
        self.metadata=json.loads(str(self.arrays.pop("metadata")))
        self.prev_p=np.zeros((49,3),np.float32)

    @staticmethod
    def _relu(x):
        return np.maximum(x,0)

    @staticmethod
    def _softmax(x):
        y=np.exp(x-x.max(-1,keepdims=True))
        return y/y.sum(-1,keepdims=True)

    def frame(self,x,p):
        """Consumes exactly one current acoustic frame and probability frame."""
        x=np.asarray(x,np.float32);p=np.asarray(p,np.float32)
        if x.shape!=(49,126) or p.shape!=(49,3):
            raise ValueError("Must receive the current causal NOTE frame")
        f=np.concatenate((x[:,PICK],p,p-self.prev_p),axis=-1).astype(np.float32)
        self.prev_p=p.copy()
        a=self.arrays
        z=np.clip((f-a["mean"])/a["scale"],-16,16)
        h=self._relu(z@a["encoder_w0"]+a["encoder_b0"])
        h=self._relu(h@a["encoder_w1"]+a["encoder_b1"])
        pool=np.concatenate((h.mean(0),h.max(0)))
        kl=pool@a["k_w"]+a["k_b"]
        pk=self._softmax(kl)
        joint=np.concatenate((h,np.broadcast_to(pk[:,None].T,(49,7))),axis=1)
        po=joint@a["pitch_w"]+a["pitch_b"]
        q=1./(1.+np.exp(-np.clip(po,-30,30)))
        return q.ravel().astype(np.float32),pk.astype(np.float32)


def variants():
    return [
      ("S614_original",Fusion()),
      ("joint_pitch_020",Fusion(.2,0,0)),
      ("joint_pitch_040",Fusion(.4,0,0)),
      ("joint_pitch_060",Fusion(.6,0,0)),
      ("joint_pitch_080",Fusion(.8,0,0)),
      ("joint_040_support025",Fusion(.4,.25,0)),
      ("joint_060_support025",Fusion(.6,.25,0)),
      ("joint_060_support050",Fusion(.6,.5,0)),
      ("joint_040_count025",Fusion(.4,0,.25)),
      ("joint_060_count025",Fusion(.6,0,.25)),
      ("joint_040_support025_count025",Fusion(.4,.25,.25)),
      ("joint_060_support025_count025",Fusion(.6,.25,.25)),
    ]
