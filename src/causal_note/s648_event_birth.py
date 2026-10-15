"""S648 — event-birth (not pitch-occupancy) head, strictly causal.

The same ORIGINAL S614 audio, probability and selector are the source
of all features. One frozen shadow S614 selector supplies pre-event state;
an entirely separate augmented selector is evaluated, preserving S614.
Ground truth and future samples never enter an inference feature.

Original S80/S614/S632 and S642 checkpoints are not modified.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from causal_note.strict_live import CausalSelection,Selection,HOP,SR
from causal_note.s618_attack_head import PICK

CHANGE_BINS=np.asarray([0,1,8,9,16,17,57,58,59],dtype=int)
HISTORY_COOLDOWN_HOPS=15
MAX_CANDIDATES_PER_FRAME=1


@dataclass(frozen=True)
class BirthPolicy:
    threshold: float=1.0
    min_rise: float=0.0
    def __post_init__(self):
        if not (0. <= self.threshold <= 1.) or not 0. <= self.min_rise <= 1.:
            raise ValueError("Invalid event-birth policy")


def frame_features(x,p,k,end,shadow,previous_x,previous_p):
    """Available features for each candidate pitch strictly at 'end'."""
    if x.shape!=(49,126) or p.shape!=(49,3) or k.shape!=(7,):
        raise ValueError("Incompatible original S614 frame")
    current=np.asarray(x,np.float32)
    prob=np.asarray(p,np.float32)
    past_x=np.asarray(previous_x,np.float32)
    past_p=np.asarray(previous_p,np.float32)
    onset=prob[:,0]
    # Competing pitches are not future observations or oracle labels.
    high=max(float(np.max(onset)),0.)
    neighbor=np.stack((np.roll(onset,1),np.roll(onset,-1)),axis=1)
    age=np.clip((end-shadow.last_on)/(SR*.3),0.,1.).astype(np.float32)
    feature=np.concatenate((
       current[:,PICK], (current-past_x)[:,CHANGE_BINS],
       prob, prob-past_p, neighbor,
       shadow.active[:,None].astype(np.float32),
       shadow.armed[:,None].astype(np.float32),
       age[:,None],
       np.full((49,1),float(shadow.active.sum())/6,np.float32),
       np.full((49,1),float(np.argmax(k))/6,np.float32),
       np.full((49,1),high,np.float32)
    ),axis=-1).astype(np.float32)
    return feature


def causal_features(row):
    """Pre-candidate S614 state sequence; all inputs same-time or past."""
    cfg=Selection(**row["selection"])
    original=CausalSelection(cfg)
    features=[]
    prev_x=np.zeros((49,126),np.float32)
    prev_p=np.zeros((49,3),np.float32)
    for end,x,p,k in zip(row["ends"],row["x"],row["p"],row["k"]):
        features.append(frame_features(x,p,k,int(end),original,prev_x,prev_p))
        original.step(p,k,int(end),int(end))
        prev_x=x
        prev_p=p
    return np.stack(features)


def new_birth_targets(row):
    """TRAIN-only exact 0..2-hop birth labels, 1st physical event per pitch.

    Different from S642's extended occupancy 40ms supervision. Labels
    alone may inspect truth; features NEVER access them.
    """
    labels=np.zeros((len(row["ends"]),49),np.bool_)
    true_starts=[]
    for truth in row["truth"]:
        pitch=int(truth["pitch"])
        if not 40<=pitch<=88:continue
        onset=float(truth["time"])*SR
        ix=np.searchsorted(row["ends"],onset,side="left")
        if ix>=len(labels):continue
        last=min(len(labels),ix+2)
        labels[ix:last,pitch-40]=True
        true_starts.append((ix,pitch-40))
    return labels,true_starts


def sampled_training_events(row,rng):
    feat=causal_features(row)
    truth,starts=new_birth_targets(row)
    t,n=truth.shape
    any_truth=truth.any(axis=1)
    near=np.zeros(t,bool)
    for offset in range(-4,5):
        if offset>=0:near[offset:] |= any_truth[:t-offset]
        else:near[:offset] |= any_truth[-offset:]
    # Hard same-time other notes and same-pitch sustained notes.
    original_active=feat[:,:,len(PICK)+len(CHANGE_BINS)+3+3+2]>.5
    difficult=((near[:,None]|original_active|
               (row["p"][:,:,0]>.3))&~truth)
    all_other=~truth&~difficult
    positive=np.flatnonzero(truth.ravel())
    hard=np.flatnonzero(difficult.ravel())
    background=np.flatnonzero(all_other.ravel())
    if len(positive)>3000:positive=rng.choice(positive,3000,replace=False)
    if len(hard)>5000:hard=rng.choice(hard,5000,replace=False)
    if len(background)>2500:background=rng.choice(background,2500,replace=False)
    selected=np.concatenate([positive,hard,background])
    rng.shuffle(selected)
    f=feat.reshape(-1,feat.shape[-1])[selected]
    y=truth.ravel()[selected].astype(np.int8)
    return f,y,dict(positive=int(len(positive)),hard=int(len(hard)),
       background=int(len(background)),true_note_starts=len(starts),
       features=feat.shape[-1],orig_active_hard=int((original_active&~truth).sum()))


def score_birth_head(model,features):
    flat=features.reshape(-1,features.shape[-1])
    prob=np.empty(len(flat),np.float32)
    for start in range(0,len(flat),8192):
        prob[start:start+8192]=model.predict_proba(flat[start:start+8192])[:,1]
    return prob.reshape(features.shape[:2])


def decode_birth(row,model,policy,return_events=True):
    """Bounded causal rule (one new pitch/hop), original S614 selection intact."""
    from scripts.s613_train_strict_live import decode
    cfg=Selection(**row["selection"])
    if policy.threshold==1.:
        events=decode(row["ends"],row["p"],row["k"],cfg,row["n"])
        return events,dict(new_candidate_frames=0,emitted_candidates=0,
                           candidate_score_max=None)
    scores=score_birth_head(model,causal_features(row))
    selector=CausalSelection(cfg)
    previous_score=np.zeros(49,np.float32)
    output=[]
    candidate_frames=0;actual=0
    for end,p,k,q in zip(row["ends"],row["p"],row["k"],scores):
        amended=p.copy()
        # Score-crossing avoids turning sustained loud q into periodic new
        # events. The ORIGINAL selector gates do not use annotation labels.
        allowed=(~selector.active)&(q>=policy.threshold)&(
            (q-previous_score)>=policy.min_rise)&(
            end-selector.last_on>=HISTORY_COOLDOWN_HOPS*HOP)
        indexes=np.flatnonzero(allowed)
        chosen=-1
        if len(indexes):
            chosen=int(indexes[np.argmax(q[indexes])])
            amended[chosen,0]=cfg.onset
            amended[chosen,1]=max(float(amended[chosen,1]),cfg.active_on)
            candidate_frames+=1
        emitted=selector.step(amended,k,int(end),int(end))
        if chosen>=0 and any(e["type"]=="note_on" and e["pitch"]==chosen+40 for e in emitted):
            actual+=1
        output.extend(emitted)
        previous_score=q
    output.extend(selector.panic(row["n"]))
    return output,dict(new_candidate_frames=candidate_frames,
         emitted_candidates=actual,candidate_score_max=float(np.max(scores)))


def stream_predict_only(features,model):
    """Per-frame model score equals the batch scorer; no lookahead."""
    return model.predict_proba(np.asarray(features,np.float32))[:,1]
