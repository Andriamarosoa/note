"""S654 — learned risk selector for S648 birth proposals, unchanged NOTE S614.

A second, specialist head chooses ACCEPT or ABSTAIN before S614's
causal selection. Trained from OFF-TRAIN S648 predictions on 11
original S648 CALIBRATION sources, divided again by composition into
risk-fit and risk-lock subsets.

This is *experimental* and never replaces the frozen S614 reference.
Truth is consulted to supervise training / audit only, not inference.
"""
from __future__ import annotations
import numpy as np
from causal_note.strict_live import Selection,CausalSelection,SR,HOP
from causal_note.s648_event_birth import (
    BirthPolicy,causal_features,score_birth_head,HISTORY_COOLDOWN_HOPS
)
from scripts.s613_train_strict_live import decode,event_score

BASE_BIRTH=BirthPolicy(.70,.10)
RISK_THRESHOLDS=(.35,.55,.75,.90)


def risk_vector(features,p,k,q,previous_q,pitch):
    """Original causal acoustic+S614 state plus pitch-specific K/harmonic risk."""
    idx=int(pitch)-40
    if not 0<=idx<49:raise ValueError("Pitch out of NOTE range")
    harmonics=[idx+d for d in (-24,-19,-12,12,19,24) if 0<=idx+d<49]
    harmonic_on=max((float(p[j,0]) for j in harmonics),default=0.)
    harmonic_sus=max((float(p[j,1]) for j in harmonics),default=0.)
    other=np.delete(q,idx)
    extra=np.asarray((
       float(q[idx]),float(q[idx]-previous_q[idx]),
       float(np.max(other)),float(np.max(other)-q[idx]),
       harmonic_on,harmonic_sus,
       float(k.argmax())/6.,float(pitch-40)/48.
    ),np.float32)
    return np.concatenate((np.asarray(features[idx],np.float32),extra))


def cached_inputs(row,birth_head):
    feats=causal_features(row)
    q=score_birth_head(birth_head,feats)
    return feats,q


def risk_training_examples(row,birth_head):
    """Feature + true incremental benefit for each S648 emitted candidate.

    Only an event newly emitted by S648 vs frozen S614 is trainable.
    Label=1 iff this *specific* new MIDI event matches a previously
    missed original reference note in official 1:1 onset matching.
    """
    from causal_note.s648_event_birth import decode_birth
    original,_=decode_birth(row,birth_head,BirthPolicy())
    proposed,_=decode_birth(row,birth_head,BASE_BIRTH)
    original_on=[e for e in original if e["type"]=="note_on"]
    extra_on=[e for e in proposed if e["type"]=="note_on"]
    baseline,_bpairs=event_score(original,row["truth"])
    _score,matches=event_score(proposed,row["truth"])
    originally_correct={int(j) for i,j in _bpairs}
    proposed_refs={int(i):int(j) for i,j in matches}
    original_keys={(int(e["decision_sample"]),int(e["pitch"])) for e in original_on}
    feats,scores=cached_inputs(row,birth_head)
    collected=[];labels=[];audit=[]
    for i,e in enumerate(extra_on):
        key=(int(e["decision_sample"]),int(e["pitch"]))
        if key in original_keys:continue
        frame=int(np.searchsorted(row["ends"],key[0],side="left"))
        if frame>=len(feats):raise ValueError("Event outside causal source")
        prev=scores[frame-1] if frame else np.zeros(49,np.float32)
        emb=risk_vector(feats[frame],row["p"][frame],row["k"][frame],
                        scores[frame],prev,key[1])
        reference=proposed_refs.get(i)
        good=int(reference is not None and reference not in originally_correct)
        collected.append(emb);labels.append(good)
        audit.append(dict(decision_sample=key[0],pitch=key[1],
                          true_added_event=bool(good),
                          matched_reference_index=reference))
    if not collected: return np.empty((0,feats.shape[-1]+8),np.float32),np.empty(0,np.int8),audit
    return np.stack(collected),np.asarray(labels,np.int8),audit


def decode_with_specialist(row,birth_head,risk_head,threshold):
    """Strict causal, max ONE S648 new candidate per frame, no audio future.

    The independent S614 original selector's model and thresholds are
    unchanged; skipped risky candidates never reach the augmented
    selector. Existing S614 probabilities always pass unchanged.
    """
    if not 0<=threshold<=1:raise ValueError("Invalid specialist risk floor")
    cfg=Selection(**row["selection"])
    feat,q_scores=cached_inputs(row,birth_head)
    chosen_selector=CausalSelection(cfg)
    previous_q=np.zeros(49,np.float32)
    events=[];proposed=0;blocked=0;allowed_count=0;actual=0
    for frame,(end,p,k,q) in enumerate(zip(row["ends"],row["p"],row["k"],q_scores)):
        proposal=(~chosen_selector.active)&(q>=BASE_BIRTH.threshold)&(
            q-previous_q>=BASE_BIRTH.min_rise)&(
            end-chosen_selector.last_on>=HISTORY_COOLDOWN_HOPS*HOP)
        indices=np.flatnonzero(proposal)
        index=-1
        amended=p.copy()
        if len(indices):
            # Exactly the original S648 priority: test the highest-q
            # candidate, never silently substitute a different candidate.
            index=int(indices[np.argmax(q[indices])])
            proposed+=1
            vec=risk_vector(feat[frame],p,k,q,previous_q,index+40)
            prob=float(risk_head.predict_proba(vec[None,:])[0,1])
            if prob>=threshold:
                allowed_count+=1
                amended[index,0]=cfg.onset
                amended[index,1]=max(float(amended[index,1]),cfg.active_on)
            else:
                blocked+=1
                index=-1
        out=chosen_selector.step(amended,k,int(end),int(end))
        if index>=0 and any(e["type"]=="note_on" and int(e["pitch"])==index+40 for e in out):
            actual+=1
        events.extend(out)
        previous_q=q
    events.extend(chosen_selector.panic(row["n"]))
    return events,dict(proposed_frames=proposed,risk_rejected=blocked,
       risk_accepted_frames=allowed_count,actual_new_notes=actual)


def streaming_specialist_decision(risk_head,feature,p,k,q,previous_q,pitch,threshold):
    """Same 1-event decision usable inside S649._frame; no truth / lookahead."""
    v=risk_vector(feature,p,k,q,previous_q,pitch)
    return float(risk_head.predict_proba(v[None,:])[0,1])>=threshold
