"""S762: combine ALL four S755 experts at each received causal fragment.

S761 independently evaluated one fixed K specialist for the entire song,
which cannot realize the user's requested per-fragment architecture.
This experiment pools per-fragment expert proposals and selects the best
independently FIT80-trained attack/complete score for that (sample,pitch).
Original S708 notes and the S707 release policy are never replaced.
Candidate future NoteOff metadata is carried ONLY for closure, never read
to decide the proposal's score, source, or rank.

Important: S762 is exploratory. Its heuristic scores are not calibrated
per-event probabilities, CAL32 has been reused, and no deployment/promotion
is permissible without a clean split, all-six >80 and event safety.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np

from causal_note.s746_instance_guard import InstanceGate,merge_events
from scripts.s761_fit_dualquality_birth_routing import (
    FAMILIES,BLEND,CONFIRM_BLOCKS,candidate_quality_features,
    positive_probability)
from scripts.s755_pitch_coherent_k_fit_cal32 import coherent_k_features,calibrated_k
from scripts.s760_causal_confirmation_delay import select_novel_births
from scripts.s731_auditory_scene_calibration import emit
from scripts.s761_fit_dualquality_birth_routing import evaluate as shared_evaluate

QUALITY=("min","geometric","complete")
THRESHOLDS=(.50,.65,.80,.90)
GRID={f"fragment_{quality}_risk{int(risk*100)}":(0,risk)
      for quality,risk in product(QUALITY,THRESHOLDS)}
assert len(GRID)==12

def quality_score(on,complete,mode):
    """Only a causal ranking; no false probability calibration claim."""
    a=np.asarray(on,dtype=np.float32)
    b=np.asarray(complete,dtype=np.float32)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() \
       or np.any(a<0) or np.any(a>1) or np.any(b<0) or np.any(b>1):
        raise ValueError("S762 invalid causal quality scores")
    if mode=="min":
        return np.minimum(a,b)
    if mode=="geometric":
        return np.sqrt(a*b)
    if mode=="complete":
        return b.copy()
    raise ValueError("S762 invalid score mode")

def route_fragment_proposals(families,native_k,mode):
    """Dynamically choose among experts at EACH emitted (sample,pitch).

    Families entries contain (pairs,indices,on_prob,full_prob,fused_k).
    No candidate NoteOff field is accessed by this function. The selected
    pair is carried opaquely into S746, which later closes/preempts it.
    Ties are resolved by the input expert index, never by future duration.
    """
    k=np.asarray(native_k,np.float32)
    if k.ndim!=2 or k.shape[1]!=7 or not np.isfinite(k).all():
        raise ValueError("S762 native K ABI")
    winners={}
    proposed=0
    for expert,(pairs,indices,on_probs,full_probs,fused_k) in enumerate(families):
        f=np.asarray(fused_k,np.float32)
        if f.shape!=k.shape or not np.isfinite(f).all():
            raise ValueError("S762 expert K ABI")
        if len(pairs)!=len(indices):
            raise ValueError("S762 candidate index ABI")
        q=quality_score(on_probs,full_probs,mode)
        if len(q)!=len(pairs):
            raise ValueError("S762 quality/candidate ABI")
        for (on,off),(frame,ix),prob in zip(pairs,indices,q):
            proposed+=1
            frame=int(frame);ix=int(ix)
            if not (0<=frame<len(k) and 0<=ix<49 and
                    int(on["pitch"])==ix+40 and
                    int(on["decision_sample"])==int(on["available_sample"])):
                raise AssertionError("S762 candidate is not causally aligned")
            key=(int(on["available_sample"]),int(on["pitch"]))
            row=(float(prob),-expert,(on,off),(frame,ix),f[frame])
            if key not in winners or row[:2]>winners[key][:2]:
                winners[key]=row
    selected=sorted(winners.values(),
                    key=lambda row:(int(row[2][0]["available_sample"]),
                                    -row[0],int(row[2][0]["pitch"])))
    # For a frame with several selected pitches, adopt the K posterior
    # of the highest-ranking expert at that frame. No future is consulted.
    combined=k.copy()
    seen=set()
    for row in selected:
        frame=row[3][0]
        if frame not in seen:
            combined[frame]=row[4]
            seen.add(frame)
    pairs=[row[2] for row in selected]
    probs=np.array([row[0] for row in selected],np.float32)
    indices=[row[3] for row in selected]
    return pairs,probs,indices,combined,dict(
        proposals=proposed,unique=len(selected),deduplicated=proposed-len(selected),
        fragments=len(seen))

def score_variants(source,post,k,item,experts,heads,policy,offset,release):
    x=coherent_k_features(item["x"],post,k,item["native_post"])
    proposals=[]
    diag=Counter()
    for expert,family in enumerate(FAMILIES):
        posterior=experts[family].predict_proba(x).astype(np.float32)
        fused=calibrated_k(k,posterior,BLEND)
        trial=emit(item,post,fused,policy,offset,release)
        pairs,ranking,indices,census=select_novel_births(
            source,trial,item["decision_ends"],post,k,fused,
            confirmation_blocks=CONFIRM_BLOCKS)
        quality_x=candidate_quality_features(
            item,post,k,fused,pairs,ranking,indices,expert)
        prob_on=positive_probability(heads["onset"],quality_x)
        prob_complete=positive_probability(heads["complete"],quality_x)
        proposals.append((pairs,indices,prob_on,prob_complete,fused))
        diag.update(census)
    streams={"S708_frozen":source};details={}
    routed={mode:route_fragment_proposals(proposals,k,mode) for mode in QUALITY}
    for mode in QUALITY:
        pairs,probabilities,indices,fused,census=routed[mode]
        for risk in THRESHOLDS:
            key=f"fragment_{mode}_risk{int(risk*100)}"
            merged,info=merge_events(
                source,pairs,probabilities,indices,fused,
                InstanceGate(risk=risk,k_tail=.10))
            streams[key]=merged
            details[key]=dict(candidate=census,gate=info,source_candidate=diag)
    if set(streams)!=set(("S708_frozen",*GRID)):
        raise AssertionError("S762 must emit all 12 complete MIDI streams")
    return streams,details

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--s708-dir",required=True,type=Path)
    p.add_argument("--offset-checkpoint",required=True,type=Path)
    p.add_argument("--specialist-dir",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    shared_evaluate(
        p.parse_args(),score_variants_impl=score_variants,grid=GRID,
        schema="S762_DYNAMIC_FRAGMENT_FOUR_EXPERT_FIT80_CAL32")

if __name__=="__main__":
    main()
