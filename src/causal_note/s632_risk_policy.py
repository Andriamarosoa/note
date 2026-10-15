"""S632: trained counterfactual action COSTS for per-note adaptive latency.

A causal observation at time t predicts three scalar future-aware *costs*:
  EMIT now: wrong note plus missed real note penalties and onset delay
  REJECT now: missed real onset penalty
  WAIT one hop: causal next observation opportunity plus waiting cost

Training targets are computed with full TRAIN annotations/trajectories by
backwards dynamic programming (hindsight supervision). At inference only
current/past NOTE native PCM and S614 heads are allowed. A finite observation
budget, actual emission time, and original NOTE on/off lifecycle remain.
This is a supervised cost-to-go approximation, NOT an RL result.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

from causal_note.strict_live import CausalSelection,HOP,SR,Selection
from causal_note.s631_adaptive_latency import (
    CandidateTracker,DecisionPolicy,FEATURE_DIM
)

FP_COST=1.2
FN_COST=1.35
WAIT_COST_PER_HOP=.020
DELAY_COST_PER_80MS=.10
TOLERANCE=.080


def causal_episodes(row,max_frames=28):
    """Extract candidate timelines without actions, stable across policies."""
    tr=CandidateTracker(DecisionPolicy(max_frames=max_frames))
    episodes=defaultdict(list)
    by_frame=[[] for _ in range(len(row["ends"]))]
    for index,(end,x,p,k) in enumerate(zip(row["ends"],row["x"],row["p"],row["k"])):
        for pitch,age,feature,birth in tr.step(p,x,k):
            record=(int(index),int(pitch),int(age),int(birth),feature)
            episodes[(int(pitch),int(birth))].append(record)
            by_frame[index].append(record)
    return episodes,by_frame


def training_costs(row):
    """Oracle supervision only on TRAIN, never used for live predictions."""
    episodes,_=causal_episodes(row)
    truth=defaultdict(list)
    for i,n in enumerate(row["truth"]):
        note_idx=int(n["pitch"])-40
        if 0<=note_idx<49:
            truth[note_idx].append((i,float(n["time"])))
    claimed=set()
    feature_rows=[];target_rows=[]
    proof=dict(episodes=len(episodes),positive_episodes=0,negative_episodes=0,
               Q_rows=0,valid_emission_states=0,oracle_wait_rows=0)
    for (pitch,birth),seq in sorted(episodes.items(),key=lambda e:(e[0][1],e[0][0])):
        birth_t=float(row["ends"][birth])/SR
        possibilities=[(abs(t-birth_t),ref,t) for ref,t in truth[pitch]
                       if -.110<=t-birth_t<=.160 and (pitch,ref) not in claimed]
        real=bool(possibilities)
        attack_t=float("inf")
        if real:
            _,ref,attack_t=min(possibilities)
            claimed.add((pitch,ref))
            proof["positive_episodes"]+=1
        else:
            proof["negative_episodes"]+=1
        n=len(seq)
        immediate=np.zeros((n,2),np.float32)
        for j,(index,_,age,_,feature) in enumerate(seq):
            actual=float(row["ends"][index])/SR
            valid=real and abs(actual-attack_t)<=TOLERANCE
            if valid:proof["valid_emission_states"]+=1
            if valid:
                immediate[j,0]=DELAY_COST_PER_80MS*max(0.,actual-attack_t)/TOLERANCE
            else:
                immediate[j,0]=FP_COST+(FN_COST if real else 0)
            immediate[j,1]=FN_COST if real else 0
        full=np.zeros((n,3),np.float32)
        for j in range(n-1,-1,-1):
            full[j,0:2]=immediate[j]
            if j==n-1:
                full[j,2]=max(FP_COST,FN_COST)*3.
            else:
                frames=seq[j+1][2]-seq[j][2]
                full[j,2]=WAIT_COST_PER_HOP*frames+float(full[j+1].min())
            if np.argmin(full[j])==2:proof["oracle_wait_rows"]+=1
        # All early states plus spaced states later; avoid high negative-frame
        # duplication and retain long waits for meaningful stop learning.
        use=np.array([r[2]<=9 or r[2] in (12,16,20,24,27) for r in seq])
        for selected,values,q in zip(use,seq,full):
            if selected:
                feature_rows.append(values[4])
                target_rows.append(q)
                proof["Q_rows"]+=1
    if not feature_rows:
        return np.zeros((0,FEATURE_DIM),np.float32),np.zeros((0,3),np.float32),proof
    return np.stack(feature_rows).astype(np.float32),np.stack(target_rows).astype(np.float32),proof


@dataclass(frozen=True)
class RiskPolicy:
    mode:str="assist"
    extra_wait_cost:float=0.
    emit_margin:float=0.
    max_emit_cost:float=.7
    max_frames:int=28

    def __post_init__(self):
        if self.mode not in ("assist","replace"):raise ValueError("Bad mode")
        if not (-.3<=self.extra_wait_cost<=.4):raise ValueError("Bad wait bias")
        if not (0<=self.emit_margin<=.5):raise ValueError("Bad emit margin")
        if not (0<self.max_emit_cost<=2.5):raise ValueError("Bad risk ceiling")
        if not 1<=self.max_frames<=28:raise ValueError("Bad horizon")


def policies():
    return [
        ("S614_original",None),
        ("S632_assist_c045",RiskPolicy(max_emit_cost=.45)),
        ("S632_assist_c065",RiskPolicy(max_emit_cost=.65)),
        ("S632_assist_c085",RiskPolicy(max_emit_cost=.85)),
        ("S632_assist_wait_m010",RiskPolicy(extra_wait_cost=-.10,max_emit_cost=.65)),
        ("S632_assist_wait_p010",RiskPolicy(extra_wait_cost=.10,max_emit_cost=.65)),
        ("S632_assist_margin015",RiskPolicy(emit_margin=.15,max_emit_cost=.65)),
        ("S632_replace_c045",RiskPolicy(mode="replace",max_emit_cost=.45)),
        ("S632_replace_c065",RiskPolicy(mode="replace",max_emit_cost=.65)),
        ("S632_replace_wait_m010",RiskPolicy(mode="replace",extra_wait_cost=-.10,max_emit_cost=.65)),
    ]


def predicted_costs(models,by_frame):
    flat=[r for frame in by_frame for r in frame]
    if not flat:
        return [[] for _ in by_frame]
    input_x=np.stack([r[4] for r in flat]).astype(np.float32)
    result=np.column_stack([np.concatenate(
        [model.predict(input_x[start:start+4096])
         for start in range(0,len(input_x),4096)])
         for model in models]).astype(np.float32)
    result=np.clip(result,0.,10.)
    out=[];offset=0
    for frame in by_frame:
        out.append([(r[1],r[2],r[3],q) for r,q in
                    zip(frame,result[offset:offset+len(frame)])])
        offset+=len(frame)
    return out


def apply_policy(row,cost_frames,selection,policy):
    if policy is None:
        from scripts.s613_train_strict_live import decode
        return decode(row["ends"],row["p"],row["k"],selection,row["n"]),dict(
            waits=0,emits=0,rejects=0,delays_ms=[])
    state=CausalSelection(selection)
    events=[];decided=set();count=defaultdict(int)
    delays=[]
    for end,p,k,observations in zip(row["ends"],row["p"],row["k"],cost_frames):
        native=np.asarray(p,np.float32)
        changed=native.copy()
        if policy.mode=="replace":
            changed[~state.active,0]=0.
        decisions=[]
        for pitch,age,birth,costs in observations:
            key=(pitch,birth)
            if key in decided:continue
            if state.active[pitch]:
                decided.add(key)
                continue
            e,r,w=(float(v) for v in costs)
            w+=policy.extra_wait_cost
            if age>=policy.max_frames:
                w=float("inf")
            if e+policy.emit_margin<min(r,w) and e<=policy.max_emit_cost:
                # Rescues require acoustic corroboration to prevent a trained
                # cost approximation from manufacturing notes out of silence.
                if native[pitch,0]>=.11 and native[pitch,1]>=.19:
                    changed[pitch,0]=max(changed[pitch,0],selection.onset)
                    changed[pitch,1]=max(changed[pitch,1],selection.active_on)
                    decisions.append((key,"emit",age,pitch))
                else:count["waits"]+=1
            elif r<w and r<e:
                decided.add(key);count["rejects"]+=1
            else:
                count["waits"]+=1
        generated=state.step(changed,k,int(end),int(end))
        on={int(x["pitch"])-40 for x in generated if x["type"]=="note_on"}
        for key,action,age,pitch in decisions:
            if pitch in on:
                decided.add(key)
                count["emits"]+=1
                delays.append(float(age*HOP/SR*1000))
        # If the original native S614 issued a note, close corresponding
        # candidate episodes to avoid repeated rescue of same acoustic event.
        for pitch,age,birth,costs in observations:
            if pitch in on:decided.add((pitch,birth))
        events.extend(generated)
    events.extend(state.panic(row["n"]))
    return events,dict(waits=count["waits"],emits=count["emits"],
                       rejects=count["rejects"],delays_ms=delays)
