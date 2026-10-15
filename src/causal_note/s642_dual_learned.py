"""S642: two genuine TRAIN-only learned heads for NOTE weak attacks + delay.

This research-only decoder does not change S614/S80/S632 weights. Both
inference heads see only current and past PCM-derived S614 features.
Ground-truth note time is consulted ONLY when constructing TRAIN labels.

Heads:
  candidate: binary physical faint onset classification (S626-compatible ABI)
  timing: classified 0/1/2/3 HOPs to hold each emitted NoteOn; the matching
          NoteOff is delayed by the SAME number of hops (duration preserved).

A delayed event retains its original causal decision_sample and reports the
real available_sample at playback; it is never retroactively time-stamped.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import numpy as np

from causal_note.strict_live import CausalSelection,Selection,SR,HOP
from causal_note.s618_attack_head import input_features,candidate_mask,DIM

MAX_DELAY_HOPS=3
HOP_SECONDS=HOP/SR


@dataclass(frozen=True)
class S642Policy:
    use_candidate: bool=False
    use_delay: bool=False
    rescue_min_confidence: float=.965

    def __post_init__(self):
        if not 0.<=self.rescue_min_confidence<=1.:
            raise ValueError("Invalid rescue confidence")


def event_embedding(features,probabilities,pitch_index,frame_index):
    """Single right-aligned current+past feature row, no oracle."""
    track=input_features(features[frame_index:frame_index+1],
                         probabilities[frame_index:frame_index+1])
    return track[0,pitch_index]


def training_delay_labels(row):
    """Train-only oracle. Only hold if actual onset begins 1..3 hops later.

    We allow only a minimal nonnegative hold to recover otherwise premature
    true attacks. False events remain immediate (they cannot become genuine
    merely by waiting arbitrary time).
    """
    from scripts.s613_train_strict_live import decode
    heads=Selection(**row["selection"])
    original=decode(row["ends"],row["p"],row["k"],heads,row["n"])
    prediction=[e for e in original if e["type"]=="note_on"]
    targets=defaultdict(list)
    for i,t in enumerate(row["truth"]):
        if 40<=t["pitch"]<=88:
            targets[int(t["pitch"])].append((float(t["time"]),i))
    used=set()
    x=input_features(row["x"],row["p"])
    feats=[]; labels=[]
    for event in prediction:
        pitch=int(event["pitch"]); current=float(event["time"])
        eligible=[(t-current,idx) for t,idx in targets.get(pitch,[])
                  if -0.080<=t-current<=MAX_DELAY_HOPS*HOP_SECONDS
                  and (pitch,idx) not in used]
        shift=0
        if eligible:
            delta,idx=min(eligible,key=lambda it:abs(it[0]))
            used.add((pitch,idx))
            if delta>0:
                shift=int(np.clip(np.ceil((delta-1e-10)/HOP_SECONDS),
                                  1,MAX_DELAY_HOPS))
        frame=np.searchsorted(row["ends"],event["decision_sample"],side="left")
        frame=min(frame,len(row["ends"])-1)
        feats.append(x[frame,pitch-40])
        labels.append(shift)
    return (np.asarray(feats,np.float32).reshape(-1,DIM),
            np.asarray(labels,np.int8),len(prediction))


def candidate_scores(row,model):
    features=input_features(row["x"],row["p"])
    flat=features.reshape(-1,DIM)
    # Replaying batched predictions is valid: every feature row is strictly
    # causal and independently predicted, with no future-dependent smoothing.
    p=np.concatenate([model.predict_proba(flat[i:i+4096])[:,1]
                      for i in range(0,len(flat),4096)])
    return p.reshape(len(row["ends"]),49).astype(np.float32)


def decode_acoustic(row,model,policy:S642Policy):
    from scripts.s613_train_strict_live import decode
    cfg=Selection(**row["selection"])
    if not policy.use_candidate:
        return decode(row["ends"],row["p"],row["k"],cfg,row["n"]),0
    confidence=candidate_scores(row,model)
    selector=CausalSelection(cfg)
    events=[]
    attempted=0
    for end,x,p,k,q in zip(row["ends"],row["x"],row["p"],row["k"],confidence):
        edited=p.copy()
        eligible=(~selector.active)&candidate_mask(x[None],p[None])[0]
        candidate=eligible & (p[:,0]<cfg.onset) & (p[:,0]>=.13) & (
            p[:,1]>=.26) & (q>=policy.rescue_min_confidence)
        # Prevent making unbounded new-note bursts at a single frame.
        positions=np.flatnonzero(candidate)
        if len(positions):
            best=positions[np.argmax(q[positions])]
            edited[best,0]=cfg.onset
            edited[best,1]=max(float(edited[best,1]),cfg.active_on)
            attempted+=1
        events.extend(selector.step(edited,k,int(end),int(end)))
    events.extend(selector.panic(row["n"]))
    return events,attempted


def apply_learned_delay(events,row,model,policy:S642Policy):
    """Schedule paired NOTE events; for mode disabled return byte-for-byte list.

    Events are complete offline in this evaluator, but the delay decision
    uses only the feature row at the event's original decision_sample.
    Per-pitch pairing and hold queues can be implemented with bounded
    3-HOP memory at actual LIVE deployment.
    """
    if not policy.use_delay:
        return list(events),dict(held_noteons=0,delays=[0])
    feature=input_features(row["x"],row["p"])
    on_positions=[i for i,e in enumerate(events) if e["type"]=="note_on"]
    emb=[]
    for event_index in on_positions:
        e=events[event_index]
        frame=min(len(row["ends"])-1,
                  int(np.searchsorted(row["ends"],e["decision_sample"],side="left")))
        emb.append(feature[frame,int(e["pitch"])-40])
    if not emb:
        return list(events),dict(held_noteons=0,delays=[])
    delays=np.asarray(model.predict(np.stack(emb)),np.int64)
    if not np.all((delays>=0)&(delays<=MAX_DELAY_HOPS)):
        raise ValueError("Trained latency policy outside fixed 0..3 hops")
    shifts={}
    active={}
    for event_index,shift in zip(on_positions,delays):
        shifts[event_index]=int(shift)*HOP
    for index,e in enumerate(events):
        pitch=int(e["pitch"])
        if e["type"]=="note_on":
            active[pitch]=shifts[index]
        elif e["type"]=="note_off":
            shifts[index]=active.pop(pitch,0)
    output=[]
    for i,e in enumerate(events):
        item=dict(e)
        new_time=int(e["available_sample"])+shifts.get(i,0)
        item["available_sample"]=new_time
        item["time"]=new_time/SR
        output.append(item)
    # Time order for actual transmitted MIDI, NoteOff before NoteOn on
    # coincident re-attacks; index order otherwise stable.
    output.sort(key=lambda e:(e["available_sample"],
                  0 if e["type"]=="note_off" else 1))
    return output,dict(held_noteons=int((delays>0).sum()),
                       delays=delays.tolist())


def evaluated_policy(row,candidate_head,delay_head,policy:S642Policy):
    decoded,attempted=decode_acoustic(row,candidate_head,policy)
    output,stats=apply_learned_delay(decoded,row,delay_head,policy)
    stats["rescued_frame_proposals"]=attempted
    return output,stats


def fixed_policies():
    return [
        ("S614_original",S642Policy()),
        ("S642_candidate_only",S642Policy(use_candidate=True)),
        ("S642_delay_only",S642Policy(use_delay=True)),
        ("S642_joint",S642Policy(use_candidate=True,use_delay=True))
    ]
