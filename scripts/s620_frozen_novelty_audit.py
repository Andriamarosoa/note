"""S620: reuse frozen S618 native evidence but require a NEW causal transition.

S618's high attack scores plateau during sustained notes and produced
1209 false positives when used naively. Test whether past-only probability
transition + physical rise suppresses those false positives. All conditions
are fixed before opening the five URMP development scores. This is NOT the
original S614 sealed test and NOT a promotion.
"""
from __future__ import annotations
from collections import Counter,deque
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import time

import joblib
import numpy as np
from causal_note.strict_live import Selection,CausalSelection,load_heads
from causal_note.s618_attack_head import input_features,DIM
from scripts.s616_live_dev_ablation import ROOT,MODEL,BASELINE_URMP,prepare_dev_source
from scripts.s613_prepare_strict_live import digest
from scripts.s613_train_strict_live import decode,event_score,summary,offset_audit


EXPECTED_S618="3a57baff6ab26ba57fe34dc9c521bf2bbdbabaf1688ce5740d3cb1c7a67c63e1"
POLICIES=(
    ("base",2.,2.,2.),
    ("novelty_080_015_008",.80,.15,.08),
    ("novelty_080_015_018",.80,.15,.18),
    ("novelty_080_025_008",.80,.25,.08),
    ("novelty_080_025_018",.80,.25,.18),
    ("novelty_090_015_008",.90,.15,.08),
    ("novelty_090_015_018",.90,.15,.18),
    ("novelty_090_025_008",.90,.25,.08),
    ("novelty_090_025_018",.90,.25,.18),
)


def decode_novelty(ends,x,p,k,config,clip_samples,q,confidence,jump,physical):
    engine=CausalSelection(config)
    events=[];activated=0
    history=deque([np.zeros(49,np.float32) for _ in range(5)],maxlen=5)
    prev_p=np.zeros(49,np.float32)
    for end,f,original,count,trained in zip(ends,x,p,k,q):
        old=history[0]
        rise=np.maximum(f[:,[57,58,59]],0).max(axis=-1)
        eligible=(trained>=confidence) & ((trained-old)>=jump) & (rise>=physical) & (
            original[:,0]>=.15) & ((original[:,0]-prev_p)>=.02) & (
            original[:,1]>=.12) & (~engine.active) & engine.armed
        edited=original.copy()
        edited[eligible,0]=np.maximum(edited[eligible,0],config.onset)
        edited[eligible,1]=np.maximum(edited[eligible,1],config.active_on)
        activated+=int(eligible.sum())
        events.extend(engine.step(edited,count,int(end),int(end)))
        history.append(trained.copy())
        prev_p=original[:,0].copy()
    events.extend(engine.panic(int(clip_samples)))
    return events,activated


def main():
    ar=argparse.ArgumentParser()
    ar.add_argument("--model",type=Path,required=True)
    ar.add_argument("--out",type=Path,required=True)
    args=ar.parse_args()
    if args.out.exists():raise FileExistsError("No overwrite of evaluation evidence")
    args.out.mkdir(parents=True)
    assert digest(args.model.read_bytes())==EXPECTED_S618
    fitted=joblib.load(args.model)
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest()==frozen["model_sha256"]
    protocol=json.loads((ROOT/"protocol.json").read_text())
    sources=[s for s in protocol["sources"] if s["split"]=="dev" and s["corpus"]=="URMP"]
    assert len(sources)==5 and len({s["composition"] for s in sources})==5
    native=load_heads(MODEL,expected_sha256=frozen["model_sha256"])
    selection=Selection(**native.metadata["selection"])
    rows={name:[] for name,*_ in POLICIES}
    clock=time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        cache=Path(tmp);(cache/"data").mkdir()
        for spec in sources:
            pack=prepare_dev_source(spec,cache)
            ends,x=pack["ends"],pack["features"]
            p,k=native.predict(x)
            q=fitted.predict_proba(input_features(x,p).reshape(-1,DIM))[:,1].reshape(len(ends),49)
            original=decode(ends,p,k,selection,pack["samples"])
            for name,conf,jump,physical in POLICIES:
                if name=="base":
                    events=original;activated=0
                else:
                    events,activated=decode_novelty(ends,x,p,k,selection,pack["samples"],q,conf,jump,physical)
                score,pairs=event_score(events,pack["truth"])
                offset=offset_audit(events,pack["truth"],pairs,pack["samples"])
                rows[name].append(dict(source=spec["name"],score=score,
                    triggered=activated,offset_correct=offset["offsets_within_max_50ms_or_20pct_duration"],
                    offset_eligible=offset["matched_noteons_with_observed_reference_offset"]))
            print("S620 SOURCE",spec["name"],"notes",len(pack["truth"]),flush=True)
    baseline=summary([x["score"] for x in rows["base"]])
    assert all(baseline[k]==v for k,v in BASELINE_URMP.items()),baseline
    result=[]
    for name,conf,jump,physical in POLICIES:
        per=rows[name];score=summary([x["score"] for x in per])
        result.append(dict(name=name,configuration=dict(confidence=conf,jump=jump,physical_rise=physical),
            TP=score["TP"],FP=score["FP"],FN=score["FN"],F1=score["F1"],
            F1_delta_points=100*(score["F1"]-baseline["F1"]),
            total_trigger_opportunities=sum(x["triggered"] for x in per),
            source_wins=sum(a["score"]["F1"]>b["score"]["F1"] for a,b in zip(per,rows["base"])),
            source_losses=sum(a["score"]["F1"]<b["score"]["F1"] for a,b in zip(per,rows["base"])),
            per_source={x["source"]:{k:x["score"][k] for k in ("TP","FP","FN","F1")} for x in per}))
    output=dict(status="S620_CAUSAL_TRANSITION_HEAD_DEV_ONLY_NO_PROMOTION",
        original_S614_model_sha256=frozen["model_sha256"],
        original_S618_auxiliary_model_sha256=EXPECTED_S618,
        original_S80_unchanged=True,baseline=baseline,variants=result,
        best_dev=max(result,key=lambda x:x["F1"])["name"],
        lookahead_frames=0,test_data_opened=False,
        new_heldout_claim=False,elapsed_seconds=round(time.perf_counter()-clock,2))
    (args.out/"report.json").write_text(json.dumps(output,indent=2)+"\n")
    print("S620 DEV SUMMARY",json.dumps([{k:x[k] for k in (
        "name","TP","FP","FN","F1","F1_delta_points","total_trigger_opportunities",
        "source_wins","source_losses")} for x in result],indent=2),flush=True)


if __name__=="__main__":main()
