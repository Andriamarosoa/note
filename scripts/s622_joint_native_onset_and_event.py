"""S622 group-OoF test of S619 event gate + S620 causal learned novelty.

Both native NOTE heads are retained; neither S80 nor S614 is changed.
S618 weights were trained on URMP TRAIN only. Five S619 event-gate fits
each exclude the scored URMP DEV composition. The S620 policy has already
been selected *on the same five DEV titles*, so this combo is an iterative
development audit, NOT fresh held-out evidence and not promotable yet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
import joblib
import numpy as np

from causal_note.strict_live import Selection,load_heads
from causal_note.s618_attack_head import input_features,DIM as FRAME_DIM
from causal_note.s619_event_gate import noteon_event_features,relay_original_events
from scripts.s620_frozen_novelty_audit import EXPECTED_S618,decode_novelty
from scripts.s619_event_gate_oof import gate_model
from scripts.s616_live_dev_ablation import ROOT,MODEL,BASELINE_URMP,prepare_dev_source
from scripts.s613_prepare_strict_live import digest
from scripts.s613_train_strict_live import decode,event_score,summary,offset_audit

POLICY_NOVELTY=(.90,.25,.08)
POLICIES={
    "S614_original":(False,None),
    "S620_novelty":(True,None),
    "S619_event_025":(False,.25),
    "combined_event_015":(True,.15),
    "combined_event_025":(True,.25),
    "combined_event_035":(True,.35),
}


def source(spec,cache,heads,trained,config):
    pack=prepare_dev_source(spec,cache)
    ends,x=pack["ends"],pack["features"]
    p,k=heads.predict(x)
    base=decode(ends,p,k,config,pack["samples"])
    plain,paired=event_score(base,pack["truth"])
    matrix,_=noteon_event_features(base,ends,x,p,k)
    labels=np.zeros(len(matrix),np.int8)
    for pi,_ in paired:labels[pi]=1
    assert len(labels)==plain["TP"]+plain["FP"] and labels.sum()==plain["TP"]
    q=trained.predict_proba(input_features(x,p).reshape(-1,FRAME_DIM))[:,1].reshape(len(ends),49)
    novel,_=decode_novelty(ends,x,p,k,config,pack["samples"],q,*POLICY_NOVELTY)
    orig_scores,_=event_score(base,pack["truth"])
    new_scores,_=event_score(novel,pack["truth"])
    return dict(source=spec["name"],composition=spec["composition"],
        acoustic=pack["source_sha256"],truth=pack["truth"],
        x=x,p=p,k=k,ends=ends,original=base,novel=novel,
        baseline=orig_scores,novel_metric=new_scores,matrix=matrix,labels=labels)


def run(model_path,out):
    if out.exists():raise FileExistsError("Immutable experiment S622")
    out.mkdir(parents=True)
    if digest(model_path.read_bytes())!=EXPECTED_S618:
        raise ValueError("Original train-only S618 head has wrong SHA256")
    frame_head=joblib.load(model_path)
    protocol=json.loads((ROOT/"protocol.json").read_text())
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest()==frozen["model_sha256"]
    selections=[s for s in protocol["sources"] if s["split"]=="dev" and s["corpus"]=="URMP"]
    assert len(selections)==5 and len({s["composition"] for s in selections})==5
    original=load_heads(MODEL,expected_sha256=frozen["model_sha256"])
    selection=Selection(**original.metadata["selection"])
    start=time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        cache=Path(td);(cache/"data").mkdir()
        loaded=[]
        for spec in selections:
            item=source(spec,cache,original,frame_head,selection)
            loaded.append(item)
            print("S622 SOURCE",item["source"],"original",item["baseline"]["TP"],
                  item["baseline"]["FP"],"novel",item["novel_metric"]["TP"],
                  item["novel_metric"]["FP"],flush=True)
    outputs={key:[] for key in POLICIES}
    for index,held in enumerate(loaded):
        development=[row for i,row in enumerate(loaded) if i!=index]
        assert len({row["composition"] for row in development})==4
        assert held["composition"] not in {x["composition"] for x in development}
        a=np.concatenate([x["matrix"] for x in development])
        b=np.concatenate([x["labels"] for x in development])
        classifier=gate_model();classifier.fit(a,b)
        for name,(novel,threshold) in POLICIES.items():
            originals=held["novel"] if novel else held["original"]
            if threshold is None:
                predicted=originals
            else:
                evidence,_=noteon_event_features(originals,held["ends"],held["x"],held["p"],held["k"])
                probabilities=classifier.predict_proba(evidence)[:,1]
                predicted=relay_original_events(originals,probabilities>=threshold)
            scored,_=event_score(predicted,held["truth"])
            outputs[name].append(dict(source=held["source"],score=scored))
    baseline=summary([r["score"] for r in outputs["S614_original"]])
    assert all(baseline[k]==v for k,v in BASELINE_URMP.items()),baseline
    expected_novelty=summary([r["score"] for r in outputs["S620_novelty"]])
    assert (expected_novelty["TP"],expected_novelty["FP"],expected_novelty["FN"]) == (272,207,227)
    results=[]
    for name,rows in outputs.items():
        score=summary([r["score"] for r in rows])
        results.append(dict(name=name,TP=score["TP"],FP=score["FP"],FN=score["FN"],
            F1=score["F1"],delta_F1_points=100*(score["F1"]-baseline["F1"]),
            improved_sources=sum(r["score"]["F1"]>o["score"]["F1"]
                 for r,o in zip(rows,outputs["S614_original"])),
            regressed_sources=sum(r["score"]["F1"]<o["score"]["F1"]
                 for r,o in zip(rows,outputs["S614_original"])),
            per_source={r["source"]:{k:r["score"][k] for k in ("TP","FP","FN","F1")} for r in rows}))
    report=dict(status="S622_DEVELOPMENT_ONLY_NATIVE_LEARNED_ONSET_AND_EVENT_GATE",
        original_S80_S614_unchanged=True,
        original_S618_sha256=EXPECTED_S618,
        ursa_group_oof_for_S619=5,test_NOT_opened=True,
        novelty_policy_preselected_using_same_development=True,
        no_new_independent_claim=True,original=baseline,
        policies=POLICIES,results=results,
        best_dev=max(results,key=lambda row:row["F1"])["name"],
        no_promotion=True,elapsed_seconds=round(time.perf_counter()-start,2))
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S622 COMBINED RESULTS",json.dumps([{k:v[k] for k in (
         "name","TP","FP","FN","F1","delta_F1_points","improved_sources","regressed_sources")}
         for v in results],indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    run(args.model,args.out)
