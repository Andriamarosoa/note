"""S632: train actual risk-to-go regressors and validate causal NOTE LIVE.

Original NOTE S614 and historic S80 unchanged. Uses train-only oracle
counterfactual cost targets from acoustic episodes (no ground truth at runtime).
Compares genuinely emitted timestamps, notes, and offsets on composition-
disjoint URMP/GuitarSet DEV. Published 22-source S614 test never opened.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
import zipfile

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from causal_note.strict_live import Selection,load_heads
from causal_note.s632_risk_policy import (
    training_costs,causal_episodes,predicted_costs,apply_policy,
    FEATURE_DIM,policies)
from scripts.s623_mixed_source_event_head import pick_original_guitar_training
from scripts.s626_train_sharp_native_onset import get_source
from scripts.s616_live_dev_ablation import ROOT,MODEL
from scripts.s613_prepare_strict_live import download,digest
from scripts.s613_train_strict_live import event_score,summary,offset_audit

CONTROL={"GuitarSet":dict(TP=998,FP=585,FN=582),
         "URMP":dict(TP=243,FP=149,FN=256)}


def aggregate_train(rows,out):
    rng=np.random.default_rng(632)
    xlist=[]; ylist=[]; proofs=[]
    for item in rows:
        x,y,proof=training_costs(item)
        assert len(x)==len(y) and x.shape[1]==FEATURE_DIM
        # Fixed source-wise cap to prevent GuitarSet repeated frames
        # from dominating URMP, especially easy silence.
        if len(x)>8500:
            chosen=rng.choice(len(x),8500,replace=False)
            x=x[chosen];y=y[chosen]
        xlist.append(x);ylist.append(y)
        proofs.append(dict(source=item["name"],corpus=item["corpus"],
                           composition=item["composition"],**proof,
                           sampled_rows=len(x)))
        print("S632 TRAIN EXAMPLES",item["name"],
              "selected",len(x),"positive_episodes",proof["positive_episodes"],
              "negative_episodes",proof["negative_episodes"],flush=True)
    features=np.concatenate(xlist)
    target=np.concatenate(ylist)
    del xlist,ylist
    assert len(features)>20000 and np.isfinite(features).all()
    assert target.shape==(len(features),3) and np.isfinite(target).all()
    checkpoints=[];models=[]
    for index,label in enumerate(("emit","reject","wait")):
        model=HistGradientBoostingRegressor(
            max_iter=100,max_leaf_nodes=18,max_depth=5,
            learning_rate=.075,l2_regularization=2.5,
            min_samples_leaf=85,early_stopping=False,random_state=632+index)
        model.fit(features,target[:,index])
        path=out/f"S632_Q_{label}_train_only.joblib"
        joblib.dump(model,path,compress=3)
        checkpoints.append(dict(name=label,
                 file=path.name,sha256=digest(path.read_bytes()),
                 train_mae=float(np.abs(model.predict(features[:3000])-
                         target[:3000,index]).mean())))
        models.append(model)
        print("S632 COST MODEL",label,"hash",checkpoints[-1]["sha256"],flush=True)
    return models,proofs,checkpoints,dict(rows=len(features),
        q_targets_mean=target.mean(0).tolist(),
        q_targets_p95=np.percentile(target,95,axis=0).tolist())


def evaluate(row,models,config):
    _,timeline=causal_episodes(row)
    costs=predicted_costs(models,timeline)
    result={}
    for name,policy in policies():
        emitted,decisions=apply_policy(row,costs,config,policy)
        measured,pairs=event_score(emitted,row["truth"])
        offsets=offset_audit(emitted,row["truth"],pairs,row["n"]/44100.)
        result[name]=dict(score=measured,offset=offsets,
                          actions=decisions)
    return result


def run(out):
    if out.exists():raise FileExistsError("Immutable experimental result path")
    out.mkdir(parents=True)
    original=json.loads((ROOT/"selection_frozen.json").read_text())
    protocol=json.loads((ROOT/"protocol.json").read_text())
    original_sha=digest(MODEL.read_bytes())
    assert original_sha==original["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==original["protocol_sha256"]
    train_u=[s for s in protocol["sources"] if s["corpus"]=="URMP" and s["split"]=="train"]
    train_g=pick_original_guitar_training(protocol)
    dev_u=[s for s in protocol["sources"] if s["corpus"]=="URMP" and s["split"]=="dev"]
    dev_g=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    assert (len(train_u),len(train_g),len(dev_u),len(dev_g))==(17,18,5,12)
    key=lambda s:(s["corpus"],s["composition"])
    assert not set(map(key,train_u+train_g))&set(map(key,dev_u+dev_g))
    assert all(s["split"]!="test" for s in train_u+train_g+dev_u+dev_g)
    old=load_heads(MODEL,expected_sha256=original_sha)
    cfg=Selection(**old.metadata["selection"])
    began=time.perf_counter()
    development=[]
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        aud=protocol["guitar_archives"]["GuitarSet_audio"]
        ann=protocol["guitar_archives"]["GuitarSet_annotations"]
        download(aud["url"],root/"data/GuitarSet_audio.zip",aud["size"],aud["sha256"])
        download(ann["url"],root/"data/GuitarSet_annotations.zip",ann["size"],ann["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as archive:
            def iter_train():
                for spec in train_u:
                    print("S632 TRAIN AUDIO URMP",spec["name"],flush=True)
                    yield get_source(spec,root,old)
                for spec in train_g:
                    print("S632 TRAIN AUDIO GUITAR",spec["name"],flush=True)
                    yield get_source(spec,root,old,archive)
            # Generator releases each entire waveform/feature array after its
            # extracted candidates have been added; fixed memory per source.
            models,proofs,checkpoints,traininfo=aggregate_train(iter_train(),out)
            for spec in dev_u:
                row=get_source(spec,root,old)
                measured=evaluate(row,models,cfg)
                development.append(dict(source=spec["name"],corpus="URMP",scores=measured))
                print("S632 DEV URMP",spec["name"],flush=True)
            for spec in dev_g:
                row=get_source(spec,root,old,archive)
                measured=evaluate(row,models,cfg)
                development.append(dict(source=spec["name"],corpus="GuitarSet",scores=measured))
                print("S632 DEV GUITAR",spec["name"],flush=True)
    variants=[]
    for name,policy in policies():
        result={}
        for corpus,baseline in CONTROL.items():
            items=[row for row in development if row["corpus"]==corpus]
            metric=summary([row["scores"][name]["score"] for row in items])
            if name=="S614_original":
                assert all(metric[k]==v for k,v in baseline.items()),metric
            true_f1=2*baseline["TP"]/(2*baseline["TP"]+baseline["FP"]+baseline["FN"])
            delays=[x for row in items for x in row["scores"][name]["actions"]["delays_ms"]]
            off_eligible=sum(row["scores"][name]["offset"]["matched_noteons_with_observed_reference_offset"] for row in items)
            off_good=sum(row["scores"][name]["offset"]["offsets_within_max_50ms_or_20pct_duration"] for row in items)
            actions=Counter()
            for row in items:
                data=row["scores"][name]["actions"]
                actions.update({k:data[k] for k in ("waits","emits","rejects")})
            result[corpus]=dict(TP=metric["TP"],FP=metric["FP"],FN=metric["FN"],
                F1=metric["F1"],delta_F1_points=100*(metric["F1"]-true_f1),
                improved_recordings=sum(r["scores"][name]["score"]["F1"]>
                    r["scores"]["S614_original"]["score"]["F1"] for r in items),
                regressed_recordings=sum(r["scores"][name]["score"]["F1"]<
                    r["scores"]["S614_original"]["score"]["F1"] for r in items),
                notes_actions=dict(actions),
                added_head_candidate_delay_ms=(
                    dict(p50=float(np.percentile(delays,50)),
                         p95=float(np.percentile(delays,95)),max=float(max(delays)))
                    if delays else None),
                matched_onset_emission_delay_ms=metric["matched_emission_delay_ms"],
                conditional_correct_noteoff=off_good/max(1,off_eligible))
        variants.append(dict(name=name,policy=asdict(policy) if policy else None,
                             corpus=result))
        print("S632 DEV F1",name,
              "GuitarSet",round(result["GuitarSet"]["F1"],5),
              "URMP",round(result["URMP"]["F1"],5),flush=True)
    gains=[r["name"] for r in variants if r["name"]!="S614_original" and
           all(r["corpus"][c]["delta_F1_points"]>0 for c in CONTROL)]
    report=dict(status="S632_TRAINED_COUNTERFACTUAL_DECISION_COSTS_DEV_ONLY",
        original_S80_S614_unchanged=True,checkpoint_S614_sha256=original_sha,
        learned_cost_models=checkpoints,
        train_source_count=35,train_proofs=proofs,train_rows=traininfo,
        held_by_composition_dev_count=17,
        teacher="backward dynamic programming of EMIT/REJECT/WAIT costs with delayed emission penalty",
        offline_teacher_sees_train_ground_truth=True,
        online_inference_never_looks_into_future=True,
        baseline_source_F1_hist={"GuitarSet":CONTROL["GuitarSet"],"URMP":CONTROL["URMP"]},
        variants=variants,
        simultaneously_positive_dev=gains,
        sealed_S614_test_not_opened=True,DEV_historically_exposed_not_independent=True,
        no_official_promotion=True,
        elapsed_seconds=round(time.perf_counter()-began,2))
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S632 RESULT",json.dumps(dict(
        learned_models=[m["sha256"] for m in checkpoints],
        candidate_policies=len(variants),simultaneously_positive=gains,
        summary=[dict(name=x["name"],G=x["corpus"]["GuitarSet"]["F1"],
                      U=x["corpus"]["URMP"]["F1"]) for x in variants]),indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    run(parser.parse_args().out)
