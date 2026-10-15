"""S618: train new NOTE *auxiliary* onset head on URMP train, audit on URMP dev.

The original S614 neural/tree heads and selectors are loaded unchanged. The
candidate head reads only current/past native PCM features and original NOTE
probabilities. This workflow does NOT claim held-out independent results and
does NOT open the already-published 22-source S614 test.

Run: PYTHONPATH=.:src python -m scripts.s618_train_dev --out /tmp/S618
"""
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from causal_note.strict_live import SR, Selection, load_heads
from causal_note.s618_attack_head import (
    DIM, input_features, candidate_mask, variants, NativeLearnedSelection)
from scripts.s616_live_dev_ablation import (
    ROOT,MODEL, BASELINE_URMP, prepare_dev_source, aggregate)
from scripts.s613_prepare_strict_live import digest
from scripts.s613_train_strict_live import decode, real_notes, event_score, offset_audit


def direct_labels(ends, truth):
    """Targets only for *training*, never infer from annotations."""
    y=np.zeros((len(ends),49),np.bool_)
    for note in truth:
        p=note["pitch"]-40
        if not 0 <= p < 49: continue
        start=np.searchsorted(ends,round(note["time"]*SR),side="left")
        stop=np.searchsorted(ends,round((note["time"]+.070)*SR),side="right")
        y[start:stop,p]=True
    return y


def samples_for_training(ends,x,p,truth,rng):
    mat=input_features(x,p)
    positive=direct_labels(ends,truth)
    eligible=candidate_mask(x,p)
    active=(eligible & ((np.arange(len(ends))[:,None] % 3)==0)) | positive
    pos_id=np.flatnonzero(active & positive)
    neg_id=np.flatnonzero(active & ~positive)
    if len(pos_id)>2500: pos_id=rng.choice(pos_id,2500,replace=False)
    if len(neg_id)>7000: neg_id=rng.choice(neg_id,7000,replace=False)
    ids=np.concatenate((pos_id,neg_id))
    rng.shuffle(ids)
    features=mat.reshape(-1,DIM)[ids]
    target=positive.reshape(-1)[ids].astype(np.int8)
    return features,target,dict(positives=int(len(pos_id)),
                                 negatives=int(len(neg_id)),
                                 original_positives=int(positive.sum()),
                                 eligible_positives=int((eligible&positive).sum()))


def evaluate_source(ends,x,p,k,truth,model,config,total_samples):
    q=model.predict_proba(input_features(x,p).reshape(-1,DIM))[:,1].reshape(len(ends),49)
    reference=decode(ends,p,k,config,total_samples)
    scores={}
    for name,policy in variants():
        selector=NativeLearnedSelection(config,policy)
        output=[]
        for t,probs,dist,qrow,feat in zip(ends,p,k,q,x):
            output.extend(selector.step(probs,dist,qrow,feat,int(t)))
        output.extend(selector.finish(total_samples))
        if name=="S614_reference" and output != reference:
            raise AssertionError("No-op S618 selector diverged from S614; abort")
        score,pairs=event_score(output,truth)
        offset=offset_audit(output,truth,pairs,total_samples)
        scores[name]=dict(score=score,
                          offsets={key:offset[key] for key in (
                              "matched_noteons_with_observed_reference_offset",
                              "offsets_within_max_50ms_or_20pct_duration",
                              "forced_or_missing_offsets")},
                          changed=int(selector.changed))
    return scores,q


def run(out):
    if out.exists():
        raise FileExistsError("Evidence must never be overwritten")
    out.mkdir(parents=True)
    protocol=json.loads((ROOT/"protocol.json").read_text())
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    model_sha=hashlib.sha256(MODEL.read_bytes()).hexdigest()
    assert model_sha==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    train=[s for s in protocol["sources"] if s["corpus"]=="URMP" and s["split"]=="train"]
    dev=[s for s in protocol["sources"] if s["corpus"]=="URMP" and s["split"]=="dev"]
    test=[s for s in protocol["sources"] if s["split"]=="test"]
    assert len(train)==17 and len(dev)==5 and len(test)==22
    assert not ({s["composition"] for s in train}&{s["composition"] for s in dev})
    assert all(s["corpus"]=="URMP" and s["split"]!="test" for s in train+dev)
    heads=load_heads(MODEL,expected_sha256=model_sha)
    original_cfg=Selection(**heads.metadata["selection"])
    rng=np.random.default_rng(618)
    X,Y=[],[]
    details=[]
    t0=time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        cache=Path(td);(cache/"data").mkdir()
        for spec in train:
            pack=prepare_dev_source(spec,cache)
            ends,x=pack["ends"],pack["features"]
            p,k=heads.predict(x)
            xs,ys,audit=samples_for_training(ends,x,p,pack["truth"],rng)
            X.append(xs);Y.append(ys)
            details.append(dict(corpus="URMP",source=spec["name"],
                                composition=spec["composition"],**audit,
                                audio_sha256=pack["source_sha256"]))
            print("S618 TRAIN",spec["name"],audit,"elapsed",round(time.perf_counter()-t0,1),flush=True)
        a=np.concatenate(X);b=np.concatenate(Y)
        del X,Y
        assert a.shape[1]==DIM and b.sum()>500 and (~b.astype(bool)).sum()>1000
        # Small CPU histogram tree: faster causal inference than a large forest;
        # no foreknowledge of target timestamps during *inference*.
        model=HistGradientBoostingClassifier(
            max_iter=85,max_leaf_nodes=15,max_depth=5,
            learning_rate=.09,min_samples_leaf=55,
            l2_regularization=2.,early_stopping=False,random_state=618)
        sample_weight=np.where(b>0,3.,1.)
        model.fit(a,b,sample_weight=sample_weight)
        save_path=out/"S618_URMP_train_only_attack_head.joblib"
        joblib.dump(model,save_path,compress=3)
        model_hash=digest(save_path.read_bytes())
        print("S618 MODEL",model_hash,"train_rows",len(b),
              "positives",int(b.sum()),"elapsed",round(time.perf_counter()-t0,1),flush=True)
        aggregated={name:[] for name,_ in variants()}
        per_source={}
        inference_timings=[]
        for spec in dev:
            pack=prepare_dev_source(spec,cache)
            ends,x=pack["ends"],pack["features"]
            p,k=heads.predict(x)
            scores,q=evaluate_source(ends,x,p,k,pack["truth"],model,original_cfg,pack["samples"])
            for key,value in scores.items():
                aggregated[key].append(dict(source=spec["name"],**value))
            per_source[spec["name"]]=dict(
                composer=spec["composition"],true_notes=len(pack["truth"]),
                audio_sha256=pack["source_sha256"],
                by_policy={n:{z:row["score"][z] for z in ("TP","FP","FN","F1")}
                           for n,row in scores.items()})
            # Estimate additional head-only inference per hop on an actual
            # 49-pitch frame; S614/FFT/driver time is NOT included.
            f=input_features(x[:160],p[:160])
            for row in f[:100]:
                start=time.perf_counter()
                model.predict_proba(row)
                inference_timings.append((time.perf_counter()-start)*1000)
            print("S618 DEV",spec["name"],
                  {n:{"TP":v["score"]["TP"],"FP":v["score"]["FP"]}
                   for n,v in scores.items()},flush=True)
    baseline=aggregate(aggregated["S614_reference"])
    assert all(baseline[key]==v for key,v in BASELINE_URMP.items()),baseline
    variants_report=[]
    for name,policy in variants():
        items=aggregated[name]
        count=aggregate(items)
        offsets=Counter()
        for record in items:
            offsets.update(record["offsets"])
        correct=offsets["offsets_within_max_50ms_or_20pct_duration"]
        eligible=offsets["matched_noteons_with_observed_reference_offset"]
        variants_report.append(dict(name=name,
            TP=count["TP"],FP=count["FP"],FN=count["FN"],F1=count["F1"],
            F1_delta_points=100*(count["F1"]-baseline["F1"]),
            conditional_offset_accuracy=correct/max(1,eligible),
            offset_counts=dict(offsets),
            source_wins=sum(int(v["score"]["F1"]>r["score"]["F1"])
                for v,r in zip(items,aggregated["S614_reference"])),
            source_losses=sum(int(v["score"]["F1"]<r["score"]["F1"])
                for v,r in zip(items,aggregated["S614_reference"]))))
    timing={"head_extra_p50_ms":float(np.percentile(inference_timings,50)),
            "head_extra_p95_ms":float(np.percentile(inference_timings,95)),
            "head_extra_p99_ms":float(np.percentile(inference_timings,99))}
    report=dict(status="S618_TRAINED_AUXILIARY_NATIVE_ATTACK_HEAD_DEV_ONLY",
                source_models_untouched=True,S80_preserved=True,S614_checkpoint_sha256=model_sha,
                learned_head_sha256=model_hash,train_samples=len(b),train_positives=int(b.sum()),
                input_dimensions=DIM,training_sources=details,
                dev_sources=per_source,baseline=baseline,variants=variants_report,
                best_dev=max(variants_report,key=lambda v:v["F1"])["name"],
                extra_head_compute=timing,combined_realtime_per_hop_unproven=True,
                train_compositions=17,dev_compositions=5,test_evidence_read=False,
                no_new_heldout_claim=True,official_promotion=False,
                development_overfitting_possible=True,elapsed_seconds=round(time.perf_counter()-t0,1))
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S618 DEV RESULT",json.dumps(dict(best_dev=report["best_dev"],compute=timing,
        variants=[{k:v[k] for k in ("name","TP","FP","FN","F1","F1_delta_points","source_wins","source_losses")}
                  for v in variants_report]),indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    run(parser.parse_args().out)
