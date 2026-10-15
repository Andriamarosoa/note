"""S642 genuine original NOTE TRAIN acoustic+timing head, DEV evaluation.

35 TRAIN recordings (17 URMP, 18 GuitarSet, distinct compositions from
17 DEV). The original checkpoint SHA and data protocol are verified.
Both new trained models are exported; each of candidate-only, delay-only
and joint is compared to identical original S614. No model is promoted.

Causal evaluation is separately reported from historical symmetric
±80ms evaluation; never confuse these or original S614 TEST F1 69.03%.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import argparse
import json
import tempfile
import time
import zipfile

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from causal_note.strict_live import Selection,load_heads
from causal_note.s618_attack_head import DIM
from causal_note.s642_dual_learned import (
    evaluated_policy,fixed_policies,training_delay_labels
)
from causal_note.s635_shared_midi_metric import causal_score
from scripts.s626_train_sharp_native_onset import get_source,choose_examples
from scripts.s623_mixed_source_event_head import pick_original_guitar_training
from scripts.s616_live_dev_ablation import ROOT,MODEL
from scripts.s613_prepare_strict_live import download,digest
from scripts.s613_train_strict_live import event_score,offset_audit

BASELINE={"GuitarSet":(998,585,582),"URMP":(243,149,256)}


def fit_candidate(samples,targets):
    if len(samples)<1000 or not 0<targets.sum()<len(targets):
        raise AssertionError("S642 candidate binary supervision missing")
    estimator=HistGradientBoostingClassifier(
        max_iter=110,max_leaf_nodes=18,max_depth=5,min_samples_leaf=40,
        learning_rate=.07,l2_regularization=5.,
        early_stopping=False,random_state=642)
    estimator.fit(samples,targets,
                  sample_weight=np.where(targets==1,5.,1.))
    if list(estimator.classes_)!=[0,1]:
        raise AssertionError("Binary attack class mapping")
    return estimator


def fit_delay(samples,targets):
    bins=Counter(int(c) for c in targets)
    if len(samples)<500 or len(bins)<2:
        raise AssertionError("Not enough observed early NoteOn supervision for learned delay")
    estimator=HistGradientBoostingClassifier(
        max_iter=90,max_leaf_nodes=14,max_depth=4,min_samples_leaf=15,
        learning_rate=.06,l2_regularization=4.,
        early_stopping=False,random_state=643,
        class_weight="balanced")
    estimator.fit(samples,targets)
    if not set(estimator.classes_)<=set(range(4)):
        raise AssertionError("Unexpected trained latency labels")
    return estimator,bins


def onsets_causal(e,truth):
    # This supports the 40..76 range of S638 and the full native
    # range separately, and respects real delayed available_sample.
    def score(lo,hi):
        refs=sorted((float(t["time"]),int(t["pitch"]),i)
                    for i,t in enumerate(truth) if lo<=t["pitch"]<=hi)
        outputs=sorted((float(ev["time"]),int(ev["pitch"]),i)
                       for i,ev in enumerate(e) if ev["type"]=="note_on" and lo<=ev["pitch"]<=hi)
        c=causal_score(refs,outputs,80.)
        return {key:c[key] for key in ("TP","FP","FN","F1")}
    return dict(full=score(40,88),common_midi_range=score(40,76))


def evaluate_one(row,candidate,delay):
    original=None
    output={}
    for name,policy in fixed_policies():
        events,internal=evaluated_policy(row,candidate,delay,policy)
        score,pairs=event_score(events,row["truth"])
        offsets=offset_audit(events,row["truth"],pairs,row["n"]/44100.)
        output[name]=dict(events=events,statistics={
            **{k:score[k] for k in ("TP","FP","FN","F1")},
            "causal":onsets_causal(events,row["truth"]),
            "NoteOff_conditional_accuracy":offsets["conditional_offset_accuracy"],
            "NoteOff_forced":offsets["forced_or_missing_offsets"],
            "candidate_proposals":int(internal["rescued_frame_proposals"]),
            "delayed_noteons":int(internal["held_noteons"]),
            "held_0_1_2_3":dict(Counter(internal["delays"]))})
        if name=="S614_original":original=events
    if original is None:raise AssertionError("Missing original policy")
    return {n:d["statistics"] for n,d in output.items()}


def run(out):
    if out.exists():raise FileExistsError("Immutable experiment result directory")
    out.mkdir(parents=True)
    start=time.perf_counter()
    protocol=json.loads((ROOT/"protocol.json").read_text())
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    sha=digest(MODEL.read_bytes())
    assert sha==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    native=load_heads(MODEL,expected_sha256=sha)
    config=dict(native.metadata["selection"])
    trainU=[s for s in protocol["sources"] if s["split"]=="train" and s["corpus"]=="URMP"]
    trainG=pick_original_guitar_training(protocol)
    devU=[s for s in protocol["sources"] if s["split"]=="dev" and s["corpus"]=="URMP"]
    devG=[s for s in protocol["sources"] if s["split"]=="dev" and s["corpus"]=="GuitarSet"]
    assert (len(trainU),len(trainG),len(devU),len(devG))==(17,18,5,12)
    group=lambda s:(s["corpus"],s["composition"])
    assert not {group(x) for x in trainU+trainG}&{group(x) for x in devU+devG}
    assert not any(s["split"]=="test" for s in trainU+trainG+devU+devG)

    rng=np.random.default_rng(642)
    ac_x=[];ac_y=[];delay_x=[];delay_y=[];audit=[]
    rows=[]
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);(root/"data").mkdir()
        arc=protocol["guitar_archives"]
        for key in ("GuitarSet_audio","GuitarSet_annotations"):
            p=arc[key]
            download(p["url"],root/"data"/(key+".zip"),p["size"],p["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as audio_zip:
            def training():
                for s in trainU:
                    yield s,get_source(s,root,native)
                for s in trainG:
                    yield s,get_source(s,root,native,audio_zip)
            for spec,data in training():
                a,b,info=choose_examples(data,rng)
                data["selection"]=config
                c,y,n=training_delay_labels(data)
                ac_x.append(a);ac_y.append(b)
                if len(c):delay_x.append(c);delay_y.append(y)
                audit.append(dict(source=spec["name"],corpus=spec["corpus"],
                                  ac_rows=len(a),positive=int(b.sum()),
                                  delay_events=n,delay_nonzero=int((y>0).sum()),
                                  trained_delay_classes=dict(Counter(int(z) for z in y))))
                print("S642 TRAIN",spec["name"],"acoustic",len(a),
                      "early_references",int((y>0).sum()),flush=True)
            ax=np.concatenate(ac_x);ay=np.concatenate(ac_y)
            dx=np.concatenate(delay_x);dy=np.concatenate(delay_y)
            assert ax.shape[1]==dx.shape[1]==DIM
            candidate=fit_candidate(ax,ay)
            delay,train_distribution=fit_delay(dx,dy)
            ca_path=out/"S642_weak_attack_train_only.joblib"
            dl_path=out/"S642_0_to_3_hops_train_only.joblib"
            joblib.dump(candidate,ca_path,compress=3)
            joblib.dump(delay,dl_path,compress=3)
            checkpoints=dict(candidate_sha256=digest(ca_path.read_bytes()),
                            delay_sha256=digest(dl_path.read_bytes()))
            print("S642 TRAINED HEADS",checkpoints,
                  "delay distribution",train_distribution,flush=True)
            del ax,ay,dx,dy,ac_x,ac_y,delay_x,delay_y
            for spec in devU:
                data=get_source(spec,root,native)
                data["selection"]=config
                rows.append(dict(source=spec["name"],corpus="URMP",
                                 policies=evaluate_one(data,candidate,delay)))
                print("S642 DEV URMP",spec["name"],flush=True)
            for spec in devG:
                data=get_source(spec,root,native,audio_zip)
                data["selection"]=config
                rows.append(dict(source=spec["name"],corpus="GuitarSet",
                                 policies=evaluate_one(data,candidate,delay)))
                print("S642 DEV GUITAR",spec["name"],flush=True)
    metrics={}
    for name,policy in fixed_policies():
        corp={}
        for corpus in ("GuitarSet","URMP"):
            group_rows=[row["policies"][name] for row in rows if row["corpus"]==corpus]
            original_rows=[row["policies"]["S614_original"]
                           for row in rows if row["corpus"]==corpus]
            summ={key:sum(v[key] for v in group_rows) for key in ("TP","FP","FN")}
            summ["F1"]=2*summ["TP"]/max(1,2*summ["TP"]+summ["FP"]+summ["FN"])
            if name=="S614_original":
                assert tuple(summ[k] for k in ("TP","FP","FN"))==BASELINE[corpus],(corpus,summ)
            ca={}
            for key in ("full","common_midi_range"):
                counts={k:sum(r["causal"][key][k] for r in group_rows)
                        for k in ("TP","FP","FN")}
                counts["F1"]=2*counts["TP"]/max(1,2*counts["TP"]+counts["FP"]+counts["FN"])
                ca[key]=counts
            summ["causal"]=ca
            summ["attempted_rescues"]=sum(r["candidate_proposals"] for r in group_rows)
            summ["held_notes"]=sum(r["delayed_noteons"] for r in group_rows)
            summ["regressed_recordings"]=sum(a["F1"]<b["F1"] for a,b in zip(group_rows,original_rows))
            summ["improved_recordings"]=sum(a["F1"]>b["F1"] for a,b in zip(group_rows,original_rows))
            summ["conditional_noteoff_accuracy"]=float(np.mean([
                r["NoteOff_conditional_accuracy"] for r in group_rows]))
            corp[corpus]=summ
        metrics[name]=corp
        print("S642 SCORE",name,"GuitarSet",round(corp["GuitarSet"]["F1"],5),
              "URMP",round(corp["URMP"]["F1"],5),
              "causalG",round(corp["GuitarSet"]["causal"]["common_midi_range"]["F1"],5),
              flush=True)
    base=metrics["S614_original"]
    winners=[name for name,group in metrics.items() if name!="S614_original"
             and all(group[c]["F1"]>base[c]["F1"] for c in ("GuitarSet","URMP"))]
    report=dict(status="S642_TRAINED_WEAK_ONSET_AND_0_3_HOP_LATENCY_DEV_ONLY",
        S614_and_S80_unchanged=True,original_checkpoint_sha256=sha,
        trained_checkpoints=checkpoints,train_sources=audit,
        train_total=35,dev_total=17,
        trained_delay_label_counts={str(k):int(v) for k,v in train_distribution.items()},
        train_Candidate_positive_count=int(sum(int(i["positive"]) for i in audit)),
        dev_per_source=rows,metrics=metrics,wins_both_corpora=winners,
        original_TEST_not_opened=True,dev_preexposed_not_independent=True,
        no_promotion=True,elapsed_seconds=round(time.perf_counter()-start,2))
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S642 VERDICT",json.dumps(dict(checkpoints=checkpoints,
        both_positive=winners,
        metrics={name:{c:dict(TP=v["TP"],FP=v["FP"],FN=v["FN"],
                 F1=v["F1"],causal=v["causal"]["common_midi_range"]["F1"])
                 for c,v in groups.items()} for name,groups in metrics.items()}),indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    run(p.parse_args().out)
