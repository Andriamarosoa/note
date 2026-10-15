"""S619 group-held event-level NOTE gate: 5 URMP dev titles, five OOF folds.

No opened S614 test, no copied S601/midi model, no synthetic oracle K,
and *no* weights/threshold changes in original S614 or historical S80.
Only emits/suppresses ORIGINAL NOTE candidate NoteOn, with valid NoteOff.
The source group is not newly independent of S614 historical tuning; the
five held titles are disjoint within this auxiliary-head cross-validation.
"""
from __future__ import annotations
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from causal_note.strict_live import Selection, load_heads
from causal_note.s619_event_gate import noteon_event_features,relay_original_events,DIM
from scripts.s616_live_dev_ablation import ROOT,MODEL,BASELINE_URMP,prepare_dev_source
from scripts.s613_prepare_strict_live import digest
from scripts.s613_train_strict_live import event_score,offset_audit,decode,summary


THRESHOLDS=(.05,.15,.25,.35,.45,.55,.65,.75,.85)
POLICY="keep original emitted NoteOn only when learned event P(valid) >= threshold"


def source_data(spec,cache,heads,config):
    pack=prepare_dev_source(spec,cache)
    ends,x=pack["ends"],pack["features"]
    p,k=heads.predict(x)
    events=decode(ends,p,k,config,pack["samples"])
    base,pairs=event_score(events,pack["truth"])
    matrix,indices=noteon_event_features(events,ends,x,p,k)
    assert len(matrix)==len(indices)==base["TP"]+base["FP"]
    label=np.zeros(len(indices),np.int8)
    for predicted_i,_ in pairs:
        label[predicted_i]=1
    assert label.sum()==base["TP"]
    return dict(source=spec["name"],composer=spec["composition"],
        matrix=matrix,labels=label,events=events,truth=pack["truth"],
        baseline=base,inputs=dict(audio_sha256=pack["source_sha256"],
                                annotation_sha256=pack["label_sha256"]))


def gate_model():
    return HistGradientBoostingClassifier(
        max_iter=90, max_leaf_nodes=8, max_depth=4,
        learning_rate=.06,min_samples_leaf=12,l2_regularization=3.,
        early_stopping=False,random_state=619)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    out=parser.parse_args().out
    if out.exists():raise FileExistsError("Immutable scoring folder")
    out.mkdir(parents=True)
    protocol=json.loads((ROOT/"protocol.json").read_text())
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    original_sha=hashlib.sha256(MODEL.read_bytes()).hexdigest()
    assert original_sha==frozen["model_sha256"] and digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    selected=[s for s in protocol["sources"] if s["corpus"]=="URMP" and s["split"]=="dev"]
    assert len(selected)==5 and len({s["composition"] for s in selected})==5
    original=load_heads(MODEL,expected_sha256=original_sha)
    selection=Selection(**original.metadata["selection"])
    begin=time.perf_counter()
    sources=[]
    with tempfile.TemporaryDirectory() as root:
        cache=Path(root);(cache/"data").mkdir()
        for spec in selected:
            d=source_data(spec,cache,original,selection)
            sources.append(d)
            print("S619 SOURCE",d["source"],"TP",d["baseline"]["TP"],
                  "FP",d["baseline"]["FP"],"rows",len(d["labels"]),flush=True)
    scores={t:[] for t in THRESHOLDS}
    reference=[]
    folds=[]
    for hold in range(5):
        training=[d for i,d in enumerate(sources) if i!=hold]
        test=sources[hold]
        a=np.concatenate([d["matrix"] for d in training])
        b=np.concatenate([d["labels"] for d in training])
        assert a.shape[1]==DIM and len(set(d["composer"] for d in training))==4
        assert test["composer"] not in {d["composer"] for d in training}
        if len(set(b))!=2:
            raise AssertionError("Training event data contains only one class")
        model=gate_model()
        model.fit(a,b)
        q=model.predict_proba(test["matrix"])[:,1]
        reference.append(test["baseline"])
        folds.append(dict(held=test["source"],composer=test["composer"],
                          train_sources=[x["source"] for x in training],
                          rows_test=len(q),positives_test=int(test["labels"].sum()),
                          probability_min=float(q.min()),probability_max=float(q.max())))
        for threshold in THRESHOLDS:
            filtered=relay_original_events(test["events"],q>=threshold)
            scored,pairs=event_score(filtered,test["truth"])
            off=offset_audit(filtered,test["truth"],pairs,
                             max(int(e["decision_sample"]) for e in test["events"])/44100)
            scores[threshold].append(dict(name=test["source"],
                                         score=scored,offset=off,kept_notes=int((q>=threshold).sum())))
        print("S619 OOF",test["source"],"score_probability",
              [round(float((q>=t).mean()),3) for t in THRESHOLDS],flush=True)
    baseline=summary(reference)
    assert all(baseline[k]==v for k,v in BASELINE_URMP.items()),baseline
    results=[]
    for threshold in THRESHOLDS:
        items=scores[threshold]
        aggregate=summary([item["score"] for item in items])
        offset=Counter()
        for item in items:
            offset.update({key:item["offset"][key] for key in (
                "matched_noteons_with_observed_reference_offset",
                "offsets_within_max_50ms_or_20pct_duration",
                "forced_or_missing_offsets")})
        denom=offset["matched_noteons_with_observed_reference_offset"]
        results.append(dict(threshold=threshold,
                            TP=aggregate["TP"],FP=aggregate["FP"],FN=aggregate["FN"],
                            F1=aggregate["F1"],F1_delta_points=100*(aggregate["F1"]-baseline["F1"]),
                            conditional_offset_accuracy=offset["offsets_within_max_50ms_or_20pct_duration"]/max(1,denom),
                            per_source={s["name"]:dict(TP=s["score"]["TP"],FP=s["score"]["FP"],
                                         FN=s["score"]["FN"],F1=s["score"]["F1"]) for s in items}))
    best=max(results,key=lambda v:v["F1"])
    result=dict(status="S619_5COMPOSITION_OOF_EVENT_GATE_ON_KNOWN_URMP_DEV_NO_PROMOTION",
        source_checkpoint_sha256=original_sha,original_S80_and_S614_unchanged=True,
        strict_causal_only_original_noteon_candidates=True,NoteOff_lifecycle_paired=True,
        test_sources_and_annotations_NOT_READ=True,previously_used_S614_dev=True,
        group_held_model_fits=5, original_reference=baseline,
        fixed_thresholds=list(THRESHOLDS),
        policy=POLICY,folds=folds,results=results,
        best_oof_threshold=best["threshold"],best_oof_F1=best["F1"],
        no_independent_promotion_claim=True,elapsed_seconds=round(time.perf_counter()-begin,2))
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("S619 OOF RESULTS",json.dumps({
        "reference":baseline, "best_oof":best,
        "thresholds":[{k:x[k] for k in ("threshold","TP","FP","FN","F1","F1_delta_points")} for x in results]
    },indent=2),flush=True)


if __name__=="__main__":
    main()
