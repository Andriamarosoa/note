"""S623 cross-instrument native NOTE event-head training on train-only sources.

Three event-level policies are trained exclusively on historical S613 train
groups: URMP-only, GuitarSet-only, and a mixed-source head. Scores are
computed on GuitarSet/URMP development sources (previously used in S614
tuning; NOT independent promotion evidence). No S80 / S614 changes.
"""
from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import zipfile

import joblib
import numpy as np
from scipy.io import wavfile
from causal_note.strict_live import SR,Selection,feature_track,load_heads
from causal_note.s619_event_gate import noteon_event_features,relay_original_events,DIM
from scripts.s619_event_gate_oof import gate_model
from scripts.s616_live_dev_ablation import ROOT,MODEL,prepare_dev_source
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import real_notes,event_score,decode,summary


THRESHOLDS=(.15,.25,.35)
CONTROL={"GuitarSet":dict(TP=998,FP=585,FN=582),
         "URMP":dict(TP=243,FP=149,FN=256)}


def guitar_record(spec,audio_zip,root,heads,selection):
    blob=audio_zip.read(spec["audio"])
    rate,sound=wavfile.read(io.BytesIO(blob))
    if rate != SR or sound.dtype not in (np.dtype("int16"),np.dtype("int32")):
        raise AssertionError("Original GuitarSet archive audio specification changed")
    values=sound.astype(np.float32)/(32768 if sound.dtype==np.int16 else 2147483648)
    if values.ndim==2:values=values.mean(1)
    values=values[:int(spec["seconds"]*SR)].copy()
    ends,x=feature_track(values)
    p,k=heads.predict(x)
    truth=real_notes(labels(spec,root)["notes"],len(values)/SR)
    events=decode(ends,p,k,selection,len(values))
    return build_evidence(spec,events,truth,ends,x,p,k)


def urmp_record(spec,root,heads,selection):
    data=prepare_dev_source(spec,root)
    ends,x=data["ends"],data["features"]
    p,k=heads.predict(x)
    events=decode(ends,p,k,selection,data["samples"])
    return build_evidence(spec,events,data["truth"],ends,x,p,k)


def build_evidence(spec,events,truth,ends,x,p,k):
    matrix,indices=noteon_event_features(events,ends,x,p,k)
    scored,pairs=event_score(events,truth)
    target=np.zeros(len(matrix),np.int8)
    for i,_ in pairs:target[i]=1
    assert len(matrix)==scored["TP"]+scored["FP"]
    assert int(target.sum())==scored["TP"]
    return dict(name=spec["name"],corpus=spec["corpus"],
        composition=spec["composition"],original=events,truth=truth,
        features=matrix,targets=target,score=scored)


def pick_original_guitar_training(protocol):
    guitar=defaultdict(list)
    for s in protocol["sources"]:
        if s["corpus"]=="GuitarSet" and s["split"]=="train":
            guitar[s["composition"]].append(s)
    assert len(guitar)==9 and all(len(rows)==4 for rows in guitar.values())
    # Two original recordings per train composition, selected by metadata
    # alone; no label or result-based sampling.
    return [row for key in sorted(guitar) for row in sorted(guitar[key],
            key=lambda r:hashlib.sha256(r["name"].encode()).hexdigest())[:2]]


def fit_group(rows,negative_weight):
    X=np.concatenate([v["features"] for v in rows])
    y=np.concatenate([v["targets"] for v in rows])
    assert X.shape[1]==DIM and len(np.unique(y))==2
    weights=np.where(y==0,negative_weight,1.)
    classifier=gate_model()
    classifier.fit(X,y,sample_weight=weights)
    return classifier,dict(rows=len(y),tp_labels=int(y.sum()),
        fp_labels=int(len(y)-int(y.sum())),
        compositions=len({(r["corpus"],r["composition"]) for r in rows}))


def run(out):
    if out.exists():raise FileExistsError("Immutable S623 results only")
    out.mkdir(parents=True)
    protocol=json.loads((ROOT/"protocol.json").read_text())
    freeze=json.loads((ROOT/"selection_frozen.json").read_text())
    sha=digest(MODEL.read_bytes())
    assert sha==freeze["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==freeze["protocol_sha256"]
    urmp_train=[s for s in protocol["sources"] if s["corpus"]=="URMP" and s["split"]=="train"]
    urmp_dev=[s for s in protocol["sources"] if s["corpus"]=="URMP" and s["split"]=="dev"]
    guitar_train=pick_original_guitar_training(protocol)
    guitar_dev=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    assert (len(urmp_train),len(urmp_dev),len(guitar_train),len(guitar_dev))==(17,5,18,12)
    train_set={(s["corpus"],s["composition"]) for s in guitar_train+urmp_train}
    dev_set={(s["corpus"],s["composition"]) for s in guitar_dev+urmp_dev}
    assert not train_set&dev_set
    old=load_heads(MODEL,expected_sha256=sha)
    config=Selection(**old.metadata["selection"])
    start=time.perf_counter()
    training=[];development=[]
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);(root/"data").mkdir()
        for spec in urmp_train:
            row=urmp_record(spec,root,old,config)
            training.append(row)
            print("S623 TRAIN URMP",row["name"],len(row["targets"]),flush=True)
        for spec in urmp_dev:
            row=urmp_record(spec,root,old,config)
            development.append(row)
            print("S623 DEV URMP",row["name"],len(row["targets"]),flush=True)
        a=protocol["guitar_archives"]["GuitarSet_audio"]
        b=protocol["guitar_archives"]["GuitarSet_annotations"]
        download(a["url"],root/"data/GuitarSet_audio.zip",a["size"],a["sha256"])
        download(b["url"],root/"data/GuitarSet_annotations.zip",b["size"],b["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as zipped:
            for spec in guitar_train:
                row=guitar_record(spec,zipped,root,old,config)
                training.append(row)
                print("S623 TRAIN GUITAR",row["name"],len(row["targets"]),flush=True)
            for spec in guitar_dev:
                row=guitar_record(spec,zipped,root,old,config)
                development.append(row)
                print("S623 DEV GUITAR",row["name"],len(row["targets"]),flush=True)
    baseline={}
    for corpus in ("GuitarSet","URMP"):
        measure=summary([r["score"] for r in development if r["corpus"]==corpus])
        assert all(measure[k]==v for k,v in CONTROL[corpus].items()),measure
        baseline[corpus]={q:measure[q] for q in ("TP","FP","FN","F1")}
    variants=[]
    for train_mix in ("URMP","GuitarSet","mixed"):
        choose=[r for r in training if train_mix=="mixed" or r["corpus"]==train_mix]
        for penalty in (1.,2.):
            model,train_evidence=fit_group(choose,penalty)
            checkpoint=out/f"trained_S623_{train_mix}_neg{int(penalty)}.joblib"
            joblib.dump(model,checkpoint,compress=3)
            model_sha=digest(checkpoint.read_bytes())
            for threshold in THRESHOLDS:
                records=[]
                for row in development:
                    confidence=model.predict_proba(row["features"])[:,1]
                    events=relay_original_events(row["original"],confidence>=threshold)
                    scored,pairs=event_score(events,row["truth"])
                    records.append(dict(corpus=row["corpus"],name=row["name"],old=row["score"],new=scored))
                aggregates={}
                for corpus in ("GuitarSet","URMP"):
                    same=[x for x in records if x["corpus"]==corpus]
                    total=summary([x["new"] for x in same])
                    aggregates[corpus]=dict(TP=total["TP"],FP=total["FP"],
                        FN=total["FN"],F1=total["F1"],
                        delta_F1_points=100*(total["F1"]-baseline[corpus]["F1"]),
                        improved_sources=sum(x["new"]["F1"]>x["old"]["F1"] for x in same),
                        regressed_sources=sum(x["new"]["F1"]<x["old"]["F1"] for x in same))
                variants.append(dict(train_corpus=train_mix,
                    negative_weight=penalty,threshold=threshold,
                    original_model_checkpoint_sha256=sha,head_sha256=model_sha,
                    training=train_evidence,by_corpus=aggregates))
                print("S623 VARIANT",train_mix,"negative",penalty,
                      "threshold",threshold,"G",round(aggregates["GuitarSet"]["F1"],4),
                      "U",round(aggregates["URMP"]["F1"],4),flush=True)
    report=dict(status="S623_NATIVE_MULTI_CORPUS_TRAIN_DEV_ONLY_NO_PROMOTION",
        original_S80_S614_unchanged=True,original_S614_checkpoint_sha256=sha,
        train_source_count=len(training),URMP_train=17,GuitarSet_train=18,
        composition_disjoint_train_dev=True,all_train_sources_originally_seen_by_S614=True,
        development_source_count=len(development),
        original=baseline,variants=variants,
        test_sources_NOT_read=True,previous_S614_dev_exposure=True,
        validation_independence_claim=False,official_promotion=False,
        elapsed_seconds=round(time.perf_counter()-start,2))
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S623 SUMMARY",json.dumps({v:report[v] for v in (
        "original","train_source_count","development_source_count","elapsed_seconds")},indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    run(parser.parse_args().out)
