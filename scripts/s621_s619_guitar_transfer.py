"""S621 cross-source transfer of the S619 event gate, fixed OOF threshold.

Fits the ORIGINAL S619 event classifier on the five URMP DEV compositions,
then checks frozen 0.25 gate (versus untouched S614) on all 12 GuitarSet DEV
recordings. GuitarSet DEV was seen by S614 development previously; this is a
transfer audit, NOT an independent promotion or another sealed test.
"""
from __future__ import annotations

import argparse
from collections import Counter
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
from scripts.s619_event_gate_oof import source_data,gate_model
from scripts.s616_live_dev_ablation import ROOT,MODEL,prepare_dev_source
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import real_notes,decode,event_score,summary,offset_audit

FROZEN_THRESHOLD=.25
EXPECTED_CONTROL=dict(TP=998,FP=585,FN=582)


def run(out):
    if out.exists():raise FileExistsError("Cannot overwrite S621 results")
    out.mkdir(parents=True)
    protocol=json.loads((ROOT/"protocol.json").read_text())
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    original_sha=hashlib.sha256(MODEL.read_bytes()).hexdigest()
    assert original_sha==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    train=[s for s in protocol["sources"] if s["split"]=="dev" and s["corpus"]=="URMP"]
    guitar=[s for s in protocol["sources"] if s["split"]=="dev" and s["corpus"]=="GuitarSet"]
    assert len(train)==5 and len(guitar)==12 and len({s["composition"] for s in guitar})==3
    assert all(s["name"].startswith("04_") for s in guitar)
    old=load_heads(MODEL,expected_sha256=original_sha)
    config=Selection(**old.metadata["selection"])
    began=time.perf_counter()
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);(root/"data").mkdir()
        fitting=[source_data(spec,root,old,config) for spec in train]
        a=np.concatenate([row["matrix"] for row in fitting])
        b=np.concatenate([row["labels"] for row in fitting])
        assert a.shape[1]==DIM and len(set(d["composer"] for d in fitting))==5
        classifier=gate_model()
        classifier.fit(a,b)
        trained_path=out/"S621_S619_URMP_DEV_original_event_gate.joblib"
        joblib.dump(classifier,trained_path,compress=3)
        model_hash=digest(trained_path.read_bytes())
        audio_proof=protocol["guitar_archives"]["GuitarSet_audio"]
        ann_proof=protocol["guitar_archives"]["GuitarSet_annotations"]
        download(audio_proof["url"],root/"data/GuitarSet_audio.zip",
                 audio_proof["size"],audio_proof["sha256"])
        download(ann_proof["url"],root/"data/GuitarSet_annotations.zip",
                 ann_proof["size"],ann_proof["sha256"])
        old_metrics=[];new_metrics=[];sources=[]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as archive:
            for spec in guitar:
                raw=archive.read(spec["audio"])
                sample_rate,pcm=wavfile.read(io.BytesIO(raw))
                assert sample_rate==SR and pcm.dtype in (np.dtype("int16"),np.dtype("int32"))
                audio=pcm.astype(np.float32)/(32768 if pcm.dtype==np.int16 else 2147483648)
                if audio.ndim==2:audio=audio.mean(axis=1)
                audio=audio[:int(spec["seconds"]*SR)].copy()
                ends,features=feature_track(audio)
                p,k=old.predict(features)
                reference=decode(ends,p,k,config,len(audio))
                acoustic_notes=labels(spec,root)
                truth=real_notes(acoustic_notes["notes"],len(audio)/SR)
                base_score,_=event_score(reference,truth)
                matrix,indices=noteon_event_features(reference,ends,features,p,k)
                prob=classifier.predict_proba(matrix)[:,1]
                filtered=relay_original_events(reference,prob>=FROZEN_THRESHOLD)
                new_score,pairs=event_score(filtered,truth)
                offs=offset_audit(filtered,truth,pairs,len(audio)/SR)
                old_metrics.append(base_score);new_metrics.append(new_score)
                sources.append(dict(source=spec["name"],composition=spec["composition"],
                    original={key:base_score[key] for key in ("TP","FP","FN","F1")},
                    revised={key:new_score[key] for key in ("TP","FP","FN","F1")},
                    net_difference=dict(TP=new_score["TP"]-base_score["TP"],
                                        FP=new_score["FP"]-base_score["FP"]),
                    emitted_events=len(matrix),accepted=int((prob>=FROZEN_THRESHOLD).sum()),
                    offset_conditional=offs["conditional_offset_accuracy"],
                    sha256=digest(raw)))
                print("S621 GUITAR",spec["name"],
                      "F1",round(base_score["F1"],4),"->",round(new_score["F1"],4),
                      "TP",base_score["TP"],new_score["TP"],
                      "FP",base_score["FP"],new_score["FP"],flush=True)
    original=summary(old_metrics)
    revised=summary(new_metrics)
    assert all(original[key]==v for key,v in EXPECTED_CONTROL.items()),original
    report=dict(status="S621_FROZEN_EVENT_GATE_CROSS_DEVELOPMENT_GUITARSET",
        S80_preserved=True,original_S614_sha256=original_sha,
        trained_head_sha256=model_hash,training="five URMP DEV compositions",
        original_model_unchanged=True,training_sources=[x["source"] for x in fitting],
        frozen_probability_threshold=FROZEN_THRESHOLD,
        guitarset_source_count=len(sources),
        original=dict(TP=original["TP"],FP=original["FP"],FN=original["FN"],F1=original["F1"]),
        candidate=dict(TP=revised["TP"],FP=revised["FP"],FN=revised["FN"],F1=revised["F1"]),
        F1_delta_points=100*(revised["F1"]-original["F1"]),
        improved_recordings=sum(x["revised"]["F1"]>x["original"]["F1"] for x in sources),
        regressed_recordings=sum(x["revised"]["F1"]<x["original"]["F1"] for x in sources),
        sources=sources, previous_S614_GuitarSet_dev_exposure=True,
        no_new_heldout_claim=True,official_promotion=False,
        elapsed_seconds=round(time.perf_counter()-began,2))
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S621 TRANSFER SUMMARY",json.dumps({k:report[k] for k in (
        "original","candidate","F1_delta_points","improved_recordings",
        "regressed_recordings")},indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    run(parser.parse_args().out)
