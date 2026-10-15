"""S624: blocked cross-instrument event-head OOF, NO TEST PROMOTION.

Train native event-quality classifiers on 7 of the 8 GuitarSet/URMP DEV
composition groups and freeze predictions on the 8th. All four versions of
the underlying NOTE S614 heads, S80 history, and acoustic features remain
unchanged. This corrects the S623 in-sample original-head false-negative
underrepresentation, but still reuses previously developed-on S614 sources.
"""
from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np
from causal_note.strict_live import Selection,load_heads
from causal_note.s619_event_gate import relay_original_events
from scripts.s619_event_gate_oof import gate_model
from scripts.s623_mixed_source_event_head import guitar_record,urmp_record
from scripts.s616_live_dev_ablation import ROOT,MODEL
from scripts.s613_prepare_strict_live import download,digest
from scripts.s613_train_strict_live import event_score,summary

THRESHOLDS=(0.,.10,.15,.20,.25,.30,.35)
PENALTIES=(1.,2.)
CONTROL={"GuitarSet":dict(TP=998,FP=585,FN=582),
         "URMP":dict(TP=243,FP=149,FN=256)}


def run(out):
    if out.exists():raise FileExistsError("No rewriting OOF record")
    out.mkdir(parents=True)
    protocol=json.loads((ROOT/"protocol.json").read_text())
    freeze=json.loads((ROOT/"selection_frozen.json").read_text())
    sha=digest(MODEL.read_bytes())
    assert sha==freeze["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==freeze["protocol_sha256"]
    urmp=[s for s in protocol["sources"] if s["split"]=="dev" and s["corpus"]=="URMP"]
    guitar=[s for s in protocol["sources"] if s["split"]=="dev" and s["corpus"]=="GuitarSet"]
    assert len(urmp)==5 and len(guitar)==12
    assert len({s["composition"] for s in guitar})==3
    native=load_heads(MODEL,expected_sha256=sha)
    selection=Selection(**native.metadata["selection"])
    started=time.perf_counter()
    rows=[]
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        for spec in urmp:
            rows.append(urmp_record(spec,root,native,selection))
        a=protocol["guitar_archives"]["GuitarSet_audio"]
        b=protocol["guitar_archives"]["GuitarSet_annotations"]
        download(a["url"],root/"data/GuitarSet_audio.zip",a["size"],a["sha256"])
        download(b["url"],root/"data/GuitarSet_annotations.zip",b["size"],b["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as z:
            for spec in guitar:
                rows.append(guitar_record(spec,z,root,native,selection))
    groups=defaultdict(list)
    for row in rows:groups[(row["corpus"],row["composition"])].append(row)
    assert len(groups)==8
    assert {key[0] for key in groups}=={"URMP","GuitarSet"}
    reference={}
    for corpus in CONTROL:
        aggregated=summary([r["score"] for r in rows if r["corpus"]==corpus])
        assert all(aggregated[k]==v for k,v in CONTROL[corpus].items()),aggregated
        reference[corpus]={k:aggregated[k] for k in ("TP","FP","FN","F1")}
    frozen={}
    folds=[]
    for group,held in sorted(groups.items()):
        training=[r for key,section in groups.items() if key!=group for r in section]
        x=np.concatenate([r["features"] for r in training])
        y=np.concatenate([r["targets"] for r in training])
        assert { (r["corpus"],r["composition"]) for r in training }.isdisjoint({group})
        assert y.sum()>100 and (len(y)-y.sum())>100
        for penalty in PENALTIES:
            model=gate_model()
            model.fit(x,y,sample_weight=np.where(y==0,penalty,1.))
            for row in held:
                q=model.predict_proba(row["features"])[:,1]
                frozen[(penalty,row["name"])]=q
            print("S624 HELD",group,"penalty",penalty,
                  "train_rows",len(y),"held",len(held),flush=True)
        folds.append(dict(corpus=group[0],composition=group[1],
            held_recordings=[r["name"] for r in held],
            train_corpus_compositions=sorted(f"{k[0]}/{k[1]}" for k in groups if k!=group),
            label_leakage_excluded=True))
    results=[]
    for penalty in PENALTIES:
        for threshold in THRESHOLDS:
            per=[]
            for row in rows:
                confidence=frozen[(penalty,row["name"])]
                if threshold==0:
                    chosen=row["original"]
                else:
                    chosen=relay_original_events(row["original"],confidence>=threshold)
                scored,pairs=event_score(chosen,row["truth"])
                per.append(dict(corpus=row["corpus"],source=row["name"],
                    old=row["score"],new=scored))
            by_corpus={}
            for corpus in ("GuitarSet","URMP"):
                arr=[r for r in per if r["corpus"]==corpus]
                score=summary([r["new"] for r in arr])
                by_corpus[corpus]=dict(TP=score["TP"],FP=score["FP"],FN=score["FN"],
                    F1=score["F1"],
                    delta_F1_points=100*(score["F1"]-reference[corpus]["F1"]),
                    improved_sources=sum(x["new"]["F1"]>x["old"]["F1"] for x in arr),
                    regressed_sources=sum(x["new"]["F1"]<x["old"]["F1"] for x in arr))
            results.append(dict(negative_weight=penalty,threshold=threshold,by_corpus=by_corpus))
            print("S624 FROZEN OOF",penalty,threshold,
                  "G",round(by_corpus["GuitarSet"]["F1"],4),
                  "U",round(by_corpus["URMP"]["F1"],4),flush=True)
    possible=[v for v in results if v["by_corpus"]["GuitarSet"]["F1"]>=reference["GuitarSet"]["F1"]
          and v["by_corpus"]["URMP"]["F1"]>=reference["URMP"]["F1"]]
    report=dict(status="S624_8GROUP_OOF_TRAIN_DEV_ONLY",
        reference_checkpoint_sha256=sha, original_S80_S614_unchanged=True,
        train_split_used=False, published_S614_test_NOT_read=True,
        previously_exposed_S614_DEV=True, held_group_count=len(groups),
        blocked_composition_crossval=True, folds=folds,original=reference,
        models_trained=8*len(PENALTIES),
        candidate_policies=results,
        number_of_policies_nonregressing_both=len(possible),
        simultaneous_no_regression=possible,
        no_independent_claim=True,no_promotion=True,
        elapsed_seconds=round(time.perf_counter()-started,2))
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S624 SUMMARY",json.dumps({k:report[k] for k in (
        "original","number_of_policies_nonregressing_both",
        "simultaneous_no_regression","elapsed_seconds")},indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    run(parser.parse_args().out)
