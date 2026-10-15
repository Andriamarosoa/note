"""S630 native causal phase bridge evaluation on composition-disjoint DEV only.

Preserves original S614 unchanged and checks all original baseline numbers.
No S80 changes and no historical published test use. ±80ms match is on
actual timestamp of emitted MIDI NoteOn (not backdated to an old onset).
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
import zipfile

from causal_note.strict_live import Selection,load_heads
from causal_note.s630_phase_bridge import CausalPhaseBridge,variants
from scripts.s626_train_sharp_native_onset import get_source
from scripts.s616_live_dev_ablation import ROOT,MODEL
from scripts.s613_prepare_strict_live import download,digest
from scripts.s613_train_strict_live import decode,event_score,offset_audit,summary


CONTROL={"GuitarSet":dict(TP=998,FP=585,FN=582),
         "URMP":dict(TP=243,FP=149,FN=256)}

def evaluate(row,selection):
    events_baseline=decode(row["ends"],row["p"],row["k"],selection,row["n"])
    rows={}
    for name,param in variants():
        if param is None:events=events_baseline;eligible=0
        else:
            bridge=CausalPhaseBridge(selection,param)
            events=[]
            for end,p,k in zip(row["ends"],row["p"],row["k"]):
                events.extend(bridge.step(p,k,int(end)))
            events.extend(bridge.finish(row["n"]))
            eligible=bridge.helped
        score,pairs=event_score(events,row["truth"])
        offset=offset_audit(events,row["truth"],pairs,row["n"]/44100.)
        rows[name]=dict(score=score,conditional_offset=offset["conditional_offset_accuracy"],
                        candidate_opportunities=eligible)
    return rows

def run(out):
    if out.exists():raise FileExistsError("S630 evidence must not be overwritten")
    out.mkdir(parents=True)
    protocol=json.loads((ROOT/"protocol.json").read_text())
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    sha=digest(MODEL.read_bytes())
    assert sha==frozen["model_sha256"]
    native=load_heads(MODEL,expected_sha256=sha)
    selection=Selection(**native.metadata["selection"])
    devU=[s for s in protocol["sources"] if s["split"]=="dev" and s["corpus"]=="URMP"]
    devG=[s for s in protocol["sources"] if s["split"]=="dev" and s["corpus"]=="GuitarSet"]
    assert len(devU)==5 and len(devG)==12
    started=time.perf_counter()
    sources=[]
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        for spec in devU:
            record=get_source(spec,root,native)
            scores=evaluate(record,selection)
            sources.append(dict(name=spec["name"],corpus="URMP",scores=scores))
            print("S630 URMP",spec["name"],flush=True)
        aud=protocol["guitar_archives"]["GuitarSet_audio"]
        ann=protocol["guitar_archives"]["GuitarSet_annotations"]
        download(aud["url"],root/"data/GuitarSet_audio.zip",aud["size"],aud["sha256"])
        download(ann["url"],root/"data/GuitarSet_annotations.zip",ann["size"],ann["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as archive:
            for spec in devG:
                record=get_source(spec,root,native,archive)
                scores=evaluate(record,selection)
                sources.append(dict(name=spec["name"],corpus="GuitarSet",scores=scores))
                print("S630 GUITAR",spec["name"],flush=True)
    results=[]
    for name,params in variants():
        corp={}
        for corpus,control in CONTROL.items():
            samples=[s for s in sources if s["corpus"]==corpus]
            v=summary([s["scores"][name]["score"] for s in samples])
            if name=="S614_baseline":assert all(v[k]==expected for k,expected in control.items())
            base_f1=2*control["TP"]/(2*control["TP"]+control["FP"]+control["FN"])
            corp[corpus]=dict(TP=v["TP"],FP=v["FP"],FN=v["FN"],F1=v["F1"],
                delta_F1_points=100*(v["F1"]-base_f1),
                opportunities=sum(s["scores"][name]["candidate_opportunities"] for s in samples),
                improved_sources=sum(s["scores"][name]["score"]["F1"]>s["scores"]["S614_baseline"]["score"]["F1"] for s in samples),
                regressed_sources=sum(s["scores"][name]["score"]["F1"]<s["scores"]["S614_baseline"]["score"]["F1"] for s in samples))
        results.append(dict(name=name,policy=vars(params) if params else None,by_corpus=corp))
        print("S630 VARIANT",name,"G",round(corp["GuitarSet"]["F1"],5),
              "U",round(corp["URMP"]["F1"],5),flush=True)
    viable=[x["name"] for x in results if all(x["by_corpus"][c]["delta_F1_points"]>0 for c in CONTROL)]
    report=dict(status="S630_PAST_ONLY_ONSET_SUSTAIN_PHASE_DEV",
        original_S80_and_S614_unchanged=True,old_checkpoint_sha256=sha,
        known_17_composition_dev_sources=True,test_22_sources_unopened=True,
        no_new_independent_claim=True,
        variants=results,positive_both_corpora=viable,
        no_promotion=True,elapsed_seconds=round(time.perf_counter()-started,2))
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S630 SUMMARY",json.dumps({k:report[k] for k in ("positive_both_corpora","elapsed_seconds")},indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    run(parser.parse_args().out)
