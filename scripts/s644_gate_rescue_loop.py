"""S644: auditable stage-by-stage native rescue gate investigation.

Read original S642 candidate head (verified SHA256) and S614 native
without changing either model. Evaluate four predeclared q floors on
TRAIN and lock one option *before* DEV replay. DEV additionally reports
diagnostic curves for transparency, but never chooses the policy.

Important: an attempted rescue is NOT an emitted MIDI NoteOn.
The causal original S614 selector (onset, sustain, confirmation,
refractory, polyphony limit) is run to the end on every variant.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import tempfile
import zipfile
import numpy as np
import joblib
from causal_note.strict_live import CausalSelection, Selection, load_heads
from causal_note.s642_dual_learned import candidate_scores
from scripts.s626_train_sharp_native_onset import get_source
from scripts.s623_mixed_source_event_head import pick_original_guitar_training
from scripts.s616_live_dev_ablation import ROOT,MODEL
from scripts.s613_prepare_strict_live import download,digest,event_score

CANDIDATE_SHA="2f7364865803bb0cab8c564911e25574203b129169f6abf5f4ecdf14acc7f256"
S614_SHA="7ccf046e3a1870eaca3a108e6f9565298aeeadf43732a2f83559542c03f5f17f"
# Prespecified before inspecting q distributions or the DEV results.
POLICIES=(("S614",None),("q50",.50),("q70",.70),("q85",.85),("q95",.95),
          ("q965_original",.965))
CORPORA=("GuitarSet","URMP")


def measure(row,head,config,threshold):
    sel=CausalSelection(config)
    events=[]
    counts=Counter()
    stage_q=[]
    active_q=[]
    for end,p,k,x,q in zip(row["ends"],row["p"],row["k"],row["x"],head):
        editable=p.copy()
        idle=~sel.active
        weak=idle&(p[:,0]<config.onset)
        lowattack=weak&(p[:,0]>=.13)
        physical=lowattack&(p[:,1]>=.26)
        passed=physical&(q>=threshold) if threshold is not None else np.zeros(49,bool)
        counts["idle_weak_frames"]+=int(weak.sum())
        counts["above_attack_013"]+=int(lowattack.sum())
        counts["above_sustain_026"]+=int(physical.sum())
        for thr in (.5,.7,.85,.95,.965):
            counts[f"q{int(thr*1000)}_pass"]+=int((physical&(q>=thr)).sum())
        if np.any(physical):
            stage_q.extend(float(v) for v in q[physical])
        if np.any(weak):
            active_q.extend(float(v) for v in q[weak])
        choices=np.flatnonzero(passed)
        selected=-1
        if len(choices):
            selected=int(choices[np.argmax(q[choices])])
            editable[selected,0]=config.onset
            editable[selected,1]=max(float(editable[selected,1]),config.active_on)
            counts["attempted_frames"]+=1
        generated=sel.step(editable,k,int(end),int(end))
        if selected>=0 and any(e["type"]=="note_on" and e["pitch"]==selected+40
                              for e in generated):
            counts["actual_candidate_noteons"]+=1
        events.extend(generated)
    events.extend(sel.panic(row["n"]))
    metric,pairs=event_score(events,row["truth"])
    counts["noteons"]=sum(e["type"]=="note_on" for e in events)
    if threshold is None and counts["attempted_frames"]:
        raise AssertionError("Original S614 cannot attempt rescued frames")
    return ({key:metric[key] for key in ("TP","FP","FN","F1")},dict(counts),
            np.asarray(stage_q,dtype=np.float32),np.asarray(active_q,dtype=np.float32))


def rowspec(protocol,corpus,split):
    return [s for s in protocol["sources"] if s["corpus"]==corpus and s["split"]==split]


def aggregates(rows,stage):
    out={}
    for name,threshold in POLICIES:
        out[name]={}
        for corpus in CORPORA:
            corpus_rows=[r for r in rows if r["corpus"]==corpus]
            met=[r["variants"][name]["metric"] for r in corpus_rows]
            totals={key:sum(v[key] for v in met) for key in ("TP","FP","FN")}
            totals["F1"]=2*totals["TP"]/max(1,2*totals["TP"]+totals["FP"]+totals["FN"])
            c=Counter()
            for r in corpus_rows:c.update(r["variants"][name]["counts"])
            totals["gates"]=dict(c)
            original=[r["variants"]["S614"]["metric"] for r in corpus_rows]
            totals["source_improvements"]=sum(a["F1"]>b["F1"] for a,b in zip(met,original))
            totals["source_regressions"]=sum(a["F1"]<b["F1"] for a,b in zip(met,original))
            out[name][corpus]=totals
    return out


def lock_on_train(results):
    baseline=results["S614"]
    # No DEV is read by this decision. Improvement must be net positive
    # for both corpora. Otherwise retain original S614; no rescue promoted.
    viable=[]
    for name,threshold in POLICIES[1:]:
        if all(results[name][corpus]["F1"]>baseline[corpus]["F1"]
               and results[name][corpus]["source_regressions"]<=2
               for corpus in CORPORA):
            viable.append((min(results[name][c]["F1"]-baseline[c]["F1"] for c in CORPORA),
                           name))
    return sorted(viable,reverse=True)[0][1] if viable else "S614"


def profile(values):
    if not len(values):return dict(n=0,median=None,p90=None,p99=None,maximum=None)
    return dict(n=int(len(values)),median=float(np.percentile(values,50)),
                p90=float(np.percentile(values,90)),
                p99=float(np.percentile(values,99)),maximum=float(np.max(values)))


def run(model_path,out):
    if out.exists():raise FileExistsError("Immutable S644 report output")
    out.mkdir(parents=True)
    if digest(model_path.read_bytes())!=CANDIDATE_SHA:
        raise ValueError("Frozen S642 candidate head fingerprint mismatch")
    model=joblib.load(model_path)
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    protocol=json.loads((ROOT/"protocol.json").read_text())
    sha=digest(MODEL.read_bytes())
    assert sha==S614_SHA==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    native=load_heads(MODEL,expected_sha256=sha)
    config=Selection(**native.metadata["selection"])
    train_u=rowspec(protocol,"URMP","train")
    train_g=pick_original_guitar_training(protocol)
    dev_u=rowspec(protocol,"URMP","dev")
    dev_g=rowspec(protocol,"GuitarSet","dev")
    assert (len(train_u),len(train_g),len(dev_u),len(dev_g))==(17,18,5,12)
    groups=lambda specs:{(x["corpus"],x["composition"]) for x in specs}
    assert not groups(train_u+train_g)&groups(dev_u+dev_g)
    assert not any(x["split"]=="test" for x in train_u+train_g+dev_u+dev_g)
    sections={}
    with tempfile.TemporaryDirectory() as temporary:
        folder=Path(temporary);(folder/"data").mkdir()
        a=protocol["guitar_archives"]
        for key in ("GuitarSet_audio","GuitarSet_annotations"):
            part=a[key]
            download(part["url"],folder/"data"/(key+".zip"),part["size"],part["sha256"])
        with zipfile.ZipFile(folder/"data/GuitarSet_audio.zip") as archive:
            for stage,sources in (("TRAIN",train_u+train_g),("DEV",dev_u+dev_g)):
                evidence=[]
                pgate=[];qweak=[]
                for item in sources:
                    data=get_source(item,folder,native,archive if item["corpus"]=="GuitarSet" else None)
                    q=candidate_scores(data,model)
                    variant={}
                    for name,threshold in POLICIES:
                        metric,counts,phys,weak=measure(data,q,config,threshold)
                        variant[name]=dict(metric=metric,counts=counts)
                        if name=="S614":
                            pgate.append(phys);qweak.append(weak)
                    evidence.append(dict(name=item["name"],corpus=item["corpus"],
                                         composition=item["composition"],variants=variant))
                    print("S644",stage,item["name"],"physical",
                          variant["S614"]["counts"]["above_sustain_026"],
                          "q965",variant["S614"]["counts"]["q965_pass"],
                          "q50",variant["S614"]["counts"]["q500_pass"],
                          flush=True)
                section=dict(sources=len(evidence),metrics=aggregates(evidence,stage),
                    samples=dict(physical_q=profile(np.concatenate(pgate)),
                                 weak_q=profile(np.concatenate(qweak))),
                    per_source=evidence)
                if stage=="TRAIN":
                    section["train_only_locked_policy"]=lock_on_train(section["metrics"])
                    print("S644 LOCKED ON TRAIN",section["train_only_locked_policy"],flush=True)
                sections[stage]=section
    selected=sections["TRAIN"]["train_only_locked_policy"]
    report=dict(status="S644_ORIGINAL_S642_CANDIDATE_GATE_AUDIT",
        original_NOTE_S614_S80_S632_unchanged=True,
        candidate_head_sha256=CANDIDATE_SHA,
        original_S614_sha256=S614_SHA,
        stage_order=["TRAIN","LOCK","DEV"],DEV_never_chooses_threshold=True,
        policies=[dict(name=n,min_q=q) for n,q in POLICIES],
        sections=sections,train_locked_policy=selected,
        dev_locked_summary=sections["DEV"]["metrics"][selected],
        no_promotion=True,original_test_not_read=True,
        DEV_historically_exposed_not_independent=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S644 FINAL GATE SUMMARY",json.dumps(dict(
        selected_from_train=selected,
        TRAIN={k:{c:dict(F1=v["F1"],TP=v["TP"],FP=v["FP"],FN=v["FN"],
                          attempts=v["gates"].get("attempted_frames",0),
                          actual_on=v["gates"].get("actual_candidate_noteons",0))
                  for c,v in rows.items()} for k,rows in sections["TRAIN"]["metrics"].items()},
        DEV={k:{c:dict(F1=v["F1"],TP=v["TP"],FP=v["FP"],FN=v["FN"],
                          attempts=v["gates"].get("attempted_frames",0),
                          actual_on=v["gates"].get("actual_candidate_noteons",0))
                  for c,v in rows.items()} for k,rows in sections["DEV"]["metrics"].items()},
        q_profiles={k:sections[k]["samples"] for k in sections}),indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.candidate,a.out)
