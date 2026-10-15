"""S645: causal score-rise weak onset rescue, audited TRAIN before DEV.

S644 proved frozen q>=.965 cannot pass original weak physical gate;
relaxing q without temporal shape floods false MIDI notes. S645 retains
the exact same S642 trained classifier and S614 model, but requires
a fresh rapid rise in q over strictly past six audio hops.

All alternative gates are predeclared. No test set opened; no model
threshold is promoted using the previously inspected development data.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import argparse
import json
import tempfile
import zipfile
import numpy as np
import joblib
from causal_note.strict_live import Selection,CausalSelection,load_heads
from causal_note.s642_dual_learned import candidate_scores
from scripts.s644_gate_rescue_loop import CANDIDATE_SHA,S614_SHA,CORPORA
from scripts.s626_train_sharp_native_onset import get_source
from scripts.s623_mixed_source_event_head import pick_original_guitar_training
from scripts.s616_live_dev_ablation import ROOT,MODEL
from scripts.s613_prepare_strict_live import download,digest
from scripts.s613_train_strict_live import event_score,decode

# q minimum; required recent q rise; minimum attack and sustain confidence.
PREDECLARED=(
    ("S614_original",None),
    ("rise098_010",(.98,.10,.05,.10,False)),
    ("rise099_010",(.99,.10,.05,.10,False)),
    ("rise099_020",(.99,.20,.02,.05,False)),
    ("rise0995_010",(.995,.10,.02,.05,False)),
    ("rise099_spectral",(.99,.10,.02,.05,True)),
)


def evaluate(row,q,cfg,policy):
    if policy is None:
        events=decode(row["ends"],row["p"],row["k"],cfg,row["n"])
        score,_=event_score(events,row["truth"])
        return dict(score={k:score[k] for k in ("TP","FP","FN","F1")},
                    attempted_frames=0,actual_rescued_onsets=0,signal_rises=0,
                    passes_acoustic=0)
    floor,jump,min_on,min_sustain,spectral=policy
    sel=CausalSelection(cfg)
    events=[]
    # Only six past frames, reset without any future input.
    history=np.zeros((6,49),np.float32)
    attempts=0;actual=0;rises=0;physical_count=0
    for end,x,p,k,predict in zip(row["ends"],row["x"],row["p"],row["k"],q):
        minimum=history.min(axis=0)
        high=predict>=floor
        rising=high & ((predict-minimum)>=jump)
        inactive=~sel.active
        physical=inactive&(p[:,0]<cfg.onset)&(p[:,0]>=min_on)&(
            p[:,1]>=min_sustain)
        if spectral:
            physical &= x[:,[57,58,59]].max(axis=1)>=.12
        eligible=rising&physical
        rises+=int(rising.sum())
        physical_count+=int(eligible.sum())
        edited=p.copy()
        indexes=np.flatnonzero(eligible)
        if len(indexes):
            chosen=int(indexes[np.argmax(predict[indexes])])
            edited[chosen,0]=max(edited[chosen,0],cfg.onset)
            edited[chosen,1]=max(edited[chosen,1],cfg.active_on)
            attempts+=1
        else:chosen=-1
        new_events=sel.step(edited,k,int(end),int(end))
        if chosen>=0 and any(e["type"]=="note_on" and e["pitch"]==chosen+40
                             for e in new_events):actual+=1
        events.extend(new_events)
        history[:-1]=history[1:]
        history[-1]=predict
    events.extend(sel.panic(row["n"]))
    metric,_=event_score(events,row["truth"])
    return dict(score={k:metric[k] for k in ("TP","FP","FN","F1")},
                attempted_frames=attempts,actual_rescued_onsets=actual,
                signal_rises=rises,passes_acoustic=physical_count)


def summarize(rows):
    result={}
    for name,policy in PREDECLARED:
        result[name]={}
        for corpus in CORPORA:
            records=[r for r in rows if r["corpus"]==corpus]
            scores=[r["policies"][name]["score"] for r in records]
            base=[r["policies"]["S614_original"]["score"] for r in records]
            totals={k:sum(x[k] for x in scores) for k in ("TP","FP","FN")}
            tp,fp,fn=(totals[k] for k in ("TP","FP","FN"))
            totals["F1"]=2*tp/max(1,2*tp+fp+fn)
            for key in ("attempted_frames","actual_rescued_onsets",
                        "signal_rises","passes_acoustic"):
                totals[key]=sum(r["policies"][name][key] for r in records)
            totals["regressed_sources"]=sum(x["F1"]<b["F1"] for x,b in zip(scores,base))
            totals["improved_sources"]=sum(x["F1"]>b["F1"] for x,b in zip(scores,base))
            result[name][corpus]=totals
    return result


def locked_policy(train):
    base=train["S614_original"]
    # Both corpora MUST improve on original 80ms symmetric metric.
    # Maximum two individual regressions per corpus.
    good=[(min(v[c]["F1"]-base[c]["F1"] for c in CORPORA),name)
          for name,v in train.items() if name!="S614_original"
          and all(v[c]["F1"]>base[c]["F1"] and v[c]["regressed_sources"]<=2
                  for c in CORPORA)]
    return max(good)[1] if good else "S614_original"


def run(checkpoint,out):
    if out.exists():raise FileExistsError("S645 never overwrites evidence")
    out.mkdir(parents=True)
    if digest(checkpoint.read_bytes())!=CANDIDATE_SHA:
        raise AssertionError("Expected original trained S642 candidate checkpoint")
    trained=joblib.load(checkpoint)
    protocol=json.loads((ROOT/"protocol.json").read_text())
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    assert digest(MODEL.read_bytes())==S614_SHA==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    heads=load_heads(MODEL,expected_sha256=S614_SHA)
    cfg=Selection(**heads.metadata["selection"])
    train_u=[s for s in protocol["sources"] if s["corpus"]=="URMP" and s["split"]=="train"]
    train_g=pick_original_guitar_training(protocol)
    dev_u=[s for s in protocol["sources"] if s["corpus"]=="URMP" and s["split"]=="dev"]
    dev_g=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    assert (len(train_u),len(train_g),len(dev_u),len(dev_g))==(17,18,5,12)
    group=lambda s:(s["corpus"],s["composition"])
    assert not set(map(group,train_u+train_g)) & set(map(group,dev_u+dev_g))
    stages={}
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);(root/"data").mkdir()
        for key in ("GuitarSet_audio","GuitarSet_annotations"):
            info=protocol["guitar_archives"][key]
            download(info["url"],root/"data"/(key+".zip"),info["size"],info["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as archive:
            for stage,sources in (("TRAIN",train_u+train_g),("DEV",dev_u+dev_g)):
                records=[]
                for spec in sources:
                    row=get_source(spec,root,heads,
                        archive if spec["corpus"]=="GuitarSet" else None)
                    q=candidate_scores(row,trained)
                    outcome={name:evaluate(row,q,cfg,policy)
                             for name,policy in PREDECLARED}
                    records.append(dict(name=spec["name"],corpus=spec["corpus"],
                                        policies=outcome))
                    print("S645",stage,spec["name"],"attack_events",
                        outcome["rise099_010"]["actual_rescued_onsets"],
                        "q099_F1",round(outcome["rise099_010"]["score"]["F1"],4),
                        flush=True)
                metrics=summarize(records)
                if stage=="TRAIN":
                    choice=locked_policy(metrics)
                    print("S645 TRAIN LOCK",choice,flush=True)
                stages[stage]=dict(n_sources=len(records),metrics=metrics,
                                   by_source=records)
    assert stages["DEV"]["metrics"]["S614_original"]["GuitarSet"]["TP"]==998
    assert stages["DEV"]["metrics"]["S614_original"]["URMP"]["TP"]==243
    report=dict(status="S645_CAUSAL_SCORE_RISE_RESCUE_TRAIN_SELECTED_DEV_EVALUATED",
      original_S614_S642_S80_unchanged=True,trained_candidate_sha256=CANDIDATE_SHA,
      original_S614_sha256=S614_SHA,
      time_window_past_hops=6,predeclared_policies=[
          dict(name=name,requirements=p) for name,p in PREDECLARED],
      selected_on_TRAIN_before_DEV=choice,
      stages=stages,TEST_never_opened=True,
      previous_DEV_exposure_no_promotion=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S645 FINAL SUMMARY",json.dumps(dict(
        train_locked=choice,
        TRAIN={name:{c:{k:v[k] for k in ("F1","TP","FP","FN","actual_rescued_onsets")}
                     for c,v in corp.items()} for name,corp in stages["TRAIN"]["metrics"].items()},
        DEV={name:{c:{k:v[k] for k in ("F1","TP","FP","FN","actual_rescued_onsets")}
                     for c,v in corp.items()} for name,corp in stages["DEV"]["metrics"].items()}
    ),indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();run(a.candidate,a.out)
