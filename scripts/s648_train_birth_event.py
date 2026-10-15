"""S648 truly trained event-BIRTH head, TRAIN fit/calibration → DEV replay.

Frozen S614 and original GuitarSet/URMP datasets. Composition grouped
TRAIN into fit vs calibration before opening any DEV source; no TEST ever
read, no references used at prediction time. Includes loss/regression audit.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
import zipfile
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from causal_note.strict_live import Selection,load_heads
from causal_note.s648_event_birth import (BirthPolicy,causal_features,
    sampled_training_events,decode_birth)
from causal_note.s635_shared_midi_metric import causal_score
from scripts.s626_train_sharp_native_onset import get_source
from scripts.s623_mixed_source_event_head import pick_original_guitar_training
from scripts.s616_live_dev_ablation import ROOT,MODEL
from scripts.s613_prepare_strict_live import download,digest
from scripts.s613_train_strict_live import event_score,offset_audit

S614_SHA="7ccf046e3a1870eaca3a108e6f9565298aeeadf43732a2f83559542c03f5f17f"
CONTROL={"GuitarSet":(998,585,582),"URMP":(243,149,256)}
# FIXED search on TRAIN composition-heldout calibration, no DEV tuning.
POLICIES=[
 ("S614_original",BirthPolicy()),
 ("birth070_rise010",BirthPolicy(.70,.10)),
 ("birth080_rise010",BirthPolicy(.80,.10)),
 ("birth085_rise015",BirthPolicy(.85,.15)),
 ("birth090_rise015",BirthPolicy(.90,.15)),
 ("birth095_rise015",BirthPolicy(.95,.15)),
]

def partition_train(train):
    by=defaultdict(list)
    for s in train:by[(s["corpus"],s["composition"])].append(s)
    chosen={}
    for corpus,number_calib in (("GuitarSet",3),("URMP",5)):
        groups=sorted([k for k in by if k[0]==corpus],
                      key=lambda k:hashlib.sha256(
                          (k[0]+":"+k[1]).encode()).hexdigest())
        cal=set(groups[:number_calib])
        chosen[corpus]=dict(fit=[s for key in groups if key not in cal for s in by[key]],
                           calibration=[s for key in groups if key in cal for s in by[key]],
                           calibration_compositions=[k[1] for k in sorted(cal)])
    fit=sum((v["fit"] for v in chosen.values()),[])
    calibration=sum((v["calibration"] for v in chosen.values()),[])
    if (len(fit),len(calibration))!=(24,11):raise AssertionError("Unexpected compositional train split")
    return fit,calibration,chosen

def causal_counts(events,truth):
    reference=[(float(n["time"]),int(n["pitch"]),i) for i,n in enumerate(truth)
               if 40<=int(n["pitch"])<=76]
    detected=[(float(e["time"]),int(e["pitch"]),i) for i,e in enumerate(events)
              if e["type"]=="note_on" and 40<=int(e["pitch"])<=76]
    stats=causal_score(reference,detected,80.)
    return {k:stats[k] for k in ("TP","FP","FN","F1")}

def evaluate(row,model):
    result={}
    for name,policy in POLICIES:
        events,internal=decode_birth(row,model,policy)
        score,pairs=event_score(events,row["truth"])
        offset=offset_audit(events,row["truth"],pairs,row["n"]/44100.)
        result[name]=dict(metrics={k:score[k] for k in ("TP","FP","FN","F1")},
            causal=causal_counts(events,row["truth"]),
            attempted=int(internal["new_candidate_frames"]),
            emitted=int(internal["emitted_candidates"]),
            noteons=sum(e["type"]=="note_on" for e in events),
            noteoffs=sum(e["type"]=="note_off" for e in events),
            conditional_offset_accuracy=offset["conditional_offset_accuracy"])
        if result[name]["noteons"]!=result[name]["noteoffs"]:
            raise AssertionError("Missing NoteOff after birth model re-evaluation")
    return result

def summarise(records):
    totals={}
    for name,_ in POLICIES:
        totals[name]={}
        for corpus in ("GuitarSet","URMP"):
            selected=[r["policies"][name] for r in records if r["corpus"]==corpus]
            base=[r["policies"]["S614_original"] for r in records if r["corpus"]==corpus]
            counts={k:sum(a["metrics"][k] for a in selected)
                    for k in ("TP","FP","FN")}
            counts["F1"]=2*counts["TP"]/max(1,2*counts["TP"]+counts["FP"]+counts["FN"])
            causal={k:sum(a["causal"][k] for a in selected)
                    for k in ("TP","FP","FN")}
            causal["F1"]=2*causal["TP"]/max(1,2*causal["TP"]+causal["FP"]+causal["FN"])
            counts["causal_common_range"]=causal
            counts["attempted"]=sum(a["attempted"] for a in selected)
            counts["emitted"]=sum(a["emitted"] for a in selected)
            counts["sources_up"]=sum(a["metrics"]["F1"]>b["metrics"]["F1"]
                                    for a,b in zip(selected,base))
            counts["sources_down"]=sum(a["metrics"]["F1"]<b["metrics"]["F1"]
                                      for a,b in zip(selected,base))
            counts["original_noteoffs"]=sum(a["noteoffs"] for a in selected)
            totals[name][corpus]=counts
    return totals

def calibration_lock(results):
    base=results["S614_original"]
    accepted=[]
    for name,_ in POLICIES[1:]:
        delta=[results[name][c]["F1"]-base[c]["F1"] for c in ("GuitarSet","URMP")]
        fp=[results[name][c]["FP"]-base[c]["FP"] for c in ("GuitarSet","URMP")]
        if all(d>0 for d in delta) and all(f<=8 for f in fp) and all(
            results[name][c]["sources_down"]<=1 for c in ("GuitarSet","URMP")):
            accepted.append((min(delta),name))
    return max(accepted)[1] if accepted else "S614_original"

def main(out):
    if out.exists():raise FileExistsError("Never overwrite frozen S648 evidence")
    out.mkdir(parents=True)
    start=time.perf_counter()
    proto=json.loads((ROOT/"protocol.json").read_text())
    fixed=json.loads((ROOT/"selection_frozen.json").read_text())
    if digest(MODEL.read_bytes())!=S614_SHA or digest((ROOT/"protocol.json").read_bytes())!=fixed["protocol_sha256"]:
        raise ValueError("Original NOTE reference and source protocol changed")
    heads=load_heads(MODEL,expected_sha256=S614_SHA)
    selection=dict(heads.metadata["selection"])
    train_u=[s for s in proto["sources"] if s["split"]=="train" and s["corpus"]=="URMP"]
    train_g=pick_original_guitar_training(proto)
    dev=[s for s in proto["sources"] if s["split"]=="dev" and s["corpus"] in ("GuitarSet","URMP")]
    assert len(dev)==17 and len(train_u)==17 and len(train_g)==18
    identity=lambda items:{(s["corpus"],s["composition"]) for s in items}
    if identity(train_u+train_g)&identity(dev):raise AssertionError("Composition overlap to DEV")
    fit,calib,groups=partition_train(train_u+train_g)
    assert not identity(fit)&identity(calib)
    rng=np.random.default_rng(648)
    training=[]
    rows=[]
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/"data").mkdir()
        for key in ("GuitarSet_audio","GuitarSet_annotations"):
            item=proto["guitar_archives"][key]
            download(item["url"],root/"data"/(key+".zip"),
                     item["size"],item["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as zip:
            for spec in fit:
                record=get_source(spec,root,heads,zip if spec["corpus"]=="GuitarSet" else None)
                record["selection"]=selection
                x,y,cnt=sampled_training_events(record,rng)
                rows.append((x,y))
                training.append(dict(source=spec["name"],corpus=spec["corpus"],**cnt))
                print("S648 FIT",spec["name"],"positive",cnt["positive"],
                      "hard",cnt["hard"],flush=True)
            x=np.concatenate([a for a,b in rows]);y=np.concatenate([b for a,b in rows])
            assert 5000<len(x)<500000 and 200<y.sum()<len(y)-5000
            estimator=HistGradientBoostingClassifier(
                max_iter=105,max_leaf_nodes=16,max_depth=5,
                min_samples_leaf=65,learning_rate=.06,
                l2_regularization=8.,early_stopping=False,random_state=648)
            estimator.fit(x,y,sample_weight=np.where(y==1,3.,1.))
            checkpoint=out/"S648_birth_event_fit_train_only.joblib"
            joblib.dump(estimator,checkpoint,compress=3)
            sha=digest(checkpoint.read_bytes())
            print("S648 FIT CHECKPOINT",sha,"positive",int(y.sum()),
                  "rows",len(x),flush=True)
            del rows,x,y
            sections={}
            for phase,specs in (("CALIBRATION",calib),("DEV",dev)):
                evals=[]
                for spec in specs:
                    row=get_source(spec,root,heads,zip if spec["corpus"]=="GuitarSet" else None)
                    row["selection"]=selection
                    values=evaluate(row,estimator)
                    evals.append(dict(name=spec["name"],corpus=spec["corpus"],
                                      audio_sha256=row["source_sha256"],policies=values))
                    print("S648",phase,spec["name"],"S614",values["S614_original"]["metrics"]["F1"],
                          "birth85",values["birth085_rise015"]["metrics"]["F1"],flush=True)
                scores=summarise(evals)
                sections[phase]=dict(n_sources=len(specs),score=scores,
                                     by_source=evals)
                if phase=="CALIBRATION":
                    locked=calibration_lock(scores)
                    print("S648 TRAIN CALIBRATION LOCKED",locked,flush=True)
    benchmark=sections["DEV"]["score"]["S614_original"]
    for corpus in ("GuitarSet","URMP"):
        assert tuple(benchmark[corpus][v] for v in ("TP","FP","FN"))==CONTROL[corpus]
    output=dict(status="S648_TRUE_CAUSAL_EVENT_BIRTH_ONSET_HEAD",
         checkpoint_sha256=sha,original_S614_sha256=S614_SHA,
         input_features=training[0]["features"],
         training_fit_sources=training,calibration_compositions=groups,
         fit_sources=len(fit),calibration_sources=len(calib),DEV_sources=len(dev),
         lock_from_CALIBRATION_before_DEV=locked,
         policies=[dict(name=n,threshold=p.threshold,min_rise=p.min_rise) for n,p in POLICIES],
         metrics=sections,original_TEST_not_accessed=True,
         historical_DEV_not_independent=True,no_model_promotion=True,
         S614_S632_S80_unchanged=True,
         elapsed_seconds=round(time.perf_counter()-start,2))
    (out/"report.json").write_text(json.dumps(output,indent=2)+"\n")
    print("S648 FINAL VERDICT",json.dumps(dict(locked=locked,checkpoint_sha=sha,
      CALIBRATION={k:{c:{key:v[key] for key in ("TP","FP","FN","F1","emitted")}
                       for c,v in corp.items()} for k,corp in sections["CALIBRATION"]["score"].items()},
      DEV={k:{c:{key:v[key] for key in ("TP","FP","FN","F1","emitted")}
                    for c,v in corp.items()} for k,corp in sections["DEV"]["score"].items()}),indent=2),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    main(ap.parse_args().out)
