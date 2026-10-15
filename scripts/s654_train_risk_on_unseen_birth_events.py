"""S654 trained risk-aware pitch selection: calibration nested by composition.

Frozen S648 birth head was FIT on 24 TRAIN recordings. Only its original
11 unseen TRAIN-CALIBRATION recordings may train the new risk expert.
Seven risk-fit recordings train logistic weights; four disjoint source
compositions tune a predeclared risk threshold; 17 already-exposed DEV
then measure corrections/regressions. Original TEST never opened.

Every variant runs the S652 one-to-one event audit before the next loop.
No promotion even if DEV improves: a genuinely new heldout cohort and
physical MIDI timing tests are still required.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import tempfile
import zipfile

import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from causal_note.strict_live import Selection,load_heads
from causal_note.s648_event_birth import BirthPolicy,decode_birth
from causal_note.s654_pitch_risk_selector import (
    RISK_THRESHOLDS,risk_training_examples,decode_with_specialist
)
from causal_note.s652_event_round_audit import (
    audit_event_round,require_event_audit_before_next_round
)
from scripts.s648_train_birth_event import partition_train
from scripts.s623_mixed_source_event_head import pick_original_guitar_training
from scripts.s626_train_sharp_native_onset import get_source
from scripts.s616_live_dev_ablation import ROOT,MODEL
from scripts.s613_prepare_strict_live import download,digest
from scripts.s613_train_strict_live import event_score

BASE_SHA="7ccf046e3a1870eaca3a108e6f9565298aeeadf43732a2f83559542c03f5f17f"
BIRTH_SHA="7613dca4aa9bc5992c96a75ab55f714d8720c55b6880ee9e223e39ea880eb4e9"
VARIANTS=("S614_original","S648_broad_birth",
          "S654_risk035","S654_risk055","S654_risk075","S654_risk090")


def partition_calibration(specs):
    by=defaultdict(list)
    for s in specs:by[(s["corpus"],s["composition"])].append(s)
    fit=[]; lock=[]
    for corpus,hold_count in (("GuitarSet",1),("URMP",2)):
        names=sorted((k for k in by if k[0]==corpus),
                     key=lambda k:hashlib.sha256(
                         ("S654:"+k[0]+":"+k[1]).encode()).hexdigest())
        hold=set(names[:hold_count])
        for k in names:
            (lock if k in hold else fit).extend(by[k])
    if (len(fit),len(lock))!=(8,3):
        raise AssertionError(f"Unexpected nested S654 risk selection split {(len(fit),len(lock))}")
    if {(s["corpus"],s["composition"]) for s in fit} & {
            (s["corpus"],s["composition"]) for s in lock}:
        raise AssertionError("Calibrator saw risk-lock compositions")
    if not any(s["corpus"]=="GuitarSet" for s in lock) or not any(s["corpus"]=="URMP" for s in lock):
        raise AssertionError("Both corpora necessary in lock set")
    return fit,lock


def risk_lock_policy(aggregate):
    base=aggregate["S614_original"]
    qualified=[]
    for name in VARIANTS[2:]:
        if all(aggregate[name][corpus]["F1"]>base[corpus]["F1"] and
               aggregate[name][corpus]["FP"]-base[corpus]["FP"]<=3 and
               aggregate[name][corpus]["regressed_sources"]<=1
               for corpus in ("GuitarSet","URMP")):
            score=min(aggregate[name][corpus]["F1"]-base[corpus]["F1"]
                      for corpus in ("GuitarSet","URMP"))
            qualified.append((score,name))
    return max(qualified)[1] if qualified else "S614_original"


def evaluate_one(row,model,risk,name,source,corpus,save_details):
    original,_=decode_birth(row,model,BirthPolicy())
    if name=="S614_original":
        proposed=original; state=dict(proposed_frames=0,risk_rejected=0,risk_accepted_frames=0,actual_new_notes=0)
    elif name=="S648_broad_birth":
        proposed,details=decode_birth(row,model,BirthPolicy(.70,.10))
        state=dict(proposed_frames=int(details["new_candidate_frames"]),
                   risk_rejected=0,risk_accepted_frames=int(details["new_candidate_frames"]),
                   actual_new_notes=int(details["emitted_candidates"]))
    else:
        floor=float(name.split("risk")[1])/1000.
        proposed,state=decode_with_specialist(row,model,risk,floor)
    proof=audit_event_round(source,corpus,row["truth"],original,proposed)
    raw,_=event_score(proposed,row["truth"])
    score={k:float(raw[k]) for k in ("TP","FP","FN","F1")}
    result=dict(name=source,corpus=corpus,
        score=score,state=state,
        event_proof=dict(corrected=proof["matched_reference_partition"]["corrected"],
          lost=proof["matched_reference_partition"]["lost"],
          delta_TP=proof["delta"]["TP"],delta_FP=proof["delta"]["FP"],
          by_K=proof["classifications"],
          false_event_types=proof["false_note_diagnostic_counts"]))
    if save_details:
        result["event_identities"]=[p for p in proof["exact_reference_labels"]
          if p["status"] in ("newly_correct","regressed_true_note")]
        result["new_false_events"]=proof["candidate_false_notes"]
    return result


def aggregate(records):
    scored={}
    for name in VARIANTS:
        scored[name]={}
        for corpus in ("GuitarSet","URMP"):
            xs=[s["variants"][name] for s in records if s["corpus"]==corpus]
            tp=sum(int(x["score"]["TP"]) for x in xs)
            fp=sum(int(x["score"]["FP"]) for x in xs)
            fn=sum(int(x["score"]["FN"]) for x in xs)
            base=[s["variants"]["S614_original"] for s in records if s["corpus"]==corpus]
            scored[name][corpus]=dict(TP=tp,FP=fp,FN=fn,
                F1=2*tp/max(1,2*tp+fp+fn),
                regressed_sources=sum(x["score"]["F1"]<b["score"]["F1"]-1e-10
                                      for x,b in zip(xs,base)),
                improved_sources=sum(x["score"]["F1"]>b["score"]["F1"]+1e-10
                                     for x,b in zip(xs,base)),
                verified_new_truths=sum(x["event_proof"]["corrected"] for x in xs),
                verified_lost_truths=sum(x["event_proof"]["lost"] for x in xs),
                rejected_candidates=sum(x["state"]["risk_rejected"] for x in xs),
                emitted_assists=sum(x["state"]["actual_new_notes"] for x in xs))
            if scored[name][corpus]["verified_new_truths"]-scored[name][corpus]["verified_lost_truths"]!=tp-sum(
              int(b["score"]["TP"]) for b in base):
                raise AssertionError("S654 net TP masks unmatched event identities")
    return scored


def main(checkpoint,out):
    if out.exists():raise FileExistsError("Cannot overwrite S654 evidence")
    out.mkdir(parents=True)
    if digest(checkpoint.read_bytes())!=BIRTH_SHA:
        raise AssertionError("Original S648 birth checkpoint changed")
    birth=joblib.load(checkpoint)
    proto=json.loads((ROOT/"protocol.json").read_text())
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    assert digest(MODEL.read_bytes())==BASE_SHA==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    native=load_heads(MODEL,expected_sha256=BASE_SHA)
    sources=[s for s in proto["sources"] if s["split"]=="train" and s["corpus"]=="URMP"]
    sources+=pick_original_guitar_training(proto)
    dev=[s for s in proto["sources"] if s["split"]=="dev" and s["corpus"] in ("GuitarSet","URMP")]
    assert len(sources)==35 and len(dev)==17
    earlier_fit,calibration,_=partition_train(sources)
    assert len(earlier_fit)==24 and len(calibration)==11
    risk_fit,risk_lock=partition_calibration(calibration)
    identity=lambda rows:{(s["corpus"],s["composition"]) for s in rows}
    assert not identity(risk_fit+risk_lock)&identity(earlier_fit+dev)
    training_rows=[];samples=[];labels=[];phases={}
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/"data").mkdir()
        for name in ("GuitarSet_audio","GuitarSet_annotations"):
            spec=proto["guitar_archives"][name]
            download(spec["url"],root/"data"/(name+".zip"),spec["size"],spec["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as archive:
            def load(spec):
                row=get_source(spec,root,native,
                        archive if spec["corpus"]=="GuitarSet" else None)
                row["selection"]=dict(native.metadata["selection"])
                return row
            for spec in risk_fit:
                row=load(spec)
                xx,yy,events=risk_training_examples(row,birth)
                if len(xx):samples.append(xx);labels.append(yy)
                training_rows.append(dict(source=spec["name"],corpus=spec["corpus"],
                    candidates=len(yy),novel_matched=int(yy.sum()),
                    hard_false=int((yy==0).sum())))
                print("S654 RISK FIT",spec["name"],"candidate_events",len(yy),
                     "new truths",int(yy.sum()),flush=True)
            x=np.concatenate(samples);y=np.concatenate(labels)
            if len(y)<100 or y.sum()<3 or (y==0).sum()<15:
                raise AssertionError("Insufficient out-of-fit event supervision")
            # Class-balanced logistic is a specialist *not* a replacement of
            # any of S614's existing neural or tree heads.
            risk=make_pipeline(StandardScaler(),
                LogisticRegression(C=.05,class_weight="balanced",
                                   max_iter=450,random_state=654))
            risk.fit(x,y)
            ckpt=out/"S654_specialist_risk_fit_only.joblib"
            joblib.dump(risk,ckpt,compress=3)
            risk_sha=digest(ckpt.read_bytes())
            print("S654 RISK TRAINED",risk_sha,
                  "train_emissions",len(y),"train_true_new_notes",int(y.sum()),flush=True)
            for stage,specs in (("RISK_LOCK",risk_lock),("DEV",dev)):
                rows=[]
                for spec in specs:
                    row=load(spec)
                    variants={}
                    # No candidate is evaluated without a full S652 audit.
                    for name in VARIANTS:
                        variants[name]=evaluate_one(row,birth,risk,name,
                                    spec["name"],spec["corpus"],save_details=True)
                        print("S654 AUDIT",stage,spec["name"],name,
                          "gain",variants[name]["event_proof"]["corrected"],
                          "lost",variants[name]["event_proof"]["lost"],
                          "FP",variants[name]["event_proof"]["delta_FP"],flush=True)
                    rows.append(dict(name=spec["name"],corpus=spec["corpus"],
                                    audio_sha256=row["source_sha256"],variants=variants))
                metrics=aggregate(rows)
                phases[stage]=dict(n_sources=len(specs),metrics=metrics,by_source=rows)
                if stage=="RISK_LOCK":
                    choice=risk_lock_policy(metrics)
                    print("S654 TRAIN ONLY FROZEN CHOICE",choice,flush=True)
    for corpus,checks in (("GuitarSet",(998,585,582)),("URMP",(243,149,256))):
        assert tuple(phases["DEV"]["metrics"]["S614_original"][corpus][k]
                     for k in ("TP","FP","FN"))==checks
    report=dict(status="S654_LEARNED_S648_BIRTH_PITCH_RISK_SPECIALIST",
        original_S614_S80_S632_unchanged=True,
        original_S614_sha256=BASE_SHA,
        original_S648_birth_sha256=BIRTH_SHA,
        trained_S654_risk_sha256=risk_sha,
        S648_fit_sources=24,S654_risk_fit_sources=len(risk_fit),
        S654_risk_lock_sources=len(risk_lock),dev_sources=len(dev),
        training_evidence=training_rows,training_positive=int(y.sum()),
        training_negative=int((y==0).sum()),
        predeclared_risk_thresholds=list(RISK_THRESHOLDS),
        frozen_choice_from_risk_lock_only=choice,
        results=phases,TEST_never_read=True,
        historical_exposed_DEV_not_new_holdout=True,
        no_model_promotion=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S654 FINAL",json.dumps(dict(risk_head_sha=risk_sha,
        choice_locked_on_TRAIN_only=choice,
        train=training_rows,
        RISK_LOCK=phases["RISK_LOCK"]["metrics"],
        DEV=phases["DEV"]["metrics"]),indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--birth-checkpoint",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    main(a.birth_checkpoint,a.out)
