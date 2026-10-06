"""LOFO validation of the low-register / low-confidence ambiguity signature.

Training/evaluation cohort:
  estimated register == low
  0.50 <= frozen residual-LR p(K2) < 0.70
  true K in {2,3}

Predeclared inference-safe signature:
  - minimum low-order harmonic-relation error among selected triplet F0s
  - selected-triplet F0 span (cents)
  - weakest/second amplitude ratio amp3_over_amp2

Validation:
  balanced logistic C=0.1, leave-one-fold-out over folds 0/1/2/4,
  fixed decision threshold p(K3)>=0.5, no threshold search.

Policy test:
  baseline residual action is frozen p(K2)>=0.60.
  Only within register=low and 0.60<=p(K2)<0.65, abstain from 3->2 when the
  LOFO ambiguity model predicts p(K3)>=0.5.

No outer fold 3. No automatic promotion.
"""
from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.audit_v273_low_band_fold2_vs_fold4 import inference_features
from scripts.v273_window_experiment import load_bundle
from scripts.v273_residual_audit import require

FOLDS=(0,1,2,4)
TRAIN_P0=.50
TRAIN_P1=.70
ACTION_P0=.60
ACTION_P1=.65
NAMES=("geom_harmonic_relation_min_error","geom_span_cents","amp3_over_amp2")

def model():
    return make_pipeline(
      StandardScaler(),
      LogisticRegression(C=.1,max_iter=3000,class_weight="balanced",random_state=27391)
    )

def load_rows(root):
    rows=[]
    for fold in FOLDS:
        p=Path(root)/f"fold-{fold}"/"replay.npz"
        with np.load(p,allow_pickle=False) as z:a={k:np.asarray(z[k]) for k in z.files}
        action=a["val_b_low"].astype(bool)&(a["val_base_k"].astype(int)==3)
        valid=a["val_valid"].astype(bool)
        ids=a["val_action_ids"][valid].astype(int)
        y=a["val_true_k"].astype(int)[action][valid]
        prob=a["val_probability"].astype(float)
        reg=np.asarray(a["val_register"]).astype(str)
        rec=a["val_recording"].astype(str)[action][valid]
        require(len(ids)==len(y)==len(prob)==len(reg)==len(rec),"length drift")
        for i in range(len(y)):
            rows.append({"fold":fold,"row_id":int(ids[i]),"true_k":int(y[i]),
                         "probability":float(prob[i]),"register":str(reg[i]),
                         "recording_id":str(rec[i])})
    return rows

def add_features(rows,bundle,config,dataset):
    cache,_,_=load_bundle(bundle,config)
    wanted_rows=[r for r in rows if r["register"]=="low" and TRAIN_P0<=r["probability"]<TRAIN_P1]
    require(len(wanted_rows)>=20,"small low-confidence cohort")
    wanted={r["recording_id"] for r in wanted_rows}
    tracks={t.annotation_member:t for t in index_guitarset(dataset) if t.annotation_member in wanted}
    require(set(tracks)==wanted,"audio coverage")
    by=defaultdict(list)
    for r in wanted_rows:
        require(str(cache["members"][r["row_id"]])==r["recording_id"],"recording mismatch")
        by[r["recording_id"]].append(r)
    for member in sorted(by):
        au=decode_pcm16_mono_wav(tracks[member].audio_zip,tracks[member].audio_member)
        samples=np.asarray(au.samples,float)/32768.
        for r in by[member]:
            start=int(cache["cluster_start_samples"][r["row_id"]])
            r["features"]=inference_features(samples,start)
        print(json.dumps({"recording":member,"rows":len(by[member])}),flush=True)
    require(all("features" in r for r in wanted_rows),"feature coverage")
    return wanted_rows

def counts(rows,mask):
    y=np.asarray([r["true_k"] for r in rows],int);m=np.asarray(mask,bool)
    c=int(np.sum(m&(y==2)));g=int(np.sum(m&(y==3)));o=int(np.sum(m&~np.isin(y,(2,3))))
    return {"actions":int(m.sum()),"corrections":c,"regressions":g,"other_k":o,"net":c-g}

def main():
    ap=argparse.ArgumentParser()
    for n in ("internal-exports","bundle","config","dataset","output"):
        ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args();require(not a.output.exists(),"overwrite")

    rows=load_rows(a.internal_exports)
    feat_rows=add_features(rows,a.bundle,a.config,a.dataset)
    train_rows=[r for r in feat_rows if r["true_k"] in (2,3)]
    require(len(train_rows)>=30,"small K2/K3 cohort")
    require(all(any(r["fold"]==f and r["true_k"]==2 for r in train_rows) for f in FOLDS),"missing K2 fold")
    require(all(any(r["fold"]==f and r["true_k"]==3 for r in train_rows) for f in FOLDS),"missing K3 fold")

    # LOFO predictions for every low-confidence K2/K3 row.
    pred={}
    per_fold_auc=[]
    for f in FOLDS:
        fit=[r for r in train_rows if r["fold"]!=f]
        val=[r for r in train_rows if r["fold"]==f]
        X=np.asarray([[r["features"][n] for n in NAMES] for r in fit],float)
        y=np.asarray([r["true_k"]==3 for r in fit],int)
        require(len(np.unique(y))==2,"fit class collapse")
        m=model();m.fit(X,y)
        Xv=np.asarray([[r["features"][n] for n in NAMES] for r in val],float)
        pv=m.predict_proba(Xv)[:,1]
        yv=np.asarray([r["true_k"]==3 for r in val],int)
        auc=float(roc_auc_score(yv,pv))
        per_fold_auc.append({"fold":f,"rows":len(val),"K2":int(np.sum(yv==0)),"K3":int(np.sum(yv==1)),"auc":auc})
        for r,pv1 in zip(val,pv):
            pred[(r["fold"],r["row_id"])]=float(pv1)

    yall=np.asarray([r["true_k"]==3 for r in train_rows],int)
    pall=np.asarray([pred[(r["fold"],r["row_id"])] for r in train_rows],float)
    global_auc=float(roc_auc_score(yall,pall))

    # Apply only inside the predeclared low-register .60-.65 action band.
    baseline=np.asarray([r["probability"]>=ACTION_P0 for r in rows],bool)
    abstain=np.zeros(len(rows),bool)
    for i,r in enumerate(rows):
        if not baseline[i]:continue
        if r["register"]!="low" or not (ACTION_P0<=r["probability"]<ACTION_P1):continue
        key=(r["fold"],r["row_id"])
        if key not in pred:
            # Only K2/K3 rows have LOFO probabilities. Other-K is left untouched,
            # so this policy evaluation cannot gain by label-dependent abstention.
            continue
        abstain[i]=pred[key]>=.5
    final=baseline&~abstain

    base_all=counts(rows,baseline); abst_all=counts(rows,abstain); final_all=counts(rows,final)
    per=[]
    for f in FOLDS:
        fm=np.asarray([r["fold"]==f for r in rows],bool)
        b=counts(rows,baseline&fm);ab=counts(rows,abstain&fm);fin=counts(rows,final&fm)
        per.append({"fold":f,"baseline":b,"abstained":ab,"abstention_gain":ab["regressions"]-ab["corrections"],"final":fin})

    # Band-level confusion for interpretation.
    band=[r for r in train_rows if ACTION_P0<=r["probability"]<ACTION_P1]
    band_pred=np.asarray([pred[(r["fold"],r["row_id"])]>=.5 for r in band],bool)
    band_y=np.asarray([r["true_k"] for r in band],int)
    band_report={
      "rows":len(band),"K2":int(np.sum(band_y==2)),"K3":int(np.sum(band_y==3)),
      "predicted_K3":int(np.sum(band_pred)),
      "K3_caught":int(np.sum(band_pred&(band_y==3))),
      "K2_sacrificed":int(np.sum(band_pred&(band_y==2))),
    }

    rep={"status":"completed","experiment":"v273_low_confidence_harmonic_signature_lofo",
         "training_cohort":{"register":"low","probability_band":[TRAIN_P0,TRAIN_P1],"rows":len(train_rows),
                            "K2":sum(r["true_k"]==2 for r in train_rows),"K3":sum(r["true_k"]==3 for r in train_rows)},
         "features":list(NAMES),"classifier":"StandardScaler + balanced LogisticRegression(C=0.1)",
         "decision_threshold_K3":0.5,"global_oof_auc":global_auc,"per_fold_auc":per_fold_auc,
         "policy":{"baseline":"residual p(K2)>=0.60","abstain_band":"low register and 0.60<=p(K2)<0.65",
                   "abstain_if":"LOFO p(K3)>=0.5"},
         "band_report":band_report,"baseline":base_all,"abstained":abst_all,"final":final_all,"per_fold":per,
         "strict_pass":{"global_not_worse":final_all["net"]>=base_all["net"],
                        "all_folds_nonnegative":all(q["final"]["net"]>=0 for q in per),
                        "fold2_nonnegative":next(q for q in per if q["fold"]==2)["final"]["net"]>=0,
                        "fold4_not_worse":next(q for q in per if q["fold"]==4)["final"]["net"]>=next(q for q in per if q["fold"]==4)["baseline"]["net"]},
         "outer_fold_3_used":False,"prediction_changes":False,"threshold_search":False,"automatic_promotion":False}

    a.output.mkdir(parents=True)
    (a.output/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
    lines=["# Low-confidence harmonic signature — LOFO validation","",
           f"Training cohort: **{len(train_rows)}** low-register K2/K3 rows with 0.50<=p<0.70.",
           f"Global OOF AUC: **{global_auc:.3f}**.","",
           "## LOFO AUC","","| fold | rows | K2 | K3 | AUC |","|---:|---:|---:|---:|---:|"]
    for q in per_fold_auc:
        lines.append(f"| {q['fold']} | {q['rows']} | {q['K2']} | {q['K3']} | {q['auc']:.3f} |")
    lines+=["","## Policy effect","","| fold | baseline net | abstained K2 | abstained K3 | gain from abstention | final net |",
            "|---:|---:|---:|---:|---:|---:|"]
    for q in per:
        ab=q["abstained"]
        lines.append(f"| {q['fold']} | {q['baseline']['net']:+d} | {ab['corrections']} | {ab['regressions']} | {q['abstention_gain']:+d} | {q['final']['net']:+d} |")
    lines+=["",
            f"Global baseline: **{base_all['net']:+d}**; final: **{final_all['net']:+d}**.",
            f"Strict pass: **{rep['strict_pass']}**.",
            "","No outer fold 3 and no automatic promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))
if __name__=="__main__":main()
