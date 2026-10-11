"""S121 compare unlabelled prediction stability of three unverified 527D S4/S5 inputs.

All ten original models are frozen, all 57 GOAT waveforms were processed
upstream without truth, and no tuning, calibration, or accuracy computation
is performed. Cross-variant agreement is NOT evidence of feature equivalence.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import joblib,numpy as np
from scripts.s120_s4_s5_416_layout_audit import variant_sets, NAMES
from scripts.s105_s5_frozen_forward import load as s5_load

def need(x,msg):
    if not x:raise ValueError(msg)
def ensemble(models,X):
    vals=np.stack([model.predict_proba(X) for model in models]).astype(np.float64)
    need(vals.shape==(len(models),len(X),7) and np.isfinite(vals).all(),
         "model probability contract")
    need(np.allclose(vals.sum(axis=2),1,atol=1e-5),"normalization drift")
    return vals
def stats(p,baseline):
    mean=p.mean(0)
    votes=p.argmax(-1)
    pred=mean.argmax(-1)
    distribution=np.bincount(pred,minlength=7).tolist()
    agreement=(votes==pred[None]).mean(0)
    entropy=-(mean*np.log(np.maximum(mean,1e-12))).sum(-1)
    return {"K_histogram":distribution,
            "changed_vs_frozen":int((pred!=baseline).sum()),
            "K0_rate":float(np.mean(pred==0)),
            "unanimity_rate":float(np.mean(agreement==1)),
            "average_vote_consensus":float(agreement.mean()),
            "average_predictive_entropy":float(entropy.mean()),
            "average_top1_probability":float(mean.max(-1).mean())},pred
def run(s97,s116,s4root,s5root,out):
    need(not out.exists(),"immutable output required")
    with np.load(s116,allow_pickle=False) as z:
        baseline=z["freeze_K"].astype(np.int8).copy()
        control=z["control111"].copy()
        keys0=z["row_id"].astype(str)
    keys,Xs=variant_sets(s97,s116)
    need(np.array_equal(keys0,keys) and control.shape==(57,111),"event mismatch")
    models={}
    ctrl={}
    for poly in (1,2):
        a=[];b=[]
        for fold in (0,1,2):
            s4=s4root/f"fold{fold}__owned__poly{poly}.joblib"
            state=joblib.load(s4)
            need(state["held_fold"]==fold and state["family"]==f"owned__poly{poly}",
                 "wrong original owned S4 weights")
            a.append(state["model"])
            c=joblib.load(s4root/f"fold{fold}__control__poly{poly}.joblib")
            need(c["family"]==f"control__poly{poly}", "wrong S4 control family")
            b.append(c["model"])
        models[f"S4_poly{poly}"]=a
        ctrl[f"S4_poly{poly}"]=ensemble(b,control)
    models["S5"]=s5_load(s5root)
    result={"status":"S121_UNLABELLED_S4_S5_FORWARD_STABILITY",
      "event_count":57,"variants":{},"S4_control_reference":{},
      "variant_pairwise_disagreement":{},"all_audio_inputs_fixed":True,
      "ground_truth_used":False,"accuracy_measured":False,
      "feature_equivalence_proven":False,"S80_promoted":False}
    control_pred={}
    for name,p in ctrl.items():
        report, pred=stats(p,baseline)
        result["S4_control_reference"][name]=report
        control_pred[name]=pred
    preds={}
    for name,X in Xs.items():
        item={}
        for fam,clfs in models.items():
            report,pred=stats(ensemble(clfs,X),baseline)
            report["changed_vs_S4_control"]=int((pred!=control_pred[fam]).sum()) if fam.startswith("S4") else None
            item[fam]=report
            preds[(name,fam)]=pred
        result["variants"][name]=item
    for i,va in enumerate(NAMES):
        for vb in NAMES[i+1:]:
            differences={}
            for fam in models:
                differences[fam]=int(np.count_nonzero(preds[(va,fam)]!=preds[(vb,fam)]))
            result["variant_pairwise_disagreement"][va+"__"+vb]=differences
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    return result
def selftest():
    zero=np.array([[1,0,0,0,0,0,0]]*3,dtype=np.float64)
    report,k=stats(zero[None],np.zeros(3,dtype=np.int8))
    need(report["unanimity_rate"]==1 and report["K0_rate"]==1
         and (k==0).all(),"basic unanimity metric")
    print("S121 PASS: distribution, consensus, entropy and label-free metrics")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("s97","s116","s4-models","s5-models","out"):
        p.add_argument("--"+name,type=Path)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        need(all(x is not None for x in
             (a.s97,a.s116,a.s4_models,a.s5_models,a.out)), "missing evidence")
        run(a.s97,a.s116,a.s4_models,a.s5_models,a.out)
