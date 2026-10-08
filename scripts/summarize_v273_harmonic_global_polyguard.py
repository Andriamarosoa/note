"""Train-only polyphonic safety gate on full native 59,309-event Exact-K.

Follow-up motivated by prior global diagnostic: corrections to K0/K1
raised overall exact at the expense of the polyphonic objective.
Three predeclared alternative policies are audited WITHOUT re-extracting
features, retuning the baseline, or evaluating player05/fold3.
This follow-up uses already-exposed folds, not independent confirmation.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scripts.yourmt3_exactk_common import FOLDS,metrics,paired,require
from scripts.summarize_v273_harmonic_global import load_cohort,load_features,get_matrix,choose_threshold

FEATURES={
  "harmonic_lifecycle":("features",("birth__","persistence__","damping__")),
  "harmonic_full":("features",tuple()),
  "fundamental_full":("fundamental",tuple()),
  "scrambled_full":("scrambled",tuple()),
}
POLICIES={
  "full7_keep_poly":np.arange(7,dtype=int),
  "poly_train5":np.arange(2,7,dtype=int),
  "poly_train3":np.arange(2,5,dtype=int),
}
def make_model():
    return make_pipeline(StandardScaler(),LogisticRegression(
        C=.1,solver="lbfgs",max_iter=3000,random_state=27402
    ))

def learn_scores(X,labels,base,fold,target,inner=False):
    # All val rows get predictions; train mask may be restricted by true label.
    fit=(np.isin(labels,target)) if int(target[0])>0 else np.ones(len(labels),bool)
    require(set(labels[fit])==set(target.tolist()),"training classes absent")
    m=make_model().fit(X[fit],labels[fit])
    require(np.array_equal(m[-1].classes_,target),"class ordering drift")
    p=m.predict_proba(X)
    best=target[np.argmax(p,axis=1)]
    basecol={int(c):i for i,c in enumerate(target)}
    if target[0]==0:
        # Argmax over 7 classes may favor K0/K1; those corrections are prohibited.
        allowed=(best>=2)
    else:
        allowed=np.ones(len(best),bool)
    prob_base=np.asarray([p[i,basecol[b]] for i,b in enumerate(base)])
    prop_prob=p[np.arange(len(best)),np.argmax(p,axis=1)]
    delta=prop_prob-prob_base
    proposed=np.where(allowed,best,base)
    return proposed,delta

def evaluate_policy(X,y,base,fold,target):
    gated=np.asarray(base).copy()
    raw=np.asarray(base).copy()
    logs={}
    for vf in FOLDS:
        tr=fold!=vf;va=fold==vf
        Y=y[tr];B=base[tr];Ft=fold[tr];Xt=X[tr]
        ip=np.full(len(Y),-1,int)
        im=np.full(len(Y),np.nan,float)
        for internal in FOLDS:
            iva=Ft==internal
            if not iva.any():continue
            it=~iva
            # Predictions must be made on excluded fold, not the fitting subset.
            trainmask=it & np.isin(Y,target)
            # Refit is intentionally performed on inner training only.
            m=make_model().fit(Xt[trainmask],Y[trainmask])
            require(np.array_equal(m[-1].classes_,target),"inner target classes")
            vprob=m.predict_proba(Xt[iva])
            pred=target[np.argmax(vprob,axis=1)]
            if target[0]==0:pred=np.where(pred>=2,pred,B[iva])
            base_index={int(c):i for i,c in enumerate(target)}
            mg=vprob[np.arange(int(iva.sum())),np.argmax(vprob,axis=1)]-np.asarray([
                vprob[i,base_index[b]] for i,b in enumerate(B[iva])])
            ip[iva]=pred;im[iva]=mg
        require(np.all(ip>=0) and np.isfinite(im).all(),"inner validation incomplete")
        cutoff=choose_threshold(Y,B,ip,im)
        eligible=tr & np.isin(y,target)
        clf=make_model().fit(X[eligible],y[eligible])
        require(np.array_equal(clf[-1].classes_,target),"outer fitted classes mismatch")
        vprob=clf.predict_proba(X[va])
        pred=target[np.argmax(vprob,axis=1)]
        if target[0]==0:pred=np.where(pred>=2,pred,base[va])
        cmap={int(c):i for i,c in enumerate(target)}
        margin=vprob[np.arange(int(va.sum())),np.argmax(vprob,axis=1)] - np.asarray([
            vprob[i,cmap[b]] for i,b in enumerate(base[va])])
        corrected=pred!=base[va]
        act=corrected&(margin>=cutoff["threshold"])
        raw[va]=pred
        gated[va]=np.where(act,pred,base[va])
        # Only K2..K6 proposals allowed; exact-K changes on K0/1 impossible.
        require(np.all((gated[va]==base[va])|(gated[va]>=2)),"poly-only violation")
        fold_net=int(np.sum(gated[va]==y[va])-np.sum(base[va]==y[va]))
        fold_poly_net=int(np.sum((gated[va]==y[va])&(y[va]>=2))-
                          np.sum((base[va]==y[va])&(y[va]>=2)))
        require(fold_net==fold_poly_net,"unexpected K0/K1 movement")
        logs[str(vf)]=dict(candidates=int(va.sum()),threshold=cutoff["threshold"],
                           inner_oof_net=cutoff["train_oof_net"],
                           changes=int(act.sum()),fold_net=fold_net,poly_net=fold_poly_net)
    return raw,gated,logs

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cohort",type=Path,required=True)
    p.add_argument("--input-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    require(not a.output.exists(),"refusing overwrite")
    y,base,idx,folds,mem,start=load_cohort(a.cohort)
    pos,rows,names,_=load_features(a.input_root,y,base,idx,folds,mem,start)
    yc=y[pos];bc=base[pos];fc=folds[pos]
    baseline=metrics(y,base)
    report=dict(status="completed",experiment="v273_native_poly_guard_followup",
        dataset_rows=len(y),eligible_rows=len(pos),original_baseline=baseline,
        policies={},caveats=[
          "Global K0..K6 measured over all 59309 original positions",
          "Application only on original baseline K2/K3/K4 predictions",
          "No proposal to K0/K1; global and poly net gains are therefore equal",
          "Train-only OOF gating per held-out fold, no labels used for eligibility",
          "Post-hoc follow-up after inspecting previous experiment: hypothesis generating only",
          "Player05/fold3 excluded; not a causally real-time model (+160ms context)",
          "No promotion of freeze_local_combo"
        ],automatic_promotion=False)
    saved=dict(global_index=idx,true_k=y,baseline=base,fold=folds)
    for family,(source,prefixes) in FEATURES.items():
        Xfeat=get_matrix(rows,names,source,prefixes)
        onehot=np.eye(3)[bc-2]
        X=np.column_stack([Xfeat,onehot])
        for pname,classes in POLICIES.items():
            key=family+"__"+pname
            raw,gated,logs=evaluate_policy(X,yc,bc,fc,classes)
            rp=base.copy();gp=base.copy()
            rp[pos]=raw;gp[pos]=gated
            gated_pair=paired(y,base,gp);raw_pair=paired(y,base,rp)
            gm=metrics(y,gp);rm=metrics(y,rp)
            require(gated_pair["global"]["net"]==gated_pair["poly"]["net"],"paired difference")
            require(gm["correct"]==baseline["correct"]+gated_pair["global"]["net"],"global difference mismatch")
            report["policies"][key]=dict(gated=gm,raw=rm,gated_paired=gated_pair,
                                          raw_paired=raw_pair,folds=logs,
                                          features=int(X.shape[1]))
            saved[key+"_gated"]=gp
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/"native-polyguard-predictions.npz",**saved)
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    lines=["# Poly-protected corrections — real native Exact-K (all 59,309 events)","",
       f"Untouched baseline **{100*baseline['exact']:.4f}%** global, **{100*baseline['poly']['exact']:.4f}%** poly.",
       "Every proposal remains K2 or higher; fold3/player05 excluded.",
       "",
       "| Family | Policy | Global Exact-K | Δ pp | Poly Exact-K | Δ pp | Corrections | Regressions | Fold2 net |",
       "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key,obj in report["policies"].items():
        fam,pol=key.split("__",1)
        gm=obj["gated"];q=obj["gated_paired"]["global"]
        lines.append(f"| {fam} | {pol} | {gm['exact']*100:.4f}% | {(gm['exact']-baseline['exact'])*100:+.4f} | {gm['poly']['exact']*100:.4f}% | {(gm['poly']['exact']-baseline['poly']['exact'])*100:+.4f} | {q['corrections']} | {q['regressions']} | {obj['folds']['2']['fold_net']:+d} |")
    lines += ["", "## Final caution"]+["- "+x for x in report["caveats"]]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)

if __name__=="__main__":main()
