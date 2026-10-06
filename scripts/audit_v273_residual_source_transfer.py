"""Residual source transfer audit: internal OOF vs exposed outer fold 3.

Compares the *raw* pair/triplet residual corrector before any post-hoc veto.

Internal:
- frozen OOF validation exports from run 37356100423, folds 0/1/2/4.

Outer:
- rebuild exactly the pooled-internal residual classifier used in outer run
  37536875343, with frozen robust base and B_low reconstruction.

Reports:
- candidate K2/K3 class mix
- joint and univariate AUCs
- threshold-0.5 TPR(K2 correction), FPR(K3 regression), action precision and net
- score/feature medians
- estimated-register slices
- counterfactual decomposition of whether the outer gain is caused by class mix
  or by better conditional action rates.

Fold 3 is historically exposed. Diagnostic only, no tuning/promotion.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.evaluate_v273_outer_residual_two_stage_p060 import (
    build_robust_model,reconstruct_blow,load_npz
)
from scripts.audit_v273_internal_b_like_boundary_corrector import predict,structural_features
from scripts.audit_v273_internal_b_low_harmonic_strata import audio_rows
from scripts.v273_residual_audit import extracted_matrix,fit_classifier
from scripts.v273_window_experiment import load_bundle,require

FEATURES=("best_pair_residual_ratio","best_triplet_residual_ratio")

def load_internal(root):
    ys=[];Xs=[];ps=[];f0s=[];folds=[]
    for fold in (0,1,2,4):
        p=Path(root)/f"fold-{fold}"/"replay.npz"
        require(p.exists(),f"missing {p}")
        with np.load(p,allow_pickle=False) as z:a={k:np.asarray(z[k]) for k in z.files}
        action=a["val_b_low"].astype(bool)&(a["val_base_k"].astype(int)==3)
        valid=a["val_valid"].astype(bool)
        y=a["val_true_k"].astype(int)[action][valid]
        X=a["val_X"].astype(float)[valid]
        pr=a["val_probability"].astype(float)
        f0=a["val_f0"].astype(float)[valid]
        require(len(y)==len(X)==len(pr)==len(f0),"internal export length mismatch")
        ys.append(y);Xs.append(X);ps.append(pr);f0s.append(f0);folds.append(np.full(len(y),fold,int))
    return np.concatenate(ys),np.vstack(Xs),np.concatenate(ps),np.concatenate(f0s),np.concatenate(folds)

def stats(y,X,p,f0,cuts=None):
    y=np.asarray(y,int);X=np.asarray(X,float);p=np.asarray(p,float);f0=np.asarray(f0,float)
    k23=np.isin(y,(2,3)); yy=y[k23]; xx=X[k23]; pp=p[k23]; ff=f0[k23]
    require(np.any(yy==2) and np.any(yy==3),"binary collapse")
    applied=pp>=.5
    n2=int(np.sum(yy==2));n3=int(np.sum(yy==3))
    c=int(np.sum(applied&(yy==2)));r=int(np.sum(applied&(yy==3)))
    tpr=c/n2;fpr=r/n3
    def q(v):return {"mean":float(np.mean(v)),"median":float(np.median(v)),
                     "q25":float(np.quantile(v,.25)),"q75":float(np.quantile(v,.75))}
    cls={}
    for k in (2,3):
        m=yy==k
        cls[str(k)]={"n":int(m.sum()),"p":q(pp[m]),
                     FEATURES[0]:q(xx[m,0]),FEATURES[1]:q(xx[m,1]),
                     "f0":q(ff[m])}
    joint=float(roc_auc_score((yy==2).astype(int),pp))
    # Orient each residual according to internal semantic K2 direction by actual class means.
    univ={}
    for j,n in enumerate(FEATURES):
        sign=1 if np.mean(xx[yy==2,j])>=np.mean(xx[yy==3,j]) else -1
        univ[n]={"orientation":int(sign),
                 "auc_oriented":float(roc_auc_score((yy==2).astype(int),sign*xx[:,j]))}
    if cuts is None:
        cuts=np.quantile(ff[np.isfinite(ff)&(ff>0)],[1/3,2/3])
    labels=np.full(len(ff),"unassigned",dtype=object)
    ok=np.isfinite(ff)&(ff>0)
    labels[ok&(ff<cuts[0])]="low"
    labels[ok&(ff>=cuts[0])&(ff<cuts[1])]="mid"
    labels[ok&(ff>=cuts[1])]="high"
    reg={}
    for name in ("low","mid","high","unassigned"):
        m=labels==name
        if not np.any(m):
            reg[name]={"n":0};continue
        ma=applied&m
        cc=int(np.sum(ma&(yy==2)));rr=int(np.sum(ma&(yy==3)))
        k2=int(np.sum(m&(yy==2)));k3=int(np.sum(m&(yy==3)))
        reg[name]={"n":int(m.sum()),"K2":k2,"K3":k3,
                   "actions":int(ma.sum()),"corrections":cc,"regressions":rr,
                   "net":cc-rr,
                   "auc":float(roc_auc_score((yy[m]==2).astype(int),pp[m])) if k2 and k3 else None}
    return {
      "K2":n2,"K3":n3,"K2_prior":n2/(n2+n3),
      "joint_auc":joint,"univariate":univ,"class_stats":cls,
      "threshold_0p5":{"corrections":c,"regressions":r,"net":c-r,
                       "TPR_K2":tpr,"FPR_K3":fpr,
                       "action_precision_K2":c/max(1,c+r),
                       "action_rate":(c+r)/(n2+n3)},
      "register_cuts_hz":[float(cuts[0]),float(cuts[1])],"registers":reg
    },np.asarray(cuts,float)

def threshold_sweep(y,p):
    y=np.asarray(y,int);p=np.asarray(p,float);m=np.isin(y,(2,3));y=y[m];p=p[m]
    out={}
    for th in (.5,.6,.7,.8,.9):
        a=p>=th;c=int(np.sum(a&(y==2)));r=int(np.sum(a&(y==3)))
        out[str(th)]={"corrections":c,"regressions":r,"net":c-r,"actions":int(a.sum())}
    return out

def main():
    ap=argparse.ArgumentParser()
    for n in ("internal-exports","bundle","config","uniform-weights","freeze-weights",
              "outer-group-predictions","outer-b-low-predictions","dataset","output"):
        ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args();require(not a.output.exists(),"overwrite")

    yi,Xi,pi,f0i,foldi=load_internal(a.internal_exports)
    internal,cuts=stats(yi,Xi,pi,f0i)
    # Frozen OOF source action identity.
    require(internal["threshold_0p5"]["corrections"]==108,"internal corr drift")
    require(internal["threshold_0p5"]["regressions"]==125,"internal reg drift")

    cache,parts,_=load_bundle(a.bundle,a.config)
    fit=np.asarray(parts["final_fit"],np.int64);outer=np.asarray(parts["outer"],np.int64)
    yall=np.minimum(cache["exact"].astype(np.int32),6);yf=yall[fit];yo=yall[outer]
    model=build_robust_model(a.uniform_weights,a.freeze_weights)
    Pf,Gf=predict(model,cache,fit);Po,Go=predict(model,cache,outer)
    frozen_group=load_npz(a.outer_group_predictions)
    np.testing.assert_array_equal(frozen_group["global_index"],outer)
    np.testing.assert_array_equal(frozen_group["predicted"],Go)
    Xf=structural_features(cache,fit,Pf);Xo=structural_features(cache,outer,Po)
    blow_f,blow_o,_,_,_,_=reconstruct_blow(Xf,Xo,yf,Gf)
    frozen_b=load_npz(a.outer_b_low_predictions)
    np.testing.assert_array_equal(frozen_b["global_index"],outer)
    require(np.array_equal(frozen_b["B_low_mask"].astype(bool),blow_o&np.isin(Go,(2,3,4))),"B_low replay drift")

    cand_f=blow_f&(Gf==3);cand_o=blow_o&(Go==3)
    idsf=fit[cand_f];idso=outer[cand_o]
    rf=audio_rows(cache,idsf,a.dataset);ro=audio_rows(cache,idso,a.dataset)
    Xrf,f0rf,vf,_=extracted_matrix(rf);Xro,f0ro,vo,_=extracted_matrix(ro)
    yfc=yf[cand_f];yoc=yo[cand_o]
    train=vf&np.isin(yfc,(2,3))
    residual=fit_classifier(Xrf[train],(yfc[train]==2).astype(int))
    po=np.full(len(idso),np.nan);po[vo]=residual.predict_proba(Xro[vo])[:,1]
    yo_valid=yoc[vo];Xo_valid=Xro[vo];f0o=f0ro[vo];po_valid=po[vo]

    outer_stats,_=stats(yo_valid,Xo_valid,po_valid,f0o,cuts)
    require(outer_stats["threshold_0p5"]["corrections"]==54,"outer corr drift")
    require(outer_stats["threshold_0p5"]["regressions"]==40,"outer reg drift")

    ir=internal["threshold_0p5"];orr=outer_stats["threshold_0p5"]
    # Counterfactual nets isolate class-mix vs conditional-rate effects.
    internal_rates_on_outer_counts=outer_stats["K2"]*ir["TPR_K2"]-outer_stats["K3"]*ir["FPR_K3"]
    outer_rates_on_internal_counts=internal["K2"]*orr["TPR_K2"]-internal["K3"]*orr["FPR_K3"]

    # Fold-level OOF diagnostics.
    per_fold={}
    for f in (0,1,2,4):
        m=foldi==f
        q,_=stats(yi[m],Xi[m],pi[m],f0i[m],cuts)
        per_fold[str(f)]=q

    rep={
      "status":"completed","experiment":"v273_residual_source_transfer",
      "internal_oof":internal,"outer_fold3":outer_stats,"internal_oof_by_fold":per_fold,
      "threshold_sweep":{"internal":threshold_sweep(yi,pi),"outer":threshold_sweep(yo_valid,po_valid)},
      "decomposition":{
        "observed_internal_net":ir["net"],"observed_outer_net":orr["net"],
        "internal_rates_applied_to_outer_class_counts":float(internal_rates_on_outer_counts),
        "outer_rates_applied_to_internal_class_counts":float(outer_rates_on_internal_counts),
        "delta_K2_prior":outer_stats["K2_prior"]-internal["K2_prior"],
        "delta_joint_auc":outer_stats["joint_auc"]-internal["joint_auc"],
        "delta_TPR_K2":orr["TPR_K2"]-ir["TPR_K2"],
        "delta_FPR_K3":orr["FPR_K3"]-ir["FPR_K3"],
      },
      "outer_fold_3_fresh_independent":False,"prediction_changes":False,"threshold_tuning":False
    }
    a.output.mkdir(parents=True);(a.output/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
    lines=["# Raw residual corrector transfer audit","",
           "Fold 3 is historically exposed; diagnostic only.","",
           "| metric | internal OOF | outer fold3 |",
           "|---|---:|---:|",
           f"| K2 candidates | {internal['K2']} | {outer_stats['K2']} |",
           f"| K3 candidates | {internal['K3']} | {outer_stats['K3']} |",
           f"| K2 prior | {internal['K2_prior']:.3f} | {outer_stats['K2_prior']:.3f} |",
           f"| LR AUC | {internal['joint_auc']:.3f} | {outer_stats['joint_auc']:.3f} |",
           f"| K2 correction rate @.5 | {ir['TPR_K2']:.3f} | {orr['TPR_K2']:.3f} |",
           f"| K3 false-correction rate @.5 | {ir['FPR_K3']:.3f} | {orr['FPR_K3']:.3f} |",
           f"| Action precision K2 | {ir['action_precision_K2']:.3f} | {orr['action_precision_K2']:.3f} |",
           f"| Net @.5 | {ir['net']:+d} | {orr['net']:+d} |","",
           "## Residual feature AUCs","",
           "| feature | internal | outer |","|---|---:|---:|"]
    for n in FEATURES:
        lines.append(f"| {n} | {internal['univariate'][n]['auc_oriented']:.3f} | {outer_stats['univariate'][n]['auc_oriented']:.3f} |")
    lines+=["","## Counterfactual decomposition","",
            f"Internal action rates on outer K2/K3 counts: **{internal_rates_on_outer_counts:+.2f} expected net**.",
            f"Outer action rates on internal K2/K3 counts: **{outer_rates_on_internal_counts:+.2f} expected net**.",
            f"K2 prior shift: **{rep['decomposition']['delta_K2_prior']:+.3f}**.",
            f"AUC shift: **{rep['decomposition']['delta_joint_auc']:+.3f}**.",
            f"TPR(K2) shift: **{rep['decomposition']['delta_TPR_K2']:+.3f}**; FPR(K3) shift: **{rep['decomposition']['delta_FPR_K3']:+.3f}**.","",
            "No tuning or promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":main()
