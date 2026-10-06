"""Joint-geometry audit for the raw residual corrector.

Measures why the two-feature LR transfers better to exposed fold 3 even though
the two residual features have similar univariate AUCs.

Also reports a fixed confidence grid for the raw LR itself. The grid is
diagnostic only; no outer tuning or promotion.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.audit_v273_residual_source_transfer import load_internal
from scripts.evaluate_v273_outer_residual_two_stage_p060 import build_robust_model,reconstruct_blow,load_npz
from scripts.audit_v273_internal_b_like_boundary_corrector import predict,structural_features
from scripts.audit_v273_internal_b_low_harmonic_strata import audio_rows
from scripts.v273_residual_audit import extracted_matrix,fit_classifier
from scripts.v273_window_experiment import load_bundle,require

GRID=(.5,.55,.6,.65,.7,.75,.8,.85,.9)
EPS=1e-12

def binary(y,X,p):
    y=np.asarray(y,int);X=np.asarray(X,float);p=np.asarray(p,float)
    m=np.isin(y,(2,3));return y[m],X[m],p[m]

def corr(a,b):
    return None if len(a)<3 or np.std(a)<1e-12 or np.std(b)<1e-12 else float(np.corrcoef(a,b)[0,1])

def geom_values(X):
    pair=X[:,0];trip=X[:,1]
    return {
      "pair_minus_triplet":pair-trip,
      "triplet_minus_pair":trip-pair,
      "pair_over_triplet":pair/(trip+EPS),
      "normalized_difference":(pair-trip)/(np.abs(pair)+np.abs(trip)+EPS),
      "mean_residual":.5*(pair+trip),
      "abs_difference":np.abs(pair-trip),
    }

def oriented_auc(y,v,sign):
    return float(roc_auc_score((y==2).astype(int),sign*np.asarray(v,float)))

def class_summary(y,v):
    out={}
    for k in (2,3):
        z=np.asarray(v)[y==k]
        out[str(k)]={"n":int(len(z)),"mean":float(np.mean(z)),"median":float(np.median(z)),
                     "std":float(np.std(z))}
    return out

def threshold_metrics(y,p):
    out={}
    for t in GRID:
        a=p>=t;c=int(np.sum(a&(y==2)));r=int(np.sum(a&(y==3)))
        out[str(t)]={"corrections":c,"regressions":r,"net":c-r,"actions":int(a.sum()),
                     "precision_K2":None if c+r==0 else c/(c+r)}
    return out

def internal_fold_grid(root):
    out={}
    for fold in (0,1,2,4):
        with np.load(Path(root)/f"fold-{fold}"/"replay.npz",allow_pickle=False) as z:
            a={k:np.asarray(z[k]) for k in z.files}
        action=a["val_b_low"].astype(bool)&(a["val_base_k"].astype(int)==3)
        valid=a["val_valid"].astype(bool)
        y=a["val_true_k"].astype(int)[action][valid]
        p=a["val_probability"].astype(float)
        m=np.isin(y,(2,3))
        out[str(fold)]=threshold_metrics(y[m],p[m])
    return out

def main():
    ap=argparse.ArgumentParser()
    for n in ("internal-exports","bundle","config","uniform-weights","freeze-weights",
              "outer-group-predictions","outer-b-low-predictions","dataset","output"):
        ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args();require(not a.output.exists(),"overwrite")

    yi,Xi,pi,_,_=load_internal(a.internal_exports)
    yi,Xi,pi=binary(yi,Xi,pi)

    cache,parts,_=load_bundle(a.bundle,a.config)
    fit=np.asarray(parts["final_fit"],np.int64);outer=np.asarray(parts["outer"],np.int64)
    yall=np.minimum(cache["exact"].astype(np.int32),6);yf=yall[fit];yo=yall[outer]
    model=build_robust_model(a.uniform_weights,a.freeze_weights)
    Pf,Gf=predict(model,cache,fit);Po,Go=predict(model,cache,outer)
    fg=load_npz(a.outer_group_predictions);np.testing.assert_array_equal(fg["global_index"],outer);np.testing.assert_array_equal(fg["predicted"],Go)
    Xf=structural_features(cache,fit,Pf);Xo=structural_features(cache,outer,Po)
    blow_f,blow_o,_,_,_,_=reconstruct_blow(Xf,Xo,yf,Gf)
    fb=load_npz(a.outer_b_low_predictions);require(np.array_equal(fb["B_low_mask"].astype(bool),blow_o&np.isin(Go,(2,3,4))),"B_low drift")
    cand_f=blow_f&(Gf==3);cand_o=blow_o&(Go==3)
    rf=audio_rows(cache,fit[cand_f],a.dataset);ro=audio_rows(cache,outer[cand_o],a.dataset)
    Xrf,_,vf,_=extracted_matrix(rf);Xro,_,vo,_=extracted_matrix(ro)
    yfc=yf[cand_f];yoc=yo[cand_o]
    tr=vf&np.isin(yfc,(2,3));m=fit_classifier(Xrf[tr],(yfc[tr]==2).astype(int))
    po=np.full(len(Xro),np.nan);po[vo]=m.predict_proba(Xro[vo])[:,1]
    yo2,Xo2,po2=binary(yoc[vo],Xro[vo],po[vo])

    gi=geom_values(Xi);go=geom_values(Xo2)
    derived={}
    for name in gi:
        # Freeze direction from internal class means.
        sign=1 if np.mean(gi[name][yi==2])>=np.mean(gi[name][yi==3]) else -1
        derived[name]={"orientation_from_internal":int(sign),
                       "internal_auc":oriented_auc(yi,gi[name],sign),
                       "outer_auc":oriented_auc(yo2,go[name],sign),
                       "internal":class_summary(yi,gi[name]),
                       "outer":class_summary(yo2,go[name])}

    corr_rep={}
    for label,yy,XX in (("internal",yi,Xi),("outer",yo2,Xo2)):
        corr_rep[label]={
          "all":corr(XX[:,0],XX[:,1]),
          "K2":corr(XX[yy==2,0],XX[yy==2,1]),
          "K3":corr(XX[yy==3,0],XX[yy==3,1]),
          "cov_K2":np.cov(XX[yy==2].T).tolist(),
          "cov_K3":np.cov(XX[yy==3].T).tolist(),
        }

    grid_internal=threshold_metrics(yi,pi)
    grid_outer=threshold_metrics(yo2,po2)
    fold_grid=internal_fold_grid(a.internal_exports)
    robust=[]
    for t in GRID:
        key=str(t);nets=[fold_grid[str(f)][key]["net"] for f in (0,1,2,4)]
        robust.append({"threshold":t,"internal_total_net":grid_internal[key]["net"],
                       "per_fold_net":dict(zip(("0","1","2","4"),nets)),
                       "all_internal_folds_nonnegative":all(x>=0 for x in nets),
                       "outer_net":grid_outer[key]["net"]})

    rep={"status":"completed","experiment":"v273_residual_joint_geometry",
         "correlation":corr_rep,"derived":derived,
         "confidence_grid":{"internal":grid_internal,"outer":grid_outer,"per_internal_fold":fold_grid,"robust_summary":robust},
         "outer_fold_3_fresh_independent":False,"prediction_changes":False,"outer_tuning":False}
    a.output.mkdir(parents=True);(a.output/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
    lines=["# Residual joint geometry + confidence audit","",
           "## Pair/triplet correlation","",
           "| population | all | K2 | K3 |","|---|---:|---:|---:|"]
    for label in ("internal","outer"):
        q=corr_rep[label];lines.append(f"| {label} | {q['all']:.3f} | {q['K2']:.3f} | {q['K3']:.3f} |")
    lines+=["","## Derived joint features","","| feature | internal AUC | outer AUC |","|---|---:|---:|"]
    for name,q in derived.items():lines.append(f"| {name} | {q['internal_auc']:.3f} | {q['outer_auc']:.3f} |")
    lines+=["","## Raw-LR fixed confidence grid","","| p threshold | internal net | f0 | f1 | f2 | f4 | all folds >=0 | outer net |",
            "|---:|---:|---:|---:|---:|---:|---|---:|"]
    for q in robust:
        d=q["per_fold_net"];lines.append(f"| {q['threshold']:.2f} | {q['internal_total_net']:+d} | {d['0']:+d} | {d['1']:+d} | {d['2']:+d} | {d['4']:+d} | {q['all_internal_folds_nonnegative']} | {q['outer_net']:+d} |")
    lines+=["","Diagnostic only; fold 3 historically exposed; no outer tuning."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":main()
