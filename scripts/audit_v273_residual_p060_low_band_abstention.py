"""Audit a single predeclared abstention rule for the raw residual corrector.

Baseline action:
  residual LR p(K2) >= 0.60

Proposed abstention:
  if estimated register == low AND 0.60 <= p(K2) < 0.65, do NOT apply 3->2.

Internal evaluation uses the frozen OOF register labels from run 37356100423.
Outer fold 3 rebuilds the pooled-internal residual model used in run 37536875343
and defines low/mid/high from pooled-internal K2/K3 median-triplet-F0 tertiles,
exactly following v273_residual_audit.py.

No threshold search. Fold 3 historically exposed. No automatic promotion.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

from scripts.evaluate_v273_outer_residual_two_stage_p060 import (
    build_robust_model,reconstruct_blow,load_npz
)
from scripts.audit_v273_internal_b_like_boundary_corrector import predict,structural_features
from scripts.audit_v273_internal_b_low_harmonic_strata import audio_rows
from scripts.v273_residual_audit import extracted_matrix,fit_classifier,register_labels
from scripts.v273_window_experiment import load_bundle,require

P_BASE=.60
P_LOW_KEEP=.65

def counts(y,mask):
    y=np.asarray(y,int);m=np.asarray(mask,bool)
    c=int(np.sum(m&(y==2)));r=int(np.sum(m&(y==3)));o=int(np.sum(m&~np.isin(y,(2,3))))
    return {"actions":int(m.sum()),"corrections":c,"regressions":r,"other_k":o,"net":c-r}

def internal_eval(root):
    per={};all_y=[];all_base=[];all_keep=[];all_lowband=[]
    for f in (0,1,2,4):
        with np.load(Path(root)/f"fold-{f}"/"replay.npz",allow_pickle=False) as z:
            a={k:np.asarray(z[k]) for k in z.files}
        action=a["val_b_low"].astype(bool)&(a["val_base_k"].astype(int)==3)
        valid=a["val_valid"].astype(bool)
        y=a["val_true_k"].astype(int)[action][valid]
        p=a["val_probability"].astype(float)
        reg=np.asarray(a["val_register"]).astype(str)
        require(len(y)==len(p)==len(reg),"internal length drift")
        base=p>=P_BASE
        lowband=base&(p<P_LOW_KEEP)&(reg=="low")
        keep=base&~lowband
        per[str(f)]={"baseline":counts(y,base),"lowband_abstained":counts(y,lowband),
                     "final":counts(y,keep)}
        all_y.append(y);all_base.append(base);all_keep.append(keep);all_lowband.append(lowband)
    y=np.concatenate(all_y);base=np.concatenate(all_base);keep=np.concatenate(all_keep);lb=np.concatenate(all_lowband)
    return {"per_fold":per,"baseline":counts(y,base),"lowband_abstained":counts(y,lb),
            "final":counts(y,keep)}, y, base, keep

def outer_eval(a):
    cache,parts,_=load_bundle(a.bundle,a.config)
    fit=np.asarray(parts["final_fit"],np.int64);outer=np.asarray(parts["outer"],np.int64)
    yall=np.minimum(cache["exact"].astype(np.int32),6);yf=yall[fit];yo=yall[outer]
    model=build_robust_model(a.uniform_weights,a.freeze_weights)
    Pf,Gf=predict(model,cache,fit);Po,Go=predict(model,cache,outer)
    fg=load_npz(a.outer_group_predictions)
    np.testing.assert_array_equal(fg["global_index"],outer);np.testing.assert_array_equal(fg["predicted"],Go)
    Xf=structural_features(cache,fit,Pf);Xo=structural_features(cache,outer,Po)
    blow_f,blow_o,_,_,_,_=reconstruct_blow(Xf,Xo,yf,Gf)
    fb=load_npz(a.outer_b_low_predictions)
    require(np.array_equal(fb["B_low_mask"].astype(bool),blow_o&np.isin(Go,(2,3,4))),"B_low replay drift")

    cand_f=blow_f&(Gf==3);cand_o=blow_o&(Go==3)
    rf=audio_rows(cache,fit[cand_f],a.dataset);ro=audio_rows(cache,outer[cand_o],a.dataset)
    Xrf,f0rf,vf,_=extracted_matrix(rf);Xro,f0ro,vo,_=extracted_matrix(ro)
    yfc=yf[cand_f];yoc=yo[cand_o]
    tr=vf&np.isin(yfc,(2,3))
    require(np.any(yfc[tr]==2) and np.any(yfc[tr]==3),"outer train collapse")
    m=fit_classifier(Xrf[tr],(yfc[tr]==2).astype(int))
    p=np.full(len(Xro),np.nan);p[vo]=m.predict_proba(Xro[vo])[:,1]

    fit_f0=f0rf[tr&np.isfinite(f0rf)&(f0rf>0)]
    require(len(fit_f0)>0,"no train f0")
    cuts=np.quantile(fit_f0,[1/3,2/3])
    reg=np.full(len(Xro),"unassigned",dtype="U10")
    reg[vo]=register_labels(f0ro[vo],cuts)

    base=vo&(p>=P_BASE)
    lowband=base&(p<P_LOW_KEEP)&(reg=="low")
    keep=base&~lowband

    # Translate local candidate masks to full outer prediction masks.
    ids=outer[cand_o]
    pos={int(g):i for i,g in enumerate(outer)}
    base_global=np.zeros(len(outer),bool);keep_global=np.zeros(len(outer),bool);lb_global=np.zeros(len(outer),bool)
    for gid,b,k,l in zip(ids,base,keep,lowband):
        j=pos[int(gid)];base_global[j]=bool(b);keep_global[j]=bool(k);lb_global[j]=bool(l)

    pred_base=Go.copy();pred_base[base_global]=2
    pred_final=Go.copy();pred_final[keep_global]=2
    def exact_metrics(pred):
        poly=yo>=2
        return {"global_exact":float(np.mean(pred==yo)),
                "poly_exact":float(np.mean(pred[poly]==yo[poly]))}
    return {
      "register_cuts_hz":{"low_upper":float(cuts[0]),"high_lower":float(cuts[1])},
      "baseline":counts(yo,base_global),"lowband_abstained":counts(yo,lb_global),"final":counts(yo,keep_global),
      "baseline_metrics":exact_metrics(pred_base),"final_metrics":exact_metrics(pred_final)
    }

def main():
    ap=argparse.ArgumentParser()
    for n in ("internal-exports","bundle","config","uniform-weights","freeze-weights",
              "outer-group-predictions","outer-b-low-predictions","dataset","output"):
        ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args();require(not a.output.exists(),"overwrite")

    internal,_,_,_=internal_eval(a.internal_exports)
    outer=outer_eval(a)

    rep={"status":"completed","experiment":"v273_residual_p060_low_band_abstention",
         "policy":{"baseline":"p>=0.60","abstain":"register=low AND 0.60<=p<0.65"},
         "internal_oof":internal,"outer_fold3":outer,
         "strict_internal_pass":{"all_folds_nonnegative":all(internal["per_fold"][str(f)]["final"]["net"]>=0 for f in (0,1,2,4)),
                                 "global_positive":internal["final"]["net"]>0},
         "outer_fold_3_fresh_independent":False,"threshold_search":False,"prediction_changes":False,
         "automatic_promotion":False}
    a.output.mkdir(parents=True);(a.output/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
    lines=["# Residual p0.60 + low-band abstention","",
           "Policy: apply 3→2 for p>=0.60, except abstain when register=low and 0.60<=p<0.65.","",
           "## Internal OOF","",
           "| fold | baseline net | abstained K2 | abstained K3 | abstention gain | final net |",
           "|---:|---:|---:|---:|---:|---:|"]
    for f in (0,1,2,4):
        q=internal["per_fold"][str(f)];lb=q["lowband_abstained"]
        lines.append(f"| {f} | {q['baseline']['net']:+d} | {lb['corrections']} | {lb['regressions']} | {lb['regressions']-lb['corrections']:+d} | {q['final']['net']:+d} |")
    lines+=["",
            f"Internal baseline: **{internal['baseline']['net']:+d}**; final: **{internal['final']['net']:+d}**.",
            f"Internal strict pass: **{rep['strict_internal_pass']}**.","",
            "## Exposed outer fold 3","",
            f"Baseline p>=.60: **{outer['baseline']['corrections']}/{outer['baseline']['regressions']}**, net **{outer['baseline']['net']:+d}**.",
            f"Abstained low-band: **{outer['lowband_abstained']['corrections']} K2 / {outer['lowband_abstained']['regressions']} K3**, gain **{outer['lowband_abstained']['regressions']-outer['lowband_abstained']['corrections']:+d}**.",
            f"Final outer: **{outer['final']['corrections']}/{outer['final']['regressions']}**, net **{outer['final']['net']:+d}**.",
            f"Outer global Exact-K: **{100*outer['baseline_metrics']['global_exact']:.3f}% → {100*outer['final_metrics']['global_exact']:.3f}%**.",
            f"Outer poly Exact-K: **{100*outer['baseline_metrics']['poly_exact']:.3f}% → {100*outer['final_metrics']['poly_exact']:.3f}%**.","",
            "Fold 3 is historically exposed; diagnostic only. No automatic promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":main()
