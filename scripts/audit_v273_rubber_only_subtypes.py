"""Audit interpretable subtypes inside Rubber-Band-only H2 actions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

FOLDS=(0,1,2,4)
SUBTYPES=("VOTE4","PAIRS","ALL_SHIFTED","STRONG","LOW_VAR",
          "STRONG_PAIRS","STRONG_VOTE4","PAIRS_LOW_VAR")
R_STEPS=np.asarray([-2,-1,0,1,2],np.int32)


def require(c,m):
    if not c:
        raise RuntimeError(m)


def load_npz(p):
    with np.load(p,allow_pickle=False) as z:
        return {k:np.asarray(z[k]) for k in z.files}


def find_fold_files(root,name,experiment):
    out={}
    for p in root.rglob(name):
        rp=p.parent/"report.json"
        if not rp.exists():
            continue
        rep=json.loads(rp.read_text())
        if rep.get("experiment")!=experiment:
            continue
        f=int(rep["validation_fold"])
        require(f in FOLDS and f not in out,"duplicate/bad fold")
        out[f]=p
    require(set(out)==set(FOLDS),f"missing {name} folds")
    return out


def margin_views(prob):
    p=np.asarray(prob,np.float64)
    return p[:,:,2]-p[:,:,3]


def librosa_h2_views(z):
    steps=np.asarray(z["steps"],np.int32)
    idx=[int(np.where(steps==s)[0][0]) for s in R_STEPS]
    return margin_views(np.asarray(z["probability"],np.float64)[:,idx,:])


def rubber_h2_views(z):
    p=np.asarray(z["probability"],np.float64)
    require(p.shape[1]==5,"Rubber H2 view count drift")
    return margin_views(p)


def features_from_views(v):
    v=np.asarray(v,np.float64)
    require(v.ndim==2 and v.shape[1]==5,"bad H2 margin views")
    return {
        "R_mean":v.mean(axis=1),
        "R_min":v.min(axis=1),
        "R_std":v.std(axis=1),
        "positive_views":np.sum(v>0,axis=1).astype(np.int32),
        "pair1_mean":v[:,[1,3]].mean(axis=1),
        "pair2_mean":v[:,[0,4]].mean(axis=1),
        "both_pairs_positive":(v[:,[1,3]].mean(axis=1)>0)&(v[:,[0,4]].mean(axis=1)>0),
        "all_shifted_positive":np.all(v[:,[0,1,3,4]]>0,axis=1),
        "center_positive":v[:,2]>0,
    }


def thresholds(feat,rubber_only):
    take=np.asarray(rubber_only,bool)
    require(int(take.sum())>0,"empty RUBBER_ONLY fit population")
    return {
        "strong_median":float(np.median(feat["R_mean"][take])),
        "low_var_median":float(np.median(feat["R_std"][take])),
    }


def subtype_masks(feat,rubber_only,thr):
    ro=np.asarray(rubber_only,bool)
    vote4=feat["positive_views"]>=4
    pairs=np.asarray(feat["both_pairs_positive"],bool)
    all_shifted=np.asarray(feat["all_shifted_positive"],bool)
    strong=feat["R_mean"]>=thr["strong_median"]
    low_var=feat["R_std"]<=thr["low_var_median"]
    raw={
        "VOTE4":vote4,
        "PAIRS":pairs,
        "ALL_SHIFTED":all_shifted,
        "STRONG":strong,
        "LOW_VAR":low_var,
        "STRONG_PAIRS":strong & pairs,
        "STRONG_VOTE4":strong & vote4,
        "PAIRS_LOW_VAR":pairs & low_var,
    }
    return {k:ro & np.asarray(v,bool) for k,v in raw.items()}


def accounting(y,action):
    y=np.asarray(y,np.int32);a=np.asarray(action,bool)
    corr=int(np.sum(a&(y==2)))
    reg=int(np.sum(a&(y==3)))
    neutral=int(np.sum(a&~np.isin(y,(2,3))))
    return {"actions":int(a.sum()),"corrections":corr,"regressions":reg,
            "neutral_other_k":neutral,"net":corr-reg}


def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p);poly=y>=2
    return {
        "rows":int(len(y)),
        "exact":float(np.mean(y==p)),
        "poly_exact":float(np.mean(y[poly]==p[poly])) if poly.any() else None,
        "under":int(np.sum(p<y)),
        "over":int(np.sum(p>y)),
    }


def select_subtype(stats,min_corr=3):
    admissible=[s for s in SUBTYPES
                if stats[s]["net"]>0 and stats[s]["corrections"]>=min_corr]
    if not admissible:
        return None
    order={s:i for i,s in enumerate(SUBTYPES)}
    return min(admissible,key=lambda s:(
        -stats[s]["net"],
        stats[s]["regressions"],
        stats[s]["actions"],
        order[s],
    ))


def combine_rows(parts,folds):
    keys=("y","L","R","Rviews")
    return {k:np.concatenate([parts[f][k] for f in folds],axis=0) for k in keys}


def evaluate_subtypes(data,thr):
    feat=features_from_views(data["Rviews"])
    ro=(data["L"].mean(axis=1)<=0)&(data["R"].mean(axis=1)>0)
    masks=subtype_masks(feat,ro,thr)
    return {s:accounting(data["y"],masks[s]) for s in SUBTYPES},masks,feat,ro


def load_internal(lib_root,rub_root):
    lf=find_fold_files(lib_root,"symmetric.npz","v273_symmetric_pitch_tta_fold")
    rf=find_fold_files(rub_root,"rubberband_h2.npz","v273_rubberband_h2_fold")
    out={}
    for f in FOLDS:
        lz=load_npz(lf[f]); rz=load_npz(rf[f])
        li=np.asarray(lz["row_id"],np.int64);ri=np.asarray(rz["row_id"],np.int64)
        require(set(li.tolist())==set(ri.tolist()),f"id mismatch fold {f}")
        rmap={int(v):i for i,v in enumerate(ri)}
        order=np.asarray([rmap[int(v)] for v in li],np.int64)
        y=np.asarray(lz["true_k"],np.int32)
        np.testing.assert_array_equal(y,np.asarray(rz["true_k"],np.int32)[order])
        Lviews=librosa_h2_views(lz)
        Rviews=rubber_h2_views(rz)[order]
        out[f]={
            "y":y,
            "L":Lviews,
            "R":Rviews,
            "Rviews":Rviews,
        }
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--librosa-internal",type=Path,required=True)
    ap.add_argument("--rubber-internal",type=Path,required=True)
    ap.add_argument("--librosa-outer",type=Path,required=True)
    ap.add_argument("--rubber-outer",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    require(not a.output.exists(),"refusing overwrite")

    parts=load_internal(a.librosa_internal,a.rubber_internal)

    cv=[]
    for vf in FOLDS:
        fit_folds=[f for f in FOLDS if f!=vf]
        fit=combine_rows(parts,fit_folds)
        fit_feat=features_from_views(fit["Rviews"])
        fit_ro=(fit["L"].mean(axis=1)<=0)&(fit["R"].mean(axis=1)>0)
        thr=thresholds(fit_feat,fit_ro)
        fit_stats,_,_,_=evaluate_subtypes(fit,thr)
        chosen=select_subtype(fit_stats,min_corr=3)

        val=parts[vf]
        val_stats,val_masks,_,val_ro=evaluate_subtypes(val,thr)
        sub_action=np.zeros(len(val["y"]),bool) if chosen is None else val_masks[chosen]
        sub_q=accounting(val["y"],sub_action)

        Lmean=val["L"].mean(axis=1);Rmean=val["R"].mean(axis=1)
        combined_action=(Lmean>0)|sub_action
        combined_q=accounting(val["y"],combined_action)
        ro_q=accounting(val["y"],val_ro)

        cv.append({
            "fold":vf,
            "thresholds":thr,
            "selected_subtype":chosen,
            "fit_stats":fit_stats,
            "val_rubber_only_all":ro_q,
            "val_selected_subtype":sub_q,
            "val_librosa_plus_selected":combined_q,
            "val_selected_stats":None if chosen is None else val_stats[chosen],
        })

    def sumq(field):
        return {k:int(sum(x[field][k] for x in cv))
                for k in ("actions","corrections","regressions","neutral_other_k","net")}

    cv_sub=sumq("val_selected_subtype")
    cv_comb=sumq("val_librosa_plus_selected")
    sub_nonneg=int(sum(x["val_selected_subtype"]["net"]>=0 for x in cv))
    comb_nonneg=int(sum(x["val_librosa_plus_selected"]["net"]>=0 for x in cv))

    all_internal=combine_rows(parts,FOLDS)
    all_feat=features_from_views(all_internal["Rviews"])
    all_ro=(all_internal["L"].mean(axis=1)<=0)&(all_internal["R"].mean(axis=1)>0)
    all_thr=thresholds(all_feat,all_ro)
    all_stats,_,_,_=evaluate_subtypes(all_internal,all_thr)
    outer_subtype=select_subtype(all_stats,min_corr=4)

    lo=load_npz(a.librosa_outer);ro=load_npz(a.rubber_outer)
    for key in ("global_index","k","base_predicted","target_global_index"):
        np.testing.assert_array_equal(lo[key],ro[key])

    y=np.asarray(lo["k"],np.int32)
    base=np.asarray(lo["base_predicted"],np.int32)
    tids=np.asarray(lo["target_global_index"],np.int64)
    Lviews=margin_views(np.asarray(lo["target_probability"],np.float64))
    Rviews=margin_views(np.asarray(ro["target_probability"],np.float64))
    require(Lviews.shape==Rviews.shape and Lviews.shape[1]==5,"outer view drift")
    outer_data={"y":np.zeros(len(tids),np.int32),"L":Lviews,"R":Rviews,"Rviews":Rviews}

    gidx=np.asarray(lo["global_index"],np.int64)
    pos={int(v):i for i,v in enumerate(gidx)}
    target_pos=np.asarray([pos[int(v)] for v in tids],np.int64)
    outer_data["y"]=y[target_pos]

    outer_stats,outer_masks,outer_feat,outer_ro=evaluate_subtypes(outer_data,all_thr)
    sub_action=np.zeros(len(tids),bool) if outer_subtype is None else outer_masks[outer_subtype]
    Lmean=Lviews.mean(axis=1)
    combined_action=(Lmean>0)|sub_action

    def apply_target_action(action):
        pp=target_pos[np.asarray(action,bool)]
        pred=base.copy();pred[pp]=2
        return pred,accounting(y[pp],np.ones(len(pp),bool))

    pred_sub,q_sub=apply_target_action(sub_action)
    pred_comb,q_comb=apply_target_action(combined_action)
    base_m=metrics(y,base);sub_m=metrics(y,pred_sub);comb_m=metrics(y,pred_comb)

    report={
        "status":"completed",
        "experiment":"v273_rubber_only_subtypes",
        "internal_cv":{
            "folds":cv,
            "selected_subtype_only_total":cv_sub,
            "selected_subtype_only_nonnegative_folds":sub_nonneg,
            "librosa_plus_selected_total":cv_comb,
            "librosa_plus_selected_nonnegative_folds":comb_nonneg,
        },
        "internal_full":{
            "rubber_only_rows":int(all_ro.sum()),
            "thresholds":all_thr,
            "subtype_stats":all_stats,
            "selected_for_outer":outer_subtype,
        },
        "outer":{
            "fold":3,
            "previously_exposed":True,
            "rubber_only_rows":int(outer_ro.sum()),
            "subtype_stats_diagnostic":outer_stats,
            "selected_subtype_only":{
                "subtype":outer_subtype,
                "accounting":q_sub,
                "metrics":sub_m,
                "delta_exact_points":float(100*(sub_m["exact"]-base_m["exact"])),
                "delta_poly_exact_points":float(100*(sub_m["poly_exact"]-base_m["poly_exact"])),
            },
            "librosa_plus_selected":{
                "accounting":q_comb,
                "metrics":comb_m,
                "delta_exact_points":float(100*(comb_m["exact"]-base_m["exact"])),
                "delta_poly_exact_points":float(100*(comb_m["poly_exact"]-base_m["poly_exact"])),
            },
            "base_metrics":base_m,
        },
        "automatic_promotion":False,
    }

    a.output.mkdir(parents=True)
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")

    lines=["# RUBBER_ONLY subtype audit","",
           "## Cross-fold validation","",
           "| fold | subtype | subtype corr/reg/net | Librosa + subtype corr/reg/net |",
           "|---:|---|---:|---:|"]
    for x in cv:
        s=x["selected_subtype"] or "abstain"
        q=x["val_selected_subtype"];qc=x["val_librosa_plus_selected"]
        lines.append(f"| {x['fold']} | {s} | {q['corrections']}/{q['regressions']}/{q['net']:+d} | {qc['corrections']}/{qc['regressions']}/{qc['net']:+d} |")
    lines+=["",
            f"Selected RUBBER_ONLY subtype CV total: **{cv_sub['corrections']}/{cv_sub['regressions']} = {cv_sub['net']:+d}**, non-negative {sub_nonneg}/4.",
            f"Librosa + selected subtype CV total: **{cv_comb['corrections']}/{cv_comb['regressions']} = {cv_comb['net']:+d}**, non-negative {comb_nonneg}/4.",
            "",
            "## Full internal subtype stats","",
            "| subtype | actions | corr | reg | net |",
            "|---|---:|---:|---:|---:|"]
    for s in SUBTYPES:
        q=all_stats[s]
        lines.append(f"| {s} | {q['actions']} | {q['corrections']} | {q['regressions']} | {q['net']:+d} |")
    lines+=["",
            f"Selected for outer: **{outer_subtype or 'none'}**.",
            "",
            "## Outer exploratory measurement","",
            "| policy | Exact-K | Poly Exact-K | delta poly | corr/reg/net |",
            "|---|---:|---:|---:|---:|",
            f"| base | {100*base_m['exact']:.3f}% | {100*base_m['poly_exact']:.3f}% | +0.000 pt | - |",
            f"| selected RUBBER_ONLY subtype | {100*sub_m['exact']:.3f}% | {100*sub_m['poly_exact']:.3f}% | {100*(sub_m['poly_exact']-base_m['poly_exact']):+.3f} pt | {q_sub['corrections']}/{q_sub['regressions']}/{q_sub['net']:+d} |",
            f"| Librosa + selected subtype | {100*comb_m['exact']:.3f}% | {100*comb_m['poly_exact']:.3f}% | {100*(comb_m['poly_exact']-base_m['poly_exact']):+.3f} pt | {q_comb['corrections']}/{q_comb['regressions']}/{q_comb['net']:+d} |",
            "",
            "Fold 3 was already exposed historically. No automatic promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__=="__main__":
    main()
