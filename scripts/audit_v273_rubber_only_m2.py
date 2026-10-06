"""Audit RUBBER_ONLY rules centered on the -2 semitone Rubber Band H2 view."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

FOLDS=(0,1,2,4)
RULES=("M2_POS","M2_M1_POS","M2_PAIR_POS","M2_CENTER_NEG","M2_P1_POS","M2_OR_M1_STRONG")
H2_STEPS=np.asarray([-2,-1,0,1,2],np.int32)


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
    idx=[int(np.where(steps==s)[0][0]) for s in H2_STEPS]
    return margin_views(np.asarray(z["probability"],np.float64)[:,idx,:])


def rubber_h2_views(z):
    p=np.asarray(z["probability"],np.float64)
    require(p.shape[1]==5,"Rubber H2 view count drift")
    return margin_views(p)


def rubber_only(L,R):
    return (np.asarray(L).mean(axis=1)<=0)&(np.asarray(R).mean(axis=1)>0)


def rule_masks(R,ro):
    R=np.asarray(R,np.float64)
    ro=np.asarray(ro,bool)
    m2,m1,c,p1,p2=[R[:,i] for i in range(5)]
    raw={
        "M2_POS":m2>0,
        "M2_M1_POS":(m2>0)&(m1>0),
        "M2_PAIR_POS":(m2>0)&(((m2+p2)/2)>0),
        "M2_CENTER_NEG":(m2>0)&(c<=0),
        "M2_P1_POS":(m2>0)&(p1>0),
        "M2_OR_M1_STRONG":(m2>0)&(((m2+m1)/2)>0),
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


def select_rule(stats,min_corr=3):
    admissible=[r for r in RULES if stats[r]["net"]>0 and stats[r]["corrections"]>=min_corr]
    if not admissible:
        return None
    order={r:i for i,r in enumerate(RULES)}
    return min(admissible,key=lambda r:(
        -stats[r]["net"],
        stats[r]["regressions"],
        stats[r]["actions"],
        order[r],
    ))


def load_internal(lib_root,rub_root):
    lf=find_fold_files(lib_root,"symmetric.npz","v273_symmetric_pitch_tta_fold")
    rf=find_fold_files(rub_root,"rubberband_h2.npz","v273_rubberband_h2_fold")
    out={}
    for f in FOLDS:
        lz=load_npz(lf[f]);rz=load_npz(rf[f])
        li=np.asarray(lz["row_id"],np.int64);ri=np.asarray(rz["row_id"],np.int64)
        require(set(li.tolist())==set(ri.tolist()),f"id mismatch fold {f}")
        rmap={int(v):i for i,v in enumerate(ri)}
        order=np.asarray([rmap[int(v)] for v in li],np.int64)
        y=np.asarray(lz["true_k"],np.int32)
        np.testing.assert_array_equal(y,np.asarray(rz["true_k"],np.int32)[order])
        L=librosa_h2_views(lz)
        R=rubber_h2_views(rz)[order]
        out[f]={"y":y,"L":L,"R":R}
    return out


def combine(parts,folds):
    return {k:np.concatenate([parts[f][k] for f in folds],axis=0) for k in ("y","L","R")}


def evaluate(data):
    ro=rubber_only(data["L"],data["R"])
    masks=rule_masks(data["R"],ro)
    return {r:accounting(data["y"],masks[r]) for r in RULES},masks,ro


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
        fit=combine(parts,[f for f in FOLDS if f!=vf])
        fit_stats,_,_=evaluate(fit)
        chosen=select_rule(fit_stats,3)

        val=parts[vf]
        val_stats,val_masks,val_ro=evaluate(val)
        action=np.zeros(len(val["y"]),bool) if chosen is None else val_masks[chosen]
        q=accounting(val["y"],action)

        Lmean=val["L"].mean(axis=1)
        combined=(Lmean>0)|action
        qc=accounting(val["y"],combined)

        cv.append({
            "fold":vf,
            "selected_rule":chosen,
            "fit_stats":fit_stats,
            "val_rubber_only_all":accounting(val["y"],val_ro),
            "val_selected_rule":q,
            "val_librosa_plus_rule":qc,
            "val_selected_stats":None if chosen is None else val_stats[chosen],
        })

    def sumfield(field):
        return {k:int(sum(x[field][k] for x in cv))
                for k in ("actions","corrections","regressions","neutral_other_k","net")}
    cv_rule=sumfield("val_selected_rule")
    cv_comb=sumfield("val_librosa_plus_rule")
    cv_rule_nonneg=int(sum(x["val_selected_rule"]["net"]>=0 for x in cv))
    cv_comb_nonneg=int(sum(x["val_librosa_plus_rule"]["net"]>=0 for x in cv))

    all_internal=combine(parts,FOLDS)
    all_stats,_,all_ro=evaluate(all_internal)
    outer_rule=select_rule(all_stats,4)

    lo=load_npz(a.librosa_outer);ro=load_npz(a.rubber_outer)
    for key in ("global_index","k","base_predicted","target_global_index"):
        np.testing.assert_array_equal(lo[key],ro[key])
    y=np.asarray(lo["k"],np.int32)
    base=np.asarray(lo["base_predicted"],np.int32)
    tids=np.asarray(lo["target_global_index"],np.int64)
    L=margin_views(np.asarray(lo["target_probability"],np.float64))
    R=margin_views(np.asarray(ro["target_probability"],np.float64))
    require(L.shape==R.shape and L.shape[1]==5,"outer view drift")

    gidx=np.asarray(lo["global_index"],np.int64)
    pos={int(v):i for i,v in enumerate(gidx)}
    tpos=np.asarray([pos[int(v)] for v in tids],np.int64)
    outer_data={"y":y[tpos],"L":L,"R":R}
    outer_stats,outer_masks,outer_ro=evaluate(outer_data)
    action=np.zeros(len(tids),bool) if outer_rule is None else outer_masks[outer_rule]
    Lmean=L.mean(axis=1)
    combined=(Lmean>0)|action

    def apply(a):
        pp=tpos[np.asarray(a,bool)]
        pred=base.copy();pred[pp]=2
        return pred,accounting(y[pp],np.ones(len(pp),bool))

    pred_rule,q_rule=apply(action)
    pred_comb,q_comb=apply(combined)
    base_m=metrics(y,base);rule_m=metrics(y,pred_rule);comb_m=metrics(y,pred_comb)

    report={
        "status":"completed",
        "experiment":"v273_rubber_only_m2",
        "internal_cv":{
            "folds":cv,
            "selected_rule_total":cv_rule,
            "selected_rule_nonnegative_folds":cv_rule_nonneg,
            "librosa_plus_rule_total":cv_comb,
            "librosa_plus_rule_nonnegative_folds":cv_comb_nonneg,
        },
        "internal_full":{
            "rubber_only_rows":int(all_ro.sum()),
            "rule_stats":all_stats,
            "selected_for_outer":outer_rule,
        },
        "outer":{
            "fold":3,
            "previously_exposed":True,
            "rubber_only_rows":int(outer_ro.sum()),
            "rule_stats_diagnostic":outer_stats,
            "selected_rule_only":{
                "rule":outer_rule,
                "accounting":q_rule,
                "metrics":rule_m,
                "delta_exact_points":float(100*(rule_m["exact"]-base_m["exact"])),
                "delta_poly_exact_points":float(100*(rule_m["poly_exact"]-base_m["poly_exact"])),
            },
            "librosa_plus_selected_rule":{
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

    lines=["# RUBBER_ONLY -2-view audit","",
           "## Cross-fold validation","",
           "| fold | selected rule | rule corr/reg/net | Librosa + rule corr/reg/net |",
           "|---:|---|---:|---:|"]
    for x in cv:
        s=x["selected_rule"] or "abstain"
        q=x["val_selected_rule"];qc=x["val_librosa_plus_rule"]
        lines.append(f"| {x['fold']} | {s} | {q['corrections']}/{q['regressions']}/{q['net']:+d} | {qc['corrections']}/{qc['regressions']}/{qc['net']:+d} |")
    lines+=["",
            f"Selected rule CV total: **{cv_rule['corrections']}/{cv_rule['regressions']} = {cv_rule['net']:+d}**, non-negative {cv_rule_nonneg}/4.",
            f"Librosa + rule CV total: **{cv_comb['corrections']}/{cv_comb['regressions']} = {cv_comb['net']:+d}**, non-negative {cv_comb_nonneg}/4.",
            "",
            "## Full internal rule stats","",
            "| rule | actions | corr | reg | net |",
            "|---|---:|---:|---:|---:|"]
    for r in RULES:
        q=all_stats[r]
        lines.append(f"| {r} | {q['actions']} | {q['corrections']} | {q['regressions']} | {q['net']:+d} |")
    lines+=["",
            f"Selected for outer: **{outer_rule or 'none'}**.",
            "",
            "## Outer exploratory measurement","",
            "| policy | Exact-K | Poly Exact-K | delta poly | corr/reg/net |",
            "|---|---:|---:|---:|---:|",
            f"| base | {100*base_m['exact']:.3f}% | {100*base_m['poly_exact']:.3f}% | +0.000 pt | - |",
            f"| selected RUBBER_ONLY rule | {100*rule_m['exact']:.3f}% | {100*rule_m['poly_exact']:.3f}% | {100*(rule_m['poly_exact']-base_m['poly_exact']):+.3f} pt | {q_rule['corrections']}/{q_rule['regressions']}/{q_rule['net']:+d} |",
            f"| Librosa + selected rule | {100*comb_m['exact']:.3f}% | {100*comb_m['poly_exact']:.3f}% | {100*(comb_m['poly_exact']-base_m['poly_exact']):+.3f} pt | {q_comb['corrections']}/{q_comb['regressions']}/{q_comb['net']:+d} |",
            "",
            "Fold 3 was already exposed historically. No automatic promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__=="__main__":
    main()
