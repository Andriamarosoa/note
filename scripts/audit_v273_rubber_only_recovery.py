"""Audit complementary recovery branches for RUBBER_ONLY H2."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

FOLDS=(0,1,2,4)
RULES=("REC_M2_M1_P2","REC_M2_M1_P2_P1NEG","REC_M2_M1_P2_CENTERNEG",
       "REC_M2_M1_P2_P1NEG_CENTERNEG","REC_M2_M1","REC_M2_P2")
STEPS=np.asarray([-2,-1,0,1,2],np.int32)

def require(c,m):
    if not c: raise RuntimeError(m)

def load_npz(p):
    with np.load(p,allow_pickle=False) as z:
        return {k:np.asarray(z[k]) for k in z.files}

def find_fold_files(root,name,experiment):
    out={}
    for p in root.rglob(name):
        rp=p.parent/"report.json"
        if not rp.exists(): continue
        rep=json.loads(rp.read_text())
        if rep.get("experiment")!=experiment: continue
        f=int(rep["validation_fold"])
        require(f in FOLDS and f not in out,"duplicate/bad fold")
        out[f]=p
    require(set(out)==set(FOLDS),f"missing {name} folds")
    return out

def margins(prob):
    p=np.asarray(prob,np.float64)
    return p[:,:,2]-p[:,:,3]

def librosa_views(z):
    steps=np.asarray(z["steps"],np.int32)
    idx=[int(np.where(steps==s)[0][0]) for s in STEPS]
    return margins(np.asarray(z["probability"],np.float64)[:,idx,:])

def rubber_views(z):
    p=np.asarray(z["probability"],np.float64)
    require(p.shape[1]==5,"Rubber view drift")
    return margins(p)

def ro_mask(L,R):
    return (L.mean(axis=1)<=0)&(R.mean(axis=1)>0)

def base_mask(R,ro):
    return ro & (R[:,0]>0) & (R[:,3]>0)

def recovery_masks(R,ro,base):
    m2,m1,c,p1,p2=[R[:,i] for i in range(5)]
    raw={
      "REC_M2_M1_P2":(m2>0)&(m1>0)&(p2>0),
      "REC_M2_M1_P2_P1NEG":(m2>0)&(m1>0)&(p2>0)&(p1<=0),
      "REC_M2_M1_P2_CENTERNEG":(m2>0)&(m1>0)&(p2>0)&(c<=0),
      "REC_M2_M1_P2_P1NEG_CENTERNEG":(m2>0)&(m1>0)&(p2>0)&(p1<=0)&(c<=0),
      "REC_M2_M1":(m2>0)&(m1>0),
      "REC_M2_P2":(m2>0)&(p2>0),
    }
    return {k:ro & (~base) & np.asarray(v,bool) for k,v in raw.items()}

def acct(y,a):
    y=np.asarray(y,np.int32);a=np.asarray(a,bool)
    corr=int(np.sum(a&(y==2))); reg=int(np.sum(a&(y==3)))
    neut=int(np.sum(a&~np.isin(y,(2,3))))
    return {"actions":int(a.sum()),"corrections":corr,"regressions":reg,
            "neutral_other_k":neut,"net":corr-reg}

def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p);poly=y>=2
    return {"rows":int(len(y)),"exact":float(np.mean(y==p)),
            "poly_exact":float(np.mean(y[poly]==p[poly])) if poly.any() else None,
            "under":int(np.sum(p<y)),"over":int(np.sum(p>y))}

def select_rule(stats,min_corr=2):
    good=[r for r in RULES if stats[r]["net"]>0 and stats[r]["corrections"]>=min_corr]
    if not good: return None
    order={r:i for i,r in enumerate(RULES)}
    return min(good,key=lambda r:(-stats[r]["net"],stats[r]["regressions"],stats[r]["actions"],order[r]))

def load_internal(lib_root,rub_root):
    lf=find_fold_files(lib_root,"symmetric.npz","v273_symmetric_pitch_tta_fold")
    rf=find_fold_files(rub_root,"rubberband_h2.npz","v273_rubberband_h2_fold")
    out={}
    for f in FOLDS:
        lz=load_npz(lf[f]);rz=load_npz(rf[f])
        li=np.asarray(lz["row_id"],np.int64);ri=np.asarray(rz["row_id"],np.int64)
        require(set(li.tolist())==set(ri.tolist()),f"id mismatch {f}")
        rmap={int(v):i for i,v in enumerate(ri)}
        order=np.asarray([rmap[int(v)] for v in li],np.int64)
        y=np.asarray(lz["true_k"],np.int32)
        np.testing.assert_array_equal(y,np.asarray(rz["true_k"],np.int32)[order])
        out[f]={"y":y,"L":librosa_views(lz),"R":rubber_views(rz)[order]}
    return out

def combine(parts,folds):
    return {k:np.concatenate([parts[f][k] for f in folds],axis=0) for k in ("y","L","R")}

def eval_data(d):
    ro=ro_mask(d["L"],d["R"]); base=base_mask(d["R"],ro)
    rec=recovery_masks(d["R"],ro,base)
    stats={r:acct(d["y"],rec[r]) for r in RULES}
    return ro,base,rec,stats

def main():
    ap=argparse.ArgumentParser()
    for n in ("librosa-internal","rubber-internal","librosa-outer","rubber-outer","output"):
        ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args(); require(not a.output.exists(),"refusing overwrite")

    parts=load_internal(a.librosa_internal,a.rubber_internal)
    cv=[]
    for vf in FOLDS:
        fit=combine(parts,[f for f in FOLDS if f!=vf])
        _,_,_,fit_stats=eval_data(fit)
        chosen=select_rule(fit_stats,2)
        val=parts[vf]
        _,base,rec,val_stats=eval_data(val)
        branch=np.zeros(len(val["y"]),bool) if chosen is None else rec[chosen]
        union=base|branch
        cv.append({
          "fold":vf,"selected_rule":chosen,"fit_stats":fit_stats,
          "val_base":acct(val["y"],base),
          "val_branch":acct(val["y"],branch),
          "val_union":acct(val["y"],union),
          "val_selected_stats":None if chosen is None else val_stats[chosen],
        })

    def sumfield(field):
        return {k:int(sum(x[field][k] for x in cv))
                for k in ("actions","corrections","regressions","neutral_other_k","net")}
    cv_base=sumfield("val_base");cv_branch=sumfield("val_branch");cv_union=sumfield("val_union")
    union_nonneg=int(sum(x["val_union"]["net"]>=0 for x in cv))

    allint=combine(parts,FOLDS)
    _,base_all,rec_all,stats_all=eval_data(allint)
    chosen_outer=select_rule(stats_all,2)

    lo=load_npz(a.librosa_outer);ro=load_npz(a.rubber_outer)
    for key in ("global_index","k","base_predicted","target_global_index"):
        np.testing.assert_array_equal(lo[key],ro[key])
    y=np.asarray(lo["k"],np.int32);basepred=np.asarray(lo["base_predicted"],np.int32)
    tids=np.asarray(lo["target_global_index"],np.int64)
    L=margins(np.asarray(lo["target_probability"],np.float64))
    R=margins(np.asarray(ro["target_probability"],np.float64))
    gidx=np.asarray(lo["global_index"],np.int64);pos={int(v):i for i,v in enumerate(gidx)}
    tpos=np.asarray([pos[int(v)] for v in tids],np.int64)
    outer={"y":y[tpos],"L":L,"R":R}
    _,base_o,rec_o,stats_o=eval_data(outer)
    branch_o=np.zeros(len(tids),bool) if chosen_outer is None else rec_o[chosen_outer]
    union_o=base_o|branch_o

    def apply(action):
        pp=tpos[np.asarray(action,bool)]
        pred=basepred.copy();pred[pp]=2
        return pred,acct(y[pp],np.ones(len(pp),bool))

    pred_base,q_base=apply(base_o)
    pred_branch,q_branch=apply(branch_o)
    pred_union,q_union=apply(union_o)
    base_m=metrics(y,basepred); mb=metrics(y,pred_base); mbr=metrics(y,pred_branch); mu=metrics(y,pred_union)

    report={
      "status":"completed","experiment":"v273_rubber_only_recovery",
      "internal_cv":{"folds":cv,"base_total":cv_base,"branch_total":cv_branch,
                     "union_total":cv_union,"union_nonnegative_folds":union_nonneg},
      "internal_full":{"base":acct(allint["y"],base_all),"rule_stats":stats_all,
                       "selected_recovery_for_outer":chosen_outer},
      "outer":{"fold":3,"previously_exposed":True,
               "rule_stats_diagnostic":stats_o,
               "base_only":{"accounting":q_base,"metrics":mb},
               "recovery_only":{"rule":chosen_outer,"accounting":q_branch,"metrics":mbr},
               "base_plus_recovery":{"accounting":q_union,"metrics":mu,
                  "delta_exact_points":float(100*(mu["exact"]-base_m["exact"])),
                  "delta_poly_exact_points":float(100*(mu["poly_exact"]-base_m["poly_exact"]))},
               "base_metrics":base_m},
      "automatic_promotion":False
    }
    a.output.mkdir(parents=True)
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    lines=["# RUBBER_ONLY complementary recovery audit","",
           "## Cross-fold validation","",
           "| fold | selected recovery | BASE net | branch net | union corr/reg/net |",
           "|---:|---|---:|---:|---:|"]
    for x in cv:
        lines.append(f"| {x['fold']} | {x['selected_rule'] or 'abstain'} | {x['val_base']['net']:+d} | {x['val_branch']['net']:+d} | {x['val_union']['corrections']}/{x['val_union']['regressions']}/{x['val_union']['net']:+d} |")
    lines+=["",
      f"BASE CV total: **{cv_base['corrections']}/{cv_base['regressions']} = {cv_base['net']:+d}**.",
      f"Recovery CV total: **{cv_branch['corrections']}/{cv_branch['regressions']} = {cv_branch['net']:+d}**.",
      f"BASE + recovery CV total: **{cv_union['corrections']}/{cv_union['regressions']} = {cv_union['net']:+d}**, non-negative {union_nonneg}/4.",
      "","## Full internal recovery stats","",
      "| recovery | actions | corr | reg | net |","|---|---:|---:|---:|---:|"]
    for r in RULES:
        q=stats_all[r];lines.append(f"| {r} | {q['actions']} | {q['corrections']} | {q['regressions']} | {q['net']:+d} |")
    lines+=["",f"Selected recovery for outer: **{chosen_outer or 'none'}**.","",
      "## Outer exploratory measurement","",
      "| policy | Exact-K | Poly Exact-K | corr/reg/net |","|---|---:|---:|---:|",
      f"| base model | {100*base_m['exact']:.3f}% | {100*base_m['poly_exact']:.3f}% | - |",
      f"| M2_P1_POS BASE | {100*mb['exact']:.3f}% | {100*mb['poly_exact']:.3f}% | {q_base['corrections']}/{q_base['regressions']}/{q_base['net']:+d} |",
      f"| recovery only | {100*mbr['exact']:.3f}% | {100*mbr['poly_exact']:.3f}% | {q_branch['corrections']}/{q_branch['regressions']}/{q_branch['net']:+d} |",
      f"| BASE + recovery | {100*mu['exact']:.3f}% | {100*mu['poly_exact']:.3f}% | {q_union['corrections']}/{q_union['regressions']}/{q_union['net']:+d} |",
      "","Fold 3 was already exposed historically. No automatic promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__": main()
