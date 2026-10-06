"""Route H2 corrections by Librosa/Rubber Band disagreement cells."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

FOLDS=(0,1,2,4)
CELLS=("BOTH","RUBBER_ONLY","LIBROSA_ONLY")


def require(c,m):
    if not c:
        raise RuntimeError(m)


def load_npz(p):
    with np.load(p,allow_pickle=False) as z:
        return {k:np.asarray(z[k]) for k in z.files}


def find_fold_files(root,name,experiment):
    out={}
    for p in root.rglob(name):
        rep_path=p.parent/"report.json"
        if not rep_path.exists():
            continue
        rep=json.loads(rep_path.read_text())
        if rep.get("experiment")!=experiment:
            continue
        f=int(rep["validation_fold"])
        require(f in FOLDS and f not in out,"duplicate/bad fold")
        out[f]=p
    require(set(out)==set(FOLDS),f"missing {name} folds")
    return out


def librosa_margin(z):
    steps=np.asarray(z["steps"],np.int32)
    take=np.abs(steps)<=2
    p=np.asarray(z["probability"],np.float64)[:,take,:]
    return np.mean(p[:,:,2]-p[:,:,3],axis=1)


def rubber_margin(z):
    return np.asarray(z["mean_margin"],np.float64)


def cell_labels(L,R):
    out=np.full(len(L),"NONE",dtype="<U12")
    out[(L>0)&(R>0)]="BOTH"
    out[(L<=0)&(R>0)]="RUBBER_ONLY"
    out[(L>0)&(R<=0)]="LIBROSA_ONLY"
    return out


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


def load_internal(lib_root,rub_root):
    lf=find_fold_files(lib_root,"symmetric.npz","v273_symmetric_pitch_tta_fold")
    rf=find_fold_files(rub_root,"rubberband_h2.npz","v273_rubberband_h2_fold")
    rows={}
    for f in FOLDS:
        lz=load_npz(lf[f]);rz=load_npz(rf[f])
        li=np.asarray(lz["row_id"],np.int64);ri=np.asarray(rz["row_id"],np.int64)
        require(set(li.tolist())==set(ri.tolist()),f"id mismatch fold {f}")
        rmap={int(v):i for i,v in enumerate(ri)}
        order=np.asarray([rmap[int(v)] for v in li],np.int64)
        y=np.asarray(lz["true_k"],np.int32)
        np.testing.assert_array_equal(y,np.asarray(rz["true_k"],np.int32)[order])
        L=librosa_margin(lz);R=rubber_margin(rz)[order]
        rows[f]={
            "row_id":li,
            "y":y,
            "L":L,
            "R":R,
            "cell":cell_labels(L,R),
        }
    return rows


def cell_stats(rows,selected_folds):
    ys=[];cells=[]
    for f in selected_folds:
        ys.append(rows[f]["y"]);cells.append(rows[f]["cell"])
    y=np.concatenate(ys);cell=np.concatenate(cells)
    out={}
    for c in CELLS:
        take=cell==c
        out[c]=accounting(y,take)
    out["NONE"]={"rows":int(np.sum(cell=="NONE"))}
    return out


def select_cells(stats,min_corrections):
    chosen=[]
    for c in CELLS:
        q=stats[c]
        if q["net"]>0 and q["corrections"]>=min_corrections:
            chosen.append(c)
    return chosen


def action_from_cells(cell,chosen):
    return np.isin(cell,np.asarray(chosen,dtype="<U12"))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--librosa-internal",type=Path,required=True)
    ap.add_argument("--rubber-internal",type=Path,required=True)
    ap.add_argument("--librosa-outer",type=Path,required=True)
    ap.add_argument("--rubber-outer",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    require(not a.output.exists(),"refusing overwrite")

    rows=load_internal(a.librosa_internal,a.rubber_internal)

    cv=[]
    for vf in FOLDS:
        fit=[f for f in FOLDS if f!=vf]
        stats=cell_stats(rows,fit)
        chosen=select_cells(stats,min_corrections=3)
        action=action_from_cells(rows[vf]["cell"],chosen)
        q=accounting(rows[vf]["y"],action)
        cv.append({
            "fold":vf,
            "fit_cell_stats":stats,
            "selected_cells":chosen,
            "val":q,
        })
    cv_total={k:int(sum(x["val"][k] for x in cv))
              for k in ("actions","corrections","regressions","neutral_other_k","net")}
    cv_nonneg=int(sum(x["val"]["net"]>=0 for x in cv))

    total_stats=cell_stats(rows,FOLDS)
    outer_selected_cells=select_cells(total_stats,min_corrections=4)

    lo=load_npz(a.librosa_outer);ro=load_npz(a.rubber_outer)
    for key in ("global_index","k","base_predicted","target_global_index"):
        np.testing.assert_array_equal(lo[key],ro[key])

    y=np.asarray(lo["k"],np.int32)
    base=np.asarray(lo["base_predicted"],np.int32)
    tids=np.asarray(lo["target_global_index"],np.int64)
    L=np.asarray(lo["target_mean_margin"],np.float64)
    R=np.asarray(ro["target_mean_margin"],np.float64)
    require(len(tids)==len(L)==len(R),"outer target drift")
    cell=cell_labels(L,R)
    target_action=action_from_cells(cell,outer_selected_cells)

    gidx=np.asarray(lo["global_index"],np.int64)
    pos={int(v):i for i,v in enumerate(gidx)}
    acted_positions=np.asarray([pos[int(v)] for v in tids[target_action]],np.int64)

    corrected=base.copy()
    corrected[acted_positions]=2
    changed=corrected!=base
    require(np.all(base[changed]==3)&np.all(corrected[changed]==2),"non 3->2 outer change")

    base_m=metrics(y,base);new_m=metrics(y,corrected)
    outer_acc=accounting(y[acted_positions],np.ones(len(acted_positions),bool))

    outer_cell_stats={}
    y_target=np.asarray([y[pos[int(v)]] for v in tids],np.int32)
    for c in CELLS:
        outer_cell_stats[c]=accounting(y_target,cell==c)
    outer_cell_stats["NONE"]={"rows":int(np.sum(cell=="NONE"))}

    report={
        "status":"completed",
        "experiment":"v273_dual_engine_disagreement_routing",
        "internal_cv":{
            "folds":cv,
            "total":cv_total,
            "nonnegative_folds":cv_nonneg,
        },
        "internal_full_cell_stats":total_stats,
        "outer_selection":{
            "selected_cells":outer_selected_cells,
            "selection_source":"all internal folds 0/1/2/4 only",
            "criterion":"cell net > 0 and corrections >= 4",
        },
        "outer":{
            "fold":3,
            "previously_exposed":True,
            "cell_stats_diagnostic":outer_cell_stats,
            "accounting":outer_acc,
            "base_metrics":base_m,
            "corrected_metrics":new_m,
            "delta_exact_points":float(100*(new_m["exact"]-base_m["exact"])),
            "delta_poly_exact_points":float(100*(new_m["poly_exact"]-base_m["poly_exact"])),
        },
        "automatic_promotion":False,
    }

    a.output.mkdir(parents=True)
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    lines=["# Dual-engine disagreement routing","",
           "## Internal cross-fold routing","",
           "| fold | selected cells | corr | reg | net |",
           "|---:|---|---:|---:|---:|"]
    for x in cv:
        label=", ".join(x["selected_cells"]) if x["selected_cells"] else "abstain"
        q=x["val"]
        lines.append(f"| {x['fold']} | {label} | {q['corrections']} | {q['regressions']} | {q['net']:+d} |")
    lines+=["",
            f"Cross-fold total: **{cv_total['corrections']} / {cv_total['regressions']} = {cv_total['net']:+d}**.",
            f"Non-negative folds: **{cv_nonneg}/4**.","",
            "## Internal full-data cell stats","",
            "| cell | actions | corr | reg | net |",
            "|---|---:|---:|---:|---:|"]
    for c in CELLS:
        q=total_stats[c]
        lines.append(f"| {c} | {q['actions']} | {q['corrections']} | {q['regressions']} | {q['net']:+d} |")
    lines+=["",f"Outer selected cells: **{', '.join(outer_selected_cells) if outer_selected_cells else 'none'}**.","",
            "| outer measure | base | routed | delta |",
            "|---|---:|---:|---:|",
            f"| Exact-K global | {100*base_m['exact']:.3f}% | {100*new_m['exact']:.3f}% | {100*(new_m['exact']-base_m['exact']):+.3f} pt |",
            f"| Poly Exact-K | {100*base_m['poly_exact']:.3f}% | {100*new_m['poly_exact']:.3f}% | {100*(new_m['poly_exact']-base_m['poly_exact']):+.3f} pt |",
            "",
            f"Outer actions: **{outer_acc['actions']}**.",
            f"Outer corrections/regressions: **{outer_acc['corrections']}/{outer_acc['regressions']}**.",
            f"Outer neutral other-K: **{outer_acc['neutral_other_k']}**.",
            f"Outer net Exact-K: **{outer_acc['net']:+d}**.",
            "",
            "Fold 3 was already exposed historically. No automatic promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__=="__main__":
    main()
