"""Diagnose how RUBBER_ONLY H2 cases differ between internal folds and outer fold 3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

FOLDS=(0,1,2,4)
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


def describe(L,R):
    L=np.asarray(L,np.float64);R=np.asarray(R,np.float64)
    require(L.shape==R.shape and R.ndim==2 and R.shape[1]==5,"bad H2 view shape")
    bits=(R>0).astype(np.int8)
    pattern=np.asarray(["".join(str(int(v)) for v in row) for row in bits],dtype="<U5")
    return {
      "pattern":pattern,
      "positive_views":bits.sum(axis=1).astype(np.int32),
      "rubber_mean_margin":R.mean(axis=1),
      "rubber_min_margin":R.min(axis=1),
      "rubber_max_margin":R.max(axis=1),
      "rubber_std_margin":R.std(axis=1),
      "center_margin":R[:,2],
      "pair1_mean":R[:,[1,3]].mean(axis=1),
      "pair2_mean":R[:,[0,4]].mean(axis=1),
      "asymmetry_1":np.abs(R[:,1]-R[:,3]),
      "asymmetry_2":np.abs(R[:,0]-R[:,4]),
      "librosa_mean_margin":L.mean(axis=1),
      "engine_gap":R.mean(axis=1)-L.mean(axis=1),
    }


def rubber_only(L,R):
    return (np.asarray(L).mean(axis=1)<=0)&(np.asarray(R).mean(axis=1)>0)


def acct(y,take):
    y=np.asarray(y,np.int32);t=np.asarray(take,bool)
    c=int(np.sum(t&(y==2)));r=int(np.sum(t&(y==3)))
    n=int(np.sum(t&~np.isin(y,(2,3))))
    return {"rows":int(t.sum()),"corrections":c,"regressions":r,
            "neutral_other_k":n,"net":c-r}


def load_internal(lib_root,rub_root):
    lf=find_fold_files(lib_root,"symmetric.npz","v273_symmetric_pitch_tta_fold")
    rf=find_fold_files(rub_root,"rubberband_h2.npz","v273_rubberband_h2_fold")
    rows=[]
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
        feat=describe(L,R)
        ro=rubber_only(L,R)
        for i in np.flatnonzero(ro):
            row={"fold":int(f),"row_id":int(li[i]),"true_k":int(y[i])}
            for k,v in feat.items():
                val=v[i]
                row[k]=str(val) if isinstance(val,(str,np.str_)) else (int(val) if np.issubdtype(np.asarray(val).dtype,np.integer) else float(val))
            row["outcome"]="correction" if y[i]==2 else ("regression" if y[i]==3 else "neutral")
            rows.append(row)
    return rows


def group_patterns(rows):
    out={}
    pats=sorted({r["pattern"] for r in rows})
    for p in pats:
        rr=[r for r in rows if r["pattern"]==p]
        y=np.asarray([r["true_k"] for r in rr],np.int32)
        q=acct(y,np.ones(len(rr),bool))
        q["folds"]=sorted({int(r["fold"]) for r in rr})
        out[p]=q
    return out


def group_positive_views(rows):
    out={}
    for v in range(1,6):
        rr=[r for r in rows if int(r["positive_views"])==v]
        if not rr:
            continue
        y=np.asarray([r["true_k"] for r in rr],np.int32)
        out[str(v)]=acct(y,np.ones(len(rr),bool))
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

    internal_rows=load_internal(a.librosa_internal,a.rubber_internal)
    require(len(internal_rows)>0,"empty internal RUBBER_ONLY")
    internal_patterns=group_patterns(internal_rows)
    internal_votes=group_positive_views(internal_rows)

    lo=load_npz(a.librosa_outer);ro=load_npz(a.rubber_outer)
    for key in ("global_index","k","target_global_index"):
        np.testing.assert_array_equal(lo[key],ro[key])
    tids=np.asarray(lo["target_global_index"],np.int64)
    L=margin_views(np.asarray(lo["target_probability"],np.float64))
    R=margin_views(np.asarray(ro["target_probability"],np.float64))
    require(L.shape==R.shape and L.shape[0]==len(tids) and L.shape[1]==5,"outer H2 view drift")
    feat=describe(L,R)
    ro_mask=rubber_only(L,R)

    gidx=np.asarray(lo["global_index"],np.int64)
    y=np.asarray(lo["k"],np.int32)
    pos={int(v):i for i,v in enumerate(gidx)}
    target_y=np.asarray([y[pos[int(v)]] for v in tids],np.int32)

    outer_rows=[]
    for i in np.flatnonzero(ro_mask):
        row={"global_index":int(tids[i]),"true_k":int(target_y[i])}
        for k,v in feat.items():
            val=v[i]
            row[k]=str(val) if isinstance(val,(str,np.str_)) else (int(val) if np.issubdtype(np.asarray(val).dtype,np.integer) else float(val))
        row["outcome"]="correction" if target_y[i]==2 else ("regression" if target_y[i]==3 else "neutral")
        p=row["pattern"]
        row["pattern_seen_internal"]=bool(p in internal_patterns)
        row["internal_pattern_stats"]=internal_patterns.get(p)
        outer_rows.append(row)

    outer_patterns={}
    for p in sorted({r["pattern"] for r in outer_rows}):
        rr=[r for r in outer_rows if r["pattern"]==p]
        yy=np.asarray([r["true_k"] for r in rr],np.int32)
        q=acct(yy,np.ones(len(rr),bool))
        q["seen_internal"]=bool(p in internal_patterns)
        q["internal_stats"]=internal_patterns.get(p)
        outer_patterns[p]=q

    outer_votes={}
    for v in range(1,6):
        rr=[r for r in outer_rows if int(r["positive_views"])==v]
        if rr:
            yy=np.asarray([r["true_k"] for r in rr],np.int32)
            outer_votes[str(v)]=acct(yy,np.ones(len(rr),bool))

    report={
      "status":"completed",
      "experiment":"v273_rubber_only_domain_shift",
      "internal":{
        "rows":len(internal_rows),
        "accounting":acct(np.asarray([r["true_k"] for r in internal_rows]),np.ones(len(internal_rows),bool)),
        "pattern_stats":internal_patterns,
        "positive_view_stats":internal_votes,
      },
      "outer":{
        "fold":3,
        "previously_exposed":True,
        "rows":len(outer_rows),
        "accounting":acct(np.asarray([r["true_k"] for r in outer_rows]),np.ones(len(outer_rows),bool)) if outer_rows else {"rows":0,"corrections":0,"regressions":0,"neutral_other_k":0,"net":0},
        "pattern_stats":outer_patterns,
        "positive_view_stats":outer_votes,
        "rows_detail":outer_rows,
      },
      "automatic_promotion":False,
      "diagnostic_only":True
    }

    a.output.mkdir(parents=True)
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")

    lines=["# RUBBER_ONLY internal -> outer diagnostic","",
           f"Internal RUBBER_ONLY: **{report['internal']['accounting']['corrections']}/{report['internal']['accounting']['regressions']} = {report['internal']['accounting']['net']:+d}** over {len(internal_rows)} rows.",
           f"Outer RUBBER_ONLY: **{report['outer']['accounting']['corrections']}/{report['outer']['accounting']['regressions']} = {report['outer']['accounting']['net']:+d}** over {len(outer_rows)} rows.","",
           "## Internal by positive-view count","",
           "| positive views | rows | corr | reg | net |",
           "|---:|---:|---:|---:|---:|"]
    for v,q in internal_votes.items():
        lines.append(f"| {v} | {q['rows']} | {q['corrections']} | {q['regressions']} | {q['net']:+d} |")
    lines+=["","## Outer RUBBER_ONLY rows","",
            "| global index | true K | outcome | pattern | +views | R mean | L mean | gap | seen internally |",
            "|---:|---:|---|---|---:|---:|---:|---:|---|"]
    for r in outer_rows:
        lines.append(f"| {r['global_index']} | {r['true_k']} | {r['outcome']} | {r['pattern']} | {r['positive_views']} | {r['rubber_mean_margin']:.5f} | {r['librosa_mean_margin']:.5f} | {r['engine_gap']:.5f} | {r['pattern_seen_internal']} |")
    lines+=["","Diagnostic only; no rule selected. Fold 3 was already exposed historically."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__=="__main__":
    main()
