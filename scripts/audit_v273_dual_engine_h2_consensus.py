"""Combine Librosa and Rubber Band H2 margins and evaluate consensus rules."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

FOLDS=(0,1,2,4)
RULES=("strict_consensus","positive_agreement_weighted","mean_engine_margin")


def require(c,m):
    if not c:
        raise RuntimeError(m)


def load_npz(p):
    with np.load(p,allow_pickle=False) as z:
        return {k:np.asarray(z[k]) for k in z.files}


def find_fold_files(root, name):
    out={}
    for p in root.rglob(name):
        z=load_npz(p)
        if name=="symmetric.npz":
            ids=z["row_id"]; y=z["true_k"]
            # fold is inferred from companion report.
            rep=json.loads((p.parent/"report.json").read_text())
            f=int(rep["validation_fold"])
        else:
            ids=z["row_id"]; y=z["true_k"]
            rep=json.loads((p.parent/"report.json").read_text())
            f=int(rep["validation_fold"])
        require(f in FOLDS and f not in out,"duplicate/bad fold")
        out[f]=(p,ids,y)
    require(set(out)==set(FOLDS),f"missing {name} folds")
    return out


def margins_librosa(z):
    steps=np.asarray(z["steps"],np.int32)
    take=np.abs(steps)<=2
    p=np.asarray(z["probability"],np.float64)[:,take,:]
    return np.mean(p[:,:,2]-p[:,:,3],axis=1)


def margins_rubber(z):
    return np.asarray(z["mean_margin"],np.float64)


def rule_action(L,R,rule):
    if rule=="strict_consensus":
        return (L>0)&(R>0)
    if rule=="mean_engine_margin":
        return ((L+R)/2.0)>0
    if rule=="positive_agreement_weighted":
        denom=np.maximum(np.maximum(np.abs(L),np.abs(R)),1e-12)
        ratio=np.minimum(L,R)/denom
        return (L>0)&(R>0)&(ratio>=0.25)
    raise ValueError(rule)


def account(y,a):
    y=np.asarray(y,np.int32);a=np.asarray(a,bool)
    c=int(np.sum(a&(y==2)));r=int(np.sum(a&(y==3)))
    neutral=int(np.sum(a&~np.isin(y,(2,3))))
    return {"actions":int(a.sum()),"corrections":c,"regressions":r,
            "neutral_other_k":neutral,"net":c-r}


def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p);poly=y>=2
    return {
      "rows":int(len(y)),
      "exact":float(np.mean(y==p)),
      "poly_exact":float(np.mean(y[poly]==p[poly])) if poly.any() else None,
      "under":int(np.sum(p<y)),
      "over":int(np.sum(p>y)),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--librosa-internal",type=Path,required=True)
    ap.add_argument("--rubber-internal",type=Path,required=True)
    ap.add_argument("--librosa-outer",type=Path,required=True)
    ap.add_argument("--rubber-outer",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    require(not a.output.exists(),"refusing overwrite")

    lf=find_fold_files(a.librosa_internal,"symmetric.npz")
    rf=find_fold_files(a.rubber_internal,"rubberband_h2.npz")

    internal={}
    for rule in RULES:
        folds={}
        for f in FOLDS:
            lz=load_npz(lf[f][0]); rz=load_npz(rf[f][0])
            li=np.asarray(lz["row_id"],np.int64);ri=np.asarray(rz["row_id"],np.int64)
            require(set(li.tolist())==set(ri.tolist()),f"id mismatch fold {f}")
            rmap={int(v):i for i,v in enumerate(ri)}
            order=np.asarray([rmap[int(v)] for v in li],np.int64)
            y=np.asarray(lz["true_k"],np.int32)
            np.testing.assert_array_equal(y,np.asarray(rz["true_k"],np.int32)[order])
            L=margins_librosa(lz);R=margins_rubber(rz)[order]
            q=account(y,rule_action(L,R,rule))
            q["mean_abs_engine_margin_difference"]=float(np.mean(np.abs(L-R)))
            folds[str(f)]=q
        total={k:int(sum(folds[str(f)][k] for f in FOLDS))
               for k in ("actions","corrections","regressions","neutral_other_k","net")}
        internal[rule]={"folds":folds,"total":total,
                        "nonnegative_folds":int(sum(folds[str(f)]["net"]>=0 for f in FOLDS))}

    order={r:i for i,r in enumerate(RULES)}
    selected=min(RULES,key=lambda r:(
        -internal[r]["total"]["net"],
        internal[r]["total"]["regressions"],
        internal[r]["total"]["actions"],
        order[r]
    ))

    lo=load_npz(a.librosa_outer)
    ro=load_npz(a.rubber_outer)
    for key in ("global_index","k","base_predicted","target_global_index"):
        np.testing.assert_array_equal(lo[key],ro[key])
    y=np.asarray(lo["k"],np.int32)
    base=np.asarray(lo["base_predicted"],np.int32)
    tids=np.asarray(lo["target_global_index"],np.int64)
    L=np.asarray(lo["target_mean_margin"],np.float64)
    R=np.asarray(ro["target_mean_margin"],np.float64)
    require(len(tids)==len(L)==len(R),"outer target drift")
    target_action=rule_action(L,R,selected)

    pos={int(v):i for i,v in enumerate(np.asarray(lo["global_index"],np.int64))}
    acted_positions=np.asarray([pos[int(v)] for v in tids[target_action]],np.int64)
    corrected=base.copy(); corrected[acted_positions]=2
    changed=corrected!=base
    require(np.all(base[changed]==3)&np.all(corrected[changed]==2),"non 3->2 outer change")

    base_m=metrics(y,base);new_m=metrics(y,corrected)
    outer_acc=account(y[acted_positions],np.ones(len(acted_positions),bool))

    # Also expose each fixed rule on outer, diagnostic only.
    outer_rules={}
    for rule in RULES:
        aa=rule_action(L,R,rule)
        pp=np.asarray([pos[int(v)] for v in tids[aa]],np.int64)
        pred=base.copy();pred[pp]=2
        outer_rules[rule]={
          "accounting":account(y[pp],np.ones(len(pp),bool)),
          "metrics":metrics(y,pred)
        }

    report={
      "status":"completed",
      "experiment":"v273_dual_engine_h2_consensus",
      "internal":internal,
      "selected_rule":selected,
      "selection_source":"internal folds 0/1/2/4 only",
      "outer_fold":3,
      "outer_fold_previously_exposed":True,
      "outer_selected":{
        "accounting":outer_acc,
        "base_metrics":base_m,
        "corrected_metrics":new_m,
        "delta_exact_points":float(100*(new_m["exact"]-base_m["exact"])),
        "delta_poly_exact_points":float(100*(new_m["poly_exact"]-base_m["poly_exact"])),
      },
      "outer_all_rules_diagnostic":outer_rules,
      "automatic_promotion":False
    }

    a.output.mkdir(parents=True)
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    lines=["# Dual-engine H2 consensus","",
      "| rule | internal corr | internal reg | internal net | non-negative folds |",
      "|---|---:|---:|---:|---:|"]
    for r in RULES:
        q=internal[r]
        lines.append(f"| {r} | {q['total']['corrections']} | {q['total']['regressions']} | {q['total']['net']:+d} | {q['nonnegative_folds']}/4 |")
    lines+=["",f"Selected internally: **{selected}**.","",
      "| outer measure | base | selected consensus | delta |",
      "|---|---:|---:|---:|",
      f"| Exact-K global | {100*base_m['exact']:.3f}% | {100*new_m['exact']:.3f}% | {100*(new_m['exact']-base_m['exact']):+.3f} pt |",
      f"| Poly Exact-K | {100*base_m['poly_exact']:.3f}% | {100*new_m['poly_exact']:.3f}% | {100*(new_m['poly_exact']-base_m['poly_exact']):+.3f} pt |",
      "",
      f"Outer actions: **{outer_acc['actions']}**.",
      f"Outer corrections/regressions: **{outer_acc['corrections']}/{outer_acc['regressions']}**.",
      f"Outer neutral other-K: **{outer_acc['neutral_other_k']}**.",
      f"Outer net Exact-K: **{outer_acc['net']:+d}**.",
      "",
      "Fold 3 was already exposed historically. No automatic promotion."
    ]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    main()
