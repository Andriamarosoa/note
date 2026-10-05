"""Aggregate B_low base-K3 K2 discriminator across internal folds."""
from __future__ import annotations
import argparse,json
from pathlib import Path
FOLDS=(0,1,2,4)

def discover(root):
    out={}
    for p in root.rglob("report.json"):
        try:r=json.loads(p.read_text())
        except Exception:continue
        if r.get("protocol",{}).get("experiment")!="v273_internal_B_low_baseK3_K2_discriminator":
            continue
        f=r["protocol"].get("validation_fold")
        if f in FOLDS:
            if f in out: raise RuntimeError(f"duplicate fold {f}")
            out[f]=r
    if set(out)!=set(FOLDS): raise RuntimeError(f"missing folds {set(FOLDS)-set(out)}")
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists(): raise RuntimeError("refusing overwrite")
    a.output.mkdir(parents=True)

    reports=discover(a.input_root)
    by={f:{x["config_id"]:x for x in reports[f]["configs"]} for f in FOLDS}
    common=set.intersection(*(set(by[f]) for f in FOLDS))
    rows=[]
    for cid in sorted(common):
        per={f:by[f][cid] for f in FOLDS}
        gs=[per[f]["global_net"] for f in FOLDS]
        ps=[per[f]["poly_net"] for f in FOLDS]
        ls=[per[f]["low_net"] for f in FOLDS]
        k2=[per[f]["by_k_net"]["2"] for f in FOLDS]
        k3=[per[f]["by_k_net"]["3"] for f in FOLDS]
        strict=(all(x>=0 for x in ps) and all(x>=0 for x in ls)
                and sum(x>0 for x in gs)>=3 and sum(gs)>0)
        first=per[FOLDS[0]]
        aucs=[per[f]["val_auc_k2_vs_rest"] for f in FOLDS if per[f]["val_auc_k2_vs_rest"] is not None]
        rows.append({
          "config_id":cid,
          "C":first["C"],
          "class_weight":first["class_weight"],
          "threshold":first["threshold"],
          "per_fold":{str(f):{
            "global_net":per[f]["global_net"],
            "poly_net":per[f]["poly_net"],
            "low_net":per[f]["low_net"],
            "corrections":per[f]["corrections"],
            "regressions":per[f]["regressions"],
            "applied_rows":per[f]["applied_rows"],
            "by_k_net":per[f]["by_k_net"],
            "auc":per[f]["val_auc_k2_vs_rest"]
          } for f in FOLDS},
          "positive_global_folds":sum(x>0 for x in gs),
          "total_global_net":sum(gs),
          "min_global_net":min(gs),
          "total_poly_net":sum(ps),
          "min_poly_net":min(ps),
          "total_k2_net":sum(k2),
          "min_k2_net":min(k2),
          "total_k3_net":sum(k3),
          "max_k3_loss":min(k3),
          "mean_auc":sum(aucs)/len(aucs) if aucs else None,
          "total_corrections":sum(per[f]["corrections"] for f in FOLDS),
          "total_regressions":sum(per[f]["regressions"] for f in FOLDS),
          "strict_robust":strict
        })
    strict=[x for x in rows if x["strict_robust"]]
    strict.sort(key=lambda x:(x["total_global_net"],x["min_global_net"],x["total_k2_net"],x["total_k3_net"]),reverse=True)
    ranked=sorted(rows,key=lambda x:(x["strict_robust"],x["positive_global_folds"],x["total_global_net"],x["min_global_net"]),reverse=True)
    selected=strict[0] if strict else None

    result={
      "status":"completed",
      "outer_fold_3_used":False,
      "folds":list(FOLDS),
      "configs_tested":len(rows),
      "strict_candidate_count":len(strict),
      "strict_candidates":strict,
      "selected_config":selected,
      "descriptive_ranking":ranked
    }
    (a.output/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")

    lines=[
      "# B_low base-K3 K2 discriminator — multi-fold summary","",
      f"Configurations evaluated: **{len(rows)}**.",
      f"Strict robust configurations: **{len(strict)}**.","",
      "| config | positive folds | total global | min global | total K2 | total K3 | mean AUC | corr | regr |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for x in ranked[:24]:
        auc="n/a" if x["mean_auc"] is None else f"{x['mean_auc']:.3f}"
        lines.append(
          f"| {x['config_id']} | {x['positive_global_folds']}/4 | "
          f"{x['total_global_net']:+d} | {x['min_global_net']:+d} | "
          f"{x['total_k2_net']:+d} | {x['total_k3_net']:+d} | {auc} | "
          f"{x['total_corrections']} | {x['total_regressions']} |"
        )
    if selected:
        lines += ["",
          f"Selected robust discriminator: **{selected['config_id']}**.",
          f"Internal total gain: **{selected['total_global_net']:+d} exacts**; minimum fold: **{selected['min_global_net']:+d}**.",
          f"K2 net: **{selected['total_k2_net']:+d}**; K3 net: **{selected['total_k3_net']:+d}**.",
          "",
          "Outer fold 3 remains untouched."
        ]
    else:
        lines += ["",
          "**No discriminator configuration satisfies the strict multi-fold rule.**",
          "",
          "No outer evaluation."
        ]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    main()
