"""Aggregate internal B-like boundary corrector across folds 0,1,2,4."""
from __future__ import annotations
import argparse,json
from pathlib import Path

FOLDS=(0,1,2,4)

def discover(root):
    out={}
    for p in root.rglob("report.json"):
        try:r=json.loads(p.read_text())
        except Exception:continue
        f=r.get("protocol",{}).get("validation_fold")
        if f in FOLDS and "configs" in r and r.get("protocol",{}).get("experiment")=="v273_internal_B_like_boundary_corrector":
            if f in out: raise RuntimeError(f"duplicate fold {f}")
            out[f]=r
    if set(out)!=set(FOLDS):
        raise RuntimeError(f"missing folds {set(FOLDS)-set(out)}")
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
        ls=[per[f]["low_net"] for f in FOLDS]
        ps=[per[f]["poly_net"] for f in FOLDS]
        strict=(all(x>=0 for x in ls) and all(x>=0 for x in ps)
                and sum(x>0 for x in gs)>=3 and sum(gs)>0)
        first=per[FOLDS[0]]
        rows.append({
          "config_id":cid,
          "C":first["C"],
          "class_weight":first["class_weight"],
          "per_fold":{str(f):{
            "global_net":per[f]["global_net"],
            "low_net":per[f]["low_net"],
            "poly_net":per[f]["poly_net"],
            "corrections":per[f]["corrections"],
            "regressions":per[f]["regressions"],
            "under_delta":per[f]["under_delta"],
            "over_delta":per[f]["over_delta"],
            "applied_rows":per[f]["val_B_like_applied_rows"],
            "by_k_net":per[f]["by_k_net"],
          } for f in FOLDS},
          "positive_global_folds":sum(x>0 for x in gs),
          "total_global_net":sum(gs),
          "min_global_net":min(gs),
          "total_low_net":sum(ls),
          "min_low_net":min(ls),
          "total_poly_net":sum(ps),
          "min_poly_net":min(ps),
          "total_corrections":sum(per[f]["corrections"] for f in FOLDS),
          "total_regressions":sum(per[f]["regressions"] for f in FOLDS),
          "strict_robust":strict,
        })
    strict=[x for x in rows if x["strict_robust"]]
    strict.sort(key=lambda x:(x["total_global_net"],x["min_global_net"],
                              x["total_corrections"]-x["total_regressions"]),reverse=True)
    ranked=sorted(rows,key=lambda x:(
      x["strict_robust"],x["positive_global_folds"],x["total_global_net"],
      x["min_global_net"],x["total_poly_net"]
    ),reverse=True)
    selected=strict[0] if strict else None

    result={
      "status":"completed",
      "outer_fold_3_used":False,
      "folds":list(FOLDS),
      "configs_tested":len(rows),
      "strict_candidate_count":len(strict),
      "strict_candidates":strict,
      "selected_config":selected,
      "fallback":"no robust B-like corrector" if selected is None else None,
      "descriptive_ranking":ranked
    }
    (a.output/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")

    lines=[
      "# Internal B-like boundary corrector — multi-fold summary","",
      f"Configurations evaluated on all internal folds: **{len(rows)}**.",
      f"Strict robust configurations: **{len(strict)}**.","",
      "| config | positive folds | total global | min global | total poly | min poly | corrections | regressions |",
      "|---|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for x in ranked[:20]:
        lines.append(
          f"| {x['config_id']} | {x['positive_global_folds']}/4 | "
          f"{x['total_global_net']:+d} | {x['min_global_net']:+d} | "
          f"{x['total_poly_net']:+d} | {x['min_poly_net']:+d} | "
          f"{x['total_corrections']} | {x['total_regressions']} |"
        )
    if selected:
        lines += ["",
          f"Selected robust B-like corrector: **{selected['config_id']}**.",
          f"Total internal gain vs [42,52,61,64]: **{selected['total_global_net']:+d} exacts**.",
          f"Minimum fold gain: **{selected['min_global_net']:+d}**.",
          "",
          "Outer fold 3 remains untouched."
        ]
    else:
        lines += ["",
          "**No B-like boundary corrector satisfies the strict multi-fold rule.**",
          "",
          "No outer evaluation."
        ]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    main()
