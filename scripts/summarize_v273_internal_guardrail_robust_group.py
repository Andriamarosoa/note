"""Aggregate internal guardrail rules across validation folds 0,1,2,4."""
from __future__ import annotations
import argparse,json
from pathlib import Path

FOLDS=(0,1,2,4)

def discover(root):
    out={}
    for p in root.rglob("report.json"):
        try:r=json.loads(p.read_text())
        except Exception:continue
        fold=r.get("protocol",{}).get("validation_fold")
        if fold in FOLDS and "rules" in r:
            if fold in out: raise RuntimeError(f"duplicate fold {fold}")
            out[fold]=r
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
    by={}
    for f,r in reports.items():
        by[f]={x["rule_id"]:x for x in r["rules"]}
    common=set.intersection(*(set(by[f]) for f in FOLDS))
    if not common: raise RuntimeError("no common rules")

    rows=[]
    for rid in sorted(common):
        per={f:by[f][rid] for f in FOLDS}
        gs=[per[f]["global_net"] for f in FOLDS]
        los=[per[f]["low_net"] for f in FOLDS]
        pos=[per[f]["poly_net"] for f in FOLDS]
        strict=(all(x>=0 for x in los) and all(x>=0 for x in pos)
                and sum(x>0 for x in gs)>=3 and sum(gs)>0)
        first=per[FOLDS[0]]
        rows.append({
          "rule_id":rid,
          "zones":first["zones"],
          "threshold":first["threshold"],
          "per_fold":{str(f):{
             "global_net":per[f]["global_net"],
             "low_net":per[f]["low_net"],
             "poly_net":per[f]["poly_net"],
             "reverted":per[f]["reverted"],
             "under_delta":per[f]["under_delta"],
             "over_delta":per[f]["over_delta"],
          } for f in FOLDS},
          "positive_global_folds":sum(x>0 for x in gs),
          "total_global_net":sum(gs),
          "min_global_net":min(gs),
          "total_low_net":sum(los),
          "min_low_net":min(los),
          "total_poly_net":sum(pos),
          "min_poly_net":min(pos),
          "strict_robust":strict,
        })

    strict=[x for x in rows if x["strict_robust"]]
    strict.sort(key=lambda x:(x["total_global_net"],x["min_global_net"],
                              x["total_poly_net"],x["total_low_net"]),reverse=True)
    ranked=sorted(rows,key=lambda x:(
        x["strict_robust"],x["positive_global_folds"],x["total_global_net"],
        x["min_global_net"],x["total_poly_net"],x["total_low_net"]
    ),reverse=True)
    selected=strict[0] if strict else None
    result={
      "status":"completed",
      "outer_fold_3_used":False,
      "folds":list(FOLDS),
      "rules_tested":len(rows),
      "strict_candidate_count":len(strict),
      "strict_candidates":strict,
      "selected_rule":selected,
      "fallback_to_cluster_B":selected is None,
      "selection_rule":{
        "low_net_nonnegative_every_fold":True,
        "poly_net_nonnegative_every_fold":True,
        "global_positive_at_least_n_folds":3,
        "total_global_net_positive":True
      },
      "descriptive_top20":ranked[:20]
    }
    (a.output/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")

    lines=[
      "# Internal guardrail multi-fold summary","",
      f"Rules tested on all internal folds: **{len(rows)}**.",
      f"Strict robust progressions: **{len(strict)}**.","",
      "| rule | positive folds | total global | min global | total low | min low | total poly | min poly |",
      "|---|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for x in ranked[:20]:
        lines.append(
          f"| {x['rule_id']} | {x['positive_global_folds']}/4 | "
          f"{x['total_global_net']:+d} | {x['min_global_net']:+d} | "
          f"{x['total_low_net']:+d} | {x['min_low_net']:+d} | "
          f"{x['total_poly_net']:+d} | {x['min_poly_net']:+d} |"
        )
    if selected:
        lines += ["",
          f"Selected internal guardrail: **{selected['rule_id']}**.",
          f"Total vs robust group: global **{selected['total_global_net']:+d}**, "
          f"low **{selected['total_low_net']:+d}**, poly **{selected['total_poly_net']:+d}**.",
          "",
          "**Progression found: do not switch to Cluster B yet.**"
        ]
    else:
        lines += ["",
          "**No strict robust progression above [42,52,61,64].**",
          "",
          "**Fallback decision: switch directly to Cluster B.**"
        ]
    lines += ["","No outer evaluation and no promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    main()
