"""Summarize exact candidate_hidden1 group audits across internal folds 0,1,2,4.

Strict group robustness:
- low_net >= 0 on every fold;
- poly_net >= 0 on every fold;
- global_net > 0 on at least 3 of 4 folds;
- total global_net > 0.

Groups are identified by their sorted neuron tuple and must have been evaluated
exactly on every internal fold. Outer fold 3 is not read.
"""
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
        if fold in FOLDS and "groups" in r:
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
    by_fold={}
    keys=None
    for f,r in reports.items():
        m={tuple(sorted(int(x) for x in g["neurons"])):g for g in r["groups"]}
        by_fold[f]=m
        keys=set(m) if keys is None else keys & set(m)
    if not keys: raise RuntimeError("no common groups")

    rows=[]
    for key in sorted(keys,key=lambda x:(len(x),x)):
        per={}
        for f in FOLDS:
            g=by_fold[f][key]
            per[f]={
              "global_net":int(g["global_net"]),
              "low_net":int(g["low_net"]),
              "poly_net":int(g["poly_net"]),
              "K_net":g["K_net"],
              "changed_predictions":int(g["changed_predictions"])
            }
        gs=[per[f]["global_net"] for f in FOLDS]
        los=[per[f]["low_net"] for f in FOLDS]
        pos=[per[f]["poly_net"] for f in FOLDS]
        strict=(all(x>=0 for x in los) and all(x>=0 for x in pos)
                and sum(x>0 for x in gs)>=3 and sum(gs)>0)
        rows.append({
          "neurons":list(key),"size":len(key),
          "per_fold":{str(f):per[f] for f in FOLDS},
          "positive_global_folds":sum(x>0 for x in gs),
          "total_global_net":sum(gs),
          "total_low_net":sum(los),
          "total_poly_net":sum(pos),
          "min_global_net":min(gs),
          "min_low_net":min(los),
          "min_poly_net":min(pos),
          "strict_robust":strict
        })

    strict=[x for x in rows if x["strict_robust"]]
    strict.sort(key=lambda x:(x["total_global_net"],x["min_global_net"],
                              x["total_poly_net"],x["total_low_net"]),reverse=True)
    all_ranked=sorted(rows,key=lambda x:(
        x["positive_global_folds"],
        x["total_global_net"],
        x["min_global_net"],
        x["total_poly_net"],
        x["total_low_net"]
    ),reverse=True)
    selected=strict[0] if strict else None

    result={
      "status":"completed",
      "outer_fold_3_used":False,
      "folds":list(FOLDS),
      "groups_tested":len(rows),
      "strict_candidate_count":len(strict),
      "strict_candidates":strict,
      "selected_group":selected,
      "descriptive_top20":all_ranked[:20],
      "selection_rule":{
        "low_net_nonnegative_every_fold":True,
        "poly_net_nonnegative_every_fold":True,
        "global_positive_at_least_n_folds":3,
        "total_global_net_positive":True
      }
    }
    (a.output/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")

    lines=[
      "# Multi-fold exact group audit — candidate_hidden1","",
      f"Exact groups evaluated on all internal folds: **{len(rows)}**.",
      f"Strict robust groups: **{len(strict)}**.","",
      "| neurons | size | positive folds | total global | min global | total low | min low | total poly | min poly |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for x in all_ranked[:20]:
        lines.append(
          f"| {x['neurons']} | {x['size']} | {x['positive_global_folds']}/4 | "
          f"{x['total_global_net']:+d} | {x['min_global_net']:+d} | "
          f"{x['total_low_net']:+d} | {x['min_low_net']:+d} | "
          f"{x['total_poly_net']:+d} | {x['min_poly_net']:+d} |"
        )
    if selected:
        lines += ["",
          f"Selected robust group: **{selected['neurons']}**.",
          f"Total global **{selected['total_global_net']:+d}**, "
          f"low **{selected['total_low_net']:+d}**, "
          f"poly **{selected['total_poly_net']:+d}**."
        ]
    else:
        lines += ["","**No exact group satisfies the strict multi-fold robustness rule.**"]
    lines += ["","No outer evaluation and no promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":main()
