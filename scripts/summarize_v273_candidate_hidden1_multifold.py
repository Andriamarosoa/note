"""Aggregate rotating inner-fold candidate_hidden1 audits.

Strict robust selection uses only folds 0,1,2,4:
- low_net >= 0 on every fold;
- poly_net >= 0 on every fold;
- global_net > 0 on at least 3 of 4 folds;
- total global_net > 0.

Among strict candidates, select the neuron with maximum total global net,
then highest minimum per-fold global net, then total poly net, then total low net.
No outer fold data is read.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path

FOLDS=(0,1,2,4)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists(): raise RuntimeError("refusing overwrite")
    a.output.mkdir(parents=True)

    reports={}
    for fold in FOLDS:
        hits=list(a.input_root.rglob(f"fold-{fold}/report.json"))
        if not hits:
            hits=[p for p in a.input_root.rglob("report.json")
                  if json.loads(p.read_text()).get("protocol",{}).get("validation_fold")==fold]
        if len(hits)!=1:
            raise RuntimeError(f"fold {fold}: expected one report, got {hits}")
        r=json.loads(hits[0].read_text())
        if r["protocol"]["validation_fold"]!=fold:
            raise RuntimeError("fold mismatch")
        if r["protocol"]["outer_fold_3_loaded_for_fit_or_validation"] is not False:
            raise RuntimeError("outer leakage flag")
        reports[fold]=r

    rows=[]
    for neuron in range(96):
        per={}
        for fold in FOLDS:
            item=[x for x in reports[fold]["single_neuron"] if x["neuron"]==neuron]
            if len(item)!=1: raise RuntimeError("neuron missing")
            per[fold]=item[0]
        global_n=[per[f]["global_net"] for f in FOLDS]
        low_n=[per[f]["low_net"] for f in FOLDS]
        poly_n=[per[f]["poly_net"] for f in FOLDS]
        strict=(all(x>=0 for x in low_n) and all(x>=0 for x in poly_n)
                and sum(x>0 for x in global_n)>=3 and sum(global_n)>0)
        nonnegative_all=(all(x>=0 for x in global_n) and all(x>=0 for x in low_n)
                         and all(x>=0 for x in poly_n) and sum(global_n)>0)
        rows.append({
          "neuron":neuron,
          "per_fold":{str(f):{
            "global_net":per[f]["global_net"],
            "low_net":per[f]["low_net"],
            "poly_net":per[f]["poly_net"],
            "K_net":per[f]["K_net"],
            "changed_predictions":per[f]["changed_predictions"]
          } for f in FOLDS},
          "positive_global_folds":sum(x>0 for x in global_n),
          "total_global_net":sum(global_n),
          "total_low_net":sum(low_n),
          "total_poly_net":sum(poly_n),
          "min_global_net":min(global_n),
          "min_low_net":min(low_n),
          "min_poly_net":min(poly_n),
          "strict_robust":strict,
          "nonnegative_all_three_metrics_every_fold":nonnegative_all
        })

    strict=[x for x in rows if x["strict_robust"]]
    strict.sort(key=lambda x:(x["total_global_net"],x["min_global_net"],
                              x["total_poly_net"],x["total_low_net"]),reverse=True)
    selected=strict[0] if strict else None

    descriptive=sorted(rows,key=lambda x:(x["positive_global_folds"],x["total_global_net"],
                                           x["total_poly_net"],x["total_low_net"]),reverse=True)

    result={
      "status":"completed",
      "training_scope":"internal folds only",
      "folds":list(FOLDS),
      "outer_fold_3_used":False,
      "selection_rule":{
        "low_net_nonnegative_every_fold":True,
        "poly_net_nonnegative_every_fold":True,
        "global_positive_at_least_n_folds":3,
        "total_global_net_positive":True,
        "ranking":"total_global_net, then min_global_net, then total_poly_net, then total_low_net"
      },
      "strict_candidate_count":len(strict),
      "strict_candidates":strict,
      "selected_neuron":selected,
      "all_neurons":rows,
      "descriptive_top20":descriptive[:20],
      "reference_by_fold":{
        str(f):reports[f]["reference"] for f in FOLDS
      }
    }
    (a.output/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")

    lines=[
      "# Multi-fold internal validation — candidate_hidden1 neurons","",
      "Validation folds: 0, 1, 2, 4. Outer fold 3 was never used.","",
      f"Strict robust candidates: **{len(strict)}**.","",
      "| neuron | positive global folds | total global | min global | total low | min low | total poly | min poly |",
      "|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for x in descriptive[:20]:
        lines.append(f"| {x['neuron']} | {x['positive_global_folds']}/4 | "
                     f"{x['total_global_net']:+d} | {x['min_global_net']:+d} | "
                     f"{x['total_low_net']:+d} | {x['min_low_net']:+d} | "
                     f"{x['total_poly_net']:+d} | {x['min_poly_net']:+d} |")
    if selected:
        lines += ["",
          f"Selected robust neuron: **{selected['neuron']}**.",
          f"Total global net: **{selected['total_global_net']:+d}**; "
          f"total low net: **{selected['total_low_net']:+d}**; "
          f"total poly net: **{selected['total_poly_net']:+d}**.",
          f"Per-fold: {selected['per_fold']}."
        ]
    else:
        lines += ["","**No neuron satisfies the strict multi-fold robustness rule.**"]
    lines += ["","No outer evaluation and no promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":main()
