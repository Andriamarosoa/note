"""Exploratory outer confirmation of the internally selected B_low transition guard.

Important protocol note:
- the guard itself is selected only from internal folds 0/1/2/4;
- however outer fold 3 was already inspected diagnostically after the first B_low
  one-shot evaluation, so this is NOT a fresh independent outer validation.

Selected internal guard from run 37318538357:
  2->4, 3->2, 3->4, 4->2, 4->3
i.e. all K2/K3/K4 B_low correction transitions except 2->3.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

from scripts.v273_window_experiment import require

ALLOWED={(2,4),(3,2),(3,4),(4,2),(4,3)}
EXPECTED_ID="2->4+3->2+3->4+4->2+4->3"

def load_npz(path:Path):
    with np.load(path,allow_pickle=False) as z:
        return {k:np.asarray(z[k]) for k in z.files}

def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p)
    poly=y>=2
    out={
      "rows":int(len(y)),
      "exact":float(np.mean(y==p)),
      "poly_exact":float(np.mean(y[poly]==p[poly])) if poly.any() else None,
      "under":int(np.sum(p<y)),
      "over":int(np.sum(p>y)),
      "by_k":{}
    }
    for k in range(7):
        m=y==k
        out["by_k"][str(k)]={
          "rows":int(m.sum()),
          "exact":float(np.mean(p[m]==k)) if m.any() else None,
          "under":int(np.sum(p[m]<k)),
          "over":int(np.sum(p[m]>k))
        }
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--b-low-predictions",type=Path,required=True)
    ap.add_argument("--selection-report",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    require(not a.output.exists(),"refusing overwrite")
    a.output.mkdir(parents=True)

    sel=json.loads(a.selection_report.read_text())
    require(sel["outer_fold_3_used"] is False,"internal selection leaked outer")
    s=sel["selected_guard"]
    require(s is not None and s["strict_robust"] is True,"no strict guard")
    require(s["guard_id"]==EXPECTED_ID,"selected guard drift")
    require(s["positive_global_folds"]==4,"expected 4/4 positive folds")
    require(s["total_global_net"]==29 and s["min_global_net"]==5,"selection totals drift")

    z=load_npz(a.b_low_predictions)
    required={"global_index","k","base_predicted","corrected_predicted","B_low_mask"}
    require(required.issubset(z),"missing prediction fields")
    gid=z["global_index"].astype(np.int64)
    y=z["k"].astype(np.int32)
    base=z["base_predicted"].astype(np.int32)
    proposed=z["corrected_predicted"].astype(np.int32)
    bmask=z["B_low_mask"].astype(bool)
    require(len(gid)==len(y)==len(base)==len(proposed)==len(bmask),"shape mismatch")

    changed=bmask&(proposed!=base)
    allowed=np.zeros(len(y),dtype=bool)
    for src,dst in ALLOWED:
        allowed |= changed&(base==src)&(proposed==dst)

    guarded=base.copy()
    guarded[allowed]=proposed[allowed]

    corrections=int(np.sum((base!=y)&(guarded==y)))
    regressions=int(np.sum((base==y)&(guarded!=y)))
    net=int(np.sum(guarded==y)-np.sum(base==y))
    by_k={str(k):int(np.sum((guarded==y)&(y==k))-np.sum((base==y)&(y==k))) for k in range(7)}

    transition_rows={}
    for src,dst in sorted(ALLOWED):
        m=allowed&(base==src)&(proposed==dst)
        transition_rows[f"{src}->{dst}"]={
          "rows":int(m.sum()),
          "corrections":int(np.sum((base[m]!=y[m])&(guarded[m]==y[m]))),
          "regressions":int(np.sum((base[m]==y[m])&(guarded[m]!=y[m]))),
          "net":int(np.sum(guarded[m]==y[m])-np.sum(base[m]==y[m]))
        }

    bm=metrics(y,base); gm=metrics(y,guarded)
    report={
      "status":"completed",
      "protocol":{
        "experiment":"v273_outer_B_low_transition_guard_exploratory",
        "selection_source_run":37318538357,
        "selected_guard":EXPECTED_ID,
        "guard_transitions":[[a,b] for a,b in sorted(ALLOWED)],
        "outer_fold":3,
        "outer_used_for_guard_selection":False,
        "fresh_independent_outer_validation":False,
        "reason_not_fresh":"fold 3 had already been inspected diagnostically after the prior B_low one-shot evaluation",
        "automatic_promotion":False
      },
      "outer":{
        "rows":int(len(y)),
        "B_low_rows":int(bmask.sum()),
        "B_low_changed_before_guard":int(changed.sum()),
        "guard_applied_rows":int(allowed.sum()),
        "base_metrics":bm,
        "guarded_metrics":gm,
        "corrections":corrections,
        "regressions":regressions,
        "net_exact":net,
        "by_k_net":by_k,
        "transition_rows":transition_rows
      }
    }
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    np.savez_compressed(
      a.output/"predictions.npz",
      global_index=gid,k=y,
      base_predicted=base,
      b_low_predicted=proposed,
      guarded_predicted=guarded,
      B_low_mask=bmask.astype(np.uint8),
      guard_applied_mask=allowed.astype(np.uint8)
    )

    lines=[
      "# Exploratory outer confirmation — B_low transition guard","",
      "**Not a fresh independent outer validation:** fold 3 had already been inspected during the prior B_low diagnostic.","",
      f"Internal selected guard: **{EXPECTED_ID}** (4/4 positive folds, +29 total, min +5).","",
      "| Measure | Base [42,52,61,64] | + guarded B_low | Delta |",
      "|---|---:|---:|---:|",
      f"| exact global | {100*bm['exact']:.3f}% | {100*gm['exact']:.3f}% | {100*(gm['exact']-bm['exact']):+.3f} pt |",
      f"| exact poly | {100*bm['poly_exact']:.3f}% | {100*gm['poly_exact']:.3f}% | {100*(gm['poly_exact']-bm['poly_exact']):+.3f} pt |",
      f"| under | {bm['under']} | {gm['under']} | {gm['under']-bm['under']:+d} |",
      f"| over | {bm['over']} | {gm['over']} | {gm['over']-bm['over']:+d} |"
    ]
    for k in range(6):
        x=bm["by_k"][str(k)]["exact"]; q=gm["by_k"][str(k)]["exact"]
        lines.append(f"| K{k} exact | {100*x:.3f}% | {100*q:.3f}% | {100*(q-x):+.3f} pt |")
    lines += ["",
      f"Guard applied rows: **{int(allowed.sum())}**.",
      f"Corrections/regressions: **{corrections}/{regressions}**.",
      f"Net exact: **{net:+d}**.",
      "By-K net: "+", ".join(f"K{k} {by_k[str(k)]:+d}" for k in range(6))+".","",
      "Transition detail:"
    ]
    for t,x in transition_rows.items():
        lines.append(f"- {t}: {x['rows']} rows, {x['corrections']} corrections, {x['regressions']} regressions, net {x['net']:+d}")
    lines += ["","No automatic promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    main()
