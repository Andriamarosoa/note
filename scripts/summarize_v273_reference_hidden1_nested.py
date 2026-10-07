"""Summarize four outer folds of fixed hidden1 nested confirmation."""
from __future__ import annotations
import argparse,json
from pathlib import Path

FOLDS=(0,1,2,4)
GROUP=[42,52,61,64]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise RuntimeError("overwrite")
    a.output.mkdir(parents=True)

    reports={}
    for p in a.input_root.rglob("report.json"):
        r=json.loads(p.read_text())
        f=r.get("protocol",{}).get("outer_validation_fold")
        if f in FOLDS:
            if f in reports:raise RuntimeError(f"duplicate fold {f}")
            reports[f]=r
    if set(reports)!=set(FOLDS):raise RuntimeError(f"missing folds {set(FOLDS)-set(reports)}")

    fixed=[];policy=[];accepted=[]
    for f in FOLDS:
        r=reports[f]
        if r["protocol"]["fixed_group"]!=GROUP:raise RuntimeError("group drift")
        if r["protocol"]["fold3_used"] or r["protocol"]["player05_used"]:raise RuntimeError("scope leak")
        accepted.append(bool(r["inner_acceptance"]["accepted"]))
        fixed.append(r["outer_evaluation"]["fixed_group_delta"])
        policy.append(r["nested_policy_delta"])

    def aggregate(rows):
        gs=[x["global_net"] for x in rows];los=[x["low_net"] for x in rows];pos=[x["poly_net"] for x in rows]
        return {
          "per_fold":{str(f):rows[i] for i,f in enumerate(FOLDS)},
          "positive_global_folds":sum(x>0 for x in gs),
          "total_global_net":sum(gs),"total_low_net":sum(los),"total_poly_net":sum(pos),
          "min_global_net":min(gs),"min_low_net":min(los),"min_poly_net":min(pos),
          "strict_original_rule":bool(all(x>=0 for x in los) and all(x>=0 for x in pos)
                                      and sum(x>0 for x in gs)>=3 and sum(gs)>0),
        }

    fixed_a=aggregate(fixed);policy_a=aggregate(policy)
    confirmed=bool(all(accepted) and fixed_a["strict_original_rule"])
    result={
      "status":"completed",
      "experiment":"v273_fixed_hidden1_group_nested_confirmation",
      "fixed_group":GROUP,
      "folds":list(FOLDS),
      "fold3_used":False,
      "player05_used":False,
      "inner_accepted_by_outer_fold":{str(f):accepted[i] for i,f in enumerate(FOLDS)},
      "accepted_all_outer_rotations":all(accepted),
      "fixed_group_outer":fixed_a,
      "nested_policy_outer":policy_a,
      "predeclared_confirmation_passed":confirmed,
      "decision":"reference_confirmed" if confirmed else "reference_not_confirmed_by_nested_validation",
      "automatic_replacement":False,
    }
    (a.output/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    lines=["# Nested confirmation — V27.3 hidden1 [42,52,61,64]","",
           f"Inner acceptance in all four rotations: **{all(accepted)}**.","",
           "| outer fold | inner accepted | fixed global | fixed low | fixed poly | policy global |",
           "|---:|---|---:|---:|---:|---:|"]
    for i,f in enumerate(FOLDS):
        d=fixed[i];p=policy[i]
        lines.append(f"| {f} | {accepted[i]} | {d['global_net']:+d} | {d['low_net']:+d} | {d['poly_net']:+d} | {p['global_net']:+d} |")
    lines += ["",
      f"Fixed group total: global **{fixed_a['total_global_net']:+d}**, low **{fixed_a['total_low_net']:+d}**, poly **{fixed_a['total_poly_net']:+d}**.",
      f"Fixed group original strict rule: **{fixed_a['strict_original_rule']}**.",
      f"Nested policy total global: **{policy_a['total_global_net']:+d}**.",
      "",
      f"## Confirmatory verdict: **{'PASS' if confirmed else 'FAIL'}**",
      "",
      "No fold 3, no player 05, no alternative group search."
    ]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":main()
