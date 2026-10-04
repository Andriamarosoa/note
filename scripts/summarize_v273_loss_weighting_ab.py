"""Summarize K1/K5 protection experiments against the frozen V27.3 uniform control."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scripts.train_v273_group_gate_ab import transitions

CONTROL="uniform"
ARMS=("protect_k1","protect_k5_tenth")

def load_arm(root, arm):
    d=root/arm
    report=json.loads((d/"report.json").read_text())
    with np.load(d/"predictions.npz",allow_pickle=False) as z:
        pred={k:np.asarray(z[k]) for k in z.files}
    return report,pred

def main(args):
    reports={}
    preds={}
    reports[CONTROL],preds[CONTROL]=load_arm(args.input,CONTROL)
    for arm in ARMS:
        reports[arm],preds[arm]=load_arm(args.input,arm)

    base=preds[CONTROL]
    k=base["k"]
    poly=k>=2
    paired={}
    for arm in ARMS:
        p=preds[arm]
        if not np.array_equal(base["global_index"],p["global_index"]) or not np.array_equal(k,p["k"]):
            raise RuntimeError(f"population mismatch: {arm}")
        paired[arm]={
            "all":transitions(k,base["predicted"],p["predicted"]),
            "poly":transitions(k[poly],base["predicted"][poly],p["predicted"][poly]),
        }

    out={"experiment":"v273_k1_k5_protection_isolation","reports":reports,"paired":paired}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")

    u=reports[CONTROL]["metrics"]
    lines=[
        "# V27.3 K1/K5 protection isolation","",
        "| Arm | Global | Δ global | Poly | Δ poly | K1 | K2 | K3 | K4 | K5 | Under | Over |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        m=reports[arm]["metrics"]
        def ex(v):
            x=m["by_k"][str(v)]["exact"]
            return "n/a" if x is None else f"{100*x:.3f}%"
        lines.append(
            f"| {arm} | {100*m['exact']:.3f}% | {100*(m['exact']-u['exact']):+.3f} pt | "
            f"{100*m['poly_exact']:.3f}% | {100*(m['poly_exact']-u['poly_exact']):+.3f} pt | "
            f"{ex(1)} | {ex(2)} | {ex(3)} | {ex(4)} | {ex(5)} | {m['under']} | {m['over']} |"
        )
        lines.append(
            f"Paired {arm}: global {paired[arm]['all']['net_correct']:+d}; "
            f"poly {paired[arm]['poly']['net_correct']:+d}."
        )
    lines += ["","Uniform is frozen from run 37208199201. No automatic promotion."]
    (args.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    main(p.parse_args())
