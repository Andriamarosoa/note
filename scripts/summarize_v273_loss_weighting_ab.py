"""Summarize matched V27.3 learned_gate loss-weighting A/B."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scripts.train_v273_group_gate_ab import transitions

ARMS=("uniform","sqrt_balanced")

def main(args):
    reports={}
    preds={}
    for arm in ARMS:
        d=args.input/arm
        reports[arm]=json.loads((d/"report.json").read_text())
        with np.load(d/"predictions.npz",allow_pickle=False) as z:
            preds[arm]={k:np.asarray(z[k]) for k in z.files}

    a=preds["uniform"]; b=preds["sqrt_balanced"]
    if not np.array_equal(a["global_index"],b["global_index"]) or not np.array_equal(a["k"],b["k"]):
        raise RuntimeError("population mismatch")
    k=a["k"]; pa=a["predicted"]; pb=b["predicted"]
    poly=k>=2

    paired={
        "all":transitions(k,pa,pb),
        "poly":transitions(k[poly],pa[poly],pb[poly]),
    }
    out={"experiment":"v273_learned_gate_loss_weighting_ab","reports":reports,"paired":paired}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")

    u=reports["uniform"]["metrics"]; w=reports["sqrt_balanced"]["metrics"]
    lines=[
        "# V27.3 learned_gate loss weighting A/B","",
        "| Mesure | Uniform | sqrt-balanced | Delta |",
        "|---|---:|---:|---:|",
        f"| exact global | {100*u['exact']:.3f}% | {100*w['exact']:.3f}% | {100*(w['exact']-u['exact']):+.3f} pt |",
        f"| exact poly | {100*u['poly_exact']:.3f}% | {100*w['poly_exact']:.3f}% | {100*(w['poly_exact']-u['poly_exact']):+.3f} pt |",
    ]
    for v in (2,3,4):
        x=u["by_k"][str(v)]["exact"]; y=w["by_k"][str(v)]["exact"]
        lines.append(f"| K{v} exact | {100*x:.3f}% | {100*y:.3f}% | {100*(y-x):+.3f} pt |")
    lines += [
        f"| under | {u['under']} | {w['under']} | {w['under']-u['under']:+d} |",
        f"| over | {u['over']} | {w['over']} | {w['over']-u['over']:+d} |",
        "",
        f"Paired global net: {paired['all']['net_correct']:+d}.",
        f"Paired poly net: {paired['poly']['net_correct']:+d}.",
        "",
        "Aucune promotion automatique.",
    ]
    (args.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    main(p.parse_args())
