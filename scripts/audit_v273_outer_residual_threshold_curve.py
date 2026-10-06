"""Replay outer fold3 residual threshold curve 0.60..0.64 from frozen p064 outputs.

No refit, no audio, no tuning. Uses candidate probabilities saved by run
37546289320 and the frozen B_low mask to reconstruct action masks.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

GRID=(.60,.61,.62,.63,.64)

def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:np.asarray(z[k]) for k in z.files}

def counts(y,m):
    c=int(np.sum(m&(y==2)));r=int(np.sum(m&(y==3)));o=int(np.sum(m&~np.isin(y,(2,3))))
    return {"actions":int(m.sum()),"corrections":c,"regressions":r,"other_k":o,"net":c-r}

def metrics(y,p):
    poly=y>=2
    return {"exact":float(np.mean(y==p)),"poly_exact":float(np.mean(y[poly]==p[poly]))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--outer-p064",type=Path,required=True)
    ap.add_argument("--outer-blow",type=Path,required=True)
    ap.add_argument("--internal-threshold-report",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise RuntimeError("overwrite")
    p64=load(a.outer_p064);b=load(a.outer_blow)
    y=p64["k"].astype(int);base=p64["base_predicted"].astype(int)
    np.testing.assert_array_equal(p64["global_index"],b["global_index"])
    blow=b["B_low_mask"].astype(bool)
    cand=blow&(base==3)
    prob=p64["candidate_probability"].astype(float)
    if len(prob)!=int(cand.sum()):raise RuntimeError("candidate probability alignment")
    internal=json.loads(a.internal_threshold_report.read_text())
    by_th={round(float(q["threshold"]),2):q for q in internal["candidates"]}

    rep={}
    cidx=np.flatnonzero(cand)
    for th in GRID:
        local=np.isfinite(prob)&(prob>=th)
        gm=np.zeros(len(y),bool);gm[cidx[local]]=True
        pred=base.copy();pred[gm]=2
        q=counts(y,gm)
        iq=by_th[round(th,2)]
        rep[f"{th:.2f}"]={"outer_actions":q,"outer_metrics":metrics(y,pred),
                         "internal_total_net":iq["total"]["net"],
                         "internal_min_fold":min(x["net"] for x in iq["folds"].values()),
                         "internal_min_player":min(x["net"] for x in iq["players"].values()),
                         "internal_min_style":min(x["net"] for x in iq["styles"].values()),
                         "internal_feasible_strict":bool(iq["feasible"])}
    out={"status":"completed","experiment":"v273_outer_residual_threshold_curve_060_064",
         "thresholds":rep,"outer_fold_3_fresh_independent":False,"outer_tuning":False}
    a.output.mkdir(parents=True);(a.output/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    lines=["# Outer residual threshold curve 0.60–0.64","",
           "Replay only. Fold 3 historically exposed; do not select a threshold from this table.","",
           "| th | internal net | min fold | min player | min style | strict internal | outer corr/reg | outer net | outer global | outer poly |",
           "|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for th in GRID:
        q=rep[f"{th:.2f}"];a0=q["outer_actions"];m=q["outer_metrics"]
        lines.append(f"| {th:.2f} | {q['internal_total_net']:+d} | {q['internal_min_fold']:+d} | {q['internal_min_player']:+d} | {q['internal_min_style']:+d} | {q['internal_feasible_strict']} | {a0['corrections']}/{a0['regressions']} | {a0['net']:+d} | {100*m['exact']:.3f}% | {100*m['poly_exact']:.3f}% |")
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":main()
