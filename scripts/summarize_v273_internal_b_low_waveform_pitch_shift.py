"""Aggregate waveform pitch-shift audit across internal folds."""
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path
FOLDS=(0,1,2,4)

def discover(root):
    out={}
    for p in root.rglob("report.json"):
        try:r=json.loads(p.read_text())
        except Exception:continue
        if r.get("protocol",{}).get("experiment")!="v273_internal_B_low_waveform_pitch_shift":
            continue
        f=r["protocol"].get("validation_fold")
        if f in FOLDS:
            if f in out:raise RuntimeError(f"duplicate fold {f}")
            out[f]=r
    if set(out)!=set(FOLDS):raise RuntimeError(f"missing folds {set(FOLDS)-set(out)}")
    return out

def avg(vals):
    vals=[x for x in vals if x is not None]
    return statistics.mean(vals) if vals else None

def amin(vals):
    vals=[x for x in vals if x is not None]
    return min(vals) if vals else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise RuntimeError("refusing overwrite")
    a.output.mkdir(parents=True)
    r=discover(a.input_root)

    reps={}
    for rep in ("x1","x2","x4"):
        same=[r[f]["results"][rep]["same_representation_auc"] for f in FOLDS]
        transfer=[r[f]["results"][rep]["natural_high_model_transfer_auc"] for f in FOLDS]
        frac=[r[f]["results"][rep]["fraction_shifted_low_above_original_high_cut"] for f in FOLDS]
        reps[rep]={
          "same_auc_by_fold":{str(f):same[i] for i,f in enumerate(FOLDS)},
          "same_mean_auc":avg(same),
          "same_min_auc":amin(same),
          "transfer_auc_by_fold":{str(f):transfer[i] for i,f in enumerate(FOLDS)},
          "transfer_mean_auc":avg(transfer),
          "transfer_min_auc":amin(transfer),
          "mean_fraction_above_old_high_cut":avg(frac),
        }

    natural_high=[r[f]["natural_high_auc"] for f in FOLDS]
    result={
      "status":"completed",
      "outer_fold_3_used":False,
      "folds":list(FOLDS),
      "natural_high_auc_by_fold":{str(f):natural_high[i] for i,f in enumerate(FOLDS)},
      "natural_high_mean_auc":avg(natural_high),
      "representations":reps,
      "delta_x2_vs_x1":None if reps["x1"]["same_mean_auc"] is None or reps["x2"]["same_mean_auc"] is None else reps["x2"]["same_mean_auc"]-reps["x1"]["same_mean_auc"],
      "delta_x4_vs_x1":None if reps["x1"]["same_mean_auc"] is None or reps["x4"]["same_mean_auc"] is None else reps["x4"]["same_mean_auc"]-reps["x1"]["same_mean_auc"],
    }
    (a.output/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    lines=["# Waveform pitch shift — multi-fold summary","",
      f"Natural-high mean AUC: **{result['natural_high_mean_auc']:.3f}**.","",
      "| view | same low rows mean AUC | min fold | high-model transfer mean AUC | min transfer | low rows above old high cut |",
      "|---|---:|---:|---:|---:|---:|"]
    for rep in ("x1","x2","x4"):
        x=reps[rep]
        def ff(v):return "n/a" if v is None else f"{v:.3f}"
        lines.append(f"| {rep} | {ff(x['same_mean_auc'])} | {ff(x['same_min_auc'])} | "
                     f"{ff(x['transfer_mean_auc'])} | {ff(x['transfer_min_auc'])} | "
                     f"{ff(x['mean_fraction_above_old_high_cut'])} |")
    if result["delta_x2_vs_x1"] is not None:
        lines.append(f"\nx2 vs x1 AUC delta: **{result['delta_x2_vs_x1']:+.3f}**.")
    if result["delta_x4_vs_x1"] is not None:
        lines.append(f"x4 vs x1 AUC delta: **{result['delta_x4_vs_x1']:+.3f}**.")
    lines += ["",
      "This test uses duration-preserving waveform pitch shifting before recomputing the STFT/harmonic features.",
      "No outer evaluation and no promotion."
    ]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    main()
