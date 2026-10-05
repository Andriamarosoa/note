"""Aggregate the minimal residual failure audit across internal folds."""
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path

FOLDS=(0,1,2,4)
FEATURES=("best_pair_residual_ratio","best_triplet_residual_ratio")

def discover(root):
    out={}
    for p in root.rglob("report.json"):
        try:
            r=json.loads(p.read_text())
        except Exception:
            continue
        if r.get("protocol",{}).get("experiment")!="v273_internal_B_low_minimal_residual_failure_audit":
            continue
        f=r["protocol"].get("validation_fold")
        if f in FOLDS:
            if f in out:
                raise RuntimeError(f"duplicate fold {f}")
            out[f]=r
    if set(out)!=set(FOLDS):
        raise RuntimeError(f"missing folds {set(FOLDS)-set(out)}")
    return out

def avg(xs):
    xs=[x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():
        raise RuntimeError("refusing overwrite")
    a.output.mkdir(parents=True)
    r=discover(a.input_root)

    rows=[]
    for f in FOLDS:
        q=r[f]
        row={
          "fold":f,
          "joint_auc":q["joint"]["val_auc"],
          "pair_auc":q["univariate"][FEATURES[0]]["val_auc"],
          "triplet_auc":q["univariate"][FEATURES[1]]["val_auc"],
          "pair_coef":q["joint"]["coefficients_standardized"][FEATURES[0]],
          "triplet_coef":q["joint"]["coefficients_standardized"][FEATURES[1]],
          "fit_corr":q["correlation"]["fit_all"],
          "val_corr":q["correlation"]["val_all"],
          "fit_corr_K2":q["correlation"]["fit_K2"],
          "fit_corr_K3":q["correlation"]["fit_K3"],
          "val_corr_K2":q["correlation"]["val_K2"],
          "val_corr_K3":q["correlation"]["val_K3"],
          "condition_number":q["standardized_fit_condition_number"],
          "registers":q["registers"],
          "pair_fit_delta":q["univariate"][FEATURES[0]]["fit_class_mean_delta"],
          "pair_val_delta":q["univariate"][FEATURES[0]]["val_class_mean_delta"],
          "triplet_fit_delta":q["univariate"][FEATURES[1]]["fit_class_mean_delta"],
          "triplet_val_delta":q["univariate"][FEATURES[1]]["val_class_mean_delta"],
        }
        rows.append(row)

    fold2=next(x for x in rows if x["fold"]==2)
    positives=[x for x in rows if x["fold"] in (1,4)]

    def sign(x):
        return 0 if abs(x)<1e-12 else (1 if x>0 else -1)

    diagnosis={
      "fold2_both_univariate_below_half":bool(fold2["pair_auc"]<0.5 and fold2["triplet_auc"]<0.5),
      "fold2_fit_to_val_pair_direction_flip":sign(fold2["pair_fit_delta"])!=sign(fold2["pair_val_delta"]),
      "fold2_fit_to_val_triplet_direction_flip":sign(fold2["triplet_fit_delta"])!=sign(fold2["triplet_val_delta"]),
      "fold2_higher_fit_correlation_than_positive_mean":bool(
        fold2["fit_corr"] > avg([x["fit_corr"] for x in positives])
      ),
      "fold2_condition_number_vs_positive_mean_ratio":float(
        fold2["condition_number"]/max(avg([x["condition_number"] for x in positives]),1e-12)
      ),
    }

    register_summary={}
    for name in ("low","mid","high"):
        vals=[]
        for x in rows:
            q=x["registers"][name]
            vals.append({
              "fold":x["fold"],
              "joint_auc":q.get("joint_auc"),
              "pair_auc":q.get("pair_auc"),
              "triplet_auc":q.get("triplet_auc"),
              "k23_exact_net":q.get("k23_exact_net"),
              "val_rows":q.get("val_rows"),
            })
        register_summary[name]=vals

    result={
      "status":"completed","outer_fold_3_used":False,
      "folds":list(FOLDS),
      "folds_detail":rows,
      "diagnosis_flags":diagnosis,
      "register_summary":register_summary,
      "means":{
        "joint_auc":avg([x["joint_auc"] for x in rows]),
        "pair_auc":avg([x["pair_auc"] for x in rows]),
        "triplet_auc":avg([x["triplet_auc"] for x in rows]),
        "fit_corr":avg([x["fit_corr"] for x in rows]),
        "condition_number":avg([x["condition_number"] for x in rows]),
      }
    }
    (a.output/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")

    lines=["# Minimal residual failure audit — multi-fold summary","",
      "| fold | joint AUC | pair AUC | triplet AUC | pair coef | triplet coef | fit corr | val corr | cond # |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in rows:
        lines.append(
          f"| {x['fold']} | {x['joint_auc']:.3f} | {x['pair_auc']:.3f} | {x['triplet_auc']:.3f} | "
          f"{x['pair_coef']:+.3f} | {x['triplet_coef']:+.3f} | {x['fit_corr']:.3f} | "
          f"{x['val_corr']:.3f} | {x['condition_number']:.1f} |"
        )
    lines += ["","## Fold 2 checks","",
      f"- pair FIT→VAL direction flip: **{diagnosis['fold2_fit_to_val_pair_direction_flip']}**",
      f"- triplet FIT→VAL direction flip: **{diagnosis['fold2_fit_to_val_triplet_direction_flip']}**",
      f"- both fold-2 univariate AUCs < 0.5: **{diagnosis['fold2_both_univariate_below_half']}**",
      f"- fold-2 fit correlation > positive-fold mean: **{diagnosis['fold2_higher_fit_correlation_than_positive_mean']}**",
      f"- fold-2 condition-number / positive-fold mean: **{diagnosis['fold2_condition_number_vs_positive_mean_ratio']:.2f}x**",
      "","## By register",""]
    for name in ("low","mid","high"):
        lines.append(f"### {name}")
        lines.append("| fold | joint AUC | pair AUC | triplet AUC | K2/K3 net |")
        lines.append("|---|---:|---:|---:|---:|")
        for q in register_summary[name]:
            def ff(v):
                return "n/a" if v is None else f"{v:.3f}"
            lines.append(f"| {q['fold']} | {ff(q['joint_auc'])} | {ff(q['pair_auc'])} | {ff(q['triplet_auc'])} | {q['k23_exact_net'] if q['k23_exact_net'] is not None else 'n/a'} |")
        lines.append("")
    lines += ["No outer evaluation and no promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    main()
