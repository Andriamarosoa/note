"""Nested native-margin gate for low-register residual actions in p=.60-.65.

The raw residual LR remains the main corrector:
  apply 3->2 when residual p(K2) >= .60.

Only inside register=low and .60 <= residual p(K2) < .65, test an independent
native confidence signal from the frozen cardinality model:
  native_k2_share = P_base(K2) / (P_base(K2) + P_base(K3)).

For each held-out fold, choose the minimum native_k2_share required to KEEP the
3->2 correction using only inner folds. A candidate gate is feasible only if its
effect versus the ungated baseline is non-negative on every inner fold. Choose
the gate with maximum total inner improvement, then higher kept-action K2
precision, then the stricter threshold. If no positive feasible gate exists,
disable the gate for that held fold.

The policy is applied to all target-band actions, including other-K.
No outer fold 3. No automatic promotion.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.v273_residual_audit import require

FOLDS=(0,1,2,4)
R0=.60
R1=.65
GRID=tuple(round(x,3) for x in np.arange(.05,.501,.025))
EPS=1e-12

def load_rows(root):
    rows=[]
    for fold in FOLDS:
        p=Path(root)/f"fold-{fold}"/"replay.npz"
        with np.load(p,allow_pickle=False) as z:a={k:np.asarray(z[k]) for k in z.files}
        action=a["val_b_low"].astype(bool)&(a["val_base_k"].astype(int)==3)
        valid=a["val_valid"].astype(bool)
        y=a["val_true_k"].astype(int)[action][valid]
        rp=a["val_probability"].astype(float)
        reg=np.asarray(a["val_register"]).astype(str)
        ids=a["val_action_ids"][valid].astype(int)
        P=a["val_base_probability"][action][valid].astype(float)
        require(len(y)==len(rp)==len(reg)==len(ids)==len(P),"length drift")
        require(P.ndim==2 and P.shape[1]>=4,"base probability shape")
        for i in range(len(y)):
            p2=float(P[i,2]);p3=float(P[i,3])
            rows.append({"fold":fold,"row_id":int(ids[i]),"true_k":int(y[i]),
                         "residual_p_k2":float(rp[i]),"register":str(reg[i]),
                         "base_p2":p2,"base_p3":p3,
                         "native_k2_share":p2/(p2+p3+EPS),
                         "native_margin_p2_minus_p3":p2-p3})
    return rows

def counts(rows,mask):
    y=np.asarray([r["true_k"] for r in rows],int);m=np.asarray(mask,bool)
    c=int(np.sum(m&(y==2)));g=int(np.sum(m&(y==3)));o=int(np.sum(m&~np.isin(y,(2,3))))
    return {"actions":int(m.sum()),"corrections":c,"regressions":g,"other_k":o,"net":c-g}

def target(r):
    return r["register"]=="low" and R0<=r["residual_p_k2"]<R1

def inner_select(fit_rows):
    band=[r for r in fit_rows if target(r) and r["true_k"] in (2,3)]
    folds=sorted({r["fold"] for r in band})
    candidates=[]
    for th in GRID:
        per=[];base_c=base_g=keep_c=keep_g=0
        for f in folds:
            rr=[r for r in band if r["fold"]==f]
            bc=sum(r["true_k"]==2 for r in rr);bg=sum(r["true_k"]==3 for r in rr)
            kept=[r for r in rr if r["native_k2_share"]>=th]
            kc=sum(r["true_k"]==2 for r in kept);kg=sum(r["true_k"]==3 for r in kept)
            baseline_net=bc-bg;kept_net=kc-kg
            effect=kept_net-baseline_net
            per.append({"fold":int(f),"baseline_K2":bc,"baseline_K3":bg,"baseline_net":baseline_net,
                        "kept_K2":kc,"kept_K3":kg,"kept_net":kept_net,"effect_vs_baseline":effect})
            base_c+=bc;base_g+=bg;keep_c+=kc;keep_g+=kg
        total_effect=(keep_c-keep_g)-(base_c-base_g)
        feasible=all(q["effect_vs_baseline"]>=0 for q in per)
        precision=None if keep_c+keep_g==0 else keep_c/(keep_c+keep_g)
        candidates.append({"threshold":float(th),"baseline_K2":base_c,"baseline_K3":base_g,
                           "kept_K2":keep_c,"kept_K3":keep_g,"effect_vs_baseline":total_effect,
                           "kept_precision_K2":precision,"feasible":feasible,"per_fold":per})
    feasible=[q for q in candidates if q["feasible"] and q["effect_vs_baseline"]>0]
    if not feasible:return None,candidates
    best=max(feasible,key=lambda q:(q["effect_vs_baseline"],
                                    -1 if q["kept_precision_K2"] is None else q["kept_precision_K2"],
                                    q["threshold"]))
    return best,candidates

def auc_report(rows):
    rr=[r for r in rows if r["true_k"] in (2,3)]
    if len(rr)<4 or len({r["true_k"] for r in rr})<2:return None
    y=np.asarray([r["true_k"]==2 for r in rr],int)
    s=np.asarray([r["native_k2_share"] for r in rr],float)
    return float(roc_auc_score(y,s))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--internal-exports",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args();require(not a.output.exists(),"overwrite")
    rows=load_rows(a.internal_exports)

    baseline=np.asarray([r["residual_p_k2"]>=R0 for r in rows],bool)
    final=baseline.copy()
    details=[]

    for f in FOLDS:
        fit=[r for r in rows if r["fold"]!=f]
        best,cands=inner_select(fit)
        th=None if best is None else float(best["threshold"])
        for i,r in enumerate(rows):
            if r["fold"]!=f or not baseline[i] or not target(r):continue
            if th is not None and r["native_k2_share"]<th:
                final[i]=False
        val_band=[r for r in rows if r["fold"]==f and target(r)]
        details.append({"fold":int(f),"selected_threshold":th,"inner_best":best,
                        "inner_candidates":cands,"outer_band_rows":len(val_band),
                        "outer_band_auc_k2":auc_report(val_band)})

    abstain=baseline&~final
    base_all=counts(rows,baseline);ab_all=counts(rows,abstain);fin_all=counts(rows,final)
    per=[]
    for f in FOLDS:
        m=np.asarray([r["fold"]==f for r in rows],bool)
        b=counts(rows,baseline&m);ab=counts(rows,abstain&m);fin=counts(rows,final&m)
        band=[r for r in rows if r["fold"]==f and target(r)]
        per.append({"fold":int(f),"baseline":b,"abstained":ab,
                    "gate_effect":fin["net"]-b["net"],"final":fin,
                    "target_band_K2":sum(r["true_k"]==2 for r in band),
                    "target_band_K3":sum(r["true_k"]==3 for r in band),
                    "target_band_other":sum(r["true_k"] not in (2,3) for r in band),
                    "target_band_auc_k2":auc_report(band),
                    "selected_threshold":next(q["selected_threshold"] for q in details if q["fold"]==f)})

    broad=[r for r in rows if r["register"]=="low" and .50<=r["residual_p_k2"]<.70]
    target_all=[r for r in rows if target(r)]
    report={"status":"completed","experiment":"v273_native_margin_low_band_nested_gate",
            "signal":"base P(K2)/(P(K2)+P(K3))","grid":list(GRID),
            "broad_low_confidence_auc_k2":auc_report(broad),
            "target_band_auc_k2":auc_report(target_all),
            "selection":"inner folds only; each fold effect>=0; maximize total improvement, then kept K2 precision, then stricter threshold",
            "baseline":base_all,"abstained":ab_all,"final":fin_all,"per_fold":per,"calibration":details,
            "strict_pass":{"global_not_worse":fin_all["net"]>=base_all["net"],
                           "global_positive":fin_all["net"]>0,
                           "all_folds_nonnegative":all(q["final"]["net"]>=0 for q in per),
                           "fold2_nonnegative":next(q for q in per if q["fold"]==2)["final"]["net"]>=0,
                           "fold4_not_worse":next(q for q in per if q["fold"]==4)["final"]["net"]>=next(q for q in per if q["fold"]==4)["baseline"]["net"]},
            "outer_fold_3_used":False,"held_fold_used_for_threshold":False,
            "prediction_changes":False,"automatic_promotion":False}

    a.output.mkdir(parents=True)
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    lines=["# Native K2/K3 margin — nested low-band gate","",
           f"Broad low-confidence AUC(K2): **{report['broad_low_confidence_auc_k2']:.3f}**.",
           f"Target .60-.65 band AUC(K2): **{report['target_band_auc_k2']:.3f}**.",
           f"Baseline net: **{base_all['net']:+d}**; gated final: **{fin_all['net']:+d}**.","",
           "| fold | native threshold | band K2/K3 | band AUC | baseline net | abstained K2 | abstained K3 | gate effect | final net |",
           "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for q in per:
        th="disabled" if q["selected_threshold"] is None else f"{q['selected_threshold']:.3f}"
        auc="n/a" if q["target_band_auc_k2"] is None else f"{q['target_band_auc_k2']:.3f}"
        ab=q["abstained"]
        lines.append(f"| {q['fold']} | {th} | {q['target_band_K2']}/{q['target_band_K3']} | {auc} | {q['baseline']['net']:+d} | {ab['corrections']} | {ab['regressions']} | {q['gate_effect']:+d} | {q['final']['net']:+d} |")
    lines+=["",f"Strict pass: **{report['strict_pass']}**.","","No outer fold 3; no automatic promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))
if __name__=="__main__":main()
