"""Compare risk-coupled NN to the ORIGINAL NN with unsupported K0/K1 disabled.

This is a DESCRIPTIVE matched output-eligibility comparison, not a causal
ablation of the risk head: both training objectives and decoders changed.
It clarifies how much, if anything, risk coupling adds *beyond* preventing
an unsupported class proposal.

Uses exact frozen cohort, all 59309 events, exact same 7493 eligible IDs.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scripts.yourmt3_exactk_common import metrics,paired,require,FOLDS

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--old",type=Path,required=True)
    p.add_argument("--risk",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    require(not a.output.exists(),"refuse overwrite")
    with np.load(a.old,allow_pickle=False) as z: old={k:z[k] for k in z.files}
    with np.load(a.risk,allow_pickle=False) as z: new={k:z[k] for k in z.files}
    for key in ("global_index","baseline","true_k","fold","eligible_indices"):
        require(np.array_equal(old[key],new[key]),"drift: "+key)
    y=new["true_k"]; b=new["baseline"]; fold=new["fold"]
    require(len(y)==59309 and len(new["eligible_indices"])==7493,"population drift")
    elig=np.isin(new["global_index"],new["eligible_indices"])
    require(np.sum(elig)==7493,"eligible identification drift")
    pred_old=old["corrected"]
    pred_risk=new["corrected"]
    pred_masked=np.where(elig&np.isin(pred_old,(0,1)),b,pred_old)
    require(np.array_equal(pred_masked[~elig],b[~elig]),"baseline outside candidates changed")
    require(np.array_equal(pred_risk[~elig],b[~elig]),"risk outside candidate changed")

    def details(p):
        return dict(score=metrics(y,p),paired=paired(y,b,p),
            good_by_K={str(k):int(np.sum((y==k)&(p==k))) for k in range(7)})
    records={key:details(pr) for key,pr in (
        ("raw_old_unsupported_K0_K1",pred_old),
        ("old_with_K0_K1_reverted_POSTHOC",pred_masked),
        ("risk_coupled_trained_nn",pred_risk))}
    comparisons={}
    for bkey,base_pred in (
        ("raw_old_unsupported_K0_K1",pred_old),
        ("old_with_K0_K1_reverted_POSTHOC",pred_masked)):
        old_good=(base_pred==y)
        new_good=(pred_risk==y)
        comparisons[bkey]=dict(
            disagreements=int(np.sum(pred_risk!=base_pred)),
            old_good_new_wrong=int(np.sum(old_good&~new_good)),
            old_wrong_new_good=int(np.sum(~old_good&new_good)),
            delta_good=int(new_good.sum()-old_good.sum()),
            by_true_K={
                str(k):{
                    "old_good_new_wrong":int(np.sum((y==k)&old_good&~new_good)),
                    "old_wrong_new_good":int(np.sum((y==k)&~old_good&new_good)),
                    "net":int(np.sum((y==k)&new_good)-np.sum((y==k)&old_good))
                } for k in range(7)
            },
            by_fold={
                str(f):dict(
                    improvement=int(np.sum((fold==f)&new_good&~old_good)),
                    regression=int(np.sum((fold==f)&~new_good&old_good)),
                    net=int(np.sum((fold==f)&new_good)-np.sum((fold==f)&old_good)))
                for f in FOLDS}
        )
    report=dict(experiment="v273_risk_coupling_matched_class_support_control",
        results=records,paired_comparisons=comparisons,
        caveats=[
            "Old K0/K1 reverted policy is POSTHOC; not a trained policy and cannot be promoted.",
            "Both training objective and final decoder changed between original and risk NN. This does not causally isolate the risk branch alone.",
            "Control matches structural K0/K1 availability for the two output policies.",
            "Previously inspected folds 0/1/2/4; not independent players.",
            "Freeze_local_combo unchanged."
        ])
    a.output.mkdir(parents=True)
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    lines=["# Risk coupling — what is gained beyond unsupported K0/K1 suppression","",
        "This is NOT a new model, only exact matched-cohort diagnostics.","",
        "| Variant | Global | Poly | Corrections | Regressions | Net vs frozen |",
        "|---|---:|---:|---:|---:|---:|"]
    for name,z in records.items():
        m=z["score"];c=z["paired"]["global"]
        lines.append(f"| {name} | {m['exact']*100:.4f}% | "
            f"{m['poly']['exact']*100:.4f}% | {c['corrections']} | "
            f"{c['regressions']} | {c['net']:+d} |")
    x=comparisons["old_with_K0_K1_reverted_POSTHOC"]
    lines+=["","## Trained risk NN versus old network with unsupported proposals suppressed","",
        f"- Prediction disagreements: {x['disagreements']}",
        f"- Corrected previously wrong old predictions: {x['old_wrong_new_good']}",
        f"- Destroyed previously correct old predictions: {x['old_good_new_wrong']}",
        f"- Net correct delta: {x['delta_good']:+d}","",
        "| True K | Newly correct | Newly wrong | Net |",
        "|---|---:|---:|---:|"]
    for k,v in x["by_true_K"].items():
        lines.append(f"| K{k} | {v['old_wrong_new_good']} | "
            f"{v['old_good_new_wrong']} | {v['net']:+d} |")
    lines+=["","## Interpretation limits"]+[
        "- "+t for t in report["caveats"]]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)

if __name__=="__main__":main()
