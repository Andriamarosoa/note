"""S615 causal hybrid ablation preserving original S613 tree expert.

Problem: frozen S613 uses a neural+ONNX-tree HYBRID, while S615 standalone
trained only the neural MLP. Comparing two is a valid end-to-end comparison,
but the backbone change confounds a conclusion about hop training.
This audit restores the original S613 tree expert (frozen, no retraining),
combining it with retrained S615 neural pitch heads.

Dev-only tuning of mixture weight and count-source; select no official model.
No test compositions, test labels or future audio are used. All input
features are the exact S613 causal 256-hop tensors.

Requires already-trained FULL S615 hop256 candidate and pinned 17 dev caches.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from causal_note.strict_live import SR, Selection, load_heads, NativeLiveHeads
from scripts.s615_compare_dev import measure


ALPHAS=(0.0,0.25,0.5,0.75,1.0)
COUNT_SOURCES=("frozen_s613","new_s615")


def merge_hybrid(frozen_path, new_path, output, *, alpha,
                 count_source="frozen_s613", selector="frozen_s613"):
    if output.exists():
        raise FileExistsError(output)
    if not 0<=alpha<=1:
        raise ValueError("Hybrid tree weight must be [0,1]")
    if count_source not in COUNT_SOURCES:
        raise ValueError("Unknown count source")
    if selector not in ("frozen_s613","new_s615"):
        raise ValueError("Unknown selector source")
    with np.load(frozen_path,allow_pickle=False) as z:
        frozen={key:z[key].copy() for key in z.files}
    with np.load(new_path,allow_pickle=False) as z:
        learned={key:z[key].copy() for key in z.files}
    frozen_meta=json.loads(str(frozen["metadata"]))
    new_meta=json.loads(str(learned["metadata"]))
    if (frozen_meta.get("architecture")!="native_causal_hybrid"
            or tuple(new_meta.get("trained_hops",()))!=(256,)):
        raise ValueError("Expected frozen S613 hybrid and separately trained S615 hop256")
    for key in ("tree0","tree1","tree2","schema","pitch.mean","pitch.scale"):
        if key not in frozen and key not in learned:
            raise ValueError("Missing native checkpoint key: "+key)
    out=dict(frozen)
    for key in tuple(out):
        if key.startswith("pitch."):
            out[key]=learned[key]
        if key.startswith("count.") and count_source=="new_s615":
            out[key]=learned[key]
    meta=dict(frozen_meta)
    meta["architecture"]="native_causal_hybrid"
    meta["tree_weight"]=alpha
    meta["trained_hops"]=[256]
    meta["experiment"]="S615_FROZEN_TREE_HYBRID_DEV_ABLATION"
    meta["new_pitch_head"]="S615_FULL_HOP256_NEURAL"
    meta["count_source"]=count_source
    meta["no_new_tree_training"]=True
    meta["selection"]=(
        frozen_meta["selection"] if selector=="frozen_s613"
        else new_meta["selection_by_hop"]["256"])
    meta.pop("selection_by_hop",None)
    out["metadata"]=np.asarray(json.dumps(meta))
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**out)
    # A missing or incompatible array should never be silently accepted.
    loaded=load_heads(output)
    if loaded.metadata.get("tree_weight")!=alpha:
        raise AssertionError("Merged hybrid checkpoint roundtrip mismatch")
    return output


def run(root,frozen,new,out):
    protocol=json.loads((root/"protocol.json").read_text())
    dev=[s for s in protocol["sources"] if s["split"]=="dev"]
    if len(dev)!=17:
        raise ValueError("Requires exact original 17-source GuitarSet+URMP development set")
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    reference=measure(root,load_heads(frozen),256,dev)
    neural=measure(root,NativeLiveHeads(new),256,dev)
    trials=[]
    for count in COUNT_SOURCES:
        for a in ALPHAS:
            for sel in ("frozen_s613","new_s615"):
                p=out/f"hybrid_count-{count}_tree-{a:.2f}_sel-{sel}.npz"
                merge_hybrid(frozen,new,p,alpha=a,count_source=count,selector=sel)
                result=measure(root,load_heads(p),256,dev)
                compact=dict(
                    count_source=count,alpha=a,selector=sel,checkpoint=str(p),
                    F1=result["noteon_80ms"]["F1"],
                    TP=result["noteon_80ms"]["TP"],
                    FP=result["noteon_80ms"]["FP"],
                    FN=result["noteon_80ms"]["FN"],
                    active_K=result["count_head_active_k"],
                    onset_K=result["onset_k"],
                )
                trials.append(compact)
                print(json.dumps(dict(
                    count=count,alpha=a,selection=sel,F1=compact["F1"],
                    TP=compact["TP"],FP=compact["FP"],FN=compact["FN"],
                    grouped_onset_K=compact["onset_K"]["onset_exact_k_rate"])),
                      flush=True)
    best=max(trials,key=lambda r:(r["F1"],-r["FP"]))
    report=dict(
        status="DEV_ONLY_HYBRID_ABLATION_NOT_PROMOTED",
        dev_sources=17,baseline_F1=reference["noteon_80ms"]["F1"],
        S615_neural_F1=neural["noteon_80ms"]["F1"],
        best=best,trials=trials,
        original_S613=reference,
        S615_neural=neural,
        warning=("Weight and selector were selected on these same 17 DEV songs."
                 " This is not new held-out evidence or a promotion. "
                 "K-active and grouped-onset Exact-K are separate."),
    )
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(
        baseline_F1=report["baseline_F1"],
        S615_neural_F1=report["S615_neural_F1"],
        best_HYBRID_F1=best["F1"],
        best_HYBRID_grouped_K=best["onset_K"]["onset_exact_k_rate"],
        best_K3=best["onset_K"]["by_true_k"].get("3"),
        best_K4=best["onset_K"]["by_true_k"].get("4"),
        best_params={k:best[k] for k in ("count_source","alpha","selector")})),
        flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--frozen",type=Path,default=Path("model/live/strict_note/NOTE_strict_live.npz"))
    parser.add_argument("--new",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    run(args.root,args.frozen,args.new,args.out)


if __name__=="__main__":
    main()
