"""S620 true causal 128+256 dual-cadence ablation, frozen S613 hybrid.

Same 17 pinned GuitarSet+URMP DEV tracks, no held-out test accessed.
Baseline S613 256 decisions are never changed. Onset proposals at 128
are *experimentally* computed by the 256-trained frozen checkpoint.
A two-source reference-first per-note merger prevents double NOTE-on/off.
The fast branch never waits for a future 256-hop frame. CPU is separately
benchmarked with actual native stream; these event scores do not estimate
real microphone/FluidSynth roundtrip.

Candidate gate config optimized on DEV => no independent promotion claim.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from causal_note.strict_live import CausalSelection, HOP, SR, Selection, load_heads
from causal_note.dual_hop import (
    DualHopConfig, DualSourceEventMerger, fast_candidate_mask,
)
from scripts.s613_train_strict_live import event_score, real_notes, summary
from scripts.s615_train_multihop import source_data
from scripts.s615_exactk_audit import exact_k_onset_audit, merge_k_reports


GATES=tuple(
    DualHopConfig(mode="guard",fast_onset=o,novelty_min=.04,
                  novelty_ratio=ratio,novelty_mode=kind)
    for o in (.45,.50,.55,.60)
    for ratio in (.35,.55,.75)
    for kind in ("lag","recent")
)


def prepared_predictions(root, heads, spec, fast_heads=None):
    end128, x128, _, _, notes, frames=source_data(root,spec,128)
    end256, x256, _, _, _, same_frames=source_data(root,spec,256)
    if frames!=same_frames:
        raise ValueError("Causal 128/256 sources differ")
    fast, kfast=(fast_heads if fast_heads is not None else heads).predict(x128)
    slow, kslow=heads.predict(x256)
    features={
        "lag":np.maximum(x128[:,:,60:63],0).max(axis=2),
        "recent":np.maximum(x128[:,:,114:117],0).max(axis=2),
    }
    # Once scored, discard the giant 126-col feature tensor. Predictions
    # and two tiny per-pitch novelty tracks suffice for all 24 variants.
    return dict(
        spec=spec,end128=end128,end256=end256,fast=fast,kfast=kfast,
        slow=slow,kslow=kslow,novelty=features,frames=frames,
        truth=real_notes(notes,frames/SR)
    )


def replay(prepared, heads, cfg=None):
    """Return all emitted events with exact source timestamps, no lookahead.

    All 256-hop decisions are processed BEFORE 128-hop decisions at the
    same timestamp, exactly like the microphone streaming implementation.
    """
    cfg=cfg or DualHopConfig(mode="off")
    sel_cfg=Selection(**heads.metadata["selection"])
    reference=CausalSelection(sel_cfg,hop=HOP)
    fast=(CausalSelection(Selection(**{
        **heads.metadata["selection"],"onset":cfg.fast_onset}),hop=128)
          if cfg.mode!="off" else None)
    merger=DualSourceEventMerger() if cfg.mode=="guard" else None
    output=[]
    source_events=Counter()
    promoted=0
    next256=0
    for row,t in enumerate(prepared["end128"]):
        t=int(t)
        staged=[]
        if next256<len(prepared["end256"]) and int(prepared["end256"][next256])==t:
            p,k=prepared["slow"][next256],prepared["kslow"][next256]
            for e in reference.step(p,k,t,t):
                source_events[e["type"]]+=1
                staged.append(("reference",e))
            next256+=1
        if fast is not None:
            pf=prepared["fast"][row].copy()
            nv=prepared["novelty"][cfg.novelty_mode][row]
            qualified=fast_candidate_mask(nv,pf,cfg,
                reference_sustain=sel_cfg.active_on)
            promoted+=int(qualified.sum())
            pf[~qualified,0]=0
            for e in fast.step(pf,prepared["kfast"][row],t,t):
                staged.append(("fast",e))
        for origin,e in staged:
            if merger is not None:
                output.extend(merger.step(origin,e))
            elif origin=="reference":
                output.append(e)
    if next256 != len(prepared["end256"]):
        raise AssertionError("Missing final 256-hop frame")
    if merger is None:
        output.extend(reference.panic(prepared["frames"]))
    else:
        reference.panic(prepared["frames"])
        fast.panic(prepared["frames"])
        output.extend(merger.finish(prepared["frames"]))
    return output,dict(promoted_pitch_frames=promoted,
                       reference_emission=dict(source_events),
                       fused_suppressed_duplicates=(
                           merger.suppressed_duplicates if merger else 0))


def evaluate(root, output, *, checkpoint, fast_checkpoint=None):
    if output.exists():
        raise FileExistsError(output)
    protocol=json.loads((root/"protocol.json").read_text())
    sources=[x for x in protocol["sources"] if x["split"]=="dev"]
    if len(sources)!=17:
        raise ValueError("Expected all 17 composition-disjoint S613 DEV songs")
    heads=load_heads(checkpoint)
    if heads.metadata["architecture"]!="native_causal_hybrid":
        raise ValueError("Require true frozen neural+ONNX S613")
    fast_heads=None
    if fast_checkpoint is not None:
        fast_heads=load_heads(fast_checkpoint)
        if 128 not in fast_heads.metadata.get("trained_hops",()):
            raise ValueError("Fast candidate checkpoint must declare trained_hops including 128")
    cached=[]
    for i,s in enumerate(sources,1):
        row=prepared_predictions(root,heads,s,fast_heads=fast_heads)
        old,metadata=replay(row,heads)
        scores,matches=event_score(old,row["truth"],.08)
        row["reference"]=old
        row["old_truth"]={j for _,j in matches}
        row["old_score"]=scores
        cached.append(row)
        print(json.dumps(dict(loaded=i,source=s["name"],
                              old_TP=scores["TP"],old_FN=scores["FN"])),flush=True)
    reference=summary([r["old_score"] for r in cached])
    reference_k=merge_k_reports([exact_k_onset_audit(r["reference"],r["truth"])
                                for r in cached])
    tested=[]
    for cfg in GATES:
        scores=[]; k_reports=[]; regained=lost=0
        by_corpus=defaultdict(list)
        candidate_frames=duplicates=0
        for r in cached:
            events,diagnostic=replay(r,heads,cfg)
            score,matches=event_score(events,r["truth"],.08)
            new_truth={j for _,j in matches}
            regained+=len(new_truth-r["old_truth"])
            lost+=len(r["old_truth"]-new_truth)
            scores.append(score)
            by_corpus[r["spec"]["corpus"]].append(score)
            k_reports.append(exact_k_onset_audit(events,r["truth"]))
            candidate_frames+=diagnostic["promoted_pitch_frames"]
            duplicates+=diagnostic["fused_suppressed_duplicates"]
        m=summary(scores)
        k=merge_k_reports(k_reports)
        result=dict(
            config=asdict(cfg),TP=m["TP"],FP=m["FP"],FN=m["FN"],
            F1=m["F1"],recall=m["TP"]/(m["TP"]+m["FN"]),
            gained_real_TP=regained,lost_old_TP=lost,
            delta_TP=m["TP"]-reference["TP"],
            delta_FP=m["FP"]-reference["FP"],
            candidate_pitch_frames=candidate_frames,
            duplicated_note_ons_suppressed=duplicates,
            grouped_onset_k=k,
            by_corpus={corpus:summary(items) for corpus,items in by_corpus.items()},
        )
        assert result["delta_TP"]==regained-lost
        tested.append(result)
        print(json.dumps(dict(
            gate=asdict(cfg),TP=m["TP"],FP=m["FP"],
            FN=m["FN"],F1=round(m["F1"],5),
            recovered=regained,lost=lost,
            K3=k["by_true_k"].get("3"),K4=k["by_true_k"].get("4")
        )),flush=True)
    best_f1=max(tested,key=lambda d:(d["F1"],d["gained_real_TP"]))
    best_recall=max(tested,key=lambda d:(d["TP"],-d["FP"]))
    report=dict(
        status="DEV_ONLY_S620_DUAL_HOP_NO_PROMOTION",
        model=str(checkpoint),reference_hop=256,fast_hop=128,
        no_extra_lookahead=True,dev_source_count=17,
        fast_checkpoint_trained_at_hop128=fast_heads is not None,
        fast_checkpoint=str(fast_checkpoint) if fast_heads is not None else None,
        source_audio_only_up_to_decision=True,
        reference=reference,reference_onset_k=reference_k,
        candidates=tested,best_f1=best_f1,best_recall=best_recall,
        warning=("Fast branch reuses frozen S613 trained at 256 and is "
                 "not independently trained for 128. These 24 DEV-tuned "
                 "policies provide no independent F1/Exact-K guarantee; "
                 "run new held-out before any change to S613 reference.")
    )
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(
        status=report["status"],reference=reference,
        best_F1={k:v for k,v in best_f1.items()
                 if k not in ("grouped_onset_k","by_corpus")},
        best_K3=best_f1["grouped_onset_k"]["by_true_k"].get("3"),
        best_K4=best_f1["grouped_onset_k"]["by_true_k"].get("4"),
    )),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--model",type=Path,
                        default=Path("model/live/strict_note/NOTE_strict_live.npz"))
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--fast-model",type=Path,
                        help="Separate 128-hop trained NOTE candidate checkpoint")
    args=parser.parse_args()
    evaluate(args.root,args.out,checkpoint=args.model,
             fast_checkpoint=args.fast_model)


if __name__=="__main__":
    main()
