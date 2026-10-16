"""S618: recover missed REAL guitar NoteOn events in frozen native S613.

Focus on FN -> TP, not deleting false positives. Run identical causal
S613 hybrid checkpoint on exactly the pinned 17 dev GuitarSet+URMP clips.
Training/test annotations NEVER enter online decision. Ground truth is
used ONLY to audit misses and measure offline recovery/regression.

A conservative counterfactual sweep changes only the *selector* thresholds,
not ONNX neural weights, harmonic model, 256-hop decision cadence or FFTs.
Full true positives/false positives, baseline recovered/lost truth identities,
and onset-grouped K1..6 are all reported. NEVER select a new official
reference on these same development songs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from causal_note.strict_live import load_heads, CausalSelection, Selection, SR
from scripts.s613_train_strict_live import event_score, real_notes, summary
from scripts.s615_train_multihop import source_data
from scripts.s615_exactk_audit import exact_k_onset_audit, merge_k_reports


CONFIGS = (
    ("reference", .60, .50),
    ("recover_onset_055", .55, .50),
    ("recover_onset_050", .50, .50),
    ("recover_onset_045", .45, .50),
    ("recover_sustain_045", .60, .45),
    ("recover_sustain_040", .60, .40),
    ("recover_joint_055_045", .55, .45),
    ("recover_joint_050_045", .50, .45),
    ("recover_joint_050_040", .50, .40),
    ("recover_joint_045_040", .45, .40),
)


def score_events(ends, probability, counts, config, frames):
    selector=CausalSelection(config,hop=256)
    events=[]
    for end, p, k in zip(ends,probability,counts):
        events.extend(selector.step(p,k,int(end),int(end)))
    events.extend(selector.panic(int(frames)))
    return events


def causes_of_misses(ends, probabilities, truth, unmatched_ids,
                     *, onset=.6, sustain=.5, tolerance=.08):
    times=np.asarray(ends)/SR
    pitch_stats=defaultdict(Counter)
    histogram=Counter()
    examples=[]
    for index in sorted(unmatched_ids):
        target=truth[index]
        if not 40 <= target["pitch"] <= 88:
            continue
        p=probabilities[:,target["pitch"]-40]
        mask=np.abs(times-float(target["time"])) <= tolerance
        if not mask.any():
            category="no_model_frames"
            max_on=max_s=max_joint=0.
        else:
            local=p[mask]
            o,s=local[:,0],local[:,1]
            max_on=float(o.max())
            max_s=float(s.max())
            qualifies=s>=sustain
            max_joint=float(o[qualifies].max()) if qualifies.any() else 0.
            if max_on<onset:
                category="onset_head_below_threshold"
            elif max_joint<onset:
                category="onset_sustain_not_simultaneously_ready"
            else:
                category="selector_state_or_event_timing"
        histogram[category]+=1
        pitch_stats[str(target["pitch"])][category]+=1
        if len(examples)<50:
            examples.append(dict(pitch=target["pitch"],truth_time_s=round(float(target["time"]),5),
                                 category=category,max_onset=round(max_on,4),
                                 max_sustain=round(max_s,4),
                                 max_joint_onset_at_sustain=round(max_joint,4)))
    return dict(causes=dict(histogram),
                by_pitch={p:dict(v) for p,v in pitch_stats.items()},
                illustrative_examples=examples)


def run(root, output, *, model=Path("model/live/strict_note/NOTE_strict_live.npz")):
    if output.exists():
        raise FileExistsError(output)
    protocol=json.loads((root/"protocol.json").read_text(encoding="utf-8"))
    specs=[s for s in protocol["sources"] if s["split"]=="dev"]
    if len(specs)!=17:
        raise ValueError("Strict comparison needs all 17 original DEV songs")
    heads=load_heads(model)  # critical: includes frozen ONNX tree component
    if heads.metadata.get("architecture")!="native_causal_hybrid":
        raise ValueError("Frozen S613 full ONNX hybrid is required")
    selection=Selection(**heads.metadata["selection"])
    if selection.onset!=.60 or selection.active_on!=.50:
        raise ValueError("Frozen source checkpoint baseline must be onset .60, sustain .50")
    prepared=[]
    for idx,spec in enumerate(specs,1):
        ends,features,_,_,notes,length=source_data(root,spec,256)
        p,k=heads.predict(features)
        truth=real_notes(notes,length/SR)
        baseline=score_events(ends,p,k,selection,length)
        baseline_score,pairs=event_score(baseline,truth,.08)
        baseline_truth={j for i,j in pairs}
        prepared.append(dict(spec=spec,ends=ends,p=p,k=k,truth=truth,length=length,
                             baseline=baseline,baseline_truth=baseline_truth))
        print(json.dumps(dict(loading=idx,total=17,source=spec["name"],
                              baseline_tp=baseline_score["TP"],
                              baseline_fn=baseline_score["FN"])),flush=True)
    misses=Counter()
    by_pitch=defaultdict(Counter)
    examples=[]
    baseline_scores=[]
    baseline_K=[]
    for rec in prepared:
        sc,_=event_score(rec["baseline"],rec["truth"],.08)
        baseline_scores.append(sc)
        baseline_K.append(exact_k_onset_audit(rec["baseline"],rec["truth"]))
        audit=causes_of_misses(rec["ends"],rec["p"],rec["truth"],
                               set(range(len(rec["truth"])))-rec["baseline_truth"])
        misses.update(audit["causes"])
        for pitch,counts in audit["by_pitch"].items():
            by_pitch[pitch].update(counts)
        for x in audit["illustrative_examples"][:4]:
            if len(examples)<60:
                examples.append(dict(source=rec["spec"]["name"],**x))
    comparisons=[]
    for label,onset,sustain in CONFIGS:
        # Keep every other NOTE selection state rule (refractory, release,
        # K inactive budget, confirmation) fixed.
        cfg=replace(selection,onset=onset,active_on=sustain)
        scores=[]
        kreports=[]
        retained=recovered=lost=0
        by_corpus=defaultdict(list)
        for rec in prepared:
            events=rec["baseline"] if label=="reference" else score_events(
                rec["ends"],rec["p"],rec["k"],cfg,rec["length"])
            score,pairs=event_score(events,rec["truth"],.08)
            new_truth={j for i,j in pairs}
            recovered+=len(new_truth-rec["baseline_truth"])
            lost+=len(rec["baseline_truth"]-new_truth)
            retained+=len(new_truth & rec["baseline_truth"])
            scores.append(score)
            by_corpus[rec["spec"]["corpus"]].append(score)
            kreports.append(exact_k_onset_audit(events,rec["truth"]))
        agg=summary(scores)
        grouped=merge_k_reports(kreports)
        row=dict(label=label,selection=asdict(cfg),
                 TP=agg["TP"],FP=agg["FP"],FN=agg["FN"],F1=agg["F1"],
                 recall=agg["TP"]/max(1,agg["TP"]+agg["FN"]),
                 precision=agg["TP"]/max(1,agg["TP"]+agg["FP"]),
                 additional_TP=agg["TP"]-summary(baseline_scores)["TP"],
                 additional_FP=agg["FP"]-summary(baseline_scores)["FP"],
                 recovered_missed_truth=recovered,
                 lost_previously_matched_truth=lost,
                 preserved_truth=retained,
                 net_recall_delta=rec_over_count(recovered,lost),
                 onset_exact_k_rate=grouped["onset_exact_k_rate"],
                 exact_pitch_set_rate=grouped["pitch_set_rate"],
                 onset_K_by_true_K=grouped["by_true_k"],
                 per_corpus={corpus:dict(F1=summary(values)["F1"],
                                        TP=summary(values)["TP"],
                                        FP=summary(values)["FP"],
                                        FN=summary(values)["FN"])
                             for corpus,values in by_corpus.items()})
        comparisons.append(row)
        print(json.dumps(dict(label=label,TP=row["TP"],FP=row["FP"],
                              FN=row["FN"],F1=round(row["F1"],5),
                              recovered=recovered,lost=lost,
                              grouped_K=round(row["onset_exact_k_rate"],4))),flush=True)
    report=dict(
        status="DEV_ONLY_FN_FIRST_SELECTION_SWEEP_NO_PROMOTION",
        sources=len(specs),
        model=str(model),
        checkpoint_architecture=heads.metadata["architecture"],
        default_hop=256,
        lookahead_samples=0,
        reference_counts=summary(baseline_scores),
        original_missed_attack_diagnosis={
            "categories":dict(misses),
            "by_pitch":{p:dict(c) for p,c in by_pitch.items()},
            "examples":examples,
            "important":"Ground truth is used only AFTER prediction, never to select pitch or event timing."
        },
        tested=comparisons,
        best_F1_label=max(comparisons,key=lambda x:x["F1"])["label"],
        best_recall_label=max(comparisons,key=lambda x:(x["TP"],-x["FP"]))["label"],
        warning=("Threshold adjustments are a diagnostic control, NOT an independent "
                 "generalization improvement or new learned head. Some recovered "
                 "truth may be offset by newly false attacks. Re-validate on "
                 "new unseen songs and preserve exact K3/K4."),
    )
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report


def rec_over_count(recovered,lost):
    return int(recovered-lost)


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--root",type=Path,required=True)
    a.add_argument("--model",type=Path,default=Path("model/live/strict_note/NOTE_strict_live.npz"))
    a.add_argument("--out",type=Path,required=True)
    args=a.parse_args()
    run(args.root,args.out,model=args.model)


if __name__=="__main__":
    main()
