"""S626 — S613 FROZEN HYBRID true misses by EXACT simultaneous note combinations.

S625 proved what note-pitch pairs/chord sets exist in TRAIN and DEV.
This audits **which ones S613 actually misses on DEV** using:
  - original full frozen S613 hybrid (neural + ONNX via load_heads);
  - same source-disjoint 17 pinned GuitarSet+URMP DEV recordings;
  - source-verified 256-hop causal features and original truth labels;
  - exact per-pitch 80ms maximum-cardinality one-to-one NoteOn matching.

Two SEPARATE meanings of combination:
A. coactive NOTE PAIRS: each two annotated notes of distinct MIDI pitches
   overlapped at least 10ms; success requires BOTH actual NoteOns detected,
   not merely a predicted K=2 or the right peak. One note in a K4 chord
   contributes to multiple pairs: these are NOT independent samples.
B. entire 30ms ONSET GROUPS (K2+): count one sample per group, with
   exact pitches and note-level hit fraction. This is NOT active K or S80
   Exact-K, and group member NoteOns are matched individually ±80ms.

Report for each exact pair/set: denominator, both-detected/full-group,
fraction missing >=1, how many note events missed (can > #pairs),
distinct source recordings AND composition groups. Show ranking by raw
misses vs failure RATE with >=10 examples and >=3 compositions separately.
Also report conditional exact pitch FP contexts for unmatched emissions,
but do not attribute a false positive to an exact truth combination
without evidence.

No threshold tuning, model training, synthetic notes or held-out TEST.
DEV-only attribution is NOT independent validation of an augmentation.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
from scipy.optimize import linear_sum_assignment
import numpy as np

from causal_note.strict_live import load_heads, CausalSelection, Selection, SR
from scripts.s613_train_strict_live import event_score,real_notes,summary
from scripts.s615_train_multihop import source_data
from scripts.s615_exactk_audit import group_onsets,exact_k_onset_audit,merge_k_reports
from scripts.s625_audit_exact_note_combinations import pitch_name,key_string


def match_truth(events,notes):
    sc,pairs=event_score(events,notes,.080)
    found={j for _,j in pairs}
    return sc,found,{i for i,_ in pairs}


def init_bucket():
    return dict(examples=0,failures=0,missed_note_instances=0,
                total_note_instances=0,complete=0,
                corpus=Counter(),source_ids=set(),composition_groups=set(),
                category_K=Counter(),examples_failed=[])


def collect(record,bucket,spec,has_failure,missed_count,member_count,
            k=None,example=None):
    bucket["examples"]+=1
    bucket["failures"]+=int(has_failure)
    bucket["complete"]+=int(not has_failure)
    bucket["missed_note_instances"]+=missed_count
    bucket["total_note_instances"]+=member_count
    bucket["corpus"][spec["corpus"]]+=1
    bucket["source_ids"].add(spec["id"])
    bucket["composition_groups"].add((spec["corpus"],spec["composition"]))
    if k is not None:
        bucket["category_K"][str(k)]+=1
    if has_failure and len(bucket["examples_failed"])<4:
        bucket["examples_failed"].append(example)


def serialize(buckets,*,min_examples=1):
    result=[]
    for key,raw in buckets.items():
        if raw["examples"]<min_examples:
            continue
        n=raw["examples"]
        x=dict(pitches=[int(k) for k in key.split(",")],
               notes="+".join(pitch_name(int(p)) for p in key.split(",")),
               examples=n,complete=int(raw["complete"]),
               failed_at_least_one=int(raw["failures"]),
               failure_rate=round(raw["failures"]/n,6),
               missed_note_instances=int(raw["missed_note_instances"]),
               note_instance_recall=round(
                   1-raw["missed_note_instances"]/raw["total_note_instances"],6),
               distinct_recordings=len(raw["source_ids"]),
               distinct_compositions=len(raw["composition_groups"]),
               breakdown_corpus=dict(raw["corpus"]),
               participating_K=dict(raw["category_K"]),
               failure_examples=raw["examples_failed"])
        result.append(x)
    return sorted(result,key=lambda r:(-r["failed_at_least_one"],-r["examples"],r["notes"]))


def pair_condition(notes,matched,spec):
    buckets=defaultdict(init_bucket)
    pending=sorted(enumerate(notes),key=lambda x:(x[1]["time"],x[1]["pitch"]))
    active=[]
    for i,n in pending:
        start=float(n["time"])
        end=start+float(n["duration"])
        active=[(j,x) for j,x in active
                if x["time"]+x["duration"]>start]
        for j,old in active:
            if old["pitch"]==n["pitch"]:
                continue
            overlap=min(end,float(old["time"])+float(old["duration"]))-start
            if overlap <.010-1e-9:
                continue
            key=key_string(tuple(sorted((int(n["pitch"]),int(old["pitch"])))))
            miss_count=int(i not in matched)+int(j not in matched)
            collect(notes,buckets[key],spec,miss_count>0,miss_count,2,
                    example=dict(source=spec["name"],
                                 two_reference_onsets_s=[round(float(old["time"]),4),
                                                         round(start,4)],
                                 missed_pitches=[p["pitch"] for ix,p in ((i,n),(j,old))
                                                 if ix not in matched]))
        active.append((i,n))
    return buckets


def group_condition(notes,matched,spec):
    buckets=defaultdict(init_bucket)
    ordered=sorted(enumerate(notes),key=lambda x:(x[1]["time"],x[1]["pitch"]))
    grouped=[]
    for i,n in ordered:
        t=float(n["time"])
        if grouped and t-grouped[-1]["start"]<=.030:
            grouped[-1]["ids"].append(i)
        else:
            grouped.append(dict(start=t,ids=[i]))
    for g in grouped:
        ids=g["ids"]
        pitchset=tuple(sorted({notes[i]["pitch"] for i in ids}))
        if len(pitchset)<2:continue
        misses=sum(i not in matched for i in ids)
        key=key_string(pitchset)
        collect(notes,buckets[key],spec,misses>0,misses,len(ids),
                k=len(pitchset),
                example=dict(source=spec["name"],start_s=round(g["start"],4),
                             missed=[pitch_name(notes[i]["pitch"])
                                     for i in ids if i not in matched]))
    return buckets


def accumulate(dest,partial):
    for key,v in partial.items():
        tgt=dest[key]
        for field in ("examples","failures","missed_note_instances",
                      "total_note_instances","complete"):
            tgt[field]+=v[field]
        tgt["corpus"].update(v["corpus"])
        tgt["source_ids"].update(v["source_ids"])
        tgt["composition_groups"].update(v["composition_groups"])
        tgt["category_K"].update(v["category_K"])
        for ex in v["examples_failed"]:
            if len(tgt["examples_failed"])<4:tgt["examples_failed"].append(ex)


def summarize_aggregate(ranked):
    return dict(
        combinations=len(ranked),
        total_combo_observations=sum(r["examples"] for r in ranked),
        top_failures=ranked[:30],
        highest_rates_supported=sorted([
            x for x in ranked if x["examples"]>=10 and
            x["distinct_compositions"]>=3],
            key=lambda x:(-x["failure_rate"],-x["failed_at_least_one"]))[:30],
        # No sample-size restriction: expose fragile 100% 1/1 estimates separately.
        rare_100pct_failures=[x for x in ranked
                   if x["failed_at_least_one"]==x["examples"] and
                   x["examples"]<=3][:30],
    )


def evaluate(root,out,checkpoint):
    if out.exists():raise FileExistsError(out)
    protocol=json.loads((root/"protocol.json").read_text())
    specs=[r for r in protocol["sources"] if r["split"]=="dev"]
    if len(specs)!=17:raise ValueError("Must use exactly original 17 DEV recordings")
    model=load_heads(checkpoint)
    if model.metadata.get("architecture")!="native_causal_hybrid":
        raise ValueError("Use full neural+ONNX S613 hybrid, not neural-only")
    select=Selection(**model.metadata["selection"])
    paired=defaultdict(init_bucket)
    groups=defaultdict(init_bucket)
    kstats=[]
    scs=[]
    missed_pitches=Counter()
    expected_pitches=Counter()
    by_corpus=defaultdict(list)
    fp_context=Counter()
    for index,spec in enumerate(specs,1):
        ends,x,_,_,raw_notes,length=source_data(root,spec,256)
        truth=real_notes(raw_notes,length/SR)
        predicted_k,predicted_count=model.predict(x)
        selector=CausalSelection(select,hop=256)
        events=[]
        for end,p,k in zip(ends,predicted_k,predicted_count):
            events.extend(selector.step(p,k,int(end),int(end)))
        events.extend(selector.panic(length))
        scored,matched,predicted_matched=match_truth(events,truth)
        scs.append(scored)
        by_corpus[spec["corpus"]].append(scored)
        kstats.append(exact_k_onset_audit(events,truth))
        for n,note in enumerate(truth):
            name=pitch_name(note["pitch"])
            expected_pitches[name]+=1
            if n not in matched:missed_pitches[name]+=1
        unmatched_pred=[
            e for i,e in enumerate([e for e in events if e["type"]=="note_on"])
            if i not in predicted_matched]
        for e in unmatched_pred:
            t=float(e["time"])
            p=int(e["pitch"])
            active={int(n["pitch"]) for n in truth
                   if n["time"] <= t < n["time"]+n["duration"]}
            if not active:
                fp_context["when_no_labeled_note_active"]+=1
            elif p in active:
                fp_context["same_pitch_as_labeled_active_note"]+=1
            elif any(abs(p-q) in (12,19,24) for q in active):
                fp_context["harmonic_interval_of_active_labeled_note"]+=1
            elif any(abs(p-q)<=2 for q in active):
                fp_context["neighbor_1_or_2_semitones_from_labeled_active_note"]+=1
            else:
                fp_context["other_pitch_while_note_active"]+=1
        accumulate(paired,pair_condition(truth,matched,spec))
        accumulate(groups,group_condition(truth,matched,spec))
        print(json.dumps(dict(index=index,total=17,
            source=spec["name"],corpus=spec["corpus"],
            TP=scored["TP"],FN=scored["FN"],FP=scored["FP"])),flush=True)
    baseline=summary(scs)
    pair_rank=serialize(paired)
    group_rank=serialize(groups)
    by_pitch=sorted([
        dict(note=p,expected=n,missed=int(missed_pitches[p]),
             recall=round((n-missed_pitches[p])/n,6))
        for p,n in expected_pitches.items()],key=lambda x:(-x["missed"],x["note"]))
    report=dict(
        status="S626_FROZEN_S613_EXACT_COMBINATION_FAILURE_AUDIT_DEV_ONLY",
        model_architecture=model.metadata["architecture"],
        checkpoint=str(checkpoint),dev_sources=len(specs),
        scope="17 original DEV recordings, GuitarSet+URMP, first 24s, MIDI40..88",
        test_sources_read=0,tolerance_ms=80,overlap_min_ms=10,
        grouped_onset_tolerance_ms=30,
        reference_noteon_80ms=baseline,
        by_corpus={k:summary(v) for k,v in by_corpus.items()},
        baseline_grouped_onset_K=merge_k_reports(kstats),
        most_missed_individual_pitches=by_pitch,
        condition_A_overlap_pairs=summarize_aggregate(pair_rank),
        condition_B_exact_onset_groups=summarize_aggregate(group_rank),
        false_positive_context=dict(fp_context),
        notes=(
            "A pair counted failed if at least one of its two annotated "
            "note attacks was not matched within 80ms, independently of "
            "predicted K. In K3/K4 one missed note can count in multiple "
            "pairs: sums across pairs MUST NOT be interpreted as total FN. "
            "Ranked high failure rates require >=10 observations across >=3 "
            "independent compositions. 17 DEV songs only; no holdout or "
            "training-conditioning claims. Harmonic-interval of a false "
            "note does not prove it is an overtone from that note."
        ),
    )
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+"\n")
    print("S626_VERIFIED_SUMMARY "+json.dumps(dict(
         baseline=baseline,
         pair_top_misses=pair_rank[:16],
         pair_top_supported_failure_rates=report[
             "condition_A_overlap_pairs"]["highest_rates_supported"][:12],
         group_top_misses=group_rank[:14],
         group_top_supported_failure_rates=report[
             "condition_B_exact_onset_groups"]["highest_rates_supported"][:10],
         top_individual_pitches=by_pitch[:15],
         fp_context=dict(fp_context)
    ),sort_keys=True),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--model",type=Path,default=Path(
        "model/live/strict_note/NOTE_strict_live.npz"))
    a=parser.parse_args()
    evaluate(a.root,a.out,a.model)


if __name__=="__main__":
    main()
