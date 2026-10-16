"""S639 forensic audit of Bass ×2 false NoteOn on *real* GuitarSet PCM.

Each missed/recovered label and each unmatched note_on is classified
ONLY AFTER running causal, frozen S614 and S637 streams. No TEST accessed.
Onset scores and learned K posterior are recorded AT DECISION TIME to
test suspected repeated note-on and inactive-K explanations.
"""
from __future__ import annotations

from collections import Counter,defaultdict
from dataclasses import replace
import argparse
import json
from pathlib import Path
import tempfile
import time
import zipfile
import numpy as np

from causal_note.strict_live import SR,Selection,CausalSelection,load_heads
from scripts.s613_prepare_strict_live import download,digest,labels
from scripts.s613_train_strict_live import real_notes,event_score,summary
from scripts.s637_guitarset_three_view_groundtruth import (
    ROOT,MODEL,acoustic_batches,corrected,raw_guitar_audio,
    GUITARSET_PINNED_CONTROL,
)
from scripts.s638_guitarset_true_overlap import overlap_groups,truth_active_pitch_k

POLICY_REFERENCE=("S614_original",None,None,False,0.)
POLICY_BASS08=("bass_080",.80,None,False,0.)
POLICY_BASS09=("bass_090",.90,None,False,0.)
POLICY_BASS08_VETO=("bass_080_octave_veto",.80,None,True,0.)
POLICY_BASS08_K35=("bass_080_rescue_K35",.80,None,False,.35)
POLICIES=(POLICY_REFERENCE,POLICY_BASS08,POLICY_BASS09,
          POLICY_BASS08_VETO,POLICY_BASS08_K35)
# Ablation: original K count threshold was disabled (1.1): test whether
# enabling native K for the *entire decoder* helps, with paired baseline.
COUNT_ABLATIONS=(("S614_K35",POLICY_REFERENCE),
                 ("bass_080_K35",POLICY_BASS08))


def active_true_pitches(truth,t,margin=.030):
    return {int(n["pitch"]) for n in truth
            if n["time"]-margin<=t<=n["time"]+n["duration"]+margin}


def classify_false_note(event,truth,*,prior_predicted_gap_s=None):
    """Explain classes with descriptive nonexclusive flags and exclusive primary.

    Crucial: 'harmonic' means coincidence with a feasible interval while
    ground truth remains audible. It is an ALIAS SUSPECT, not proof of source
    acoustics: another true note onset might exist beyond the ±80ms scorer.
    """
    pitch=int(event["pitch"]);t=float(event["time"])
    active=active_true_pitches(truth,t)
    onset_same=[abs(float(n["time"])-t)
                for n in truth if int(n["pitch"])==pitch]
    near_same=min(onset_same,default=float("inf"))
    exact_active=pitch in active
    octave_active=any(abs(pitch-q)==12 for q in active)
    other_harmonic=any(abs(pitch-q) in (19,24) for q in active)
    any_active=bool(active)
    earlier_same_gt=any(int(n["pitch"])==pitch
                        and float(n["time"])<t-.080
                        and t<=float(n["time"])+float(n["duration"])+.030
                        for n in truth)
    close_other_attack=any(int(n["pitch"])!=pitch
                           and abs(float(n["time"])-t)<=.080
                           for n in truth)
    if earlier_same_gt:
        reason="sustain_retrigger_suspect"
    elif near_same<=.080:
        reason="duplicate_within_truth_match_window"
    elif octave_active:
        reason="octave_alias_suspect"
    elif other_harmonic:
        reason="higher_harmonic_alias_suspect"
    elif close_other_attack:
        reason="cross_pitch_onset_leakage"
    elif any_active:
        reason="unexplained_during_other_sustain"
    else:
        reason="outside_annotated_note_activity"
    return dict(
        reason=reason,
        original_pitch_active_at_prediction=exact_active,
        octave_related_active_truth=octave_active,
        third_or_fourth_harmonic_interval_with_active_truth=other_harmonic,
        time_within_80ms_of_same_pitch_true_onset=near_same<=.080,
        active_truth_polyphony=len(active),
        other_pitch_attack_within_80ms=close_other_attack,
        prior_same_pitch_predicted_interval_ms=(
            round(prior_predicted_gap_s*1000,2)
            if prior_predicted_gap_s is not None else None),
        active_ground_truth_pitches=sorted(active),
    )


def trace_source(spec,root,archive,heads,sel):
    pcm,sha=raw_guitar_audio(spec,archive)
    truth=real_notes(labels(spec,root)["notes"],len(pcm)/SR)
    overlap=overlap_groups(truth)
    configs={n:sel for n,*_ in POLICIES}
    configs.update({n:replace(sel,count_confidence=.35)
                    for n,_ in COUNT_ABLATIONS})
    policies={p[0]:p for p in POLICIES}
    policies.update(dict(COUNT_ABLATIONS))
    streams={name:CausalSelection(configs[name]) for name in configs}
    events={name:[] for name in configs}
    traces={name:[] for name in configs}
    promotion_count=Counter()
    for end,native,k,bass,high,valid in acoustic_batches(pcm,heads):
        for name,state in streams.items():
            modified,nproposed=corrected(native,bass,high,k,state,policies[name],valid)
            promotion_count[name]+=nproposed
            num_active=int(state.active.sum())
            emitted=state.step(modified,k,int(end),int(end))
            for e in emitted:
                if e["type"]!="note_on":continue
                idx=int(e["pitch"])-40
                k_requirement=min(6,num_active+1)
                traces[name].append(dict(
                    event=e,
                    is_rescue_promoted=bool(valid and
                         native[idx,0]<sel.onset and
                         modified[idx,0]>=sel.onset),
                    native_onset=float(native[idx,0]),
                    native_sustain=float(native[idx,1]),
                    bass_shifted_onset=float(bass[idx,0]),
                    bass_shifted_sustain=float(bass[idx,1]),
                    original_K_mode=int(np.argmax(k)),
                    original_K_max=float(np.max(k)),
                    original_K_p_at_least_next=float(k[k_requirement:].sum()),
                    active_selected_notes_before=num_active,
                    pitch_active_before=bool(state.active[idx]),
                    pitch_armed_before=True if e.get("reason")!="restrike" else False,
                ))
            events[name].extend(emitted)
    for name in streams:
        events[name].extend(streams[name].panic(len(pcm)))
    result={}
    base,pairs_baseline=event_score(events["S614_original"],truth)
    truth_base={j for _,j in pairs_baseline}
    baseline_predictions=[e for e in events["S614_original"] if e["type"]=="note_on"]
    base_fp=[e for i,e in enumerate(baseline_predictions) if i not in {i for i,_ in pairs_baseline}]
    for name in configs:
        score,pairs=event_score(events[name],truth)
        matched_pred={i for i,_ in pairs}
        matched_truth={j for _,j in pairs}
        prior={}
        diagnoses=[]
        predicted=[e for e in events[name] if e["type"]=="note_on"]
        assert len(traces[name])==len(predicted)
        for i,e in enumerate(predicted):
            pitch=e["pitch"]
            previous_gap=(e["time"]-prior[pitch]) if pitch in prior else None
            prior[pitch]=float(e["time"])
            if i in matched_pred:
                continue
            info=classify_false_note(e,truth,prior_predicted_gap_s=previous_gap)
            baseline_near=any(
                b["pitch"]==pitch and abs(b["time"]-e["time"])<=.080
                for b in base_fp)
            trace=traces[name][i]
            info.update(dict(
                pitch=pitch,
                time_s=round(float(e["time"]),5),
                matched_to_baseline_false_note=bool(baseline_near),
                **{k:v for k,v in trace.items() if k!="event"},
            ))
            diagnoses.append(info)
        by_reason=Counter(d["reason"] for d in diagnoses)
        added_unmatched=[d for d in diagnoses if not d["matched_to_baseline_false_note"]]
        by_reason_added=Counter(d["reason"] for d in added_unmatched)
        more=sorted(added_unmatched,key=lambda d:d["time_s"])
        new_labels=matched_truth-truth_base
        lost_labels=truth_base-matched_truth
        groups={}
        for g in ("onset_overlap","onset_single","onset_K2","onset_K3plus"):
            indices=overlap[g]
            groups[g]=dict(
                newly_recovered=len(new_labels & indices),
                lost=len(lost_labels & indices),
                true_count=len(indices))
        result[name]=dict(
            score=score,policies=policies[name],all_promotions=promotion_count[name],
            sources_onset_labels=len(truth),
            newly_recovered=len(new_labels),lost_correct=len(lost_labels),
            groups=groups,
            false_positive_reasons=dict(by_reason),
            unmatched_no_baseline_FP_at_80ms=len(added_unmatched),
            novel_FP_reasons=dict(by_reason_added),
            novel_FP_k_mode_below_needed=sum(
                int(d["original_K_mode"]<=d["active_selected_notes_before"])
                for d in added_unmatched),
            novel_FP_with_K_next_prob_below_35pct=sum(
                int(d["original_K_p_at_least_next"]<.35) for d in added_unmatched),
            novel_FP_native_onset_below_60pct=sum(
                int(d["native_onset"]<sel.onset) for d in added_unmatched),
            novel_FP_rescue_promoted=sum(int(d["is_rescue_promoted"]) for d in added_unmatched),
            novel_FP_native_sustain_below_25pct=sum(
                int(d["native_sustain"]<sel.active_off) for d in added_unmatched),
            novel_FP_native_sustain_below_50pct=sum(
                int(d["native_sustain"]<sel.active_on) for d in added_unmatched),
            novel_FP_repeated_same_pitch_within_120ms=sum(
                int(d["prior_same_pitch_predicted_interval_ms"] is not None
                    and d["prior_same_pitch_predicted_interval_ms"]<=120)
                for d in added_unmatched),
            novel_FP_repeated_same_pitch_within_300ms=sum(
                int(d["prior_same_pitch_predicted_interval_ms"] is not None
                    and d["prior_same_pitch_predicted_interval_ms"]<=300)
                for d in added_unmatched),
            examples=more[:28],
        )
    return dict(name=spec["name"],recording_type="comp" if "_comp" in spec["name"] else "solo",
                composition=spec["composition"],source_sha256=sha,
                reference_S614=base,labels=len(truth),
                policies=result)


def aggregate(records):
    output={}
    for name in records[0]["policies"]:
        rows=[r["policies"][name] for r in records]
        score=summary([r["score"] for r in rows])
        baseline=summary([r["policies"]["S614_original"]["score"] for r in records])
        reasons=Counter();new_reasons=Counter()
        totals=Counter()
        by_recording=[]
        for source,row in zip(records,rows):
            reasons.update(row["false_positive_reasons"])
            new_reasons.update(row["novel_FP_reasons"])
            for field in ("all_promotions","newly_recovered","lost_correct",
                          "unmatched_no_baseline_FP_at_80ms",
                          "novel_FP_k_mode_below_needed",
                          "novel_FP_with_K_next_prob_below_35pct",
                          "novel_FP_native_onset_below_60pct",
                          "novel_FP_rescue_promoted",
                          "novel_FP_native_sustain_below_25pct",
                          "novel_FP_native_sustain_below_50pct",
                          "novel_FP_repeated_same_pitch_within_120ms",
                          "novel_FP_repeated_same_pitch_within_300ms"):
                totals[field]+=row[field]
            by_recording.append(dict(source=source["name"],
                                     delta_TP=row["score"]["TP"]-source["reference_S614"]["TP"],
                                     delta_FP=row["score"]["FP"]-source["reference_S614"]["FP"]))
        by_group={}
        for group in ("onset_overlap","onset_single","onset_K2","onset_K3plus"):
            by_group[group]=dict(
                newly_recovered=sum(r["groups"][group]["newly_recovered"] for r in rows),
                lost=sum(r["groups"][group]["lost"] for r in rows))
        output[name]=dict(
            score=score,
            delta_TP=score["TP"]-baseline["TP"],
            delta_FP=score["FP"]-baseline["FP"],
            delta_F1_points=100*(score["F1"]-baseline["F1"]),
            false_positive_reasons=dict(reasons),
            nominally_novel_false_positive_reasons=dict(new_reasons),
            diagnostic_totals=dict(totals),
            correction_groups=by_group,
            by_recording=by_recording,
        )
    return output


def run(out):
    if out.exists():
        raise FileExistsError("S639 refuses to overwrite analysis evidence")
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    assert digest(MODEL.read_bytes())==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    protocol=json.loads((ROOT/"protocol.json").read_text())
    train={(s["corpus"],s["composition"]) for s in protocol["sources"] if s["split"]=="train"}
    test={(s["corpus"],s["composition"]) for s in protocol["sources"] if s["split"]=="test"}
    dev=[s for s in protocol["sources"] if s["split"]=="dev" and s["corpus"]=="GuitarSet"]
    assert len(dev)==12 and len({s["composition"] for s in dev})==3
    assert not {("GuitarSet",s["composition"]) for s in dev}&(train|test)
    heads=load_heads(MODEL,expected_sha256=frozen["model_sha256"])
    sel=Selection(**heads.metadata["selection"])
    assert sel.count_confidence==1.1 # baseline K ceiling really DISABLED
    start=time.perf_counter()
    rows=[]
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        for key,filename in (("GuitarSet_audio","GuitarSet_audio.zip"),
                             ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            item=protocol["guitar_archives"][key]
            download(item["url"],root/"data"/filename,item["size"],item["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as audio_zip:
            for i,s in enumerate(dev,1):
                row=trace_source(s,root,audio_zip,heads,sel)
                rows.append(row)
                print("S639 forensic",i,"/",len(dev),row["name"],
                      "baselineFP",row["policies"]["S614_original"]["score"]["FP"],
                      "bass08FP",row["policies"]["bass_080"]["score"]["FP"],
                      "novelReasons",row["policies"]["bass_080"]["novel_FP_reasons"],flush=True)
    scores=aggregate(rows)
    baseline=scores["S614_original"]["score"]
    assert all(baseline[k]==v for k,v in GUITARSET_PINNED_CONTROL.items()),baseline
    assert scores["bass_080"]["delta_TP"]==46
    assert scores["bass_080"]["delta_FP"]==249
    report=dict(
        status="S639_FORENSIC_REAL_GUITARSET_DEV_BASS_RESCUE_FP_CAUSES",
        real_acoustic_and_ground_truth=True,synthetic=False,
        original_S614_frozen=True,
        original_K_count_confidence=sel.count_confidence,
        native_K_count_veto_disabled_in_original=True,
        interpretation="post-hoc annotated FP correlates, not causal source separation proof",
        exact_match_tol_ms=80,
        fp_baseline_overlap_tol_ms=80,
        test_never_read=True,already_exposed_DEV=True,no_promotion=True,
        recordings=rows,
        summary=scores,seconds_elapsed=round(time.perf_counter()-start,2))
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S639 FINAL",json.dumps(dict(
        baseline_count_confidence=sel.count_confidence,
        measures={name:{
            "delta_TP":v["delta_TP"],
            "delta_FP":v["delta_FP"],
            "delta_F1_points":round(v["delta_F1_points"],4),
            "novel_FP_reasons":v["nominally_novel_false_positive_reasons"],
            "diagnostic_totals":v["diagnostic_totals"]}
                  for name,v in scores.items()}
    )),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        run(args.out)
