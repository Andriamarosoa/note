"""S627 NOTE forensic audit: compare matched TP vs missed FN vs false FP at all available levels.

Audits SAME original 17 DEV songs, frozen full S613 neural+ONNX and
original causal 256-hop front end. This is an analysis tool, NOT a fusion
rule, model, train update, augmentation, or independent test score.

Ground truth is used only AFTER original frozen causal predictions to
label TP/FN/FP and stratify examples. Actual feature interpretations
match original src/causal_note/strict_live.py (not S619's approximate
descriptions of the temporal lag feature slices).

Dimensions:
 - corpus, composition, pitch/register, duration, onset K1..6;
 - TRUE active pitches and intervals (1/2, octave 12, twelfth 19,
   two octaves 24), recent independent attacks, prior same-pitch attack;
 - per-frame onset/sustain/release, count-K distribution/peak,
   selector armed/active/cooldown before emitting;
 - per-pitch logarithmic 3-resolution H1 and H2/H3/harmonic content;
 - temporal lag 2/4/8/16 x original hop, spectral activity/flux;
 - annotation labels vs emitted NoteOn identity, duplicate/late FPs;
 - TP-vs-FN conditional miss rates with denominators and composition
   coverage; within-source same-pitch matched controls where possible;
 - exact note onset groups and FPs, no equivalence to historical S80 K.
No fabricated causation. Equal normalized harmonic pitch interval is
ONLY a relation, NOT proof of an overtone. Small samples labelled fragile.

The original training DATA are counted separately by S624/S625.
This audit does not use held-out TEST data. Its DEV patterns are
exploratory and cannot themselves establish augmentation efficacy.
"""
from __future__ import annotations

from collections import Counter,defaultdict
from pathlib import Path
import argparse,json,math

import numpy as np

from causal_note.strict_live import CausalSelection,Selection,SR,load_heads
from scripts.s613_train_strict_live import event_score,real_notes,summary
from scripts.s615_train_multihop import source_data
from scripts.s625_audit_exact_note_combinations import pitch_name


def interval_relation(pitch,active):
    intervals={abs(int(pitch)-int(q)) for q in active if int(q)!=int(pitch)}
    if 1 in intervals or 2 in intervals:
        return "close_1_2"
    if intervals.intersection((12,19,24)):
        return "harmonic_interval_12_19_24"
    if 7 in intervals:
        return "fifth_interval_7"
    if intervals:
        return "other_coactive"
    return "none"


def time_bin(delta):
    if delta is None or not math.isfinite(delta):return "none"
    if delta<=.010:return "up_to_10ms"
    if delta<=.030:return "10_to_30ms"
    if delta<=.080:return "30_to_80ms"
    if delta<=.250:return "80_to_250ms"
    return "after_250ms"


def k_bin(n):
    return str(n) if n<=4 else "5_plus"


def duration_bin(duration):
    if duration<=.1:return "under_100ms"
    if duration<=.35:return "100_to_350ms"
    if duration<=.8:return "350_to_800ms"
    return "over_800ms"


def pitch_register(pitch):
    if pitch < 52:return "low_E2_to_Dsharp3"
    if pitch < 64:return "middle_E3_to_Dsharp4"
    if pitch < 76:return "high_E4_to_Dsharp5"
    return "very_high_E5_plus"


def feature_view(x):
    """Causal 126-column NOTE ABI at ONE real frame for ONE pitch.

    0..23 three FFT windows by 8 harmonics, 24..47 relative harmonic
    profiles, 48..53 spectral pitch score and fundamental strength,
    54..59 2-hop lag, 60..65 4-hop lag, 66..71 8-hop lag,
    72..77 16-hop lag, 78..101 exact pitch-vs-neighbor contrasts,
    102..107 recent mean, 108..113 whole history mean,
    114..119 current minus recent minimum,
    120..122 spectral flux, 123 short RMS, 124 long-window RMS,
    125 normalized MIDI pitch. ALL features are past-only.
    """
    x=np.asarray(x)
    if x.shape!=(126,):
        raise ValueError("Expected exactly one 126-dim original feature row")
    h=x[:24].reshape(3,8)
    mean_h=np.maximum(h.mean(axis=0),0)
    fundamental=float(mean_h[0])
    harmonic=float(mean_h[1:4].mean())
    return dict(
        h1=float(fundamental),
        h2=float(mean_h[1]),h3=float(mean_h[2]),
        h2_to_h1=float(mean_h[1]/max(fundamental,.05)),
        h234_to_h1=float(harmonic/max(fundamental,.05)),
        pitch_score=float(x[48:51].mean()),
        lag_2hop_score_delta=float(x[54:57].max()),
        lag_4hop_score_delta=float(x[60:63].max()),
        lag_8hop_score_delta=float(x[66:69].max()),
        lag_16hop_score_delta=float(x[72:75].max()),
        recent_positive_novelty=float(x[114:117].max()),
        spectral_flux_mean=float(x[120:123].mean()),
        short_rms_log=float(x[123]),long_rms_log=float(x[124]),
        relative_harmonic_prominence=float(x[24:48].max()),
    )


def build_context(truth,i):
    n=truth[i]; t=float(n["time"]); pitch=int(n["pitch"])
    different_active=[]
    recent_other=[]; prior_same=[]
    onset_near=[j for j,r in enumerate(truth)
                 if j!=i and abs(float(r["time"])-t)<=.030
                 and int(r["pitch"])!=pitch]
    for j,other in enumerate(truth):
        if i==j:continue
        q=int(other["pitch"]); a=float(other["time"])
        b=a+float(other["duration"])
        if q!=pitch and a<=t<b:
            different_active.append(q)
        if q!=pitch and a<=t and t-a<=.25:
            recent_other.append(t-a)
        if q==pitch and a<=t:
            prior_same.append(t-a)
    active_distinct=sorted(set(different_active+[pitch]))
    return dict(
        corpus_note_onset_time_s=t,pitch=pitch,pitch_name=pitch_name(pitch),
        register=pitch_register(pitch),
        duration_s=float(n["duration"]),duration_band=duration_bin(n["duration"]),
        active_K=k_bin(len(active_distinct)),
        active_K_raw=len(active_distinct),
        onset_K=k_bin(len(set(int(truth[j]["pitch"]) for j in onset_near)|{pitch})),
        onset_K_raw=len(set(int(truth[j]["pitch"]) for j in onset_near)|{pitch}),
        coactive_interval=interval_relation(pitch,different_active),
        simultaneous_pitches=active_distinct,
        nearest_other_onset=time_bin(min(recent_other) if recent_other else None),
        previous_same_pitch=time_bin(min(prior_same) if prior_same else None),
        has_prior_different_active=bool(different_active),
        note_ends_near_clip=bool(t+float(n["duration"])>=23.98),
    )


def frame_metrics(ends,x,probs,counts,selector_states,pitch,t,*,timing="first_available"):
    """Frame strictly AT/AFTER expected true onset; <= 5.8ms later.

    Source/GT onset timestamps select which original causal frame to
    AUDIT. This is NOT a proposed online algorithm or an oracle input.
    """
    ix=int(np.searchsorted(ends,t*SR,side="left"))
    if ix>=len(ends):
        ix=len(ends)-1
    pindex=int(pitch)-40
    one=np.asarray(probs[ix,pindex])
    count=np.asarray(counts[ix])
    row=dict(
        first_frame_time_s=round(float(ends[ix]/SR),6),
        first_frame_relative_to_annotation_ms=round(
            float((ends[ix]/SR-t)*1000),4),
        p_onset=float(one[0]),p_sustain=float(one[1]),p_release=float(one[2]),
        predicted_K=int(count.argmax()),K_confidence=float(count.max()),
        selector_active_before=bool(selector_states["active"][ix,pindex]),
        selector_armed_before=bool(selector_states["armed"][ix,pindex]),
        selector_prior_high_frames=int(selector_states["high"][ix,pindex]),
        selector_prior_low_frames=int(selector_states["low"][ix,pindex]),
        prior_note_on_age_ms=(
            float((ends[ix]-selector_states["last_on"][ix,pindex])/SR*1000)
            if selector_states["last_on"][ix,pindex]>=0 else None),
    )
    row.update(feature_view(x[ix,pindex]))
    return row


def posthoc_head_evidence(ends,p,pitch,t):
    """Diagnostic ±80ms frame maxima, NOT causal at t."""
    target=(int(pitch)-40)
    indexes=np.flatnonzero(np.abs(ends/SR-t)<=.080)
    if indexes.size==0:
        return dict(classification="no_frame_in_80ms",peak_onset=None)
    subset=p[indexes,target]
    peak_on=float(subset[:,0].max())
    peak_sus=float(subset[:,1].max())
    concomitant=(subset[:,0]>=.60)&(subset[:,1]>=.50)
    if peak_on<.60: reason="weak_onset_head"
    elif not concomitant.any(): reason="onset_sustain_asynchronous"
    else: reason="passed_heads_selector_or_timing"
    return dict(classification=reason,peak_onset=peak_on,
                peak_sustain=peak_sus,joint_threshold_frame=bool(concomitant.any()))


def selector_trace(ends,p,k,selection,length):
    s=CausalSelection(selection,hop=256)
    states=dict(active=[],armed=[],high=[],low=[],last_on=[])
    events=[]
    for end,prob,cnt in zip(ends,p,k):
        for attr in states:
            states[attr].append(getattr(s,attr).copy())
        events.extend(s.step(prob,cnt,int(end),int(end)))
    events.extend(s.panic(int(length)))
    return events,{key:np.stack(val) for key,val in states.items()}


def binned(value):
    if value is None:return "missing"
    if value<.15:return "lt_015"
    if value<.35:return "015_035"
    if value<.50:return "035_050"
    if value<.60:return "050_060"
    if value<.80:return "060_080"
    return "ge_080"


def stratified_rates(rows,dimensions,*,min_cases=15):
    results={}
    for dimension in dimensions:
        groups=defaultdict(list)
        for row in rows:
            val=row.get(dimension)
            if isinstance(val,bool):val=str(val).lower()
            if isinstance(val,float):
                val=round(val,3)
            if val is None:val="missing"
            groups[str(val)].append(row)
        output=[]
        for key,group in groups.items():
            total=len(group)
            misses=sum(not r["matched"] for r in group)
            recordings=len(set(r["source"] for r in group))
            compositions=len(set((r["corpus"],r["composition"]) for r in group))
            output.append(dict(condition=key,total=total,matched=total-misses,
                   missed=misses,miss_rate=round(misses/total,5),
                   source_recordings=recordings,independent_compositions=compositions,
                   limited_evidence=bool(total<min_cases or compositions<3)))
        results[dimension]=sorted(output,key=lambda x:(-x["missed"],-x["total"]))
    return results


def matched_controls(rows):
    """Matched control contrasts conditioned on exact source/pitch/active K.

    Greedy 1:1 without replacement; do not mistake these correlated
    observational comparisons for measured causes.
    """
    remaining={i for i,r in enumerate(rows) if r["matched"]}
    pairs=[]
    used_fallback=Counter()
    for missed in (r for r in rows if not r["matched"]):
        levels=(
            ("same_source_pitch_K",lambda r:
             r["source"]==missed["source"] and r["pitch"]==missed["pitch"]
             and r["active_K"]==missed["active_K"]),
            ("same_source_pitch",lambda r:
             r["source"]==missed["source"] and r["pitch"]==missed["pitch"]),
            ("same_corpus_pitch_K",lambda r:
             r["corpus"]==missed["corpus"] and r["pitch"]==missed["pitch"]
             and r["active_K"]==missed["active_K"]),
            ("same_corpus_pitch",lambda r:
             r["corpus"]==missed["corpus"] and r["pitch"]==missed["pitch"]),
        )
        chosen=None; method=None
        for level,predicate in levels:
            candidates=[i for i in remaining if predicate(rows[i])]
            if candidates:
                chosen=min(candidates,key=lambda i:abs(
                    rows[i]["duration_s"]-missed["duration_s"]))
                method=level;break
        if chosen is None:continue
        remaining.remove(chosen)
        control=rows[chosen]
        metrics={}
        for name in (
            "p_onset","p_sustain","p_release",
            "h1","h2_to_h1","h234_to_h1",
            "lag_2hop_score_delta","lag_4hop_score_delta",
            "recent_positive_novelty","short_rms_log","K_confidence",
        ):
            metrics[name]=float(missed[name])-float(control[name])
        pairs.append(metrics)
        used_fallback[method]+=1
    result={}
    for metric in (pairs[0] if pairs else {}):
        vector=[p[metric] for p in pairs]
        result[metric]=dict(median_FN_minus_TP=float(np.median(vector)),
                            mean_FN_minus_TP=float(np.mean(vector)),
                            fraction_FN_higher=float(np.mean(np.array(vector)>0)))
    return dict(paired_samples=len(pairs),match_level=dict(used_fallback),
                unpaired_missed_cases=sum(not r["matched"] for r in rows)-len(pairs),
                metrics=result,
                caveat="Observational matched comparisons without causal identification; fallback matches weaker and residual confounding likely.")


def make_fp_row(event,truth,ends,x,p,k,states,spec):
    t=float(event["time"]); pitch=int(event["pitch"])
    active={int(n["pitch"]) for n in truth
            if n["time"]<=t<n["time"]+n["duration"]}
    nearest=(min((abs(float(n["time"])-t) for n in truth
                  if int(n["pitch"])==pitch),default=None))
    source=interval_relation(pitch,active)
    context=(
      "same_pitch_labeled_active" if pitch in active else
      "possible_outside_80ms_timing" if nearest is not None and nearest<=.25 else
      "harmonic_interval" if source=="harmonic_interval_12_19_24" else
      "close_pitch_interval" if source=="close_1_2" else
      "another_note_active" if active else "no_annotated_note_active"
    )
    acoustic=frame_metrics(ends,x,p,k,states,pitch,t)
    return dict(source=spec["name"],corpus=spec["corpus"],
                composition=spec["composition"],pitch=pitch,pitch_name=pitch_name(pitch),
                time_s=t,reason=event["reason"],active_labeled_pitches=sorted(active),
                context_priority=context,
                pitch_interval=source,
                nearest_same_pitch_truth_delta_ms=(
                    float(nearest*1000) if nearest is not None else None),
                **acoustic)


def audit(root,out,model_file,*,include_rows=True):
    if out.exists():
        raise FileExistsError("Refusing to overwrite S627 audit output")
    protocol=json.loads((root/"protocol.json").read_text())
    sources=[s for s in protocol["sources"] if s["split"]=="dev"]
    if len(sources)!=17:raise ValueError("Original 17 DEV sources required")
    model=load_heads(model_file)
    if model.metadata.get("architecture")!="native_causal_hybrid":
        raise ValueError("S627 requires true original ONNX+neural S613")
    selection=Selection(**model.metadata["selection"])
    all_truth=[]; all_fp=[]; source_scores=[]; by_corpus=defaultdict(list)
    for index,spec in enumerate(sources,1):
        ends,x,_,_,annotations,length=source_data(root,spec,256)
        p,k=model.predict(x)
        events,state=selector_trace(ends,p,k,selection,length)
        truth=real_notes(annotations,length/SR)
        scored,pairs=event_score(events,truth,.080)
        ons=[ev for ev in events if ev["type"]=="note_on"]
        matched_truth={j for _,j in pairs}
        matched_pred={i for i,_ in pairs}
        source_scores.append(scored);by_corpus[spec["corpus"]].append(scored)
        for i,note in enumerate(truth):
            t=float(note["time"])
            row=dict(source=spec["name"],corpus=spec["corpus"],
                     composition=spec["composition"],matched=i in matched_truth)
            row.update(build_context(truth,i))
            row.update(frame_metrics(ends,x,p,k,state,int(note["pitch"]),t))
            row.update(posthoc_head_evidence(ends,p,int(note["pitch"]),t))
            row["first_frame_onset_bin"]=binned(row["p_onset"])
            row["first_frame_sustain_bin"]=binned(row["p_sustain"])
            row["local_head_evidence_class"]=row.pop("classification")
            row["max_onset_within_plus_minus_80ms"]=row.pop("peak_onset")
            row["max_sustain_within_plus_minus_80ms"]=row.pop("peak_sustain")
            row["same_frame_joint_threshold_within_80ms"]=row.pop("joint_threshold_frame",False)
            row["predicted_K_vs_active_K"]=(row["predicted_K"]-row["active_K_raw"])
            row["real_positive_novelty_flag"]=bool(row["recent_positive_novelty"]>.04)
            row["matched_noteon_delta_ms"]=None
            if i in matched_truth:
                matched_index=next(a for a,b in pairs if b==i)
                row["matched_noteon_delta_ms"]=round(
                    (ons[matched_index]["time"]-t)*1000,4)
            all_truth.append(row)
        for i,event in enumerate(ons):
            if i not in matched_pred:
                all_fp.append(make_fp_row(event,truth,ends,x,p,k,state,spec))
        print(json.dumps(dict(
            source_index=index,total_sources=len(sources),source=spec["name"],
            TP=scored["TP"],FP=scored["FP"],FN=scored["FN"],
        )),flush=True)
    score=summary(source_scores)
    assert (score["TP"],score["FN"],score["FP"])==(1241,838,734),score
    dimensions=[
      "corpus","register","pitch_name","duration_band","active_K",
      "onset_K","coactive_interval","nearest_other_onset",
      "previous_same_pitch","selector_active_before","selector_armed_before",
      "local_head_evidence_class","first_frame_onset_bin",
      "first_frame_sustain_bin","real_positive_novelty_flag",
      "note_ends_near_clip",
    ]
    counts=Counter(r["local_head_evidence_class"] for r in all_truth if not r["matched"])
    fp_counts=Counter(row["context_priority"] for row in all_fp)
    report=dict(
       status="S627_MULTIDIMENSIONAL_FORENSIC_AUDIT_S613_17DEV_NO_PROMOTION",
       frozen_hybrid_baseline=score,
       sources=17,test_sources_read=0,note_events=len(all_truth),
       emitted_false_note_on=len(all_fp),
       true_notes_stratified=stratified_rates(all_truth,dimensions),
       matched_FN_vs_TP_controls=matched_controls(all_truth),
       missed_head_evidence_classes=dict(counts),
       false_positive_context_priority=dict(fp_counts),
       false_positive_by_pitch=dict(Counter(r["pitch_name"] for r in all_fp)),
       false_positive_by_corpus=dict(Counter(r["corpus"] for r in all_fp)),
       feature_layout_provenance="src/causal_note/strict_live.py CausalPitchFeatures._frame original 126-dimensional ABI",
       definitions={
          "tp_fn":"maximum cardinality same-MIDI-note matching within +/-80ms",
          "first_decision_frame":"first causal hop256 ending at or after an annotated attack (at most 5.8ms later)",
          "posthoc_head":"maximum over +/-80ms uses future evidence ONLY for audit, cannot be directly used by strict LIVE",
          "active_K":"annotation-derived # unique pitches sustaining at NOTE onset",
          "onset_K":"unique annotated pitches within +/-30ms onset cluster, not historical S80 K",
          "harmonic_interval":"octave/twelfth/two-octave pitch distance, never proof of acoustically harmonic causation",
          "RMS":"original model's transformed log-RMS feature, not raw dBFS",
          "counts_denominator":"individual truth notes for TP/FN; each note counted ONCE (unlike S626 pair counts)",
          "correlation":"differences between FN and TP are association, not established cause",
       },
       limitations=[
          "DEV descriptions and same-source/other-source paired controls are exploratory and correlated",
          "Direct comparison of absolute model scores to spectral energy across instruments may be confounded by level/instrument",
          "No independent held-out test songs and no microphone or FluidSynth latency measured",
          "No model or selector changed, no new dataset generated",
          "S627 is not all historical NOTE sources, only pinned S613 original 17 DEV songs",
       ],
    )
    if include_rows:
        report["per_note_truth_rows"]=all_truth
        report["per_false_predicted_event_rows"]=all_fp
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+"\n")
    print("S627_TOPLINE "+json.dumps(dict(
       TP=score["TP"],FN=score["FN"],FP=score["FP"],
       missed_head_evidence=report["missed_head_evidence_classes"],
       false_positive_context=report["false_positive_context_priority"],
       matched_controls=report["matched_FN_vs_TP_controls"]["match_level"],
       corpus=report["true_notes_stratified"]["corpus"],
       active_K=report["true_notes_stratified"]["active_K"],
       coactive_interval=report["true_notes_stratified"]["coactive_interval"],
       onset_K=report["true_notes_stratified"]["onset_K"],
       recent_attacks=report["true_notes_stratified"]["nearest_other_onset"],
       original_onset_confidence=report["true_notes_stratified"]["first_frame_onset_bin"],
    ),sort_keys=True),flush=True)
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--model",type=Path,default=Path(
        "model/live/strict_note/NOTE_strict_live.npz"))
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--summary-only",action="store_true")
    a=ap.parse_args()
    audit(a.root,a.out,a.model,include_rows=not a.summary_only)


if __name__=="__main__":
    main()
