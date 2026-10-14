"""S512 diagnostic score of S511 *new independent RAW WAV pitch generator*.

S511 original physical note candidates were generated and SHA frozen
BEFORE opening any original IDMT XML note labels. Compare to original
S488 45 matched XML attacks (active pitches at +30ms), and SHA original
independent pretrained BP (S487). All 459 native events without
verified XML pairing are UNKNOWN, not negative labels.

Metric is note pitch F1 CONDITIONAL on known positive note attacks,
not note-event onset F1 on continuous audio. Also show:
- paired onset recall of 45 true attacks (one-to-one)
- new verified note candidates added beyond raw BP
- oracle candidate coverage (posthoc with labels) NOT model accuracy
- complete exact note sets (and poly) and every regression
- unpaired physical attacks, UNKNOWN true/false.
No thresholds or top6/top12 are chosen from XML after freeze.
"""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
def need(x,msg):
    if not x:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def match_events(truth,notes):
    if not truth or not notes:return {}
    ts=np.array([x[0] for x in truth],float)
    pt=np.array([x[0] for x in notes],float)
    D=np.abs(ts[:,None]-pt[None,:])
    cost=np.concatenate([np.where(D<=.080,D/0.08*.50,1e6),
                       np.ones((len(ts),len(ts)))],axis=1)
    ii,jj=linear_sum_assignment(cost)
    return {int(i):int(j) for i,j in zip(ii,jj) if j<len(pt) and D[i,j]<=.080}
def score(true,baseline,policies,matched,unknown_groups):
    names=["BP_original_active_plus30","H13_top6_raw_only","H13_top12_raw_only",
        "BP_plus_H13_top6","BP_plus_H13_top12","BP_plus_H13_top1"]
    out={}
    base=[set(x) for x in baseline]
    Y=[set(x) for x in true]
    for name in names:
        proposed=[set(x) for x in policies[name]]
        tp=sum(len(a&b) for a,b in zip(Y,proposed))
        fp=sum(len(b-a) for a,b in zip(Y,proposed))
        fn=sum(len(a-b) for a,b in zip(Y,proposed))
        correct0=np.array([a==b for a,b in zip(Y,base)])
        correct1=np.array([a==b for a,b in zip(Y,proposed)])
        oldK=np.array([len(v) for v in base]);newK=np.array([len(v) for v in proposed])
        trueK=np.array([len(v) for v in Y]);poly=trueK>=2
        out[name]=dict(true_annotated_onsets=len(Y),
          poly_true_annotated_onsets=int(poly.sum()),
          original_BP_note_sets_correct=int(correct0.sum()),
          proposed_exact_full_note_sets=int(correct1.sum()),
          full_note_fixes=int(np.sum(~correct0&correct1)),
          full_note_regressions=int(np.sum(correct0&~correct1)),
          poly_exact_note_sets=int(np.sum(poly&correct1)),
          poly_full_note_fixes=int(np.sum(poly&~correct0&correct1)),
          poly_full_note_regressions=int(np.sum(poly&correct0&~correct1)),
          exactK_correct=int(np.sum(newK==trueK)),
          poly_exactK_correct=int(np.sum(poly&(newK==trueK))),
          note_true_positives=int(tp),note_false_positives_on_true_attacks=int(fp),
          note_false_negatives_on_true_attacks=int(fn),
          note_precision_conditional_on_known_true_attack=tp/max(1,tp+fp),
          note_recall_conditional_on_known_true_attack=tp/max(1,tp+fn),
          note_F1_conditional_on_known_true_attack=2*tp/max(1,2*tp+fp+fn),
          physical_groups_matched_one_to_one=matched,
          unpaired_physical_groups_truth_unknown=unknown_groups,
          cannot_estimate_full_stream_note_onset_precision=True)
    return out
def run(a):
    need(not a.out.exists(),"S512 no overwriting original candidate and XML after freeze")
    manifest=json.loads(a.generator_report.read_text())
    guitar=json.loads(a.guitar_report.read_text())
    bp=json.loads(a.bp_report.read_text())
    need(manifest["status"]=="S511_NEW_INDEPENDENT_RAW_HARMONIC_PITCH_ONSET_CANDIDATES_PREMIDI"
         and manifest["candidates_sha256"]==sha(a.generator)
         and manifest["fixed_two_top6_top12_policies_before_ground_truth"]
         and manifest["no_XML_or_MIDI_ground_truth_read_in_training_or_inference"]
         and guitar["status"]=="S488_PAIRED_IDMT_45_TRUE_GUITAR_ATTACKS_MULTISOURCE_BASIC_PITCH"
         and guitar["matched_rows_SHA256"]==sha(a.guitar)
         and bp["status"]=="S487_FIVE_GUITAR_IDMT_REAL_WAV_PRETRAINED_BASIC_PITCH_FROZEN_NOXML"
         and bp["all_output_note_events_sha256"]==sha(a.bp)
         and bp["original_XML_NOT_READ_before_note_freeze"],
         "S512 all original sound→note candidates must be SHA fixed before labels")
    labels=[json.loads(z) for z in a.guitar.read_text().splitlines()]
    independent=[json.loads(z) for z in a.generator.read_text().splitlines()]
    old=[json.loads(z) for z in a.bp.read_text().splitlines()]
    need(len(labels)==45 and sum(len(z["true_active"])>=2 for z in labels)==21,
         "S512 original 45 IDMT true onsets/21 poly drifted")
    names=sorted({z["source_wav"] for z in labels})
    need(len(names)==5 and len({x["WAV_filename_SHA256"] for x in independent})==5,
         "S512 original same five genuine guitar sources changed")
    per_source={};paired={}
    local_truth=[];local_baseline=[];policy_predictions=collections.defaultdict(list)
    oracle=collections.Counter()
    for source in names:
        tr=[z for z in labels if z["source_wav"]==source]
        raw=[z for z in independent if z["WAV_filename_SHA256"]==source]
        p6=sorted([z for z in raw if z["policy"]=="top6"],
                  key=lambda z:z["independent_physical_onset_s"])
        p12=sorted([z for z in raw if z["policy"]=="top12"],
                   key=lambda z:z["independent_physical_onset_s"])
        need(len(p6)==len(p12) and
             [x["physical_event_id"] for x in p6]==
             [x["physical_event_id"] for x in p12],
             "S512 original audio generator top6/top12 spectral onset grid drift")
        bps=[x for x in old if x["WAV_filename_SHA256"]==source]
        assignments=match_events([(x["original_XML_anchor"],x) for x in tr],
                  [(x["independent_physical_onset_s"],x) for x in p6])
        oracle["matched_physical"]+=len(assignments)
        oracle["unmatched_true"]+=len(tr)-len(assignments)
        oracle["unmatched_physical_unknown"]+=len(p6)-len(assignments)
        for i,label in enumerate(tr):
            true=set(map(int,label["true_active"]))
            time=float(label["original_XML_anchor"])+.030
            bp_active={int(v["pitch_MIDI"]) for v in bps if
                 float(v["predicted_onset_s"])<=time<float(v["predicted_offset_s"])}
            top6=set(map(int,p6[assignments[i]]["candidate_MIDI"])) if i in assignments else set()
            top12=set(map(int,p12[assignments[i]]["candidate_MIDI"])) if i in assignments else set()
            ordered=p6[assignments[i]]["candidate_MIDI"] if i in assignments else []
            raw_top1=set(map(int,ordered[:1]))
            local_truth.append(true);local_baseline.append(bp_active)
            pred=dict(BP_original_active_plus30=bp_active,
                H13_top6_raw_only=top6,H13_top12_raw_only=top12,
                BP_plus_H13_top6=bp_active|top6,
                BP_plus_H13_top12=bp_active|top12,
                BP_plus_H13_top1=bp_active|raw_top1)
            for k,value in pred.items():policy_predictions[k].append(value)
            oracle["true_pitch_total"]+=len(true)
            oracle["bp_supported_truth_pitches"]+=len(true&bp_active)
            oracle["H13_raw_top6_supported_truth_pitches"]+=len(true&top6)
            oracle["H13_raw_top12_supported_truth_pitches"]+=len(true&top12)
            oracle["BP_union_H13_top6_supported_truth_pitches"]+=len(true&(top6|bp_active))
            oracle["BP_union_H13_top12_supported_truth_pitches"]+=len(true&(top12|bp_active))
            oracle["new_true_pitch_top6_not_in_BP"]+=len((true&top6)-bp_active)
            oracle["new_true_pitch_top12_not_in_BP"]+=len((true&top12)-bp_active)
            oracle["all_true_pitches_in_BP_union_H13_top6"]+=int(true<=(top6|bp_active))
            oracle["all_true_pitches_in_BP_union_H13_top12"]+=int(true<=(top12|bp_active))
        per_source[source]=dict(real_annotated_onsets=len(tr),
            independent_physical_attack_candidates=len(p6),
            matched_to_true_attack=len(assignments),
            unmatched_physical_attacks_unknown=len(p6)-len(assignments))
    scoring=score(local_truth,local_baseline,policy_predictions,
                  oracle["matched_physical"],oracle["unmatched_physical_unknown"])
    a.out.mkdir(parents=True)
    report=dict(status="S512_FROZEN_RAW_GUITAR_NOTE_CANDIDATE_VS_ORIGINAL_XML_AUDIT",
         source_model_raw_acoustic_harmonic_is_not_trained_YourMT3=True,
         source_exposure_previous_trials_not_new_blind=True,
         true_active_notes_at_original_XML_plus30ms=True,
         all_45_original_real_attack_denominator_preserved=True,
         use_one_to_one_onset_matching_80ms=True,
         note_precision_F1_are_CONDITIONAL_on_known_positive_attacks=True,
         matched_raw_harmonic_physical_onsets_are_not_full_stream_event_precision=True,
         unmatched_model_groups_are_UNKNOWN_not_verified_false_births=True,
         true_pitch_oracle_upper_bound_not_an_inference_score=True,
         no_threshold_policy_selected_on_this_XML=True,
         official_S80_exactK_and_Live_unchanged=True,
         no_promotion=True,
         counts=dict(oracle),source_details=per_source,policies=scoring,
         SHA256=dict(original_S511_before_true_notes=sha(a.generator),
                     original_S487_BP_before_true_notes=sha(a.bp),
                     original_S488_45_real_XML_labels=sha(a.guitar)))
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S512 PREDECLARED REAL GUITAR AUDIO PITCH TEST",json.dumps(
        dict(coverage=dict(oracle),policies={
           name:{k:v for k,v in data.items() if
                  k in ("proposed_exact_full_note_sets","full_note_fixes",
                       "full_note_regressions","note_F1_conditional_on_known_true_attack",
                       "note_recall_conditional_on_known_true_attack","poly_exact_note_sets")}
             for name,data in scoring.items()})),flush=True)
def selftest():
    truth=[{52,55}];baseline=[{52}]
    p={key:[{52,55}] for key in (
       "BP_original_active_plus30","H13_top6_raw_only","H13_top12_raw_only",
       "BP_plus_H13_top6","BP_plus_H13_top12","BP_plus_H13_top1")}
    p["BP_original_active_plus30"]=[{52}]
    z=score(truth,baseline,p,1,3)
    need(z["BP_plus_H13_top6"]["full_note_fixes"]==1
         and z["BP_plus_H13_top6"]["note_F1_conditional_on_known_true_attack"]==1.,
         "S512 true candidate discovery should count new pitch correctness")
    print("S512 SELFTEST PASS audit independent genuine notes without ascribing unknown false onsets")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for k in ("generator","generator-report","guitar","guitar-report",
              "bp","bp-report","out"):
        p.add_argument("--"+k,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for k in ("generator","generator_report","guitar","guitar_report",
                  "bp","bp_report","out"):need(getattr(a,k)is not None,"S512 missing --"+k)
        run(a)
