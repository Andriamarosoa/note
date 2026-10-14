"""S452: independent report-only decomposition of source-K vs pitch identity.

Never read a new label, retrain, or change frozen S450/S451 predictions.
The fundamental upper bound of any K-locked pitch head is the number
of original source S80 K-correct events on an annotation-matched cohort.
Scores exclude unmatched acoustic events; they are NOT true K0.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path

def need(x,m):
    if not x:raise ValueError(m)

def run(s450,s451,out):
    need(not out.exists(),"S452 immutable causal-ceiling audit exists")
    source=json.loads(s450.read_text())
    audit=json.loads(s451.read_text())
    need(source["status"]=="S450_ORIGINAL_S80_EXTERNAL_IDMT_506_NOTE_NAMES_FROZEN"
         and audit["status"]=="S451_ORIGINAL_S80_EXTERNAL_IDMT_TRUE_NOTE_IDENTITY_PAIRED"
         and source["no_XML_MIDI_truth_opened"] and audit["exactK_regressions"]==0
         and source["original_S80_predictions_unchanged"]
         and audit["no_refit_no_threshold_tuning_on_IDMT"],
         "S452 invalid original source or labels audit")
    scores=audit["scored"]
    detail={}
    for k in ("all_matched","true_poly_K2plus",
              "true_K1","true_K2","true_K3","true_K4","true_K5","true_K6"):
        d=scores[k]
        n=int(d["n"]);correct_k=int(d["original_S80_K_correct"])
        b=int(d["baseline_exact_notes"]);c=int(d["candidate_exact_notes"])
        need(0<=b<=correct_k<=n and 0<=c<=correct_k<=n
             and c-b==d["net"],
             "S452 K-locked upper bound contradicted by note identity scores")
        detail[k]=dict(n=n,exactK_correct=correct_k,
            unavoidable_wrong_K=n-correct_k,
            maximum_whole_chords_with_any_K_locked_pitch_head=correct_k,
            original_S430_correct_whole_chords=b,
            S446_correct_whole_chords=c,
            K_right_but_S446_pitches_wrong=correct_k-c,
            max_possible_K_locked_rate=(correct_k/n if n else None),
            actual_S446_whole_chord_rate=(c/n if n else None),
            conditional_pitch_accuracy_if_K_correct=(c/correct_k if correct_k else None),
            sourceK_wrong_fraction=(1-correct_k/n if n else None),
            S446_vs_baseline_net=c-b)
    need(detail["all_matched"]["n"]==audit["pitches_in_supported_MIDI40to88"]
         and detail["true_poly_K2plus"]["n"]<=detail["all_matched"]["n"],
         "S452 scored population drift")
    result=dict(status="S452_EXTERNAL_K_LOCKED_NOTE_IDENTITY_CEILING_AUDIT",
        event_population_total=int(audit["n_external_native_V88_events"]),
        events_with_matching_XML=int(audit["matched_annotation_onsets"]),
        events_scored_in_MIDI40_88=int(audit["pitches_in_supported_MIDI40to88"]),
        unmatched_not_K0=int(audit["unknown_or_unmatched_events_excluded"]),
        original_K0_not_blindly_scored=True,
        original_K_predictions_unchanged=True,
        S446_note_heads_cannot_repair_incorrect_K_without_new_K_model=True,
        upper_bound_is_only_for_this_annotated_subset=True,
        not_blind_new_instrument_validation=True,
        no_promotions_or_training=True,
        metrics=detail)
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("S452 CEILING",json.dumps({
        "matched_MIDI_events":detail["all_matched"],
        "poly_trueK2plus":detail["true_poly_K2plus"],
        "by_trueK":{k:detail[k] for k in (
            "true_K1","true_K2","true_K3","true_K4")}}),flush=True)
    print("S452 PASS no benchmark inflation through unmatched K0",flush=True)

def selftest():
    a=dict(n=21,original_S80_K_correct=7,
           baseline_exact_notes=4,candidate_exact_notes=4,net=0)
    assert 0<=a["baseline_exact_notes"]<=a["original_S80_K_correct"]<=a["n"]
    assert a["original_S80_K_correct"]-a["candidate_exact_notes"]==3
    print("S452 SELFTEST PASS source K cardinality upper bound")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for n in ("s450-report","s451-report","out"):p.add_argument("--"+n,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for n in ("s450_report","s451_report","out"):
            need(getattr(a,n)is not None,"S452 missing --"+n)
        run(a.s450_report,a.s451_report,a.out)
