"""S497 — independent paired loss audit for S496 joint onset/notes/Exact-K.

Original frozen S80 guitar 45 true XML note events (21 poly) and original
V88 piano 322 MIDI new-note events (73 poly). Compare THREE predeclared
graphs with each source baseline, score fixes AND regressions for K0/K1+
and poly, and count every native activation without aligned annotation.
No events with unknown truth are labelled false or true negatives.
Read S488/S476 truth only AFTER S496 SHA-frozen models are validated.
No source training, model threshold selection, live change or promotion.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
def need(c,msg):
    if not c:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def score(y,baseline,candidate):
    truth=np.asarray(y,int);original=np.asarray(baseline,int);proposed=np.asarray(candidate,int)
    need(len(truth)==len(original)==len(proposed),"S497 paired predicted/native event rows changed")
    before=original==truth;after=proposed==truth
    poly=truth>=2
    under=proposed<truth;over=proposed>truth
    return dict(n=len(y),true_poly=int(poly.sum()),
        source_correct=int(before.sum()),candidate_correct=int(after.sum()),
        fixes=int(np.sum(~before&after)),regressions=int(np.sum(before&~after)),
        net=int(after.sum()-before.sum()),
        poly_source_correct=int(np.sum(before&poly)),
        poly_candidate_correct=int(np.sum(after&poly)),
        poly_fixes=int(np.sum(~before&after&poly)),
        poly_regressions=int(np.sum(before&~after&poly)),
        under_count=int(under.sum()),over_count=int(over.sum()),
        original_K0_missed_true_attack=int(np.sum(original==0)),
        original_K0_recovered_correct=int(np.sum((original==0)&after)),
        original_K0_changed_to_wrong_positive=int(np.sum((original==0)&(proposed>0)&~after)),
        source_Kpositive_regressed=int(np.sum((original>0)&before&~after)))
def run(frozen,freeze_report,guitar,guitar_report,piano,piano_report,out):
    need(not out.exists(),"S497 immutable paired audit already exists")
    spec=json.loads(freeze_report.read_text())
    g=json.loads(guitar_report.read_text());p=json.loads(piano_report.read_text())
    need(spec["status"]=="S496_TWO_SOURCE_JOINT_GRAPH_K_ONSET_PITCH_SHADOW_PRELABEL"
         and spec["true_note_annotations_not_read_in_S496"]
         and spec["original_source_S80_or_V88_files_unchanged"]
         and spec["no_promotion"]
         and g["status"]=="S488_PAIRED_IDMT_45_TRUE_GUITAR_ATTACKS_MULTISOURCE_BASIC_PITCH"
         and g["matched_rows_SHA256"]==sha(guitar)
         and p["status"]=="S476_THIRD_MAESTRO_PIANO_NEW_ONSET_K_VS_SUSTAINING_K_READONLY"
         and p["SHA256"]["paired_label_comparison"]==sha(piano),
         "S497 only genuine S496 source model freeze and original external XML/MIDI can be scored")
    rowsG=[json.loads(line) for line in guitar.read_text().splitlines()]
    rowsP=[json.loads(line) for line in piano.read_text().splitlines()]
    need(len(rowsG)==45 and len(rowsP)==322,"S497 scored original true attack cohort changed")
    summary={}
    for corpus,records,truth_field,k_field,correct,correctpoly,poly_n in (
        ("guitar_S80",rowsG,"true_active","original_S80_K",25,7,21),
        ("piano_V88",rowsP,"true_new_onset_MIDI_pitches","original_V88_K",42,7,73)):
        path=frozen/f"S496_{corpus}_JOINT_SHADOW_FROZEN.npz"
        need(spec["output_files"][corpus]["frozen_source_SHA256"]==sha(path),
             "S497 frozen source S496 K decisions changed after reading external labels")
        with np.load(path,allow_pickle=False) as z:
            ids=z["row_id"].astype(str)
            source=z["source_K"].astype(int)
            tested={key:z[key].astype(int) for key in (
                "guarded_sourceK","joint_Knew_shadow","joint_Kactive_shadow")}
            ambiguities=z["ambiguous_event_conflict"].astype(bool)
            competitors=z["competing_original_positiveK"].astype(bool)
            ownership=z["note_group_assigned_to_native"].astype(int)
        index={k:i for i,k in enumerate(ids)}
        need(len(index)==len(ids) and all(x["row_id"] in index for x in records),
             "S497 every positive truth must match exact frozen native ID")
        originalidx=np.asarray([index[x["row_id"]] for x in records],int)
        actual=np.asarray([len(x[truth_field]) for x in records],int)
        source_onsets=np.asarray([x[k_field] for x in records],int)
        need(np.array_equal(source[originalidx],source_onsets)
             and int(np.sum(source_onsets==actual))==correct
             and int(np.sum((source_onsets==actual)&(actual>=2)))==correctpoly
             and int(np.sum(actual>=2))==poly_n,
             "S497 source S80 versus piano V88 historical Exact-K identity mismatch")
        known=np.zeros(len(ids),bool);known[originalidx]=True
        policies={}
        for policy,after in tested.items():
            baseline=score(actual,source[originalidx],after[originalidx])
            unknown=(~known)&(after!=source)
            labeled=(known)&(after!=source)
            touchedpositive=known&(source>0)&(after!=source)
            policies[policy]=dict(
                true_annotated_attack_paired_K=baseline,
                true_attack_native_rows_changed=int(labeled.sum()),
                true_attack_original_Kpositive_changed=int(touchedpositive.sum()),
                UNKNOWN_native_rows_changed=int(unknown.sum()),
                UNKNOWN_K0_to_positive=int(np.sum(unknown&(source==0)&(after>0))),
                UNKNOWN_Kpositive_to_other=int(np.sum(unknown&(source>0)&(after!=source))),
                changed_UNKNOWN_also_marked_ambiguous=int(np.sum(unknown&ambiguities)),
                changed_UNKNOWN_competes_with_source_positive=int(np.sum(unknown&competitors)),
                total_nonmatched_pitch_native_rows=int(np.sum(ownership<0)),
                report_UNKNOWN_as_truth_Never=True)
        need(np.array_equal(tested["guarded_sourceK"][source>0],source[source>0]),
             "S497 guarded original Kpositive safety rule broken")
        summary[corpus]=dict(source_native_rows=len(ids),
             annotated_note_birth_attacks=len(records),true_poly_events=poly_n,
             original_source_model="complete_S80" if corpus=="guitar_S80" else "V88_only",
             policies=policies)
        print("S497 PAIRED",json.dumps(dict(corpus=corpus,
           guarded=policies["guarded_sourceK"],
           joint_birth=policies["joint_Knew_shadow"],
           joint_active=policies["joint_Kactive_shadow"])),flush=True)
    out.mkdir(parents=True)
    rep=dict(status="S497_JOINT_GRAPH_EXACTK_BIRTH_ACTIVE_PAIRED_TWO_SOURCE_REVIEW",
        independent_original_guitar=S80_NOTE_K_FAMILY if False else "full original S80",
        independent_original_piano="original V88 only",
        K_label_definitions_different_guitar_active_vs_piano_new=True,
        no_model_training_no_label_threshold_selection=True,
        prior_external_MIDI_XML_exposure_remains=True,
        original_S80_and_Live_untouched=True,
        unknown_native_events_never_asserted_false_births=True,
        protected_original_source_Kpositive_guarded=True,
        experimental_not_promoted=True,no_promotion=True,
        comparisons=summary,
        sha256=dict(S496_unlabelled_frozen=sha(freeze_report),
                    IDMT_external_XML=sha(guitar),MAESTRO_original_MIDI=sha(piano)))
    (out/"report.json").write_text(json.dumps(rep,indent=2)+"\n")
def selftest():
    d=score([2,1,2,0],[0,1,2,0],[2,2,2,1])
    need(d["fixes"]==1 and d["regressions"]==2 and
         d["original_K0_changed_to_wrong_positive"]==1,
         "S497 count fixes and regressions even on true K0 non-attack")
    print("S497 SELFTEST PASS per-example correction, regression and false K0 risks")
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--self-test",action="store_true")
    for k in ("frozen","freeze-report","guitar","guitar-report","piano","piano-report","out"):
        ap.add_argument("--"+k,type=Path)
    a=ap.parse_args()
    if a.self_test:selftest()
    else:
        for k in ("frozen","freeze_report","guitar","guitar_report","piano","piano_report","out"):
            need(getattr(a,k)is not None,"S497 missing --"+k)
        run(a.frozen,a.freeze_report,a.guitar,a.guitar_report,a.piano,a.piano_report,a.out)
