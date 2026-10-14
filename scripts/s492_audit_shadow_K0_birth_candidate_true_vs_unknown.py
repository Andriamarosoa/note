"""S492: evaluate frozen, true source K0-only proposed rescue per source.

Only AFTER SHA-pinned S491 unlabelled freeze, compare to separately held
original S488 true IDMT guitar notes and S476 true piano MIDI onsets.
Score both pre-registered policies; no picking on test truth. On genuine
annotated onset rows true K>=1, so original K0 cases rescued to correct K
cannot regress BUT the unannotated native V88 events can be false K0
candidates. Report every activated UNKNOWN row separately, never count
them as true or false labels. No original S80 or V88 source changes.
"""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
def need(c,m):
    if not c:raise ValueError(m)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def score_sample(source,truth,guess):
    truth=np.asarray(truth,int)
    source=np.asarray(source,int)
    guess=np.asarray(guess,int)
    assert len(truth)==len(source)==len(guess)
    before=source==truth
    after=guess==truth
    poly=truth>=2
    return dict(n=len(truth),
        true_poly=int(np.sum(poly)),
        original_exactK=int(before.sum()),
        proposed_exactK=int(after.sum()),
        fixed=int(np.sum(~before&after)),
        regressed=int(np.sum(before&~after)),
        net=int(after.sum()-before.sum()),
        true_poly_original_exactK=int(np.sum(before&poly)),
        true_poly_proposed_exactK=int(np.sum(after&poly)),
        true_poly_fixed=int(np.sum(~before&after&poly)),
        true_poly_regressed=int(np.sum(before&~after&poly)),
        source_true_K0_collapse=int(np.sum(source==0)),
        recovered_true_K0_exactK=int(np.sum((source==0)&after)),
        source_K0_proposed_wrong_exactK=int(np.sum((source==0)&(guess>0)&~after)),
        source_true_K1_or_higher_protected=int(np.sum(source>0)),
        new_source_K0_rescues=int(np.sum((source==0)&(guess>0))))
def run(guitar,piano,source_report,s488,s488report,s476,s476report,out):
    need(not out.exists(),"S492 historical shadow scoring output exists")
    src=json.loads(source_report.read_text())
    g=json.loads(s488report.read_text())
    p=json.loads(s476report.read_text())
    need(src["status"]=="S491_TWO_SOURCE_K0_ONLY_ONE_TO_ONE_EXTERNAL_PITCH_BIRTH_FROZEN_NO_TRUTH"
         and src["no_midi_or_xml_label_read_during_freeze"]
         and src["original_IDMT_source_is_final_S80"]
         and src["original_MAESTRO_source_is_V88_NOT_S80"]
         and g["status"]=="S488_PAIRED_IDMT_45_TRUE_GUITAR_ATTACKS_MULTISOURCE_BASIC_PITCH"
         and g["matched_rows_SHA256"]==sha(s488)
         and p["status"]=="S476_THIRD_MAESTRO_PIANO_NEW_ONSET_K_VS_SUSTAINING_K_READONLY"
         and p["SHA256"]["paired_label_comparison"]==sha(s476),
         "S492 source predictions must be SHA-sealed before external MIDI and XML notes")
    rows_g=[json.loads(x) for x in s488.read_text().splitlines()]
    rows_p=[json.loads(x) for x in s476.read_text().splitlines()]
    need(len(rows_g)==45 and len(rows_p)==322,
         "S492 original external genuine attack ground truth population drift")
    source_spec=[
      ("guitar_S80",guitar,rows_g,"true_active","original_S80_K",25,7,21),
      ("piano_V88",piano,rows_p,"true_new_onset_MIDI_pitches","original_V88_K",42,7,73)]
    bysource={}
    for key,path,real,truth_key,original_key,baseline,polybaseline,polyn in source_spec:
        need(src["source_output_SHA256"][key]["sha256"]==sha(path),
             "S492 frozen source K0 proposal file changed after prelabel experiment")
        with np.load(path,allow_pickle=False) as z:
            ids=z["row_id"].astype(str)
            ref=z["source_K"].astype(np.int8)
            groups=z["independent_BP_note_onset_group_matched"].astype(bool)
            proposals={policy:z["proposed_only_K0_"+policy].astype(np.int8)
                       for policy in ("birth","active30")}
        index={k:i for i,k in enumerate(ids)}
        need(len(ids)==src["by_corpus"][key]["source_native_event_count"]
             and len(index)==len(ids),
             "S492 exact native event ID uniqueness broken")
        labelindex=np.asarray([index[v["row_id"]] for v in real])
        truth=np.asarray([len(v[truth_key]) for v in real],int)
        need(np.all(truth>=1) and np.sum(truth>=2)==polyn
             and np.array_equal(ref[labelindex],
                 np.asarray([v[original_key] for v in real],int))
             and int(np.sum(ref[labelindex]==truth))==baseline
             and int(np.sum((ref[labelindex]==truth)&(truth>=2)))==polybaseline,
             "S492 historical original exact K denominator/true source V88/S80 contamination")
        known=set(labelindex.tolist())
        annotated=np.zeros(len(ids),bool)
        annotated[labelindex]=True
        bypolicy={}
        for name,pred in proposals.items():
            truthscore=score_sample(ref[labelindex],truth,pred[labelindex])
            # True positive onsets only: a source K0 was necessarily wrong.
            # Not every rescued native event has XML/MIDI ground truth!
            unknown=np.flatnonzero(~annotated&(ref==0)&(pred>0))
            bypolicy[name]=dict(
                annotated_true_attack_K_metrics=truthscore,
                original_K0_changed_at_known_positive_onsets=int(np.sum(
                    annotated&(ref==0)&(pred>0))),
                original_K0_changed_at_UNANNOTATED_native_events=int(len(unknown)),
                unknown_change_predictions_K_hist={
                    str(int(k)):int(np.sum(pred[unknown]==k)) for k in range(1,7)},
                source_K_nonzero_ever_modified=int(np.sum(pred[ref>0]!=ref[ref>0])),
                independent_onset_groups_matched_n=int(np.sum(groups)),
                native_candidates_total=len(ids),
                original_K0_total=int(np.sum(ref==0)),
                no_unknown_event_class_assumed=True)
        bysource[key]=bypolicy
    out.mkdir(parents=True)
    result=dict(status="S492_SHA_FROZEN_SOURCE_K0_ONLY_POSITIVE_VS_UNANNOTATED_AUDIT",
        independent_models="historical_S80_or_V88_plus_pretrained_BasicPitch",
        IDMT_guitar_original_source="final S80 actual",
        MAESTRO_piano_original_source="V88 not final S80",
        label_guitar="notes active at original XML attack plus30ms",
        label_piano="new MIDI notes at original note onset group",
        annotations_originally_exposed_prior_to_S491=True,
        no_new_blind_validation_claim=True,
        no_unknown_trueK0_or_false_event_assumption=True,
        no_deployed_suppression_and_no_official_S80_modifications=True,
        two_policy_results_both_reported_no_selection=True,
        read_only_counterfactual=True,no_promotion=True,
        by_source=bysource,
        SHA256=dict(S491_report=sha(source_report),
                    S488_IDMT_real_attacks=sha(s488),
                    S476_piano_real_attacks=sha(s476)))
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("S492 COUNTERFACTUAL SOURCE K0",json.dumps(bysource),flush=True)
def selftest():
    actual=np.array([1,2,1])
    old=np.array([0,0,1])
    fix=np.array([1,2,1])
    z=score_sample(old,actual,fix)
    need(z["fixed"]==2 and z["regressed"]==0
         and z["true_poly_fixed"]==1 and z["recovered_true_K0_exactK"]==2,
         "S492 paired original K0 rescue audit incorrect")
    print("S492 SELFTEST PASS count true poly K0 rescue and no annotations on unknown rows")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for k in ("guitar","piano","source-report","s488","s488report","s476","s476report","out"):
        p.add_argument("--"+k,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for k in ("guitar","piano","source_report","s488","s488report","s476","s476report","out"):
            need(getattr(a,k)is not None,"S492 missing --"+k)
        run(a.guitar,a.piano,a.source_report,a.s488,a.s488report,
            a.s476,a.s476report,a.out)
