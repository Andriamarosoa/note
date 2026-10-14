"""S495 independent SOURCE K0-first priority pitch evidence risk audit.

Score exactly the same genuine 45 guitar S80 and 322 piano V88 onsets,
AFTER SHA-verified original S494 frozen source evidence (no MIDI opened
at inference). Compare both previously declared policies alongside S491.
Count rescued true onset attacks and *unlabelled* native activations and
nearby nonzero original K competition separately. No oracle routing
or claim that unknown candidates have true K0.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
def need(c,m):
    if not c:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pair(truth,original,candidate):
    y=np.asarray(truth,int);base=np.asarray(original,int);p=np.asarray(candidate,int)
    correct=base==y;after=p==y
    poly=y>=2
    return dict(n=len(y),poly=int(poly.sum()),
         original_correct=int(correct.sum()),new_correct=int(after.sum()),
         fixed=int(np.sum(~correct&after)),regressed=int(np.sum(correct&~after)),
         poly_original_correct=int(np.sum(poly&correct)),
         poly_new_correct=int(np.sum(poly&after)),
         poly_fixed=int(np.sum(poly&~correct&after)),
         poly_regressed=int(np.sum(poly&correct&~after)),
         original_wrong_K0_true_note=int(np.sum(base==0)),
         original_K0_recovered_to_correct=int(np.sum((base==0)&after)),
         original_K0_positive_but_proposed_wrong_K=int(np.sum((base==0)&(p>0)&~after)))
def run(s494,s494_report,idmt,idmt_report,piano,piano_report,out):
    need(not out.exists(),"S495 immutable K0 priority paired source audit exists")
    v=json.loads(s494_report.read_text())
    a=json.loads(idmt_report.read_text())
    b=json.loads(piano_report.read_text())
    need(v["status"]=="S494_SOURCE_K0_FIRST_PRETRAINED_ONSET_EVIDENCE_PRELABEL_SHADOW"
         and v["read_no_XML_or_MIDI_truth_during_inference"]
         and v["no_promotion"]
         and a["status"]=="S488_PAIRED_IDMT_45_TRUE_GUITAR_ATTACKS_MULTISOURCE_BASIC_PITCH"
         and a["matched_rows_SHA256"]==sha(idmt)
         and b["status"]=="S476_THIRD_MAESTRO_PIANO_NEW_ONSET_K_VS_SUSTAINING_K_READONLY"
         and b["SHA256"]["paired_label_comparison"]==sha(piano),
         "S495 cannot score non-frozen K0 router or altered original MIDI XML event labels")
    g=[json.loads(x) for x in idmt.read_text().splitlines()]
    p=[json.loads(x) for x in piano.read_text().splitlines()]
    need(len(g)==45 and len(p)==322,"S495 original annotation cohort changed")
    results={}
    for key,records,truth_field,old_field,base_expected,poly_expected in (
        ("guitar_S80",g,"true_active","original_S80_K",25,7),
        ("piano_V88",p,"true_new_onset_MIDI_pitches","original_V88_K",42,7)):
        expected=v["by_source"][key]["SHA256"]["pretruth_policy_file"]
        file=s494/f"S494_{key}_K0_FIRST_PRIORITY_SHADOW_PRETRUTH.npz"
        need(expected==sha(file),"S495 original pre-MIDI K0 candidate model changed")
        with np.load(file,allow_pickle=False) as z:
            ids=z["row_id"].astype(str)
            src=z["source_K"].astype(int)
            props={k:z[k].astype(int) for k in (
                "proposed_K0_priority_BP_birth_K",
                "proposed_K0_priority_BP_activeAtBPonsetPlus30_K")}
        lookup={name:i for i,name in enumerate(ids)}
        need(len(lookup)==len(src) and
             all(x["row_id"] in lookup for x in records),
             "S495 original onsets not mapped to source-native identity")
        chosen=np.asarray([lookup[x["row_id"]] for x in records],int)
        truth=np.asarray([len(x[truth_field]) for x in records],int)
        trueold=np.asarray([int(x[old_field]) for x in records],int)
        need(np.array_equal(src[chosen],trueold) and
             int(np.sum(src[chosen]==truth))==base_expected and
             int(np.sum((src[chosen]==truth)&(truth>=2)))==poly_expected,
             "S495 original V88 or final S80 baseline drifted")
        used=np.zeros(len(src),bool);used[chosen]=True
        out={}
        for name,allpred in props.items():
            known=pair(truth,src[chosen],allpred[chosen])
            unknown_idx=np.flatnonzero((~used)&(src==0)&(allpred>0))
            out[name]=dict(
               score_true_annotated_positive_attacks=known,
               new_positive_source_K0_activations=int(np.sum(
                   used&(src==0)&(allpred>0))),
               extra_UNKNOWN_source_K0_activations=int(len(unknown_idx)),
               original_source_K>=1_modified=int(np.sum(
                   allpred[src>0]!=src[src>0])),
               unknown_extra_K_hist={str(j):int(np.sum(allpred[unknown_idx]==j))
                   for j in (1,2,3,4,5,6)},
               unknown_trueK_and_pitch_not_assumed=True)
        results[key]=dict(
            policies=out,
            original_K0_native_count=int((src==0).sum()),
            competing_already_nonzero_K_near_groups=v["by_source"][key][
                "competing_source_K_ge1_near_BP_attacks"],
            competing_original_source_K_strictly_closer_than_K0=v["by_source"][key][
                "competing_source_K_ge1_strictly_closer_to_BP_onset_n"])
        print("S495 ORIGINAL SOURCE POSITIVE VS UNKNOWN",
              json.dumps(dict(name=key,**results[key])),flush=True)
    out.mkdir(parents=True)
    result=dict(status="S495_TWO_SOURCES_SOURCE_K0_PRIORITY_POSTFREEZE_PAIRED_RISK_AUDIT",
      original_source_guitar_full_S80=True,
      original_source_piano_only_V88=True,
      no_training_or_threshold_tuning_on_true_labels=True,
      all_old_nonzero_K_predictions_are_protected=True,
      no_unknown_459_guitar_events_declared_true_K0=True,
      original_test_source_prior_exposure_precludes_blind_promotion=True,
      cannot_guarantee_no_false_notes_on_unannotated_candidates=True,
      nondeployed_study_only=True,no_promotion=True,
      by_source=results,
      SHA256=dict(S494_pretruth_report=sha(s494_report),
         true_IDMT_guitar=sha(idmt),true_MAESTRO_piano=sha(piano)))
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
def selftest():
    d=pair([1,2,0],[0,0,0],[1,2,1])
    need(d["fixed"]==2 and d["regressed"]==1,
         "S495 true K0 incorrect false positive must be counted as regression")
    print("S495 SELFTEST PASS false real K0 correction counted as regression")
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--self-test",action="store_true")
    for k in ("s494","s494-report","idmt","idmt-report","piano","piano-report","out"):
        ap.add_argument("--"+k,type=Path)
    a=ap.parse_args()
    if a.self_test:selftest()
    else:
        for k in ("s494","s494_report","idmt","idmt_report","piano","piano_report","out"):
            need(getattr(a,k)is not None,"S495 missing "+k)
        run(a.s494,a.s494_report,a.idmt,a.idmt_report,a.piano,a.piano_report,a.out)
