"""S483 independent post-S482-freeze paired Original V88 IDMT XML K0 audit.

Never train/refit on IDMT. True annotation opened only AFTER the source
S482 model ensemble predictions are SHA frozen, with exact original S453
source row identities and real 45 MIDI40..88, 21 poly events. Events
unmatched to XML are not true K0 and counted as unknown candidate changes.
"""
from __future__ import annotations
import argparse,json,hashlib,collections
from pathlib import Path
import numpy as np
def need(c,m):
    if not c:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def paired(true,old,new):
    k=np.asarray(true,int);a=np.asarray(old,int);b=np.asarray(new,int)
    right=a==k;changed=b==k
    return dict(n=int(len(k)),baseline_correct=int(right.sum()),
                proposed_correct=int(changed.sum()),
                fixes=int(np.sum(~right&changed)),
                regressions=int(np.sum(right&~changed)),
                net=int(changed.sum()-right.sum()),
                under_before=int(np.sum(a<k)),under_after=int(np.sum(b<k)),
                over_before=int(np.sum(a>k)),over_after=int(np.sum(b>k)))
def run(freeze,report,s453,s453_report,unknown,unknown_report,out):
    need(not out.exists(),"S483 paired external audit already exists")
    f=json.loads(report.read_text())
    t=json.loads(s453_report.read_text())
    u=json.loads(unknown_report.read_text())
    need(f["status"]=="S482_IDMT_ORIGINAL_V88_K0_RESCUE_AFTER_GUITARSET_TRAIN_NOXML"
         and f["frozen_predictions_SHA256"]==sha(freeze)
         and f["IDMT_XML_or_labels_NEVER_OPENED"]
         and t["status"]=="S453_ORIGINAL_S5_S8_S18_S80_EXTERNAL_IDMT_WRONGK_STAGE_AUDIT"
         and t["annotated_rows_SHA256"]==sha(s453)
         and u["status"]=="S462_IDMT_EXTERNAL_ORIGINAL_ONSET_XML_MATCHING_FORENSIC"
         and u["all_IDMT_sources_pinned_SHA256"]["nearest_source_ledger"]==sha(unknown),
         "S483 original label-free ensemble freeze must precede independent XML score")
    with np.load(freeze,allow_pickle=False) as z:
        ids=z["row_id"].astype(str)
        ref=z["original_V88_K"].astype(int)
        p3=z["proposed_V88_K_consensus3"].astype(int)
        p4=z["proposed_V88_K_consensus4"].astype(int)
    need(len(ids)==506 and len(set(ids))==506 and
         np.array_equal(ref[ref>0],p3[ref>0]) and
         np.array_equal(ref[ref>0],p4[ref>0]),
         "S483 frozen original source K>0 must be protected")
    ix={name:i for i,name in enumerate(ids)}
    known=[json.loads(v) for v in s453.read_text().splitlines()]
    noxml=[json.loads(v) for v in unknown.read_text().splitlines()]
    need(len(known)==47 and len(noxml)==506 and
         set(ix)=={v["row_id"] for v in noxml},
         "S483 real five WAV IDMT source population differs")
    evaluated=[v for v in known if v["within_supported_notes"]]
    matched=np.asarray([ix[v["source_row_id"]] for v in evaluated],int)
    true=np.asarray([int(v["actual_K"]) for v in evaluated],int)
    need(len(matched)==45 and int(np.sum(true>=2))==21
         and np.array_equal(ref[matched],np.asarray([
             int(v["each_original_stage_K"]["V88_freeze"])
             for v in evaluated],int))
         and int(np.sum(ref[matched]==true))==22,
         "S483 independent original V88 paired external 22/45 baseline drift")
    matched_any={ix[v["source_row_id"]] for v in known}
    result={}
    for key,new in [("original_no_change",ref),("three_fold_vote",p3),
                    ("four_fold_vote",p4)]:
        changed=np.flatnonzero(new!=ref)
        result[key]=dict(
            all_inrange_45=paired(true,ref[matched],new[matched]),
            true_poly_21=paired(true[true>=2],ref[matched][true>=2],
                                new[matched][true>=2]),
            each_real_K={str(k):paired(true[true==k],ref[matched][true==k],
                           new[matched][true==k]) for k in sorted(set(true))},
            source_V88_K0_to_K1=int(np.sum((ref==0)&(new==1))),
            source_V88_K0_to_K2=int(np.sum((ref==0)&(new==2))),
            proposed_change_on_47_matched_true_attack_events=int(np.sum(
                [i in matched_any for i in changed])),
            proposed_change_on_459_unmatched_UNKNOWN_LABEL_events=int(np.sum(
                [i not in matched_any for i in changed])))
    out.mkdir(parents=True)
    output=dict(status="S483_REAL_IDMT_PAIRED_45_TRUE_NOTES_AFTER_GUITARSET_K0_RESCUE_FREEZE",
        original_native_events=506,
        original_valid_true_attacks=45,
        original_true_poly_events=21,
        unmatched_459_events_not_groundtruth_K0=True,
        original_V88_source_K_not_altered=True,
        actual_S80_complete_NOT_run_or_changed=True,
        precommit_predictions_SHA256=sha(freeze),
        IDMT_truth_checked_only_after_source_frozen=True,
        no_external_retrain_or_threshold_selection=True,
        previously_exposed_IDMT_XML_not_prospective_blind=True,
        experiments=result,
        no_promotion=True,
        SHA256=dict(original_S453=sha(s453),original_S462=sha(unknown)))
    (out/"report.json").write_text(json.dumps(output,indent=2)+"\n")
    print("S483 PAIR IDMT",json.dumps(dict(
        original_45=22,original_poly_21=4,
        three=result["three_fold_vote"],four=result["four_fold_vote"])),flush=True)
def selftest():
    old=np.array([0,0,2]);new=np.array([1,2,2]);truth=np.array([1,0,2])
    z=paired(truth,old,new)
    need(z["fixes"]==1 and z["regressions"]==1 and z["net"]==0,
         "S483 must count true K0 false rescue as regression")
    print("S483 SELFTEST PASS paired fixes minus regression on real annotated onset")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for name in ("freeze","report","s453","s453-report","unknown","unknown-report","out"):
        p.add_argument("--"+name,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for n in ("freeze","report","s453","s453_report","unknown","unknown_report","out"):
            need(getattr(a,n)is not None,"S483 missing "+n)
        run(a.freeze,a.report,a.s453,a.s453_report,a.unknown,a.unknown_report,a.out)
