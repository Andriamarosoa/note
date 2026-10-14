"""S456 -- independent paired known-IDMT K audit of S455 frozen candidate.

Read S453's independently XML-derived, SHA-frozen row ledger ONLY AFTER
S455 has committed all 506 original-vs-challenger source decisions without
seeing XML. Onset unmatched rows are NOT true K0. No fitting/tuning.
This is a retrospective previously exposed external IDMT set, NOT blind.
"""
from __future__ import annotations
import argparse,collections,json
from pathlib import Path
import numpy as np
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha
from scripts.s402_k_input_first_supervised_midi_pitch_network import need

def compare(truth,old,new,mask):
    ix=np.flatnonzero(mask)
    if not len(ix):
        return dict(n=0,old_correct=0,new_correct=0,
            corrections=0,regressions=0,net=0,old_K0=0,new_K0=0)
    a=old[ix]==truth[ix]
    b=new[ix]==truth[ix]
    wins=int(np.sum(~a&b));losses=int(np.sum(a&~b))
    return dict(n=len(ix),old_correct=int(a.sum()),new_correct=int(b.sum()),
        corrections=wins,regressions=losses,net=wins-losses,
        old_K0=int(np.sum(old[ix]==0)),new_K0=int(np.sum(new[ix]==0)),
        source_K_miscount_under=int(np.sum(old[ix]<truth[ix])),
        new_K_miscount_under=int(np.sum(new[ix]<truth[ix])),
        source_K_miscount_over=int(np.sum(old[ix]>truth[ix])),
        new_K_miscount_over=int(np.sum(new[ix]>truth[ix])))

def run(frozen,frozen_report,source,source_report,out):
    need(not out.exists(),"S456 independent audit immutable")
    a=json.loads(frozen_report.read_text())
    b=json.loads(source_report.read_text())
    need(a["status"]=="S455_REAL_EXTERNAL_IDMT_MORPHOLOGY_K_CANDIDATE_FROZEN_NO_XML"
         and a["annotations_XML_MIDI_never_read"]
         and a["original_S80_K_predictions_preserved_separately"]
         and a["checksums"]["frozen_output"]==sha(frozen)
         and b["status"]=="S453_ORIGINAL_S5_S8_S18_S80_EXTERNAL_IDMT_WRONGK_STAGE_AUDIT"
         and b["matched_note_annotation_events"]==47
         and b["valid_pitch_supported_events"]==45
         and b["annotated_rows_SHA256"]==sha(source),
         "S456 original frozen S455 model inference or S453 independent truth SHA drift")
    with np.load(frozen,allow_pickle=False) as z:
        ids=z["row_id"].astype(str)
        old=z["original_S80_K"].astype(np.int8)
        new=z["S454_candidate_K"].astype(np.int8)
        fold=z["four_fold_candidate_K"].astype(np.int8)
        votes=z["consensus_votes"].astype(np.int8)
    need(len(ids)==506 and len(set(ids))==506
         and old.shape==new.shape==(506,)
         and fold.shape==(4,506)
         and np.isin(old,range(7)).all()
         and np.isin(new,range(7)).all()
         and np.array_equal(new[votes<3],old[votes<3]),
         "S456 source decisions changed when trained fold quorum absent")
    idx={v:i for i,v in enumerate(ids)}
    labeled=[json.loads(l) for l in source.read_text().splitlines()]
    need(len(labeled)==47 and len({x["source_row_id"] for x in labeled})==47
         and all(x["source_row_id"] in idx for x in labeled),
         "S456 S453 note-annotated source event identity drift")
    truth=np.full(506,-1,np.int8)
    supported=np.zeros(506,bool)
    for row in labeled:
        i=idx[row["source_row_id"]]
        need(int(old[i])==row["original_S80_K"],
             "S456 claimed source original S80 K differs from stage S453")
        truth[i]=int(row["actual_K"])
        supported[i]=bool(row["within_supported_notes"])
    matched=truth>=1
    valid=matched&supported
    need(int(matched.sum())==47 and int(valid.sum())==45
         and int((valid&(truth>=2)).sum())==21,
         "S456 original S453 47/45/21 annotation population changed")
    scores=dict(all_supported=compare(truth,old,new,valid),
        true_poly=compare(truth,old,new,valid&(truth>=2)),
        by_true_K={str(k):compare(truth,old,new,valid&(truth==k)) for k in range(1,7)},
        by_original_predicted_K={
            str(k):compare(truth,old,new,valid&(old==k)) for k in range(7)})
    need(scores["all_supported"]["old_correct"]==25
         and scores["true_poly"]["old_correct"]==7,
         "S456 must reproduce original S452 ceiling and not count unmatched K0")
    out.mkdir(parents=True)
    report=dict(status="S456_INDEPENDENT_RETROSPECTIVE_EXTERNAL_IDMT_K_CANDIDATE_PAIRED_AUDIT",
       population=506,annotated_original_matches=47,valid_evaluable=45,
       original_true_poly_evaluable=21,
       original_S80_predictions_preserved=True,
       original_S80_metrics_25_of_45_and_7_of_21_reproduced=True,
       unknown_unmatched_not_relabelled_as_K0=True,
       new_candidate_is_only_research_model=True,
       candidate_GuitarSet_training_only=True,
       previously_exposed_IDMT_XML_truth=True,
       no_new_independent_blind_dataset=False,
       threshold_not_tuned_to_external_IDMT=True,
       no_promotion=True,paired=scores,
       source_sha256=dict(candidate_frozen=sha(frozen),
                          frozen_original_XML_matching=sha(source)))
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S456 INDEPENDENT",json.dumps(dict(
         all_supported=scores["all_supported"],
         true_poly=scores["true_poly"],
         by_true_K=scores["by_true_K"])),flush=True)

def selftest():
    y=np.array([3,2,2,-1],np.int8)
    b=np.array([4,2,0,0],np.int8)
    c=np.array([3,3,2,0],np.int8)
    d=compare(y,b,c,y>=1)
    need(d["n"]==3 and d["corrections"]==2 and
         d["regressions"]==1 and d["net"]==1,
         "S456 independent paired score/K unknown handling wrong")
    print("S456 SELFTEST PASS K0 unknown exclusions and paired score")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for arg in ("frozen","frozen-report","source","source-report","out"):
        p.add_argument("--"+arg,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for arg in ("frozen","frozen_report","source","source_report","out"):
            need(getattr(a,arg)is not None,"S456 missing --"+arg)
        run(a.frozen,a.frozen_report,a.source,a.source_report,a.out)
