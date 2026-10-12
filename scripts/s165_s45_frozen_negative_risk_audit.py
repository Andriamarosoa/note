"""S165: explain S163 five-source S45 zero-action result using frozen scores.

No threshold sweep; exactly original S45 lambda1,cut0. Analyze original
Ridge score (correction risk minus regression risk), eligibility and
independent source votes among only FIVE of sixteen heads, never MIDI.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
# Independent reproduction of the original seven-line S45 choose_best rule.
# Avoid importing its training-only PyTorch dependencies in this read-only audit.
def choose_best(parent,proposal,risk,lam,cut):
    score=risk[:,:,0]-float(lam)*risk[:,:,1]
    score=np.where(np.isfinite(score),score,-np.inf)
    j=np.argmax(score,axis=1)
    top=score[np.arange(len(parent)),j]
    good=np.isfinite(top)&(top>cut)
    return (np.where(good,proposal[np.arange(len(parent)),j],parent).astype(np.int8),
            np.where(good,j,-1).astype(np.int8))
def need(c,msg):
    if not c:raise ValueError(msg)
def audit(source,out):
    need(not out.exists(),"fail closed: output already exists")
    with np.load(source,allow_pickle=False) as z:
        expected={"row_id","S18_parent_K","freeze_K",
            "partial_five_family_ids","partial_proposal_K",
            "partial_agreement_votes","partial_risk_two_scores",
            "partial_research_candidate_K","partial_research_selected_head"}
        need(set(z.files)==expected,"S163 evidence source changed")
        ids=z["row_id"].astype(str)
        old=z["S18_parent_K"].astype(np.int8)
        families=list(map(str,z["partial_five_family_ids"]))
        prop=z["partial_proposal_K"].astype(np.int8)
        vote=z["partial_agreement_votes"].astype(np.int16)
        risk=z["partial_risk_two_scores"].astype(np.float64)
        output=z["partial_research_candidate_K"].astype(np.int8)
        selected=z["partial_research_selected_head"].astype(np.int8)
    need(len(ids)==57 and len(families)==5 and
         prop.shape==(57,5) and risk.shape==(57,5,2),"partial source shape")
    changed=prop!=old[:,None]
    need(np.isfinite(risk[changed]).all(),"every changed proposal must be trained risk-scored")
    need(np.isnan(risk[~changed]).all(),"unchanged row has no S45 risk and cannot be selected")
    replay,route=choose_best(old,prop,risk,1.,0.)
    need(np.array_equal(replay,output) and np.array_equal(route,selected),
         "S163 original S45 lambda1/cut0 decision policy changed")
    score=risk[:,:,0]-risk[:,:,1]
    rows=[]
    for i,j in zip(*np.nonzero(changed)):
        rows.append({"row_id":ids[i],"family":families[j],
             "S18_parent_K":int(old[i]),"proposal_K":int(prop[i,j]),
             "other_available_family_votes":int(vote[i,j]),
             "trained_expected_fix":float(risk[i,j,0]),
             "trained_expected_regression":float(risk[i,j,1]),
             "original_lambda1_score":float(score[i,j]),
             "original_threshold0_passed":bool(score[i,j]>0),
             "selected_by_partial_policy":bool(selected[i]==j)})
    n=len(rows)
    unique=len(set(row["row_id"] for row in rows))
    reasons={
       "no_active_source_proposal_families":int(sum(not np.any(changed[:,j]) for j in range(5))),
       "candidate_proposals":n,
       "unique_native_events_with_proposal":unique,
       "positive_risk_score_proposals":sum(r["original_threshold0_passed"] for r in rows),
       "accepted_actions":int(np.sum(output!=old)),
       "mean_original_risk_score":float(np.mean([r["original_lambda1_score"] for r in rows])) if rows else None}
    report={"status":"S165_LABEL_FREE_CAUSAL_DIAGNOSTIC_S45_PARTIAL_ABSTENTION",
       "new_audio_events":57,"evaluated_original_source_families":5,
       "trained_risk_sources_complete_16":False,
       "original_95_checkpoint_S163_provenance_unchanged":True,
       "original_S45_policy_lambda":1.0,"original_S45_policy_cut":0.0,
       "source_artifact_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
       "reasons":reasons,"proposal_details":rows,
       "original_16_family_source_agreement_known":False,
       "thresholds_retuned":False,"GOAT_truth_or_MIDI_accessed":False,
       "full_original_S80_accuracy_measured":False,"official_promotion":False}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
def selftest():
    p=np.asarray([[1,3],[2,1]],np.int8)
    old=np.asarray([1,2],np.int8)
    risk=np.full((2,2,2),np.nan)
    risk[0,1]=[.25,.4]
    proposed,idx=choose_best(old,p,risk,1.,0.)
    need(proposed.tolist()==[1,2] and idx.tolist()==[-1,-1],
         "negative net risk should preserve parent")
    print("S165 PASS: original S45 net-risk veto no threshold tuning")
if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--input",type=Path)
    a.add_argument("--out",type=Path)
    a.add_argument("--self-test",action="store_true")
    x=a.parse_args()
    if x.self_test:selftest()
    else:
        need(x.input and x.out,"original S163 evidence required")
        audit(x.input,x.out)
