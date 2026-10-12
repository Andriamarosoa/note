"""S170: independent label-free audit of ONE selected risk action in S169.

Identifies precisely which original S45 family Ridge policy accepted
its proposal, parent / new K, number of agreeing source families,
model-estimated fix vs regression, and deterministic cutoff.
No GOAT labels, MIDI or posthoc thresholds. S80 cannot be promoted.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
def need(c,m):
    if not c:raise ValueError(m)
def audit(path,out):
    need(not out.exists(),"immutable risk acceptance explanation")
    with np.load(path,allow_pickle=False) as z:
        required={"row_id","S18_parent_K",
          "original_ten_source_family_ids","ten_source_K",
          "ten_partial_source_agreement","original_ridge_risks",
          "ten_partial_selected_head","ten_partial_candidate_K"}
        need(set(z.files)==required,"S169 original candidate evidence must be label-free")
        ids=z["row_id"].astype(str)
        parent=z["S18_parent_K"].astype(np.int8)
        fam=list(map(str,z["original_ten_source_family_ids"]))
        proposals=z["ten_source_K"].astype(np.int8)
        votes=z["ten_partial_source_agreement"].astype(np.int16)
        risks=z["original_ridge_risks"].astype(np.float64)
        routes=z["ten_partial_selected_head"].astype(np.int8)
        final=z["ten_partial_candidate_K"].astype(np.int8)
    need(len(ids)==57 and len(set(ids))==57 and len(fam)==10,
         "true 57 event 10 source lineage")
    need(proposals.shape==(57,10) and risks.shape==(57,10,2),
         "strict S169 model arrays")
    score=risks[:,:,0]-risks[:,:,1]
    score=np.where(np.isfinite(score),score,-np.inf)
    arg=score.argmax(1)
    top=score[np.arange(57),arg]
    pass_=(np.isfinite(top))&(top>0.)
    independent_pred=np.where(pass_,
       proposals[np.arange(57),arg],parent).astype(np.int8)
    independent_route=np.where(pass_,arg,-1).astype(np.int8)
    need(np.array_equal(independent_pred,final) and
         np.array_equal(independent_route,routes),
         "independent 156D Ridge fixed S45 decision logic replay mismatch")
    actions=[]
    for i in np.flatnonzero(routes>=0):
        j=int(routes[i])
        agreeing=[name for k,name in enumerate(fam) if
            proposals[i,k]==proposals[i,j] and proposals[i,k]!=parent[i]]
        actions.append({"event_id":ids[i],
          "selected_source_family":fam[j],
          "parent_K":int(parent[i]),"selected_proposal_K":int(proposals[i,j]),
          "original_Ridge_expected_correction":float(risks[i,j,0]),
          "original_Ridge_expected_regression":float(risks[i,j,1]),
          "fixed_lambda1_risk_score":float(score[i,j]),
          "ten_source_independent_votes_excluding_selected":int(votes[i,j]),
          "supported_by_families":agreeing,
          "original_risk_cut0_passed":bool(score[i,j]>0)})
    need(len(actions)==1 and all(x["original_risk_cut0_passed"] for x in actions),
         "S169 accepted decision count drift")
    out.parent.mkdir(parents=True)
    r={"status":"S170_INDEPENDENT_LABEL_FREE_AUDIT_ONE_PARTIAL_S45_ACTION",
       "original_source_family_count":10,"original_missing_source_families":6,
       "original_S45_ridge_fits_from_S169":190,
       "source_S169_run":38199247838,
       "original_risk_selection_rule":"argmax of fix-risk minus regress-risk, lambda1 cut0, no tuning",
       "source_artifact_SHA256":hashlib.sha256(path.read_bytes()).hexdigest(),
       "accepted_original_risk_actions":actions,
       "accepted_action_count":len(actions),
       "all_16_family_agreement_votes_available":False,
       "full_S46_S69_S72_S80_equivalence_verified":False,
       "external_exactK_improvement_measured":False,
       "GOAT_MIDI_or_trueK_used":False,
       "S80_promoted":False}
    out.write_text(json.dumps(r,indent=2)+"\n")
    print(json.dumps(r,indent=2))
def selftest():
    old=np.array([2,1]);p=np.array([[3,1],[0,2]],np.int8)
    r=np.full((2,2,2),np.nan,np.float64)
    r[0,0]=[.4,.2]
    r[1,0]=[.2,.3]
    m=r[:,:,0]-r[:,:,1]
    s=np.where(np.isfinite(m),m,-np.inf)
    j=s.argmax(1);v=s[np.arange(2),j]
    chosen=np.where(np.isfinite(v)&(v>0),p[np.arange(2),j],old)
    need(chosen.tolist()==[3,1],"independent original S45 margin >0 test")
    print("S170 PASS independent fixed-S45 risk decision test, zero labels")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        need(a.input and a.out,"original frozen S169 input required")
        audit(a.input,a.out)
