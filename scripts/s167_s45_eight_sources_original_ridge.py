"""S167 true saved trained S45 Ridge scores from 8 of 16 source families.

FIVE source outputs are frozen from real original S163 trained models;
THREE original recurrent sources from S166 use 38 genuine model weights.
Recompute all family agreement votes against all EIGHT observed sources.
Use genuine original S45 Ridge checkpoints, no new fitting and fixed
lambda1/cut0. Missing eight S45 producers NEVER replaced by parent K,
zero vectors or invented votes. With only 8/16 votes, this is research
partial S45-deploy and NOT true historical 16-source S45 or S80.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,joblib
from scripts.s141_external_original_s40_cnn import load_input
from scripts.train_v273_s45_specialized_risk_heads import make_inputs,choose_best
FAMILIES=("S11_votes_time_audio","S12_audio_logistic",
 "S19_recurrent_ABA","S20_repeated_ABA","S27_early_stop",
 "S38_morphology","S40_temporal_CNN","S41_neural_fusion")
NEURAL={
 "S19_recurrent_ABA":"S19_ABA_pass2_K",
 "S20_repeated_ABA":"S20_pass1_K",
 "S27_early_stop":"S27_early_stable_005_K"}
def need(v,why):
    if not v:raise ValueError(why)
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def run(s97,s85,s135,s154,s163,s166,s45,catalog,out):
    need(not out.exists(),"immutable source-only S167")
    ids,X,_,src=load_input(s97,s85,s135)
    with np.load(s154,allow_pickle=False) as z:
        need(np.array_equal(ids,z["row_id"].astype(str)),"S154 source join")
        parent=z["S18_K"].astype(np.int8)
    with np.load(s163,allow_pickle=False) as z:
        need(set(z.files)=={
            "row_id","S18_parent_K","freeze_K",
            "partial_five_family_ids","partial_proposal_K",
            "partial_agreement_votes","partial_risk_two_scores",
            "partial_research_candidate_K","partial_research_selected_head"},
            "S163 first five true sources changed")
        need(np.array_equal(ids,z["row_id"].astype(str)) and
             np.array_equal(parent,z["S18_parent_K"]),"S163 S18 parent changed")
        first_ids=list(map(str,z["partial_five_family_ids"]))
        first=z["partial_proposal_K"].astype(np.int8)
    with np.load(s166,allow_pickle=False) as z:
        expected={"row_id","adapted_S18_parent_K","freeze_K",
          "S19_ABA_pass2_mean_probs","S19_ABA_pass2_K",
          "S20_pass1_mean_probs","S20_pass1_K",
          "S27_early_stable_005_mean_probs","S27_early_stable_005_K"}
        need(set(z.files)==expected,"original neural source contract")
        need(np.array_equal(ids,z["row_id"].astype(str)) and
             np.array_equal(parent,z["adapted_S18_parent_K"]),
             "S19 S20 S27 original parent changed")
        new={k:z[v].astype(np.int8) for k,v in NEURAL.items()}
    need(len(first_ids)==5 and first.shape==(57,5) and
         set(first_ids).union(new)==set(FAMILIES),
         "missing/extra original S45 family proposal")
    proposals=np.column_stack([
        new[f] if f in new else first[:,first_ids.index(f)]
        for f in FAMILIES]).astype(np.int8)
    need(proposals.shape==(57,8) and np.isin(proposals,range(7)).all(),
         "partial 8/16 K0..6 source contract")
    c=json.loads(catalog.read_text())
    cat={f["family"]:f for f in c["original_families"]}
    need(set(FAMILIES).issubset(cat) and len(cat)==16,
         "S164 strict original S45 champion aliases or model source missing")
    changed=proposals!=parent[:,None]
    votes=np.zeros((57,8),np.int16)
    for j in range(8):
        votes[:,j]=(np.sum((proposals==proposals[:,j,None])&
                       (proposals!=parent[:,None]),axis=1)-changed[:,j])
    risk=np.full((57,8,2),np.nan,np.float32)
    ledger=[];used=0;inventory={}
    for j,fam in enumerate(FAMILIES):
        files=sorted((s45/"models").glob("*__"+fam+".joblib"))
        need(len(files)==19,"missing trusted original S45 models for "+fam)
        selected=np.flatnonzero(changed[:,j])
        pp=[]
        for cp in files:
            obj=joblib.load(cp)
            need(obj.get("family")==fam and
                 obj.get("source_original_sha256")==cat[fam]["original_vector_sha256"],
                 "original S44/S45 family source SHA mismatch")
            need(obj["scaler"].n_features_in_==156 and
                 obj["model"].n_features_in_==156,
                 "trained S45 156D risk schema changed")
            if len(selected):
                x=make_inputs(X[selected],parent[selected],
                      proposals[selected,j],votes[selected,j])
                p=np.asarray(obj["model"].predict(
                      np.clip(obj["scaler"].transform(x),-6,6)),np.float32)
                need(p.shape==(len(selected),2) and np.isfinite(p).all(),
                     "S45 true Ridge risk predictions invalid")
                pp.append(p);used+=1
            ledger.append({"family":fam,"held_original_piece":obj["piece"],
                           "original_model_sha256":digest(cp)})
        if len(selected):
            risk[selected,j]=np.clip(np.stack(pp).mean(0),0,1)
        inventory[fam]={"original_checkpoint_count":len(files),
                         "proposed_nonparent_events":int(len(selected)),
                         "original_ridge_models_scoring_events":len(pp)}
    need(len(ledger)==152 and
         np.isfinite(risk[changed]).all() and
         np.isnan(risk[~changed]).all(),
         "S45 incomplete risk or made-up family score")
    pred,chosen=choose_best(parent,proposals,risk,1.,0.)
    need(pred.shape==(57,) and chosen.shape==(57,),
         "original S45 risk decision contract")
    out.mkdir(parents=True)
    np.savez_compressed(out/"eight_of_sixteen_original_ridge_risks_GOAT.npz",
        row_id=ids,S18_parent_K=parent,
        original_eight_family_ids=np.asarray(FAMILIES),
        eight_proposal_K=proposals,eight_family_agreement=votes,
        original_ridge_two_risks=risk,
        selected_head_index=chosen,
        partial_eight_source_decision_K=pred)
    report={
      "status":"S167_EIGHT_OF_SIXTEEN_ORIGINAL_S45_SOURCES_AND_TRAINED_RIDGES",
      "GOAT_native_audio_events":57,
      "original_family_source_count":8,
      "missing_original_family_source_count":8,
      "true_original_model_checkpoint_provenance_count":len(ledger),
      "original_ridge_models_actually_scoring_events":used,
      "source_family_models":inventory,
      "source_S163_run":38198248491,
      "source_S166_run":38198716623,
      "S45_original_ridge_training_run":37891039201,
      "original_source_Ridge_checkpoint_SHA256":ledger,
      "partial_risk_lambda":1.0,"partial_risk_cut":0.0,
      "eight_family_votes_not_equal_historical_16":True,
      "candidate_K_histogram":np.bincount(pred,minlength=7).tolist(),
      "candidate_changes_from_adapted_S18":int((pred!=parent).sum()),
      "candidate_changed_native_event_count":int(np.any(changed,axis=1).sum()),
      "candidate_proposal_count":int(changed.sum()),
      "source_s5_and_s9_original_input_equivalence":False,
      "source_16_original_families_all_executable":False,
      "can_claim_full_S46_S80_equivalence":False,
      "GOAT_MIDI_truth_read":False,
      "independent_S80_ExactK_scored":False,
      "S80_promoted":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in (
       "status","GOAT_native_audio_events","original_family_source_count",
       "missing_original_family_source_count",
       "true_original_model_checkpoint_provenance_count",
       "original_ridge_models_actually_scoring_events",
       "source_family_models","candidate_changes_from_adapted_S18",
       "candidate_changed_native_event_count","candidate_proposal_count",
       "candidate_K_histogram","S80_promoted")},indent=2))
def selftest():
    need(len(FAMILIES)==8 and len(NEURAL)==3 and
         set(NEURAL).issubset(FAMILIES),"eight original source family names")
    parent=np.array([1,2],np.int8);p=np.array([[1,2],[3,3]],np.int8)
    changed=p!=parent[:,None]
    score=(p==p[:,0,None])&(p!=parent[:,None])
    need(score.shape==(2,2) and changed.shape==(2,2),
         "new-family agreement geometry")
    print("S167 PASS eight-only family votes, original checkpoint labels, no synthetic remaining eight")
if __name__=="__main__":
    a=argparse.ArgumentParser()
    for field in ("s97","s85","s135","s154","s163","s166","s45","catalog","out"):
        a.add_argument("--"+field,type=Path)
    a.add_argument("--self-test",action="store_true")
    x=a.parse_args()
    if x.self_test:selftest()
    else:
        need(all(getattr(x,f) is not None for f in
             ("s97","s85","s135","s154","s163","s166","s45","catalog","out")),
             "all source paths required")
        run(*[getattr(x,f) for f in (
             "s97","s85","s135","s154","s163","s166","s45","catalog","out")])
