"""S163: REAL original S45 Ridge checkpoints on FIVE supported original family sources.

Five genuinely trained proposal heads executed on NEW WAV-derived GOAT
events: S11, S12, S38, S40, S41, joined against the real S18-weight
(adapted upstream S9) parent. 95 source S45 Ridge fits recovered from
ORIGINAL 37891039201; no S45 model refit, no labels opened.

WARNING: this is a 5/16-family diagnostic with *partial votes*.
The original 16-family agreement feature used by the S45 checkpoints
is UNKNOWN for the other 11 families. The predictions, family risk,
and partial decision MUST NOT masquerade as full S45 / S80, historic
equivalence, true external accuracy, or promotion.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np
from scripts.s141_external_original_s40_cnn import load_input
from scripts.train_v273_s45_specialized_risk_heads import make_inputs,choose_best
from scripts.train_v273_s38_poly_harmonic_expert import score_predictions
FAMILIES=(
 "S11_votes_time_audio","S12_audio_logistic",
 "S38_morphology","S40_temporal_CNN","S41_neural_fusion")
S11_NAME="series11__votes_time_audio__hgb__all__cost0"
S12_NAME="series12__logistic__all__cost0"
def need(c,m):
    if not c:raise ValueError(m)
def strict(p,fields):
    with np.load(p,allow_pickle=False) as z:
        need(set(z.files)==set(fields) and
             not set(z.files).intersection({"true_K","truth","label","labels","midi"}),
             "source input contained invalid/ground-truth keys: "+str(p))
        return {k:z[k].copy() for k in z.files}
def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def make_proposals(s97,s85,s135,s11,s12,s38,s40,s41,s154,s157):
    ids,X,_,p=load_input(s97,s85,s135)
    b=p["official_freeze"].astype(np.int8)
    n=len(ids)
    z11=strict(s11,("row_id","variant_ids","predicted_K",
        "probability_variant_ids","original_model_mean_probabilities","freeze_K"))
    need(np.array_equal(ids,z11["row_id"].astype(str)) and
         np.array_equal(b,z11["freeze_K"]),"S11 alignment")
    v11=list(map(str,z11["variant_ids"]))
    need(S11_NAME in v11,"historical S11 family champion missing")
    s11out=z11["predicted_K"][:,v11.index(S11_NAME)].astype(np.int8)

    z12=strict(s12,("row_id","means","predicted_K","per_heldpiece_probability",
       "parent_freeze_K","parent_S135_K","parent_original_PR16_22_K"))
    need(np.array_equal(ids,z12["row_id"].astype(str)) and
         np.array_equal(b,z12["parent_freeze_K"]),"S12 alignment")
    q12=z12["means"].astype(np.float32)
    need(q12.shape==(n,7) and np.isfinite(q12).all(),"S12 posterior source")

    z38=strict(s38,("row_id","fold_ids","fold_probabilities","mean_probabilities",
       "original_S38_proposed_K","adapted_K1plus_K","source_parent_freeze_K",
       "parent_s135_K","parent_policy22_K"))
    need(np.array_equal(ids,z38["row_id"].astype(str)),"S38 alignment")
    p38=z38["mean_probabilities"].astype(np.float32)
    need(p38.shape==(n,7),"S38 native posteriors")

    z40=strict(s40,("row_id","fold_ids","fold_probabilities","mean_probabilities",
       "predicted_K","parent_freeze_K","parent_s135_K","parent_policy22_K"))
    need(np.array_equal(ids,z40["row_id"].astype(str)),"S40 alignment")
    p40=z40["mean_probabilities"].astype(np.float32)
    need(p40.shape==(n,7),"S40 native posteriors")

    z41=strict(s41,("row_id","freeze_K","adapted_S18_parent_K",
         "proposals_6_K","gate_6_probabilities","final_policy_K",
         "frozen_S144_wrong_parent_K"))
    need(np.array_equal(ids,z41["row_id"].astype(str)),"S41 alignment")
    s41out=z41["final_policy_K"].astype(np.int8)

    z154=strict(s154,("row_id","freeze_K","adapted_S9_K",
                      "adapted_S15_K","S16_K","S17_K","S18_K"))
    need(np.array_equal(ids,z154["row_id"].astype(str)) and
         np.array_equal(b,z154["freeze_K"]),"S18 native IDs")
    parent=z154["S18_K"].astype(np.int8)
    need(np.array_equal(parent,z41["adapted_S18_parent_K"]),
         "S41 authentic S18 parent source must be unchanged")

    z157=strict(s157,("row_id","freeze_K","adapted_global_S135_K",
           "adapted_poly_PR16_K","S8_original_weight_K",
           "S9_original_formula_K","S8_learned_advantage"))
    need(np.array_equal(ids,z157["row_id"].astype(str)) and
         np.array_equal(z157["S9_original_formula_K"],z154["adapted_S9_K"]),
         "original-weight adapted S9 source drift")
    s9=z157["S9_original_formula_K"].astype(np.int8)
    # Historical S12 uses source S9 as proposal and freeze as rollback. Only
    # logistic posterior is used by this exact selected policy.
    # The selected ORIGINAL S12 policy is logistic-only: rollback S9 to
    # the frozen baseline iff the original logistic margin is > 0.
    # Do not fabricate an HGB posterior for unused sibling policies.
    ii=np.arange(n)
    margin=q12[ii,b]-q12[ii,s9]
    s12out=s9.copy()
    veto=(s9!=b)&(margin>0.)
    s12out[veto]=b[veto]
    # Source S38 and S40 original champion uses S18 parent for eligible K1+.
    s38out=score_predictions(parent,p38.argmax(1).astype(np.int8),p38,
                               "source_K1to6","argmax")
    s40out=np.where(parent>=1,p40.argmax(1),parent).astype(np.int8)
    proposals=np.column_stack((s11out,s12out,s38out,s40out,s41out))
    need(proposals.shape==(57,5) and np.isin(proposals,range(7)).all(),
         "partial 5-family original-weight source proposal integrity")
    return ids,X,parent,b,proposals

def diagnose(s97,s85,s135,s11,s12,s38,s40,s41,s154,s157,s45,out):
    need(not out.exists(),"immutable original S45 five-family diagnostic")
    ids,X,parent,b,proposals=make_proposals(
        s97,s85,s135,s11,s12,s38,s40,s41,s154,s157)
    n=len(ids)
    moved=proposals!=parent[:,None]
    # This deliberately means AGREEMENT AMONG FIVE KNOWN HEADS ONLY.
    # It is not the original S45 all-16 family agreement!
    agreement=np.zeros((n,5),np.int16)
    for j in range(5):
        agreement[:,j]=np.sum(
          (proposals==proposals[:,j,None])&(proposals!=parent[:,None]),
          axis=1).astype(np.int16)-moved[:,j].astype(np.int16)
    risks=np.full((n,5,2),np.nan,np.float32)
    history=[]
    scored_models=0
    for j,family in enumerate(FAMILIES):
        candidates=sorted((s45/"models").glob("*__"+family+".joblib"))
        need(len(candidates)==19,"expected original 19 per-piece S45 fits for "+family)
        selected=np.flatnonzero(moved[:,j])
        batch=[]
        for cp in candidates:
            bundle=joblib.load(cp) # only fixed 37891039201 archive
            need(bundle["family"]==family,"historical S45 family metadata drift")
            model=bundle["model"];scaler=bundle["scaler"]
            need(model.n_features_in_==scaler.n_features_in_==156,
                 "S45 original trained 156D Ridge schema")
            if selected.size:
                design=make_inputs(X[selected],parent[selected],
                    proposals[selected,j],agreement[selected,j])
                estimate=np.asarray(model.predict(
                    np.clip(scaler.transform(design),-6,6)),np.float32)
                need(estimate.shape==(len(selected),2) and
                     np.isfinite(estimate).all(),"S45 trained risk vector invalid")
                batch.append(estimate)
            history.append({"family":family,"checkpoint":cp.name,
                            "sha256":digest(cp)})
        if selected.size:
            risks[selected,j]=np.clip(np.mean(np.stack(batch),axis=0),0,1)
            scored_models+=len(batch)
    need(len(history)==95,"five families require 95 unchanged original S45 models")
    # A research-only what-if, NEVER emit it as full S80 or a 16-family policy.
    candidate,selected_head=choose_best(parent,proposals,risks,1.,0.)
    change_count=int((candidate!=parent).sum())
    out.mkdir(parents=True)
    np.savez_compressed(out/"five_of_sixteen_original_weight_risk_GOAT.npz",
        row_id=ids,S18_parent_K=parent,freeze_K=b,
        partial_five_family_ids=np.asarray(FAMILIES),
        partial_proposal_K=proposals,partial_agreement_votes=agreement,
        partial_risk_two_scores=risks,
        partial_research_candidate_K=candidate,
        partial_research_selected_head=selected_head)
    r={"status":"S163_ONLY_5_OF_16_S45_SOURCES_WITH_GENUINE_95_RIDGE_MODELS",
       "new_audio_native_events":n,
       "five_of_original_sixteen_families":list(FAMILIES),
       "original_ridge_checkpoint_source_run":37891039201,
       "genuine_original_ridge_checkpoints_verified":len(history),
       "checkpoints":history,
       "genuine_original_ridge_checkpoints_that_scored_events":scored_models,
       "original_S45_training_feature_width":156,
       "partial_proposal_nonparent_by_family":{
         f:int(moved[:,j].sum()) for j,f in enumerate(FAMILIES)},
       "partial_source_agreement_only_five":True,
       "original_16_family_agreement_available":False,
       "partial_candidate_changes_vs_S18":change_count,
       "partial_research_head_counts":{f:int((selected_head==j).sum())
            for j,f in enumerate(FAMILIES)},
       "partial_candidate_K_histogram":np.bincount(candidate,minlength=7).tolist(),
       "original_S5_S9_equivalence":False,
       "original_S45_all16_equivalence":False,
       "full_S46_S69_S80_input_valid":False,
       "GOAT_truth_or_MIDI_read":False,
       "independent_exactK_score":None,"official_S80_promotion":False}
    (out/"report.json").write_text(json.dumps(r,indent=2)+"\n")
    print(json.dumps({k:r[k] for k in
      ("status","new_audio_native_events","five_of_original_sixteen_families",
      "genuine_original_ridge_checkpoints_verified",
      "partial_proposal_nonparent_by_family",
      "partial_candidate_changes_vs_S18","partial_candidate_K_histogram",
      "original_16_family_agreement_available","official_S80_promotion")},indent=2))

def selftest():
    old=np.array([1,2,2,0],np.int8)
    c=np.array([[1,2],[3,3],[2,3],[1,0]],np.int8)
    changed=c!=old[:,None]
    v=np.array([(np.sum((c==c[:,j,None])&
                   (c!=old[:,None]),axis=1)-changed[:,j]) for j in range(2)]).T
    need(v.shape==(4,2) and v[1].tolist()==[1,1] and v[2].tolist()==[0,0],
         "only AVAILABLE source votes should be counted")
    r=np.full((2,2,2),np.nan,np.float32)
    r[0,0]=[.7,.1]
    prediction,chosen=choose_best(np.array([0,1],np.int8),
         np.array([[2,3],[2,3]],np.int8),r,1.,0.)
    need(prediction.tolist()==[2,1] and chosen.tolist()==[0,-1],
         "S45 original decision function returns BOTH K and chosen head")
    need(len(FAMILIES)==5 and len(set(FAMILIES))==5,
         "partial S45 family names must be explicit")
    print("S163 PASS: 5/16 source-only risk schema, no fake absent families")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for p in ("s97","s85","s135","s11","s12","s38","s40",
              "s41","s154","s157","s45","out"):
        ap.add_argument("--"+p,type=Path)
    ap.add_argument("--self-test",action="store_true")
    a=ap.parse_args()
    if a.self_test:selftest()
    else:
        need(all(getattr(a,p) is not None for p in (
           "s97","s85","s135","s11","s12","s38","s40",
           "s41","s154","s157","s45","out")),"missing frozen input")
        diagnose(*[getattr(a,p) for p in (
            "s97","s85","s135","s11","s12","s38","s40",
            "s41","s154","s157","s45","out")])
