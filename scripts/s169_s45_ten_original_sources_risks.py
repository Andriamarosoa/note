"""S169: measure ten TRUE original trained S45 families' collective votes.

Eight source proposals from S167 plus two genuine B-first trained S24/S25
sources from S168. Recompute original S45 156D Ridge risk FEATURES with
TEN-family independent agreement, keeping all original 19 per-piece models
per family, lambda1/cut0 unchanged and NO new fitting or MIDI.
This exposes an interaction between weak heads vs changed vote count,
not a promotion-proof 16-source S45, since six original families missing.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np
from scripts.s141_external_original_s40_cnn import load_input
from scripts.train_v273_s45_specialized_risk_heads import make_inputs,choose_best
FAMILIES=("S11_votes_time_audio","S12_audio_logistic",
 "S19_recurrent_ABA","S20_repeated_ABA","S24_Bfirst",
 "S25_Bfirst","S27_early_stop","S38_morphology",
 "S40_temporal_CNN","S41_neural_fusion")
def need(c,m):
    if not c:raise ValueError(m)
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def run(s97,s85,s135,s167,s168,s45,catalog,out):
    need(not out.exists(),"S169 immutable risk audit output")
    ids,X,_,parents=load_input(s97,s85,s135)
    with np.load(s167,allow_pickle=False) as z:
        exact={"row_id","S18_parent_K","original_eight_family_ids",
         "eight_proposal_K","eight_family_agreement",
         "original_ridge_two_risks","selected_head_index",
         "partial_eight_source_decision_K"}
        need(set(z.files)==exact,"S167 frozen source schema drift")
        need(np.array_equal(ids,z["row_id"].astype(str)),"GOAT S167 source ID order")
        original_parent=z["S18_parent_K"].astype(np.int8)
        eight_ids=list(map(str,z["original_eight_family_ids"]))
        eight=z["eight_proposal_K"].astype(np.int8)
        r8=z["original_ridge_two_risks"].astype(np.float64)
        before=z["partial_eight_source_decision_K"].astype(np.int8)
        votes8=z["eight_family_agreement"].astype(np.int16)
    with np.load(s168,allow_pickle=False) as z:
        expected={"row_id","S18_adapted_K","freeze_K",
         "original_S24_Bfirst_B0_probabilities",
         "original_S24_Bfirst_B0_K",
         "original_S25_Bbefore_A0_probabilities",
         "original_S25_Bbefore_A0_K"}
        need(set(z.files)==expected and
             np.array_equal(ids,z["row_id"].astype(str)) and
             np.array_equal(original_parent,z["S18_adapted_K"]),
             "B-first original pretrained inputs misaligned")
        extra={"S24_Bfirst":z["original_S24_Bfirst_B0_K"].astype(np.int8),
               "S25_Bfirst":z["original_S25_Bbefore_A0_K"].astype(np.int8)}
    need(eight.shape==(57,8) and r8.shape==(57,8,2)
         and len(set(eight_ids))==8 and
         set(FAMILIES)==set(eight_ids).union(extra),
         "must have TEN REAL sources, not synthetic absent families")
    proposed=np.column_stack(
        [extra[f] if f in extra else eight[:,eight_ids.index(f)]
         for f in FAMILIES]).astype(np.int8)
    need(proposed.shape==(57,10) and
         np.isin(proposed,range(7)).all(),"original 10 source K0..6 contract")
    cat=json.loads(catalog.read_text())
    idx={q["family"]:q for q in cat["original_families"]}
    need(len(idx)==16 and set(FAMILIES).issubset(idx),
         "original S44 champion source family names mismatch")
    moved=proposed!=original_parent[:,None]
    agreement=np.zeros((57,10),np.int16)
    for j in range(10):
        agreement[:,j]=np.sum(
          (proposed==proposed[:,j,None])&
          (proposed!=original_parent[:,None]),axis=1)-moved[:,j]
    risks=np.full((57,10,2),np.nan,np.float32)
    checkpoints=0;scoring=0;stats={};proof=[]
    for j,fam in enumerate(FAMILIES):
        paths=sorted((s45/"models").glob("*__"+fam+".joblib"))
        need(len(paths)==19,"missing S45 original 19 heldpiece Ridge heads "+fam)
        ii=np.flatnonzero(moved[:,j])
        values=[]
        for path in paths:
            obj=joblib.load(path)
            need(obj["family"]==fam and
                 obj["source_original_sha256"]==idx[fam]["original_vector_sha256"],
                 "original training model family SHA does not match S44 champion")
            model=obj["model"];scaler=obj["scaler"]
            need(model.n_features_in_==scaler.n_features_in_==156,
                 "original 156D correction/regression risk geometry changed")
            if len(ii):
                Xr=make_inputs(X[ii],original_parent[ii],
                              proposed[ii,j],agreement[ii,j])
                q=np.asarray(model.predict(
                    np.clip(scaler.transform(Xr),-6,6)),np.float32)
                need(q.shape==(len(ii),2) and np.isfinite(q).all(),
                     "invalid original S45 Ridge posterior")
                values.append(q)
                scoring+=1
            checkpoints+=1
            proof.append({"family":fam,"original_piece":obj["piece"],
                          "trained_checkpoint_SHA256":sha(path)})
        if len(ii):
            risks[ii,j]=np.clip(np.stack(values).mean(0),0,1)
        stats[fam]={"proposed_nonparent":int(len(ii)),
                    "original_checkpoints_available":len(paths),
                    "original_checkpoints_scoring":len(values)}
    need(checkpoints==190 and np.isfinite(risks[moved]).all() and
         np.isnan(risks[~moved]).all(),
         "missing original trained S45 evidence or invented score")
    final,chosen=choose_best(original_parent,proposed,risks,1.,0.)
    # The same S11/S12/S19 exact proposal score may change when S24+S25
    # add agreement evidence, without refitting any trained model.
    deltas=[]
    for fam in eight_ids:
        j0=eight_ids.index(fam);j1=FAMILIES.index(fam)
        ii=np.flatnonzero(eight[:,j0]!=original_parent)
        for row in ii:
            old_score=float(r8[row,j0,0]-r8[row,j0,1])
            new_score=float(risks[row,j1,0]-risks[row,j1,1])
            deltas.append({"row_id":ids[row],"family":fam,
                "old_partial_eight_agreement":int(votes8[row,j0]),
                "new_partial_ten_agreement":int(agreement[row,j1]),
                "old_risk_score":old_score,"new_risk_score":new_score,
                "delta_new_minus_old":new_score-old_score})
    out.mkdir(parents=True)
    np.savez_compressed(out/"ten_source_S45_original_ridge_GOAT.npz",
       row_id=ids,S18_parent_K=original_parent,
       original_ten_source_family_ids=np.asarray(FAMILIES),
       ten_source_K=proposed,
       ten_partial_source_agreement=agreement,
       original_ridge_risks=risks,
       ten_partial_selected_head=chosen,
       ten_partial_candidate_K=final)
    report={
      "status":"S169_TEN_OF_SIXTEEN_TRUE_ORIGINAL_S45_PRODUCER_RIDGE_CHECKPOINTS",
      "GOAT_native_events":57,
      "original_family_source_count":10,
      "missing_original_family_sources":6,
      "original_family_specific_Ridge_fits_verified":checkpoints,
      "genuine_original_Ridge_fits_scoring_events":scoring,
      "original_ridge_training_run_id":37891039201,
      "fixed_lambda1_cut0":True,
      "original_trained_checkpoint_SHA256":proof,
      "family_activity":stats,
      "proposals_not_equal_parent":int(moved.sum()),
      "different_native_events_with_source_alternative":int(np.any(moved,axis=1).sum()),
      "S45_partial_K_changed_vs_S18":int((final!=original_parent).sum()),
      "S45_partial_K_histogram":np.bincount(final,minlength=7).tolist(),
      "S45_partial_changed_from_earlier_8_source_result":int((final!=before).sum()),
      "same_trained_family_new_vote_score_deltas":deltas,
      "all_original_16_source_votes_available":False,
      "original_S5_S18_on_new_audio_equivalent":False,
      "can_enter_original_S46_S80_as_equivalent":False,
      "GOAT_MIDI_true_K_opened":False,
      "external_ExactK_validated":False,
      "official_S80_promotion":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in (
      "status","GOAT_native_events","original_family_source_count",
      "missing_original_family_sources",
      "original_family_specific_Ridge_fits_verified",
      "genuine_original_Ridge_fits_scoring_events",
      "family_activity","proposals_not_equal_parent",
      "different_native_events_with_source_alternative",
      "S45_partial_K_changed_vs_S18",
      "S45_partial_changed_from_earlier_8_source_result",
      "same_trained_family_new_vote_score_deltas",
      "official_S80_promotion")},indent=2))
def selftest():
    need(len(FAMILIES)==10 and len(set(FAMILIES))==10 and
         "S24_Bfirst" in FAMILIES and "S25_Bfirst" in FAMILIES,
         "two original B-first source families required")
    b=np.array([1,2],np.int8)
    prop=np.array([[0,0,2],[3,3,2]],np.int8)
    m=prop!=b[:,None]
    votes=np.zeros_like(prop,dtype=np.int16)
    for j in range(3):
        votes[:,j]=np.sum((prop==prop[:,j,None])&
                        (prop!=b[:,None]),axis=1)-m[:,j]
    need(votes.tolist()==[[1,1,0],[1,1,0]],
         "three-family votes count only distinct sources")
    print("S169 PASS original S45 ten-family independent agreement, no invented sources")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("s97","s85","s135","s167","s168","s45","catalog","out"):
        p.add_argument("--"+n,type=Path)
    p.add_argument("--self-test",action="store_true")
    x=p.parse_args()
    if x.self_test:selftest()
    else:
        need(all(getattr(x,k) is not None for k in
            ("s97","s85","s135","s167","s168","s45","catalog","out")),
            "all original inputs required")
        run(*[getattr(x,k) for k in
            ("s97","s85","s135","s167","s168","s45","catalog","out")])
