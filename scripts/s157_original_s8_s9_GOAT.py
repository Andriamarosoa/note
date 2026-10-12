"""S157: genuine ORIGINAL learned S8 classifier + deterministic S9 on GOAT.

S8 source ORIGINAL 'series8__poly_parent__votes__C0.3__cost0.5'.
Trained original 28D vote model weights from run 37851172129.
S9 exact veto 'P>=5 and G<=3' from original posthoc script.
BUT original S5 global/poly parents are unavailable on new GOAT;
g=S135 student, p=original PR16 #22 are NEW substitutes.
No K truth/MIDI read, no re-training, no threshold tuning.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np
from scripts.s141_external_original_s40_cnn import load_input
from scripts.loop_v273_learned_risk import design,decision_advantage
POLICY="series8__poly_parent__votes__C0.3__cost0.5"
def need(c,m):
    if not c:raise ValueError(m)
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def source_s9(g,p,s8):
    return np.where((p>=5)&(g<=3),p,s8).astype(np.int8)
def run(s97,s85,s135,s8,out):
    need(not out.exists(),"immutable original S8/S9 output")
    ids,_,_,parents=load_input(s97,s85,s135)
    b=parents["official_freeze"].astype(np.int8)
    g=parents["student_S135"].astype(np.int8)
    p=parents["original_PR16_policy22"].astype(np.int8)
    with np.load(s97,allow_pickle=False) as z:
        need(np.array_equal(ids,z["row_id"].astype(str)),"S97 GOAT event alignment")
        summaries=z["summary58"].astype(np.float32)
    views=design({"features":summaries},b,g,p)
    need(views["votes"].shape==(57,28) and
         views["votes_acoustic"].shape==(57,86),"original S8 genuine 28D/86D")
    src=s8/"models"
    files=sorted(src.glob("*__votes__C0.3.joblib"))
    need(len(files)>=10,"missing original S8 original trained weight inventory")
    prob=[];proof=[]
    for f in files:
        obj=joblib.load(f)
        need(obj["arm"]=="votes" and abs(float(obj["C"])-.3)<1e-9,
             "genuine S8 decision parent provenance")
        need(obj["fold"] in (0,1,2,4),"original S8 heldfold")
        need(obj["parent_global"] and obj["parent_poly"],"trained S8 parent IDs")
        model=obj["model"];scale=obj["scaler"]
        need(model.n_features_in_==scale.n_features_in_==28,"original S8 feature schema")
        scores=decision_advantage(model,scale,views["votes"])
        need(scores.shape==(57,) and np.isfinite(scores).all(),
             "original S8 decision margin invalid")
        prob.append(scores)
        proof.append({"source_file":f.name,"fold":int(obj["fold"]),"sha256":sha(f)})
    mean=np.stack(prob).mean(0)
    agree=g!=p
    choose=agree&(mean<-.5)
    s8K=p.copy();s8K[choose]=g[choose]
    s9K=source_s9(g,p,s8K)
    need(np.isin(s8K,range(7)).all() and np.isin(s9K,range(7)).all(),
         "S8/S9 K0..6 output integrity")
    need(np.all((s9K==g)|(s9K==p)),"S9 source never invents K")
    out.mkdir(parents=True)
    np.savez_compressed(out/"s8_s9_original_trained_veto_GOAT_adapted_sources.npz",
        row_id=ids,freeze_K=b,adapted_global_S135_K=g,
        adapted_poly_PR16_K=p,S8_original_weight_K=s8K,
        S9_original_formula_K=s9K,
        S8_learned_advantage=mean)
    rep={
      "status":"S157_EXECUTED_GENUINE_ORIGINAL_S8_WEIGHTS_AND_S9_FORMULA_ON_GOAT",
      "real_new_audio_events":57,
      "S8_original_run":37851172129,
      "original_trained_S8_models_executed":len(proof),
      "saved_checkpoint_checksums":proof,
      "original_S8_variant":POLICY,
      "original_S8_threshold_minus_0p5_preserved":True,
      "original_S9_veto_P_ge5_and_G_le3_preserved":True,
      "S8_global_poly_input_adapted_not_original_S5":True,
      "S8_trained_disagreement_mask_events":int(agree.sum()),
      "S8_selected_global_events":int(choose.sum()),
      "S9_modified_S8_decisions":int((s9K!=s8K).sum()),
      "S9_changed_from_prior_PR16_22_adapter":int((s9K!=p).sum()),
      "S9_K_histogram":np.bincount(s9K,minlength=7).tolist(),
      "historically_equivalent_to_original_S9":False,
      "new_music_S80_independent_score_measured":False,
      "GOAT_MIDI_read":False,"promoted":False}
    (out/"report.json").write_text(json.dumps(rep,indent=2)+"\n")
    print(json.dumps({k:rep[k] for k in
      ("status","real_new_audio_events","original_trained_S8_models_executed",
       "S8_trained_disagreement_mask_events","S8_selected_global_events",
       "S9_modified_S8_decisions","S9_changed_from_prior_PR16_22_adapter",
       "S9_K_histogram","historically_equivalent_to_original_S9","promoted")},indent=2))
def selftest():
    g=np.array([1,2,3,4,5],np.int8)
    p=np.array([5,5,6,6,4],np.int8)
    s8=g.copy()
    actual=source_s9(g,p,s8)
    need(actual.tolist()==[5,5,6,4,5],"S9 exact archived high K veto math drift")
    print("S157 PASS original S8 28D schema and S9 P>=5 G<=3 rule")
if __name__=="__main__":
    a=argparse.ArgumentParser()
    for k in ("s97","s85","s135","s8","out"):
        a.add_argument("--"+k,type=Path)
    a.add_argument("--self-test",action="store_true")
    args=a.parse_args()
    if args.self_test:selftest()
    else:
        need(all(getattr(args,k) is not None for k in ("s97","s85","s135","s8","out")),
             "frozen source inputs required")
        run(args.s97,args.s85,args.s135,args.s8,args.out)
