"""S161: re-run ORIGINAL trained S41 six-action gate on the S154 S18 parent.

Fix a documented S144 interface mismatch: original S41 training d['parent']
is the S18 decision, not freeze_local_combo. Source 335D acoustic/vote
features still inherit the original FREEZE source 32D votes, as in S144.
Reuse frozen authentic 16x S38 and 4x S40 posteriors on exactly the same
57 WAV-derived GOAT events; all 19 S41 genuine trained networks are
loaded unchanged, with no labels or tuning.
Parent S18 is *adapted S9*, not the genuine unseen-audio historic S18.
Therefore this is S80-deploy-v1 research, not historical S80.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np,torch
from scripts.s141_external_original_s40_cnn import load_input
from scripts.train_v273_s41_neural_fusion import (
    FusionGate,make_proposals,gated_prediction,ACTION_NAMES)
def need(c,m):
    if not c:raise ValueError(m)
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(s97,s85,s135,s144,s154,s41,out):
    need(not out.exists(),"immutable S161 experiment")
    ids,X,_,original=load_input(s97,s85,s135)
    base_freeze=original["official_freeze"].astype(np.int8)
    with np.load(s144,allow_pickle=False) as z:
        expected={"row_id","parent_K","proposal_K","gate_scores",
                  "proposed_K","S38_four_expert_probabilities",
                  "S40_cnn_probabilities","selected_route"}
        need(set(z.files)==expected,"S144 expert-only forward schema")
        need(np.array_equal(ids,z["row_id"].astype(str)) and
             np.array_equal(base_freeze,z["parent_K"]),"S144 source native/baseline alignment")
        tab=z["S38_four_expert_probabilities"].astype(np.float32)
        cnn=z["S40_cnn_probabilities"].astype(np.float32)
        previous=z["proposed_K"].astype(np.int8)
        old_scores=z["gate_scores"].astype(np.float32)
    with np.load(s154,allow_pickle=False) as z:
        need(set(z.files)=={"row_id","freeze_K","adapted_S9_K",
                            "adapted_S15_K","S16_K","S17_K","S18_K"},
             "S154 original adapted S18 source identity")
        need(np.array_equal(ids,z["row_id"].astype(str)) and
             np.array_equal(base_freeze,z["freeze_K"]),"S154 matching native baseline")
        adapted_s18=z["S18_K"].astype(np.int8)
    need(X.shape==(57,335) and tab.shape==(57,4,7) and
         cnn.shape==(57,7),"original S41 audio and S38/S40 experts")
    need(np.isfinite(X).all() and np.isfinite(tab).all()
         and np.isfinite(cnn).all(),"nonfinite S41 trained features")
    props=make_proposals(adapted_s18,tab,cnn)
    models=sorted((s41/"models").glob("*__sixhead_fusion.pt"))
    need(len(models)==19,"all 19 original S41 trained model checkpoints required")
    scores=[];proof=[]
    torch.set_num_threads(2)
    for path in models:
        saved=torch.load(path,map_location="cpu",weights_only=False)
        need(saved["held_piece"] in path.name and
             saved["fold"] in (0,1,2,4),"original S41 training file provenance")
        mean=np.asarray(saved["scaler_mean"],np.float32)
        scale=np.asarray(saved["scaler_scale"],np.float32)
        need(mean.shape==scale.shape==(335,) and
             np.isfinite(scale).all() and np.all(scale>0),
             "original S41 fit scaler width/positive")
        normalized=np.clip((X-mean)/scale,-5,5).astype(np.float32)
        f=np.column_stack((normalized,np.eye(7,dtype=np.float32)[adapted_s18],
                            tab.reshape(57,28),cnn)).astype(np.float32)
        need(f.shape==(57,377),"true original S41 gate feature width")
        net=FusionGate()
        net.load_state_dict(saved["model"],strict=True);net.eval()
        with torch.no_grad():
            prob=torch.softmax(net(torch.from_numpy(f)),dim=1).numpy()
        need(prob.shape==(57,6) and np.isfinite(prob).all() and
             np.allclose(prob.sum(1),1,atol=1e-5),"S41 authentic gate posteriors")
        scores.append(prob)
        proof.append({"checkpoint_sha256":sha(path),
                      "held_original_piece":saved["held_piece"],
                      "held_original_fold":int(saved["fold"])})
    mixture=np.stack(scores).mean(0)
    new=gated_prediction(adapted_s18,props,mixture,"source_K1to6",0.)
    routes=mixture.argmax(1)
    keep=routes==0
    eligible=adapted_s18>=1
    has_alt=(props[:,1:]!=adapted_s18[:,None]).any(1)
    need(len(new)==57 and np.isin(new,range(7)).all(),"original new music K range")
    out.mkdir(parents=True)
    np.savez_compressed(out/"original_S41_gate_with_authentic_weight_S18_parent_GOAT.npz",
       row_id=ids,freeze_K=base_freeze,
       adapted_S18_parent_K=adapted_s18,
       proposals_6_K=props,gate_6_probabilities=mixture,
       final_policy_K=new,
       frozen_S144_wrong_parent_K=previous)
    result={
       "status":"S161_ORIGINAL_S41_TRAINED_MODELS_WITH_ADAPTED_S18_PARENT",
       "original_checkpoint_count":len(proof),
       "original_checkpoint_source_run":37881191015,
       "checkpoint_provenance":proof,
       "source_S38_S40_scores_unchanged":True,
       "GOAT_events":57,
       "adapted_S18_parent_K_histogram":np.bincount(adapted_s18,minlength=7).tolist(),
       "new_S41_output_K_histogram":np.bincount(new,minlength=7).tolist(),
       "new_S41_pred_changes_from_adapted_S18":int((new!=adapted_s18).sum()),
       "new_S41_pred_changes_from_old_S144_wrong_parent":int((new!=previous).sum()),
       "new_gate_argmax_KEEP":int(keep.sum()),
       "new_gate_argmax_nonKEEP":int((~keep).sum()),
       "eligible_K1plus_with_alternative_source_proposal":int((eligible&has_alt).sum()),
       "eligible_K1plus_with_nonKEEP_gate_preference":int((eligible&(~keep)).sum()),
       "new_gate_mean_KEEP_probability":float(mixture[:,0].mean()),
       "old_S144_gate_mean_KEEP_probability":float(old_scores[:,0].mean()),
       "genuine_S18_full_original_new_audio":False,
       "genuine_full_original_S45_16families":False,
       "GOAT_truth_MIDI_read":False,
       "new_music_Exact_K_measured":False,
       "official_S80_promoted":False
    }
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in (
       "status","original_checkpoint_count","GOAT_events",
       "adapted_S18_parent_K_histogram","new_S41_output_K_histogram",
       "new_S41_pred_changes_from_adapted_S18",
       "new_S41_pred_changes_from_old_S144_wrong_parent",
       "new_gate_argmax_KEEP","new_gate_mean_KEEP_probability",
       "old_S144_gate_mean_KEEP_probability","official_S80_promoted")},indent=2))
def selftest():
    b=np.array([1,2,3],dtype=np.int8)
    tab=np.zeros((3,4,7));tab[:,:,1]=1
    cnn=np.zeros((3,7));cnn[:,1]=1
    props=make_proposals(b,tab,cnn)
    need(props.shape==(3,6) and np.array_equal(props[:,0],b),
         "Original S41 parent is S18, not freeze")
    print("S161 PASS: S41 KEEP refers original S18 parent; S38/S40 experts retained")
if __name__=="__main__":
    a=argparse.ArgumentParser()
    for name in ("s97","s85","s135","s144","s154","s41","out"):
        a.add_argument("--"+name,type=Path)
    a.add_argument("--self-test",action="store_true")
    v=a.parse_args()
    if v.self_test:selftest()
    else:
        need(all(getattr(v,k) is not None for k in
                 ("s97","s85","s135","s144","s154","s41","out")),
                 "missing original checkpoint or GOAT audio")
        run(v.s97,v.s85,v.s135,v.s144,v.s154,v.s41,v.out)
