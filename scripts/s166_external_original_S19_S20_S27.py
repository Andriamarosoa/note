"""S166 real ORIGINAL S19/S20 recurrent weights, S27 early-stable source on GOAT.

19 original piece-held S19 RecurrentSelector checkpoints; 19 original
piece-held S20 ResidualAB checkpoints. S27 'early_stable_005' recomputes
real second A with a learned B message, from saved original S20 weights.
Ensemble the piece-held posteriors equally on a new composition, a NEW
deployment rule; source input X335 has an adapted S5 global/poly head and
adapted S18 parent. No labels, retraining or S80 accuracy claims.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from scripts.s141_external_original_s40_cnn import load_input
from scripts.train_v273_aba_recurrent import RecurrentSelector
from scripts.train_v273_aba_residual import ResidualAB
from scripts.audit_v273_partial_revisits import run_policy
def need(c,m):
    if not c:raise ValueError(m)
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def network_checkpoints(path,pattern):
    p=sorted((path/"models").glob(pattern))
    need(len(p)==19,"need 19 genuine original trained held-piece models: "+pattern)
    return p
def run(s97,s85,s135,s154,s19,s20,out):
    need(not out.exists(),"immutable original S19/S20/S27 neural outputs")
    ids,X,seq,source=load_input(s97,s85,s135)
    b=source["official_freeze"].astype(np.int8)
    with np.load(s154,allow_pickle=False) as z:
        need(set(z.files)=={"row_id","freeze_K","adapted_S9_K",
                            "adapted_S15_K","S16_K","S17_K","S18_K"},
             "S154 original fit source contract")
        need(np.array_equal(ids,z["row_id"].astype(str)) and
             np.array_equal(b,z["freeze_K"]),"native source chronology")
        parent=z["S18_K"].astype(np.int8)
    need(X.shape==(57,335) and np.isfinite(X).all(),
         "335D original trained recurrent acoustic source")
    torch.set_num_threads(2)
    P19=[];P20=[];P27=[];costs=[];model_proof=[]
    for path in network_checkpoints(s19,"*__A_B_A.pt"):
        p=torch.load(path,map_location="cpu",weights_only=False)
        need(p["piece"] in path.name and p["fold"] in (0,1,2,4),
             "S19 original training source provenance")
        mean=np.asarray(p["scaler_mean"],np.float32)
        scale=np.asarray(p["scaler_scale"],np.float32)
        need(mean.shape==scale.shape==(335,) and np.all(scale>0),
             "S19 original scaler feature schema")
        xx=np.clip((X-mean)/scale,-5,5).astype(np.float32)
        net=RecurrentSelector()
        net.load_state_dict(p["state_dict"],strict=True)
        net.eval()
        with torch.no_grad():
            prob=net(torch.from_numpy(xx),passes=2)[2][1].numpy()
        need(prob.shape==(57,7) and np.isfinite(prob).all()
             and np.allclose(prob.sum(1),1,atol=1e-5),"S19 actual ABA2 posterior")
        P19.append(prob)
        model_proof.append({"source":"S19_ABA_pass2",
                            "piece":p["piece"],"fold":int(p["fold"]),
                            "original_checkpoint_SHA256":digest(path)})
    for path in network_checkpoints(s20,"*__residual_A_B.pt"):
        p=torch.load(path,map_location="cpu",weights_only=False)
        need(p["piece"] in path.name and p["fold"] in (0,1,2,4),
             "S20 original training source provenance")
        mean=np.asarray(p["scaler_mean"],np.float32)
        scale=np.asarray(p["scaler_scale"],np.float32)
        need(mean.shape==scale.shape==(335,) and np.all(scale>0),
             "S20 original scaler feature schema")
        xx=np.clip((X-mean)/scale,-5,5).astype(np.float32)
        net=ResidualAB()
        net.load_state_dict(p["model"],strict=True)
        net.eval()
        with torch.no_grad():
            A=net(torch.from_numpy(xx),torch.from_numpy(parent.astype(np.int64)),passes=1)[2][0].numpy()
            stable,usage,_=run_policy(net,torch.from_numpy(xx),
                      torch.from_numpy(parent.astype(np.int64)),"early_stable_005")
            stable=stable.numpy()
        for proba in (A,stable):
            need(proba.shape==(57,7) and np.isfinite(proba).all()
                  and np.allclose(proba.sum(1),1,atol=1e-5),
                  "S20 or S27 original recurrent posteriors")
        P20.append(A);P27.append(stable)
        costs.append(usage)
        model_proof.append({"source":"S20_and_S27",
                            "piece":p["piece"],"fold":int(p["fold"]),
                            "original_checkpoint_SHA256":digest(path)})
    need(len(P19)==len(P20)==len(P27)==19 and len(model_proof)==38,
         "neural recurrent original checkpoint completeness")
    mean19=np.stack(P19).mean(0)
    mean20=np.stack(P20).mean(0)
    mean27=np.stack(P27).mean(0)
    pred19=mean19.argmax(1).astype(np.int8)
    pred20=mean20.argmax(1).astype(np.int8)
    pred27=mean27.argmax(1).astype(np.int8)
    out.mkdir(parents=True)
    np.savez_compressed(out/"authentic_S19_S20_S27_original_weights_GOAT.npz",
      row_id=ids,adapted_S18_parent_K=parent,freeze_K=b,
      S19_ABA_pass2_mean_probs=mean19,S19_ABA_pass2_K=pred19,
      S20_pass1_mean_probs=mean20,S20_pass1_K=pred20,
      S27_early_stable_005_mean_probs=mean27,
      S27_early_stable_005_K=pred27)
    report={
      "status":"S166_THREE_GENUINE_ORIGINAL_RECURRENT_SOURCES_ON_UNSEEN_GOAT",
      "original_S19_checkpoint_run":37858017067,
      "original_S20_checkpoint_run":37858383973,
      "GOAT_audio_events":57,
      "source_original_saved_models_S19":19,
      "source_original_saved_models_S20":19,
      "original_S27_early_stable_005_policy_replayed":True,
      "original_neural_checkpoint_SHA256":model_proof,
      "original_ensemble_rule":"mean posterior from 19 original held-piece models per source; preregistered",
      "source_S19_ABA2_K_histogram":np.bincount(pred19,minlength=7).tolist(),
      "source_S20_A1_K_histogram":np.bincount(pred20,minlength=7).tolist(),
      "source_S27_early005_K_histogram":np.bincount(pred27,minlength=7).tolist(),
      "S19_changed_vs_S18":int(np.sum(pred19!=parent)),
      "S20_changed_vs_S18":int(np.sum(pred20!=parent)),
      "S27_changed_vs_S18":int(np.sum(pred27!=parent)),
      "S27_total_real_A_calls":int(sum(z["A"] for z in costs)),
      "S27_total_real_B_calls":int(sum(z["B"] for z in costs)),
      "full_original_S5_S9_S18_audio_equivalence":False,
      "original_16_S45_source_producers_complete":False,
      "new_audio_exact_K_scored":False,
      "GOAT_truth_MIDI_read":False,
      "official_S80_promotion":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in (
      "status","GOAT_audio_events",
      "source_original_saved_models_S19","source_original_saved_models_S20",
      "original_S27_early_stable_005_policy_replayed",
      "source_S19_ABA2_K_histogram","source_S20_A1_K_histogram",
      "source_S27_early005_K_histogram",
      "S19_changed_vs_S18","S20_changed_vs_S18",
      "S27_changed_vs_S18","official_S80_promotion")},indent=2))

def selftest():
    net=ResidualAB()
    torch.set_num_threads(1)
    X=torch.zeros((3,335))
    parent=torch.tensor([0,1,2],dtype=torch.long)
    with torch.no_grad():
        p=net(X,parent,passes=1)[2][0]
        q,stat,_=run_policy(net,X,parent,"early_stable_005")
    need(p.shape==q.shape==(3,7) and stat["A"]>=6 and
         np.isfinite(q.numpy()).all(),"real second A/B adaptive stop must run")
    print("S166 PASS original S19/S20/S27 neural source graph, NO labels")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("s97","s85","s135","s154","s19","s20","out"):
        p.add_argument("--"+k,type=Path)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        need(all(getattr(a,k) is not None for k in
          ("s97","s85","s135","s154","s19","s20","out")),
          "all original audio, checkpoint and parent files required")
        run(a.s97,a.s85,a.s135,a.s154,a.s19,a.s20,a.out)
