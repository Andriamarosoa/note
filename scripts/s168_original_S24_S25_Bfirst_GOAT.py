"""S168: original S24 Bfirst_B0 and S25 Bbefore_A0 source weights on GOAT.

Two authentic historical S45 champion families via 19 S20 residual
checkpoints reused IN REVERSE order (S24) and 19 saved B-first
TrainedBfirst original weights (S25). Ensemble each held-piece
multi-label B posterior mean over nineteen trained original models:
NEW piece aggregation; source S5 and S18 upstream still adapted.
Never read MIDI labels or score/promote full S80.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np,torch
from scripts.s141_external_original_s40_cnn import load_input
from scripts.train_v273_aba_residual import ResidualAB
from scripts.train_v273_Bfirst_recurrent import TrainedBfirst
from scripts.audit_v273_inverse_ab_order import infer_reverse
def need(c,m):
    if not c:raise ValueError(m)
def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def load_weights(path,filename,modelclass,features,parent):
    files=sorted((path/"models").glob(filename))
    need(len(files)==19,"need 19 original pretrained piece-held checkpoints: "+filename)
    all_b=[];ledger=[]
    for file in files:
        trained=torch.load(file,map_location="cpu",weights_only=False)
        need(trained["piece"] in file.name and
             trained["fold"] in (0,1,2,4),"original checkpoint historical provenance")
        mean=np.asarray(trained["scaler_mean"],np.float32)
        std=np.asarray(trained["scaler_scale"],np.float32)
        need(mean.shape==std.shape==(335,) and
             np.isfinite(mean).all() and np.all(std>0),
             "original recurrent acoustic 335D normalizer")
        scaled=np.clip((features-mean)/std,-5,5).astype(np.float32)
        net=modelclass()
        net.load_state_dict(trained["model"],strict=True)
        net.eval()
        with torch.no_grad():
            signal=torch.from_numpy(scaled)
            ps=torch.from_numpy(parent.astype(np.int64))
            if filename=="*__residual_A_B.pt":
                _,B=infer_reverse(net,signal,ps,passes=1)
                q=torch.sigmoid(B[0]).numpy()
            else:
                _,B,_=net(signal,ps,passes=1)
                q=torch.sigmoid(B[0]).numpy()
        need(q.shape==(57,7) and np.isfinite(q).all() and
             np.all((q>=0)&(q<=1)),"original B-first BCE head logits invalid")
        all_b.append(q)
        ledger.append({"piece":trained["piece"],"fold":int(trained["fold"]),
            "original_checkpoint_SHA256":checksum(file)})
    mean=np.stack(all_b).mean(0)
    return mean,ledger

def run(s97,s85,s135,s154,s20,s25,out):
    need(not out.exists(),"immutable original S24/S25 original-weight outputs")
    ids,X,_,source=load_input(s97,s85,s135)
    b=source["official_freeze"].astype(np.int8)
    with np.load(s154,allow_pickle=False) as z:
        need(set(z.files)=={"row_id","freeze_K","adapted_S9_K",
             "adapted_S15_K","S16_K","S17_K","S18_K"},
             "original trained S18 native output changed")
        need(np.array_equal(ids,z["row_id"].astype(str)) and
             np.array_equal(b,z["freeze_K"]),"identical native GOAT events required")
        parent=z["S18_K"].astype(np.int8)
    torch.set_num_threads(2)
    s24,proof24=load_weights(s20,"*__residual_A_B.pt",ResidualAB,X,parent)
    s25,proof25=load_weights(s25,"*__trained_Bfirst.pt",TrainedBfirst,X,parent)
    k24=s24.argmax(1).astype(np.int8)
    k25=s25.argmax(1).astype(np.int8)
    need(k24.shape==k25.shape==(57,) and
         len(proof24)==len(proof25)==19,"true B-first source family integrity")
    out.mkdir(parents=True)
    np.savez_compressed(out/"original_S24_S25_38_trained_Bfirst_GOAT.npz",
       row_id=ids,S18_adapted_K=parent,freeze_K=b,
       original_S24_Bfirst_B0_probabilities=s24,
       original_S24_Bfirst_B0_K=k24,
       original_S25_Bbefore_A0_probabilities=s25,
       original_S25_Bbefore_A0_K=k25)
    report={
     "status":"S168_GENUINE_ORIGINAL_S24_S25_BFIRST_TRAINED_CHECKPOINTS_GOAT",
     "native_new_audio_events":57,
     "S24_original_source_training_run_S20":37858383973,
     "S25_original_trained_source_run":37862417522,
     "original_S20_fixed_checkpoint_count_for_S24":19,
     "original_S25_trained_checkpoint_count":19,
     "S24_checkpoint_sha":proof24,"S25_checkpoint_sha":proof25,
     "S24_Bfirst_B0_K_distribution":np.bincount(k24,minlength=7).tolist(),
     "S25_Bbefore_A0_K_distribution":np.bincount(k25,minlength=7).tolist(),
     "S24_changed_K_vs_S18":int(np.sum(k24!=parent)),
     "S25_changed_K_vs_S18":int(np.sum(k25!=parent)),
     "B_initial_is_auxiliary_BCE_not_accuracy_ranked":True,
     "original_S5_input_source_equivalent":False,
     "original_S45_complete_16":False,
     "new_music_S80_ExactK_scored":False,
     "GOAT_labels_MIDI_opened":False,
     "S80_promoted":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in (
      "status","native_new_audio_events",
      "original_S20_fixed_checkpoint_count_for_S24",
      "original_S25_trained_checkpoint_count",
      "S24_Bfirst_B0_K_distribution","S25_Bbefore_A0_K_distribution",
      "S24_changed_K_vs_S18","S25_changed_K_vs_S18","S80_promoted")},indent=2))
def selftest():
    torch.set_num_threads(1)
    b=ResidualAB();c=TrainedBfirst()
    x=torch.zeros(4,335);p=torch.tensor([0,1,2,3],dtype=torch.long)
    with torch.no_grad():
        a24,B24=infer_reverse(b,x,p,passes=1)
        a25,B25,P25=c(x,p,passes=1)
    need(len(B24)==2 and len(B25)==1 and
         torch.sigmoid(B24[0]).shape==
         torch.sigmoid(B25[0]).shape==(4,7),
         "S24 S25 actual B-before-A trained head output schema")
    print("S168 PASS: genuine original S24 reverse S20 weights and S25 trained-Bfirst shape")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("s97","s85","s135","s154","s20","s25","out"):
        p.add_argument("--"+k,type=Path)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        need(all(getattr(a,k) is not None for k in
         ("s97","s85","s135","s154","s20","s25","out")),"all original sources required")
        run(a.s97,a.s85,a.s135,a.s154,a.s20,a.s25,a.out)
