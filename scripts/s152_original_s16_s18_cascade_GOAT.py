"""S152: original S16->S17->S18 trained checkpoint cascade on new GOAT audio.

Original 3x108 fixed veto variants, original saved K12/K23, K32/K34,
K43/K21 binary risk classifiers/scalers. Original full S5/S9/S13/S15
parent still unavailable on GOAT; START at adapted S149 S11 risk output.
Pre-fixed policy names exactly match historic S16/S17/S18 parent lineage.
No MIDI/labels used; not an original S80 deployment score.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,joblib
from scripts.s141_external_original_s40_cnn import load_input
from scripts.loop_v273_k12_k23 import make_policies as s16_policies
from scripts.loop_v273_k32_k34 import make_policies as s17_policies
from scripts.loop_v273_k43_k21 import make_policies as s18_policies
from scripts.train_v273_aba_recurrent import S18 as S18_ORIGINAL_NAME
SOURCE_PARENT={
 "S11":"series11__votes_time_audio__hgb__all__cost0",
 "S16":"series16__flow_logistic__k23__pbase_gt0.99",
 "S17":"series17__flow_logistic__k32__pbase_gt0.99",
 "S18":S18_ORIGINAL_NAME
}
DIRECTIONS={
 "S16":("k12","k23"),
 "S17":("k32","k34"),
 "S18":("k43","k21")
}
def need(p,m):
    if not p:raise ValueError(m)
def checksum(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def probabilities(directory,features,prefixes):
    pool={};manifest=[]
    for direction in prefixes:
        for representation in ("summary","flow"):
            for kind in ("logistic","hgb"):
                needle=direction+"__*__"+representation+"_"+kind+".joblib"
                paths=sorted((directory/"models").glob(needle))
                need(len(paths)>0,"missing trained source head "+str(directory)+"/"+needle)
                scores=[]
                for p in paths:
                    trained=joblib.load(p)
                    need(trained["direction"]==direction and
                        trained["piece"] in p.name and
                        trained["fold"] in (0,1,2,4),
                        "trained source model identity mismatch")
                    model=trained["model"];scaler=trained["scaler"]
                    X=features[representation]
                    need(model.n_features_in_==scaler.n_features_in_==X.shape[1],
                         "original 90D/335D source schema mismatch")
                    z=model.predict_proba(np.clip(scaler.transform(X),-6,6))
                    classes=list(map(int,model.classes_))
                    need(1 in classes,"trained source risk target class missing")
                    q=np.asarray(z[:,classes.index(1)],np.float64)
                    need(q.shape==(len(X),) and np.isfinite(q).all() and
                         ((q>=0)&(q<=1)).all(),"original risk probabilities")
                    scores.append(q)
                    manifest.append({"direction":direction,
                        "representation":representation,"model":kind,
                        "original_piece":trained["piece"],"original_fold":int(trained["fold"]),
                        "sha256":checksum(p)})
                pool[f"{direction}__{representation}_{kind}"]=np.stack(scores).mean(0)
        pool[f"{direction}__flow_both"]=np.minimum(
            pool[f"{direction}__flow_logistic"],pool[f"{direction}__flow_hgb"])
        pool[f"{direction}__flow_mean"]=(
            pool[f"{direction}__flow_logistic"]+
            pool[f"{direction}__flow_hgb"])/2
    return pool,manifest
def run(s97,s85,s135,s11,s16,s17,s18,out):
    need(not out.exists(),"immutable original checkpoint cascade output")
    ids,X,_,parents=load_input(s97,s85,s135)
    b=parents["official_freeze"].astype(np.int8)
    features={"summary":X[:,:90],"flow":X}
    with np.load(s11,allow_pickle=False) as z:
        need(np.array_equal(ids,z["row_id"].astype(str)) and
             np.array_equal(b,z["freeze_K"]),"S11/S16 native identity")
        names=list(map(str,z["variant_ids"]))
        need(SOURCE_PARENT["S11"] in names,"fixed initial policy not in S11")
        parent=z["predicted_K"][:,names.index(SOURCE_PARENT["S11"])].astype(np.int8)
    stage_runs={"S16":37855705158,"S17":37856164138,"S18":37856499212}
    models={"S16":s16,"S17":s17,"S18":s18}
    decoders={"S16":s16_policies,"S17":s17_policies,"S18":s18_policies}
    ancestors={"freeze_K":b,"adapted_S11_initial_K":parent.copy()}
    audit={};evidence=[]
    for stage in ("S16","S17","S18"):
        probs,model_evidence=probabilities(models[stage],features,DIRECTIONS[stage])
        policies,eligible=decoders[stage](b,parent,probs)
        need(len(policies)==110 and
             SOURCE_PARENT[stage] in policies,
             "original stage policy population")
        selected=policies[SOURCE_PARENT[stage]]
        need(np.isin(selected,range(7)).all(),"veto created unknown K")
        evidence.extend({"stage":stage,**v} for v in model_evidence)
        audit[stage]={
          "original_source_run":stage_runs[stage],
          "original_model_count":len(model_evidence),
          "original_108_veto_policies_executed":True,
          "candidate_eligibility":{k:int(v.sum()) for k,v in eligible.items()},
          "selected_policy":SOURCE_PARENT[stage],
          "parent_K_histogram":np.bincount(parent,minlength=7).tolist(),
          "stage_K_histogram":np.bincount(selected,minlength=7).tolist(),
          "changes_from_input_parent":int((parent!=selected).sum()),
          "posthoc_new_music_policy_selection":False}
        ancestors[stage+"_K"]=selected.astype(np.int8)
        print(json.dumps({"stage":stage,**audit[stage]}),flush=True)
        parent=selected.astype(np.int8)
    need(len(ids)==57 and len(evidence)>=12,"authentic saved checkpoints missing")
    out.mkdir(parents=True)
    np.savez_compressed(out/"S11_to_S18_adapted_original_checkpoint_cascade_GOAT.npz",
       row_id=ids,**ancestors)
    result={"status":"S152_EXECUTED_S11_S16_S17_S18_ORIGINAL_TRAINED_WEIGHTS_CASCADE",
            "GOAT_real_audio_events":len(ids),
            "genuine_trained_models_S16_S17_S18_total":len(evidence),
            "source_checkpoints":evidence,
            "three_veto_stages":audit,
            "K_final_distribution":np.bincount(parent,minlength=7).tolist(),
            "K_final_changed_vs_official_freeze":int((parent!=b).sum()),
            "original_S16_S17_S18_fixed_head_names_reproduced":True,
            "original_S11_and_S15_input_equivalence":False,
            "new_piece_ensemble_rule":"arithmetic mean over authentic piece-excluded risk checkpoints per head",
            "S13_S14_S15_original_K0_birth_policies_connected":False,
            "authentic_S18_historical_equivalence_verified":False,
            "MIDI_truth_read":False,"independent_exactK":None,
            "S80_promoted":False}
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in
       ("status","GOAT_real_audio_events",
        "genuine_trained_models_S16_S17_S18_total",
        "K_final_distribution","K_final_changed_vs_official_freeze",
        "S13_S14_S15_original_K0_birth_policies_connected","S80_promoted")},indent=2))
def selftest():
    need(SOURCE_PARENT["S16"]=="series16__flow_logistic__k23__pbase_gt0.99",
         "original S17 expected source")
    need(SOURCE_PARENT["S17"]=="series17__flow_logistic__k32__pbase_gt0.99",
         "original S18 expected source")
    need(SOURCE_PARENT["S18"]=="series18__flow_logistic__k21__pbase_gt0.95",
         "original S45 parent mismatch")
    print("S152 PASS original saved S16 S17 S18 mother/child source policy identities")
if __name__=="__main__":
    a=argparse.ArgumentParser()
    for field in ("s97","s85","s135","s11","s16","s17","s18","out"):
        a.add_argument("--"+field,type=Path)
    a.add_argument("--self-test",action="store_true")
    arg=a.parse_args()
    if arg.self_test:selftest()
    else:
        need(all(getattr(arg,n) is not None for n in
           ("s97","s85","s135","s11","s16","s17","s18","out")),
           "all eight source paths required")
        run(arg.s97,arg.s85,arg.s135,arg.s11,arg.s16,arg.s17,arg.s18,arg.out)
