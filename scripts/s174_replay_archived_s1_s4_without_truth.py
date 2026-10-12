"""S174: truth-free numerical REPLAY of RESTORED ORIGINAL S1/S2/S3/S4 models.

Replays only historical stored input tensors, never trains/fits or reads true_K,
truth, MIDI, labels, fold-dependent target selection. Historical outer-piece
model assignment/choice already used development labels; this test merely
checks numerical integrity, it is NOT independent evaluation or GOAT inference.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np
from scipy.special import expit
from scripts.v273_regression_loops import FAMILIES,COSTS,design,predict,decode
from scripts.nonlinear_v273_regression_loops import nonlinear_features,tree_factors,blended,BLENDS
from scripts.guard_v273_regression_loops import WEIGHTS

SOURCE=Path("analysis/evidence/v273-regression-loops")
SAFE_INPUTS=("baseline","available","z","logits","features","feature_names",
             "group_context","group_names","original","piece","fold",
             "native_position","global_index")
def need(ok,why):
    if not ok:raise ValueError(why)
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path,keys):
    with np.load(path,allow_pickle=False) as z:
        need(set(keys).issubset(z.files),f"missing fields {path}")
        return {k:z[k].copy() for k in keys}
def run(root,restored,out):
    need(not out.exists(),"S174 refuses to overwrite evidence")
    pre=root/SOURCE/"prepared"
    firstroot=root/SOURCE/"series1"
    data=read(pre/"inputs.npz",SAFE_INPUTS)
    need(len(data["baseline"])==7493,"original eligible cohort drift")
    need(data["group_context"].shape==(7493,7,20) and
         data["features"].shape==(7493,58) and
         data["logits"].shape==(7493,7),"original S1 feature contracts")
    need(np.isfinite(data["group_context"]).all() and
         np.isfinite(data["features"]).all(),"corrupt source feature evidence")
    first=read(firstroot/"predictions.npz",("predictions","variant_ids"))
    probs=read(firstroot/"probabilities.npz",("baseline_correct","class_probability"))
    positions=data["native_position"].astype(int)
    need(len(set(positions.tolist()))==7493,"duplicate historical eligible positions")
    rawfirst=first["predictions"][positions]
    need(rawfirst.shape[1]==56,"original S1 policy matrix")
    splits=json.loads((firstroot/"splits.json").read_text())
    states=json.loads((restored/"S1_models.json").read_text())
    need(len(states)>0 and len(splits["blocks"])==19,"S1 states/outer blocks missing")
    replays=np.full((7493,49),-1,np.int8)
    proofs=[]
    for fi,family in enumerate(FAMILIES):
        xr,xq,rn,qn=design(data,family)
        need(rn==splits["designs"][family]["r_names"] and
             qn==splits["designs"][family]["q_names"],
             "original S1 feature ordering drift "+family)
        assignments=[a for a in splits["assignments"] if
                     a["family"]==family and a["role"]=="outer"]
        need(len(assignments)==19,"original 19 held-piece assignments "+family)
        for a in assignments:
            t=np.flatnonzero(data["piece"]==a["test_piece"])
            m=states[a["model"]]
            need(m["family"]==family and len(t)>0 and
                 a["test_piece"] not in m["train_pieces"],
                 "S1 actual checkpoint target contamination")
            r,cp,other,logit=predict(data,xr,xq,t,m)
            for j,cost in enumerate(COSTS):
                replays[t,fi*7+j]=decode(data["baseline"][t],data["available"][t],
                                         r,cp,logit,cost)
        need(np.isin(replays[:,fi*7:(fi+1)*7],range(7)).all(),
             "unreplayed S1 held pieces")
    need(np.array_equal(replays,rawfirst[:,:49]),
         "S1 original saved model decisions NOT reproduced")
    proofs.append({"family":"S1_acoustic","rows":7493,
      "families":7,"model_assignments":7*19,
      "historical_original_decision_mismatches":0})

    second=read(root/SOURCE/"series2/predictions.npz",
                ("predictions","chosen_policy","variant_ids"))
    selected=second["chosen_policy"].astype(int)
    need(selected.shape==(7493,3) and
         np.all((selected>=0)&(selected<49)),
         "S2 frozen held-piece router choices wrong")
    replay2=rawfirst[np.arange(7493)[:,None],selected]
    need(np.array_equal(replay2,second["predictions"][positions]),
         "S2 exact stored source-policy decode mismatch")
    proofs.append({"family":"S2_source","rows":7493,
                   "historical_original_decision_mismatches":0,
                   "original_nested_choice_was_supervised":True})
    # Restore S3 original fitted nonlinear models (no refitting).
    states3=joblib.load(restored/"S3_repair_models_xz.joblib")
    rec3=json.loads((root/SOURCE/"series3/model-records.json").read_text())
    third=read(root/SOURCE/"series3/predictions.npz",("predictions","variant_ids"))
    actual3=third["predictions"][positions]
    need(len(actual3)==7493 and len(rec3["assignments"])>0,"S3 replay source")
    replay3=np.full((7493,42),-1,np.int8)
    for ri,rep in enumerate(("group_audit","acoustic58")):
        x,names=nonlinear_features(data,rep)
        need(names==rec3["features"][rep],"S3 original trained feature name drift")
        for a in [a for a in rec3["assignments"] if a["representation"]==rep and
                  a["role"]=="outer"]:
            t=np.flatnonzero(data["piece"]==a["test_piece"])
            need(len(t)>0 and a["model"] in states3,"S3 outer source model missing")
            rtree,qtree=tree_factors(data,x,t,states3[a["model"]])
            for ai,alpha in enumerate(BLENDS):
                r,cp,_,lg=blended(data,t,rtree,qtree,alpha)
                for ci,cost in enumerate(COSTS):
                    replay3[t,(ri*3+ai)*7+ci]=decode(
                       data["baseline"][t],data["available"][t],r,cp,lg,cost)
    need(np.isin(replay3,range(7)).all(),"S3 outer models missing events")
    need(np.array_equal(replay3,actual3[:,:42]),
         "S3 original restored nonlinear weights decision mismatch")
    proofs.append({"family":"S3_repair","rows":7493,
                   "trained_original_model_states":len(states3),
                   "historical_original_decision_mismatches":0})
    # Recreate S4 historic fitted M/N/neutral guard, using only its
    # pre-calibration outcomes and model inputs; never open target labels.
    states4=joblib.load(restored/"S4_repair_models_xz.joblib")
    rec4=json.loads((root/SOURCE/"series4/model-records.json").read_text())
    fourth=read(root/SOURCE/"series4/predictions.npz",("predictions","variant_ids"))
    actual4=fourth["predictions"][positions]
    need(actual4.shape[1]>=56 and len(states4)>0,"S4 saved guard source")
    b=data["baseline"].astype(int);old=data["original"].astype(int)
    changed=old!=b;rows=np.arange(len(b))
    base_prob=np.asarray(probs["class_probability"][:,0],np.float64)[rows,old]
    prior=np.column_stack([base_prob,probs["baseline_correct"][:,0],
                           np.maximum(0.,1-base_prob-probs["baseline_correct"][:,0])])
    prior[~changed]=[0,0,1]
    common=np.column_stack([prior,np.eye(7)[b][:,2:5],np.eye(7)[old][:,2:7],
                              data["group_context"][rows,old]])
    replay4=np.full((7493,56),-1,np.int8)
    for ri,representation in enumerate(("group","acoustic58")):
        x=common if ri==0 else np.column_stack([common,data["features"]])
        for a in [a for a in rec4["assignments"] if
                  a["representation"]==representation and a["role"]=="outer"]:
            t=np.flatnonzero(data["piece"]==a["test_piece"])
            need(len(t)>0 and a["model"] in states4,
                 "S4 original held-piece guard model missing")
            model=states4[a["model"]]
            empirical=np.zeros((len(t),3))
            empirical[:,model.classes_.astype(int)]=model.predict_proba(x[t])
            empirical[~changed[t]]=[0,0,1]
            for ai,w in enumerate(WEIGHTS):
                q=(1-w)*prior[t]+w*empirical
                for ci,cost in enumerate(COSTS):
                    replay4[t,(ri*4+ai)*7+ci]=np.where(
                       changed[t] & (q[:,0]>cost*q[:,1]),old[t],b[t])
    need(np.isin(replay4,range(7)).all(),"S4 outer models missing events")
    need(np.array_equal(replay4,actual4[:,:56]),
         "S4 original restored guard weights decision mismatch")
    proofs.append({"family":"S4_repair","rows":7493,
                   "trained_original_model_states":len(states4),
                   "historical_original_decision_mismatches":0})
    report={
      "status":"S174_RESTORED_ORIGINAL_S1_S2_S3_S4_NUMERICAL_REPLAY_NO_NEW_WAV",
      "data":"original exposed 7493 eligible events only; no native GOAT",
      "historical_regression_family_proofs":proofs,
      "historical_models_restored_and_forwarded":True,
      "historical_trueK_or_MIDI_read":False,
      "historical_255_group_inputs_used":True,
      "historical_risk_logits_used":True,
      "original_risk_logits_reproduced_on_new_WAV":False,
      "original_255_group_proposals_on_new_WAV":False,
      "new_audio_family_count_still":11,
      "original_S80_validated":False,
      "independent_ExactK_measured":False,
      "official_S80_promotion":False}
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in (
       "status","historical_regression_family_proofs",
       "new_audio_family_count_still","original_S80_validated")},indent=2))
    return report

def selftest():
    need(len(FAMILIES)==7 and len(COSTS)==7,"original S1 49 source policies")
    need(len(SAFE_INPUTS)==13 and not any(
       x.lower() in {"truth","true_k","y","labels","midi"} for x in SAFE_INPUTS),
       "label-bearing data loaded")
    need(len(BLENDS)==3 and len(WEIGHTS)==4,"original S3/S4 policies changed")
    print("S174 PASS label-blind archived model replay source contracts")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path("."))
    p.add_argument("--restored",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        need(a.out and a.restored,"require recovered original S172 model states")
        run(a.root,a.restored,a.out)
