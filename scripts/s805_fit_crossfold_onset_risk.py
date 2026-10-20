"""S805: causal risk experts for synthetic NoteOn, real GuitarSet FIT80 OOF.

Scientific shadow only. Crossfit each risk head on S802 labels from OTHER
GuitarSet compositions; never train on a target composition, CAL32, HELD, TEST,
future frames, closure reason, or an oracle feature. Frozen independent
five-head maps remain intact. No source MIDI or native decoder mutation.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from causal_note.s795_activek_experts import HEADS, decode
from causal_note.s797_fragment_router import CUT
from scripts.s801_fit_fragment_ablations import route
from scripts.s800_oof_trace import SCHEMA as TRACE_SCHEMA, verify_snapshot
from scripts.s802_retrigger_forensics import SCHEMA as S802_SCHEMA
from scripts.s803_causal_frame_replay import SCHEMA as S803_SCHEMA
from scripts.s704_joint_metrics import SCORES, metric, aggregate
from scripts.s795_train_activek_midi256 import COUNT_KEYS

SCHEMA="S805_FIT80_CROSSFIT_CAUSAL_ONSET_RISK_V1"
POLICIES=("spectral_balance","risk_acoustic","risk_joint")
BASE={"spectral_balance":"consensus","risk_acoustic":"consensus",
      "risk_joint":"adaptive"}
FEATURES_ACOUSTIC=("pitch","on","sustain","off","on_rise",
                   "sustain_rise","harmonic_parent")
FEATURES_JOINT=FEATURES_ACOUSTIC+(
    "general_p","bass_p","polyphonic_p","transient_p","harmonic_p")
FEATURE_INDEX={"on":0,"sustain":1,"off":2,"on_rise":5,
               "sustain_rise":6,"harmonic_parent":10}
THRESHOLDS={"risk_acoustic":.30,"risk_joint":.30}
SOURCE_RUNS={"S800":39352880035,"S802":39359182007,"S803":39363062723}

def feature_matrix(X,P,columns):
    """Only the CURRENT acoustic frame and frozen independent expert heads."""
    if X.ndim!=3 or X.shape[1:]!=(49,23):
        raise ValueError("invalid S795 current-frame features")
    cells=X.shape[:2]
    vectors=[]
    for feature in columns:
        if feature=="pitch":
            value=np.broadcast_to(np.arange(40,89,dtype=np.float32)[None,:],cells)
        elif feature in FEATURE_INDEX:
            value=X[:,:,FEATURE_INDEX[feature]]
        else:
            name=feature.removesuffix("_p")
            if name not in HEADS or name not in P:raise ValueError("unknown head feature")
            value=P[name]
        if value.shape!=cells or not np.isfinite(value).all():
            raise ValueError("nonfinite current-frame feature")
        vectors.append(value)
    return np.stack(vectors,-1).reshape(-1,len(columns)).astype(np.float32)

def train_risk(foren,holdout,source_policy,columns):
    """Composition-OFF-fold supervised current-frame risk classifier."""
    if foren.get("schema")!=S802_SCHEMA or not foren.get("complete"):
        raise AssertionError("signed S802 observations unavailable")
    if source_policy not in ("consensus","adaptive"):
        raise ValueError("undeclared source policy")
    all_rows=foren["observations"][source_policy]
    train_rows=[x for x in all_rows if x["composition"]!=holdout]
    if len(train_rows)<50:
        raise AssertionError("insufficient independent training events")
    if any(x["composition"]==holdout for x in train_rows):
        raise AssertionError("current fold leaked into risk head")
    array=np.asarray([[float(x[k]) for k in columns] for x in train_rows],
                     dtype=np.float32)
    labels=np.asarray([int(x["synthetic_matched_truth"]) for x in train_rows],
                      dtype=np.int8)
    if set(np.unique(labels))!={0,1} or not np.isfinite(array).all():
        raise AssertionError("invalid training rows/classes")
    model=make_pipeline(StandardScaler(),
        LogisticRegression(C=.25,class_weight="balanced",max_iter=800,
                           random_state=805,solver="lbfgs"))
    model.fit(array,labels)
    witness=hashlib.sha256(np.ascontiguousarray(array).tobytes()+
                           np.ascontiguousarray(labels).tobytes()).hexdigest()
    provenance={"training_rows":len(train_rows),"pos":int(labels.sum()),
                "neg":int(len(labels)-labels.sum()),
                "training_compositions":sorted({x["composition"] for x in train_rows}),
                "excluded_evaluation_composition":holdout,
                "columns":list(columns),"training_features_sha256":witness,
                "C":.25,"class_weight":"balanced",
                "oracle_used_at_runtime":False,"heldout_annotations_opened":False}
    if holdout in provenance["training_compositions"]:
        raise AssertionError("crossfit leakage")
    return model,provenance

def gate(X,P,policy,model=None):
    """Do not delay MIDI; only mask current-frame proposals."""
    if policy not in POLICIES:raise ValueError("unregistered experiment")
    routed,masks,_=route(X,P,BASE[policy])
    if policy=="spectral_balance":
        accept=(X[:,:,0]-X[:,:,2]>=.08) | (X[:,:,6]>=.045) | (X[:,:,10]>=.4)
    else:
        if model is None:raise ValueError("risk head missing")
        cols=(FEATURES_ACOUSTIC if policy=="risk_acoustic" else FEATURES_JOINT)
        feats=feature_matrix(X,P,cols)
        confidence=model.predict_proba(feats)[:,1].reshape(X.shape[:2])
        if not np.isfinite(confidence).all():raise AssertionError("invalid risk map")
        accept=confidence>=THRESHOLDS[policy]
    blocked=int(np.count_nonzero((masks!=0)&~accept))
    gated={h:np.where(accept,routed[h],0.).astype(np.float32)
           for h in HEADS}
    return gated,blocked

def evaluate(manifest,traces,forensic,prior):
    manifest=json.loads(Path(manifest).read_text())
    forensic=json.loads(Path(forensic).read_text())
    prior=json.loads(Path(prior).read_text())
    if manifest.get("schema")!=TRACE_SCHEMA or not manifest.get("complete") or (
            not manifest.get("FIT_source_parity_S795")):
        raise AssertionError("signed S800 source missing")
    if forensic.get("schema")!=S802_SCHEMA or not forensic.get("complete"):
        raise AssertionError("S802 signed labels missing")
    if prior.get("schema")!=S803_SCHEMA or not prior.get("complete"):
        raise AssertionError("S803 signed metric baseline missing")
    if len(manifest.get("fit_files",[]))!=80 or (
        len({x["index"] for x in manifest["fit_files"]})!=80):
        raise AssertionError("not authentic FIT80 OOF")
    for x in (manifest,forensic,prior):
        if any(x.get(k,False) for k in ("CAL_opened","HELD_opened","TEST_opened")):
            raise AssertionError("reserved split inspected")
    comps=sorted({x["composition"] for x in manifest["fit_files"]})
    if len(comps)!=5:raise AssertionError("five GuitarSet OOF compositions required")
    models={};training={}
    for comp in comps:
        acoustic,ta=train_risk(forensic,comp,"consensus",FEATURES_ACOUSTIC)
        joint,tj=train_risk(forensic,comp,"adaptive",FEATURES_JOINT)
        models[comp]={"risk_acoustic":acoustic,"risk_joint":joint}
        training[comp]={"risk_acoustic":ta,"risk_joint":tj}
    scored={p:[] for p in POLICIES}
    baseline=[];blocked={p:0 for p in POLICIES};new_on={p:0 for p in POLICIES}
    for entry in sorted(manifest["fit_files"],key=lambda z:z["index"]):
        info=verify_snapshot(entry,traces)
        with np.load(Path(traces)/f"fit_{entry['index']:03d}.npz",
                     allow_pickle=False) as z:
            X=z["features"].astype(np.float32)
            times=z["decision_ends"]
            ends=z["metric_ends"]
            P={h:z["head_"+h].astype(np.float32) for h in HEADS}
        native=metric(info["source_events"],info["fit_truth"],ends)
        if any(int(native[k])!=int(info["source_counts"][k]) for k in COUNT_KEYS):
            raise AssertionError("signed original MIDI metric parity broken")
        baseline.append((info["composition"],native))
        for policy in POLICIES:
            m=models[info["composition"]].get(policy)
            prob,b=gate(X,P,policy,m)
            events,delta=decode(info["source_events"],times,X,prob,31,CUT,
                                int(info["samples"]))
            if any(delta.get(k,0)!=0 for k in (
                "max_added_native_delay_samples","extra_lookahead_samples",
                "extra_hold_samples","extra_confirmation_samples","extra_buffer_samples")):
                raise AssertionError("causal latency contract violated")
            if delta["new_onsets"]!=delta["new_offsets"]:
                raise AssertionError("synthetic NoteOn/NoteOff imbalance")
            scores=metric(events,info["fit_truth"],ends)
            scored[policy].append((info["composition"],scores))
            blocked[policy]+=b
            new_on[policy]+=int(delta["new_onsets"])
        print("S805_REAL_GUITARSET_FIT",entry["index"],info["composition"],flush=True)
    old=aggregate([x for _,x in baseline])
    if any(int(old[k])!=int(prior["source_counts"][k]) or
           int(old[k])!=int(manifest["S795_archived_fit_counts"][k])
           for k in COUNT_KEYS):raise AssertionError("baseline raw counters drift")
    if any(abs(old[k]-prior["baseline_fit"][k])>1e-12 for k in SCORES):
        raise AssertionError("baseline S803 six-metric parity drift")
    results={}
    for policy,rows in scored.items():
        curr=aggregate([x for _,x in rows])
        global_delta={k:float(curr[k]-old[k]) for k in SCORES}
        comp={}
        for group in comps:
            prev=aggregate([x for c,x in baseline if c==group])
            now=aggregate([x for c,x in rows if c==group])
            comp[group]={k:float(now[k]-prev[k]) for k in SCORES}
        safe=all(v>=-1e-12 for v in global_delta.values()) and all(
            all(v>=-1e-12 for v in d.values()) for d in comp.values())
        improved=any(v>1e-12 for v in global_delta.values())
        results[policy]={
            "six_scores":{k:float(curr[k]) for k in SCORES},
            "counts":{k:int(curr[k]) for k in COUNT_KEYS},
            "delta":global_delta,"by_composition_delta":comp,
            "blocked_current_frame_candidates":int(blocked[policy]),
            "emitted_synthetic_onsets":int(new_on[policy]),
            "no_regression_on_all_six_and_all_five_compositions":bool(safe),
            "fit_exploratory_eligible":bool(safe and improved)}
    return {"schema":SCHEMA,"complete":True,
        "dataset":"GuitarSet real FIT80 OOF, MIDI256 emitted events",
        "tracks":80,"compositions":comps,
        "source_runs":SOURCE_RUNS,
        "source_counts":{k:int(old[k]) for k in COUNT_KEYS},
        "baseline_fit":{k:float(old[k]) for k in SCORES},
        "policies":list(POLICIES),"policy_results":results,
        "train_provenance":training,"training_on_evaluation_composition":False,
        "S802_truth_used_for_crossfold_training_only":True,
        "features_current_frame_only":True,"risk_heads_independent_per_fold":True,
        "production_LIVE_modified":False,"physical_live_latency_measured":False,
        "added_hold_samples":0,"added_lookahead_samples":0,
        "official_reference":"S708","official_promotion":False,
        "CAL_opened":False,"HELD_opened":False,"TEST_opened":False,
        "threshold_search_performed":False,
        "fit_provisional_eligible":[k for k,v in results.items()
                                    if v["fit_exploratory_eligible"]]}

def main():
    p=argparse.ArgumentParser()
    for name in ("manifest","traces","forensic","prior","out"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("immutable scientific output exists")
    r=evaluate(a.manifest,a.traces,a.forensic,a.prior)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(r,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    print("S805_COMPLETED_REAL_FIT80",json.dumps({
        "eligible":r["fit_provisional_eligible"],
        "baseline":r["baseline_fit"],
        "metrics":{k:v["six_scores"] for k,v in r["policy_results"].items()}},
        ensure_ascii=False),flush=True)

if __name__=="__main__":main()
