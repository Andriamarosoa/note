"""S806: genuinely new physical-fundamental *contrast* risk head from S800.

Only three ORIGINAL native PCM-spectrum summary components (S613 feature
indices 0/8/24 retained as S795 X[...,16:19]) and three history features
are available. This is not a newly computed full harmonic spectrum. The
remaining S800 fields are existing neural posteriors and cannot supply
missing raw overtones. The experiment tests a new cross-pitch, current/past
acoustic *representation* and leave-one-composition-out acoustic head.
Production live, source NoteOn/NoteOff and reference S708 stay untouched.

CAUTION: supervised source FIT onset labels may be used for training only on
the other four compositions. No CAL/HELD/TEST, no future frames or oracle
gating in inference. All 6 MIDI256 metrics must be evaluated without loss.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from causal_note.s795_activek_experts import HEADS,FEATURES,PITCHES,decode,regions
from causal_note.s797_fragment_router import CUT
from scripts.s800_oof_trace import SCHEMA as S800_SCHEMA,verify_snapshot
from scripts.s803_causal_frame_replay import SCHEMA as S803_SCHEMA
from scripts.s801_fit_fragment_ablations import route
from scripts.s704_joint_metrics import SCORES,metric,aggregate
from scripts.s795_train_activek_midi256 import COUNT_KEYS

SCHEMA="S806_FIT80_FUNDAMENTAL_OCTAVE_CONTRAST_OOF_V1"
POLICIES=("fundamental_consensus","fundamental_adaptive","fundamental_strict")
POLICY_BASE={"fundamental_consensus":"consensus",
             "fundamental_adaptive":"adaptive",
             "fundamental_strict":"consensus"}
# Predeclared BEFORE opening metrics; no fitting/tuning of threshold on FIT.
THRESHOLDS={"fundamental_consensus":.50,
            "fundamental_adaptive":.35,
            "fundamental_strict":.70}
FEATURE_NAMES=(
    "fundamental_2048","fundamental_4096","relative_fundamental_2048",
    "fundamental_mean","fundamental_interresolution_gap",
    "parent_12_contrast","parent_19_contrast","upper_12_contrast",
    "low_parent_peak","upper_peak",
    "fundamental_rise_1","fundamental_rise_2",
    "relative_fundamental_rise","physical_history_64",
    "physical_history_96","relative_pitch",)
MAX_POS_PER_TRACK=300
MAX_NEG_PER_TRACK=1800

def shift_pitch(a,semitones):
    """Current-frame frequency at a different pitch; zero-padded boundary."""
    out=np.zeros_like(a)
    if semitones>0:
        out[:,semitones:]=a[:,:-semitones]
    elif semitones<0:
        out[:,:semitones]=a[:,-semitones:]
    else:
        out[:]=a
    return out

def acoustic_features(X):
    """[frames,49,16] causal. X[...,16:22] are ORIGINAL native acoustic values.

    No posterior, expert head, annotation or next-frame value is a feature.
    Neighbour pitch at same time is received PCM, not a future event.
    """
    x=np.asarray(X,np.float32)
    if x.ndim!=3 or x.shape[1:]!=(PITCHES,FEATURES) or not np.isfinite(x).all():
        raise ValueError("S800 signed current-frame feature ABI")
    f2048=x[:,:,16]
    f4096=x[:,:,17]
    rel2048=x[:,:,18]
    fundamental=.5*(f2048+f4096)
    parent12=shift_pitch(fundamental,12)
    parent19=shift_pitch(fundamental,19)
    upper12=shift_pitch(fundamental,-12)
    past1=np.concatenate((fundamental[:1],fundamental[:-1]),0)
    past2=np.concatenate((fundamental[:1],fundamental[:1],fundamental[:-2]),0)[:len(x)]
    oldrel=np.concatenate((rel2048[:1],rel2048[:-1]),0)
    physical64=x[:,:,19]
    physical96=x[:,:,20]
    logcontrast=lambda base,parent:base-parent
    columns=(f2048,f4096,rel2048,fundamental,
             np.abs(f2048-f4096),
             logcontrast(fundamental,parent12),
             logcontrast(fundamental,parent19),
             logcontrast(fundamental,upper12),
             np.maximum(parent12,parent19),
             upper12,
             np.maximum(fundamental-past1,0),
             np.maximum(fundamental-past2,0),
             np.maximum(rel2048-oldrel,0),
             physical64,physical96,
             np.broadcast_to(np.arange(PITCHES,dtype=np.float32)[None,:]/48,
                             fundamental.shape))
    out=np.stack(columns,-1).astype(np.float32)
    if out.shape!=(len(x),PITCHES,len(FEATURE_NAMES)) or not np.isfinite(out).all():
        raise AssertionError("noncausal or wrong fundamental feature shape")
    return out

def rows_for_fit(X,truth,source,seed):
    """FIT truth is TRAINING-only. Deterministic balanced audit sample."""
    true=np.asarray(truth)
    if true.shape!=X.shape[:2] or true.dtype.kind not in "biu":
        raise AssertionError("signed FIT onset labels invalid")
    active=(X[:,:,12]>.5)
    if np.any(true & active):
        raise AssertionError("FIT onset target must exclude existing source voice")
    # The same eligibility definition is frozen on OTHER folds, no future.
    eligible=regions(X,active)["general"]
    acoustic=acoustic_features(X).reshape(-1,len(FEATURE_NAMES))
    pos=np.flatnonzero((eligible & (true>0)).ravel())
    neg=np.flatnonzero((eligible & (true==0)).ravel())
    rng=np.random.default_rng(seed)
    if len(pos)>MAX_POS_PER_TRACK:pos=rng.choice(pos,MAX_POS_PER_TRACK,replace=False)
    if len(neg)>MAX_NEG_PER_TRACK:neg=rng.choice(neg,MAX_NEG_PER_TRACK,replace=False)
    selected=np.sort(np.concatenate((pos,neg)))
    return acoustic[selected],np.isin(selected,pos).astype(np.uint8),{
        "positive_available":int(np.count_nonzero(eligible & (true>0))),
        "negative_available":int(np.count_nonzero(eligible & (true==0))),
        "positive_training":int(len(pos)),"negative_training":int(len(neg)),
        "selected_rows":int(len(selected))}

def fit_heldout(training,composition):
    picked=[r for r in training if r["composition"]!=composition]
    if len({r["composition"] for r in picked})!=4:
        raise AssertionError("not four independent FIT training compositions")
    xx=np.concatenate([r["X"] for r in picked],0)
    yy=np.concatenate([r["y"] for r in picked],0)
    if xx.shape[1]!=len(FEATURE_NAMES) or set(np.unique(yy))!={0,1}:
        raise AssertionError("incorrect acoustic onset fit sample")
    model=make_pipeline(StandardScaler(),
        LogisticRegression(C=.16,class_weight="balanced",
                           max_iter=500,random_state=806,solver="liblinear"))
    model.fit(xx,yy)
    return model,{
        "eval_composition":composition,
        "training_compositions":sorted({r["composition"] for r in picked}),
        "track_count":len(picked),
        "train_rows":len(yy),"positive":int(yy.sum()),
        "negative":int(len(yy)-yy.sum()),
        "training_features_sha256":hashlib.sha256(xx.tobytes()+yy.tobytes()).hexdigest(),
        "feature_names":list(FEATURE_NAMES),
        "no_future_or_target_composition":True,
        "threshold_frozen_before_scoring":True}

def gate(X,P,model,policy):
    """Masks only already-current proposals; native outputs are immutable."""
    if policy not in POLICIES:raise ValueError("unregistered policy")
    routed,masks,_=route(X,P,POLICY_BASE[policy])
    posterior=model.predict_proba(
        acoustic_features(X).reshape(-1,len(FEATURE_NAMES)))[:,1].reshape(X.shape[:2])
    if not np.isfinite(posterior).all():raise AssertionError("acoustic posterior finite")
    accepted=posterior>=THRESHOLDS[policy]
    routed={name:np.where(accepted,routed[name],0).astype(np.float32)
            for name in HEADS}
    return routed,int(np.count_nonzero((masks!=0)&~accepted))

def evaluate(manifest_path,traces,prior_path):
    m=json.loads(Path(manifest_path).read_text())
    prior=json.loads(Path(prior_path).read_text())
    if m.get("schema")!=S800_SCHEMA or not m.get("complete") or not m.get("FIT_source_parity_S795"):
        raise AssertionError("original S800 signed real FIT OOF needed")
    if prior.get("schema")!=S803_SCHEMA or not prior.get("complete"):
        raise AssertionError("genuine S803 baseline missing")
    if len(m.get("fit_files",[]))!=80 or len({x["index"] for x in m["fit_files"]})!=80:
        raise AssertionError("not 80 different signed FIT tracks")
    if any(obj.get(flag,False) for obj in (m,prior) for flag in
           ("CAL_opened","HELD_opened","TEST_opened")):
        raise AssertionError("forbidden evaluation material")
    fit_comps=sorted({x["composition"] for x in m["fit_files"]})
    if len(fit_comps)!=5:raise AssertionError("not five disjoint FIT compositions")
    train=[];stats=[]
    # A first pass builds NEW physical spectral examples from 80 independent
    # signed native acoustic snapshots. No labels on eval composition go into
    # its model, including scaling and sampling.
    for entry in sorted(m["fit_files"],key=lambda x:x["index"]):
        info=verify_snapshot(entry,traces)
        with np.load(Path(traces)/f"fit_{entry['index']:03d}.npz",
                     allow_pickle=False) as z:
            x=z["features"].astype(np.float32)
            label=z["onset_targets_fit"]
        xx,yy,s=rows_for_fit(x,label,x[:,:,12]>.5,806+int(entry["index"]))
        train.append(dict(composition=info["composition"],name=info["name"],
                          X=xx,y=yy,index=int(entry["index"])))
        stats.append(dict(index=int(entry["index"]),composition=info["composition"],
                          **s))
    models={};provenance={}
    for comp in fit_comps:
        models[comp],provenance[comp]=fit_heldout(train,comp)
    per={policy:[] for policy in POLICIES}
    native=[];blocked={policy:0 for policy in POLICIES}
    added={policy:0 for policy in POLICIES}
    for entry in sorted(m["fit_files"],key=lambda x:x["index"]):
        info=verify_snapshot(entry,traces)
        with np.load(Path(traces)/f"fit_{entry['index']:03d}.npz",allow_pickle=False) as z:
            X=z["features"].astype(np.float32)
            decisions=z["decision_ends"]
            metric_ends=z["metric_ends"]
            P={h:z["head_"+h].astype(np.float32) for h in HEADS}
        baseline=metric(info["source_events"],info["fit_truth"],metric_ends)
        if any(int(baseline[k])!=int(info["source_counts"][k]) for k in COUNT_KEYS):
            raise AssertionError("authentic source counters drift")
        native.append((info["composition"],baseline))
        for policy in POLICIES:
            routed,b=gate(X,P,models[info["composition"]],policy)
            events,delta=decode(info["source_events"],decisions,X,routed,
                                31,CUT,int(info["samples"]))
            if any(delta.get(k,0) for k in (
                "max_added_native_delay_samples","extra_lookahead_samples",
                "extra_hold_samples","extra_buffer_samples",
                "extra_confirmation_samples")):
                raise AssertionError("causal LIVE scheduling broken")
            if delta["new_onsets"]!=delta["new_offsets"]:
                raise AssertionError("unpaired synthetic NOTE")
            r=metric(events,info["fit_truth"],metric_ends)
            per[policy].append((info["composition"],r))
            blocked[policy]+=b;added[policy]+=int(delta["new_onsets"])
        print("S806_REAL_FIT80",entry["index"],info["composition"],flush=True)
    baseline=aggregate([x for _,x in native])
    if any(int(baseline[k])!=int(prior["source_counts"][k]) or
           int(baseline[k])!=int(m["S795_archived_fit_counts"][k])
           for k in COUNT_KEYS):raise AssertionError("12 signed source counts drift")
    if any(abs(baseline[k]-prior["baseline_fit"][k])>1e-12 for k in SCORES):
        raise AssertionError("source FIT six metrics mismatch")
    rows={}
    for policy,trial in per.items():
        scored=aggregate([r for _,r in trial])
        diff={k:float(scored[k]-baseline[k]) for k in SCORES}
        by_comp={}
        for comp in fit_comps:
            ref=aggregate([r for c,r in native if c==comp])
            now=aggregate([r for c,r in trial if c==comp])
            by_comp[comp]={k:float(now[k]-ref[k]) for k in SCORES}
        safe=all(x>=-1e-12 for x in diff.values()) and all(
            all(x>=-1e-12 for x in v.values()) for v in by_comp.values())
        improved=any(x>1e-12 for x in diff.values())
        rows[policy]=dict(
            metrics={k:float(scored[k]) for k in SCORES},
            counts={k:int(scored[k]) for k in COUNT_KEYS},
            delta=diff,by_composition_delta=by_comp,
            no_regression=bool(safe),
            fit_exploratory_eligible=bool(safe and improved),
            blocked_proposal_cells=int(blocked[policy]),
            synthetic_noteons=int(added[policy]))
    return dict(schema=SCHEMA,complete=True,dataset="GuitarSet REAL FIT80 OOF MIDI256",
        source_provenance={"S800":39352880035,"S803":39363062723},
        source_counts={k:int(baseline[k]) for k in COUNT_KEYS},
        baseline_fit={k:float(baseline[k]) for k in SCORES},
        track_count=80,compositions=fit_comps,heads_unchanged=list(HEADS),
        added_head="physical_fundamental_octave_contrast",
        feature_dimension=len(FEATURE_NAMES),
        features=list(FEATURE_NAMES),
        input_truth_on_target_composition_at_training=False,
        full_harmonic_spectrum_available=False,
        physical_spectral_proxy_from_frozen_S613_features=True,
        independent_fit80_acoustic_heads=5,
        training=provenance,sampling_stats=stats,
        policies=list(POLICIES),policy_results=rows,
        fit_provisional_eligible=[k for k,v in rows.items()
                                  if v["fit_exploratory_eligible"]],
        threshold_search_performed=False,
        runtime_truth_or_future_evidence=False,
        production_LIVE_modified=False,
        physical_live_latency_measured=False,
        added_latency_hold_samples=0,added_lookahead_samples=0,
        CAL_opened=False,HELD_opened=False,TEST_opened=False,
        official_reference="S708",official_promotion=False)

def main():
    a=argparse.ArgumentParser()
    for k in ("manifest","traces","prior","out"):
        a.add_argument("--"+k,type=Path,required=True)
    args=a.parse_args()
    if args.out.exists():raise FileExistsError("S806 immutable results already written")
    report=evaluate(args.manifest,args.traces,args.prior)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("S806_FIT80_COMPLETE",json.dumps(
        dict(eligible=report["fit_provisional_eligible"],
             native=report["baseline_fit"],
             trials={k:v["metrics"] for k,v in report["policy_results"].items()})),
             flush=True)

if __name__=="__main__":main()
