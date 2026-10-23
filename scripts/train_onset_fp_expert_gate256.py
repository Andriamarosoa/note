"""Learn 69D/74D expert arbitration on CAL-only disagreement cases, PCM256.

Issue #143 H_FP_EXPERT_GATE_02. Source experts are ALREADY trained and
SHA-frozen: three TRAIN compositions, one CAL, one OUTER. For each OUTER,
this script learns a genuine small neural gate on the 8 alternating CAL
tracks not reserved for gate validation; only the other 8 CAL tracks pick
a threshold. NO OUTER label, training feature, inference or benchmark
ever selects a weight, activation decision, or threshold.

The initial 69D,70D,74D controls themselves were CAL-calibrated on the
entire CAL composition before the gate experiment: this is disclosed.
The gate extra split is nested within CAL, not an external test.
"""
from __future__ import annotations
import argparse,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from onset256.fp_expert_gate import (
    SCHEMA,GATE_FEATURES,training_events)
from onset256.fp_discriminator import load_fp_discriminator,VARIANTS
from scripts.train_onset_fp_discriminator256 import (
    COMPS,load_signals,load_reference,f1)

SEED=20261023
CUTS=(.1,.2,.3,.4,.5,.6,.7,.8,.9,.95)
MIN_TRAIN_PER_CLASS=6
MIN_DISAGREEMENTS=20
MIN_VAL_FP_SAVED=2

def score_choice(rows,base_ref,valid_tracks,accepted):
    # Fixed-event CAL proxy only. Truth never enters the LIVE gate.
    actual=[r for r in rows if r["track"] in valid_tracks]
    if len(accepted)!=len(actual):
        raise AssertionError("gate predictions missing CAL validation proposal")
    val_base=[base_ref[t]["base"] for t in valid_tracks]
    tp=sum(r["tp"] for r in val_base)
    fp=sum(r["fp"] for r in val_base)
    fn=sum(r["fn"] for r in val_base)
    removed_tp=sum(r["y"] for r,keep in zip(actual,accepted) if not keep)
    removed_fp=sum(1-r["y"] for r,keep in zip(actual,accepted) if not keep)
    return dict(tp=tp-removed_tp,fp=fp-removed_fp,fn=fn+removed_tp,
                removed_tp=removed_tp,removed_fp=removed_fp,
                f1=f1(tp-removed_tp,fp-removed_fp,fn+removed_tp))

def load_frozen_controls(path,outer):
    root=Path(path)/"checkpoints"
    models={}
    cuts={}
    shas={}
    for name in VARIANTS:
        stem=root/f"fp_{name}_outer_{outer}"
        meta=json.loads(stem.with_suffix(".json").read_text())
        if (meta["outer_composition"]!=outer or
            outer in meta["train_compositions"] or
            outer==meta["cal_composition"] or
            meta["variant"]!=name):
            raise ValueError("frozen FP experts violate original 3TRAIN+CAL+OUTER")
        models[name]=load_fp_discriminator(
            stem.with_suffix(".joblib"),stem.with_suffix(".json"),
            expected_sha256=meta["checkpoint_sha256"])
        cuts[name]=float(meta["calibrated_cut"])
        shas[name]=meta["checkpoint_sha256"]
    return models,cuts,shas

def scores_and_features(rows,indices,models):
    if not indices:
        return np.empty((0,GATE_FEATURES),np.float32),{}
    matrix=np.stack([rows[i]["X"] for i in indices])
    p={}
    for variant in VARIANTS:
        if variant=="control69":z=matrix[:,:69]
        elif variant=="control70":z=np.column_stack((matrix[:,:69],matrix[:,73]))
        else:z=matrix
        p[variant]=np.asarray(models[variant].model.predict_proba(z)[:,1],np.float32)
    g=training_events(
        matrix,p["control69"],p["control70"],p["temporal74"],
        np.asarray([rows[i]["pitch"] for i in indices]))
    return g,p

def train_small_network(X,Y,seed):
    if (X.ndim!=2 or X.shape[1]!=GATE_FEATURES
        or int((Y==1).sum())<MIN_TRAIN_PER_CLASS
        or int((Y==0).sum())<MIN_TRAIN_PER_CLASS
        or len(Y)<MIN_DISAGREEMENTS):
        return None,dict(status="insufficient_class_distinct_disagreement",
                         events=len(Y),tp=int((Y==1).sum()),fp=int((Y==0).sum()))
    # Balance only CAL-gate TRAIN; never reweight OUTER or gate-validation.
    rng=np.random.default_rng(seed)
    i_pos=np.flatnonzero(Y==1)
    i_neg=np.flatnonzero(Y==0)
    n=max(len(i_pos),len(i_neg))
    sampled=np.concatenate((rng.choice(i_pos,size=n,replace=True),
                            rng.choice(i_neg,size=n,replace=True)))
    rng.shuffle(sampled)
    net=make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(16,8),alpha=.08,
          learning_rate_init=.0008,max_iter=120,batch_size=64,
          early_stopping=False,tol=.00005,n_iter_no_change=15,
          random_state=seed))
    net.fit(X[sampled],Y[sampled])
    if net.n_features_in_!=GATE_FEATURES:
        raise AssertionError("gate network not genuinely trained on native256 11D")
    return net,dict(status="genuine_neural_trained_CAL_only",
                    events=len(Y),tp=len(i_pos),fp=len(i_neg),
                    balanced_rows=len(sampled))

def selection(q69,q74,cut69,cut74,gate_q,gate_cut,enabled):
    left=np.asarray(q69,np.float32)>=cut69
    right=np.asarray(q74,np.float32)>=cut74
    if not enabled:return left
    g=np.asarray(gate_q,np.float32)
    if g.shape!=left.shape:raise ValueError("gate decision shape mismatch")
    return np.where(left!=right,g>=gate_cut,left)

def run(signals,source_fp_models,reference,out):
    out=Path(out)
    if out.exists():raise FileExistsError("do not rewrite results after OUTER")
    rows=load_signals(signals)
    ref=load_reference(reference)
    if ({r["track"] for r in rows}!=set(ref) or
        any(ref[r["track"]]["comp"]!=r["comp"] for r in rows)):
        raise AssertionError("FP calibration labels not on original exact 80 WAV")
    out.mkdir(parents=True)
    ck=out/"checkpoints";ck.mkdir()
    folds=[]
    for fold_i,outer in enumerate(COMPS):
        cal=COMPS[(fold_i+1)%len(COMPS)]
        source,cuts,sha=load_frozen_controls(source_fp_models,outer)
        tracks=sorted({r["track"] for r in rows if r["comp"]==cal})
        if len(tracks)!=16:
            raise AssertionError("CAL must contain exactly 16 signed tracks")
        gate_train=set(tracks[::2])
        gate_validate=set(tracks[1::2])
        assert gate_train.isdisjoint(gate_validate)
        train_indices=[i for i,r in enumerate(rows) if r["track"] in gate_train]
        valid_indices=[i for i,r in enumerate(rows) if r["track"] in gate_validate]
        X,q=scores_and_features(rows,train_indices,source)
        disagree=((q["control69"]>=cuts["control69"]) !=
                  (q["temporal74"]>=cuts["temporal74"]))
        features=X[disagree]
        labels=np.asarray([rows[i]["y"] for i in train_indices],np.int8)[disagree]
        neural,training=train_small_network(features,labels,SEED+fold_i)
        Xval,qv=scores_and_features(rows,valid_indices,source)
        control=selection(
            qv["control69"],qv["temporal74"],
            cuts["control69"],cuts["temporal74"],
            np.ones(len(Xval),np.float32),1.,False)
        ordered=[rows[i] for i in valid_indices]
        # Reorder source row list exactly as score_choice expects:
        # load_signals grouped by track content, valid_indices increasing,
        # which is the order preserved by the list comprehension.
        val_rows=[r for r in rows if r["track"] in gate_validate]
        if len(ordered)!=len(val_rows) or any(
            x is not y for x,y in zip(ordered,val_rows)):
            raise AssertionError("CAL gate event ordering drift")
        base=score_choice(rows,ref,gate_validate,control)
        choice=None
        val_candidates=0
        if neural is not None:
            disagree_val=((qv["control69"]>=cuts["control69"]) !=
                          (qv["temporal74"]>=cuts["temporal74"]))
            val_candidates=int(disagree_val.sum())
            neural_q=neural.predict_proba(Xval)[:,1]
            choices=[]
            for cut in CUTS:
                picked=selection(qv["control69"],qv["temporal74"],
                    cuts["control69"],cuts["temporal74"],
                    neural_q,cut,True)
                score=score_choice(rows,ref,gate_validate,picked)
                # Hard preregistered CAL-only safeguard. Not an OUTER score.
                if (score["fp"]<=base["fp"]-MIN_VAL_FP_SAVED and
                    score["f1"]>base["f1"]+1e-10):
                    choices.append(dict(cut=cut,score=score))
            if choices:
                choice=max(choices,key=lambda x:(x["score"]["f1"],
                     base["fp"]-x["score"]["fp"],-x["score"]["removed_tp"]))
        enabled=choice is not None
        gate_cut=float(choice["cut"]) if enabled else 1.
        checkpoint_file=ck/f"fp_gate_outer_{outer}.joblib"
        digest=None
        if neural is not None:
            joblib.dump(neural,checkpoint_file,compress=3)
            digest=hashlib.sha256(checkpoint_file.read_bytes()).hexdigest()
        metadata=dict(schema=SCHEMA,outer_composition=outer,
            original_cal_composition=cal,
            original_control_source_shas=sha,
            original_three_train_compositions=tuple(c for c in COMPS
                         if c not in (outer,cal)),
            gate_train_tracks=sorted(gate_train),
            gate_validation_tracks=sorted(gate_validate),
            gate_train_disagreement_count=int(disagree.sum()),
            gate_val_disagreement_count=int(val_candidates),
            trained_hop_samples=256,gate_input_dim=GATE_FEATURES,
            control69_cut=cuts["control69"],
            temporal74_cut=cuts["temporal74"],
            calibrated_cut=gate_cut,
            enabled=enabled,
            fitted_network_available=neural is not None,
            checkpoint_sha256=digest,
            trained_only_on_CAL_train=True,
            threshold_selected_only_on_CAL_validation=True,
            never_tuned_on_OUTER=True,
            training_report=training,cal_control69_proxy=base,
            cal_gate_proxy=choice["score"] if enabled else base,
            selected_only_if_val_f1_and_fp_improved=True,
            outer_replay_not_seen_before_selection=True)
        (ck/f"fp_gate_outer_{outer}.json").write_text(
            json.dumps(metadata,indent=2)+"\n")
        folds.append(metadata)
        print("ONSET_FP_GATE_CAL_SPLIT_FROZEN",json.dumps(dict(
            outer=outer,original_cal=cal,train=training,enabled=enabled,
            gate_cut=gate_cut,gate_val_disagreements=val_candidates,
            baseline_69_CAL=base,
            selected_GATE_CAL=metadata["cal_gate_proxy"])),flush=True)
    result=dict(schema=SCHEMA,complete=True,
         scientific_hypothesis_issue=143,
         PCM256_native=True,training_hop_samples=256,
         five_gate_CAL_splits=True,
         trained_gate_networks=sum(f["fitted_network_available"] for f in folds),
         enabled_gate_folds=sum(f["enabled"] for f in folds),
         CAL_never_OUTER_for_gate_training=True,
         pretrained_expert_thresholds_had_used_full_CAL=True,
         archived_previous_four_head=dict(tp=8160,fp=4605,fn=3107),
         archived_control69=dict(tp=8127,fp=4519,fn=3140),
         folds=folds,
         real_WAV_replay_completed=False,
         proxy_only_CAL_no_heldout_scoring=True,
         five_other_axes_untouched=True,
         MIDI_production_unchanged=True,promoted=False)
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    return result

def main():
    p=argparse.ArgumentParser()
    for arg in ("signals","source-fp-models","reference","out"):
        p.add_argument("--"+arg,type=Path,required=True)
    a=p.parse_args()
    return run(a.signals,a.source_fp_models,a.reference,a.out)
if __name__=="__main__":main()
