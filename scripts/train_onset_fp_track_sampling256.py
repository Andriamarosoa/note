"""Hypothesis #144: CONTROLLED native256 FP 69D training sampling experiments.

Both choices fit genuine independent 69D neural models with EXACTLY the
original architecture and same number of TRAIN rows, only changing the
sampling distribution. Original 70D/74D weights and all other heads remain
unchanged. Splits 3TRAIN/1CAL/1OUTER by GuitarSet composition, cuts on CAL
ONLY. This script MUST NOT inspect model predictions on OUTER until replay.
"""
from __future__ import annotations
import argparse,hashlib,json,shutil
from collections import Counter,defaultdict
from pathlib import Path
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from onset256.fp_discriminator import SCHEMA,VARIANTS
from scripts.train_onset_fp_discriminator256 import (
    load_signals,load_reference,approximate_score,f1,
    COMPS,CUTS,SEED,MIN_CAL_FP_REMOVAL,MAX_CAL_TP_LOSS_FRACTION)

MODES=("track_uniform","track_class")
REAL_FROZEN_69D=dict(tp=8127,fp=4519,fn=3140)
REFERENCE_FOUR=dict(tp=8160,fp=4605,fn=3107)

def samples_train_only(rows,indices,mode,seed):
    if mode not in MODES:
        raise ValueError("unknown pre-registered training ablation")
    rng=np.random.default_rng(seed)
    train=[rows[i] for i in indices]
    counts=Counter(r["track"] for r in train)
    total=len(indices)
    if len(counts)!=48 or total<4000:
        raise AssertionError("expect exactly 48 TRUE TRAIN tracks")
    if mode=="track_uniform":
        p=np.asarray([1.0/counts[r["track"]] for r in train],np.float64)
        p/=p.sum()
        chosen=np.asarray(rng.choice(indices,size=total,replace=True,p=p),
                          dtype=np.int64)
    else:
        groups={}
        for lab in (0,1):
            ids=[i for i in indices if rows[i]["y"]==lab]
            bytrack=Counter(rows[i]["track"] for i in ids)
            if len(bytrack)<15:
                raise ValueError("FP class not represented in enough real tracks")
            prob=np.asarray([1.0/bytrack[rows[i]["track"]] for i in ids],
                            np.float64)
            prob/=prob.sum()
            k=total//2 if lab==1 else total-total//2
            groups[lab]=rng.choice(ids,size=k,replace=True,p=prob)
        chosen=np.concatenate((groups[0],groups[1])).astype(np.int64)
        rng.shuffle(chosen)
    if len(chosen)!=len(indices):
        raise AssertionError("training size differs from historical architecture")
    allowed_compositions={rows[j]["comp"] for j in indices}
    if any(rows[i]["comp"] not in allowed_compositions for i in chosen):
        raise AssertionError("training labels from an unrelated composition")
    return chosen,dict(
        mode=mode,trained_rows=int(len(chosen)),
        unique_examples_reused=int(len(set(chosen.tolist()))),
        original_training_events=int(len(indices)),
        original_unique_tracks=len(counts),
        selected_false_attacks=int(sum(rows[i]["y"]==0 for i in chosen)),
        selected_true_attacks=int(sum(rows[i]["y"]==1 for i in chosen)),
        max_original_track_events=max(counts.values()),
        min_original_track_events=min(counts.values()))

def fit_neural(rows,selected,seed):
    X=np.stack([rows[i]["X"][:69] for i in selected]).astype(np.float32)
    Y=np.asarray([rows[i]["y"] for i in selected],np.int8)
    if X.shape!=(len(selected),69) or set(Y.tolist())!={0,1}:
        raise ValueError("actual native256 69D training set not complete")
    estimator=MLPClassifier(
        hidden_layer_sizes=(64,32),alpha=.012,
        activation="relu",solver="adam",batch_size=256,
        learning_rate_init=.001,max_iter=65,tol=.0001,
        n_iter_no_change=12,early_stopping=True,validation_fraction=.12,
        random_state=seed)
    net=make_pipeline(StandardScaler(),estimator)
    net.fit(X,Y)
    if net.n_features_in_!=69:
        raise AssertionError("mismatch with frozen original learned 69D ABI")
    return net

def choose_CAL_cut(rows,ref,cal,model):
    ix=[i for i,r in enumerate(rows) if r["comp"]==cal]
    X=np.stack([rows[i]["X"][:69] for i in ix])
    prediction=np.ones(len(rows),np.float32)
    prediction[ix]=model.predict_proba(X)[:,1]
    base=approximate_score(rows,ref,cal,prediction,0.)
    possible=[]
    correct=sum(rows[i]["y"] for i in ix)
    for cut in CUTS:
        score=approximate_score(rows,ref,cal,prediction,cut)
        if (cut==0 or
            (score["removed_fp"]>=MIN_CAL_FP_REMOVAL and
             score["removed_tp"]<=MAX_CAL_TP_LOSS_FRACTION*correct and
             score["f1"]>base["f1"])):
            possible.append(dict(cut=cut,score=score))
    choice=max(possible,key=lambda x:(x["score"]["f1"],
                                -x["score"]["removed_tp"],-x["cut"]))
    return choice,base

def run(signals,four_reference,original_fp_models,mode,out):
    if mode not in MODES:
        raise ValueError("experimental arm must be predefined")
    out=Path(out)
    if out.exists():
        raise FileExistsError("scientific result must be immutable")
    rows=load_signals(signals)
    ref=load_reference(four_reference)
    if ({r["track"] for r in rows}!=set(ref) or
        any(ref[r["track"]]["comp"]!=r["comp"] for r in rows)):
        raise AssertionError("signed GT not 80 original GuitarSet WAV")
    out.mkdir(parents=True)
    root=out/"checkpoints";root.mkdir()
    original=Path(original_fp_models)/"checkpoints"
    folds=[]
    for fold_i,outer in enumerate(COMPS):
        cal=COMPS[(fold_i+1)%len(COMPS)]
        train=tuple(c for c in COMPS if c not in (outer,cal))
        ix=[i for i,r in enumerate(rows) if r["comp"] in train]
        selected,sampling=samples_train_only(rows,ix,mode,SEED+fold_i+144)
        neural=fit_neural(rows,selected,SEED+fold_i)
        choice,baseline=choose_CAL_cut(rows,ref,cal,neural)
        dst=root/f"fp_control69_outer_{outer}.joblib"
        joblib.dump(neural,dst,compress=3)
        sha=hashlib.sha256(dst.read_bytes()).hexdigest()
        meta=dict(
            schema=SCHEMA,variant="control69",trained_hop_samples=256,
            predecision_hop_samples=256,feature_dim=69,
            checkpoint_sha256=sha,calibrated_cut=float(choice["cut"]),
            trained_only_on_TRAIN=True,calibrated_only_on_CAL=True,
            frozen_before_OUTER=True,train_compositions=train,
            cal_composition=cal,outer_composition=outer,
            model="StandardScaler+MLPClassifier(64,32)",
            sampling_hypothesis_issue=144,
            sampling_mode=mode,
            independent_model_trained256=True,
            original_69D_only_sampling_changed=True,
            original_70D_and_74D_models_frozen=True,
            sampling_report=sampling,cal_only_proxy=choice["score"],
            original_CAL_proxy=baseline)
        dst.with_suffix(".json").write_text(json.dumps(meta,indent=2)+"\n")
        # Verify immutable controls: do NOT retrain 70D or temporal74.
        for v in ("control70","temporal74"):
            stem=original/f"fp_{v}_outer_{outer}"
            source=stem.with_suffix(".joblib")
            info=json.loads(stem.with_suffix(".json").read_text())
            if info["outer_composition"]!=outer or info["variant"]!=v:
                raise ValueError("control checkpoint does not match original split")
            if hashlib.sha256(source.read_bytes()).hexdigest()!=info["checkpoint_sha256"]:
                raise ValueError("frozen 70D/74D source model SHA drift")
            shutil.copy2(source,root/source.name)
            shutil.copy2(stem.with_suffix(".json"),
                         root/(stem.name+".json"))
        folds.append(dict(outer=outer,cal=cal,train=train,
            selected_cut=choice["cut"],selected_CAL_proxy=choice["score"],
            source_CAL_no_veto_proxy=baseline,
            sampler=sampling,checkpoint_sha256=sha))
        print("ONSET_FP_69D_TRACK_SAMPLING_TRUE_MLP_TRAINED",json.dumps(dict(
            outer=outer,mode=mode,cut=choice["cut"],
            sampled=sampling,cal_proxy=choice["score"])),flush=True)
    summary=dict(schema="ONSET_FP_TRACK_SAMPLING_TRUE256_ISSUE144_V1",
       complete=True,mode=mode,models_really_trained=5,
       training_hop_samples=256,
       only_training_sampling_changed=True,
       original_70D_74D_SHA_frozen=True,
       composition_holdout_untouched=True,
       baseline_control69=REAL_FROZEN_69D,
       baseline_four=REFERENCE_FOUR,
       folds=folds,
       candidate_CAL_proxy_only=True,
       real_WAV_replay_completed=False,
       source_69D_training_five_composition_folds=True,
       other_five_axes_unchanged=True,
       production_MIDI_unchanged=True,
       promoted=False)
    (out/"report.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary

def main():
    p=argparse.ArgumentParser()
    for arg in ("signals","four-reference","original-fp-models","out"):
        p.add_argument("--"+arg,required=True,type=Path)
    p.add_argument("--mode",required=True,choices=MODES)
    a=p.parse_args()
    run(a.signals,a.four_reference,a.original_fp_models,a.mode,a.out)
if __name__=="__main__":main()
