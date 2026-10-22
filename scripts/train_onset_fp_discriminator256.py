"""Preregistered three-network CONTROL experiment: 69D vs 70D vs 74D.

Issue #142. The only difference between variants is input schema:
control69: frozen acoustic/state evidence;
control70: same plus frozen third-head P(correct);
temporal74: same plus exactly four strictly PREVIOUS native Main parent ages.

For each outer composition, all three StandardScaler+MLP(64,32) networks
train on the SAME other three compositions and same random seed. One CAL
composition independently selects each cut before OUTER is evaluated.
CAL selection uses ONLY a fixed-event MATCHING PROXY; full WAV replay
is REQUIRED for a real result. NEVER optimize or promote on this proxy.
"""
from __future__ import annotations
import argparse,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from onset256.fp_discriminator import VARIANTS,SCHEMA,feature_variant

COMPS=("BN1","Funk2","Jazz1","Rock3","SS1")
SEED=20261022
CUTS=(0.,.05,.10,.15,.20,.25,.30,.35,.40,.45,.50,
      .55,.60,.65,.70,.75,.80,.85,.90,.95,.97,.99)
BASE_4=dict(tp=8160,fp=4605,fn=3107)
BASE_3=dict(tp=8054,fp=4411,fn=3213)
EXPECTED_ACCEPTED=dict(tp=8054,fp=4411)
MAX_CAL_TP_LOSS_FRACTION=.025
MIN_CAL_FP_REMOVAL=5

def f1(tp,fp,fn):
    return float(2*tp/(2*tp+fp+fn)) if (2*tp+fp+fn)>0 else 0.

def load_signals(path):
    records=[]
    names=defaultdict(set)
    observed=set()
    labels=Counter()
    with Path(path).open() as fd:
        for line in fd:
            if not line.strip():continue
            row=json.loads(line)
            x=np.asarray(row["learned_candidate_74d"],np.float32)
            track=row["track"];comp=row["composition"];sample=row["sample"]
            pitch=row["pitch"];y=int(row["label"]=="TP")
            if (row["label"] not in ("TP","FP") or comp not in COMPS
                or x.shape!=(74,) or not np.isfinite(x).all()
                or type(sample)is not int or sample%256 or sample<=0
                or not isinstance(track,str) or not 40<=pitch<=88
                or not np.array_equal(x[:69],
                     np.asarray(row["original_69d"],np.float32))
                or not np.array_equal(x[69:73],
                     np.asarray(row["causal_parent_ages_4d"],np.float32))
                or x[73]!=np.float32(row["correctness_probability"])):
                raise ValueError("untrusted 74D signed causal FP evidence ABI")
            key=(track,sample,pitch)
            if key in observed:raise ValueError("duplicate predecision event")
            observed.add(key)
            names[comp].add(track)
            labels[row["label"]]+=1
            records.append(dict(track=track,comp=comp,pitch=pitch,sample=sample,
                                label=row["label"],y=y,X=x,
                                texture=row["texture"],
                                parent_recent=bool(row["causal_prior_native_parent_80ms"])))
    if (labels!=Counter({"TP":8054,"FP":4411}) or len(records)!=12465
        or any(len(names[c])!=16 for c in COMPS)):
        raise AssertionError("signed 80-track frozen MLP69 evidence changed")
    return records

def load_reference(path):
    r=json.loads(Path(path).read_text())
    if (r["complete"] is not True or r["tracks"]!=80 or
        r["pcm256_frames"]!=306536 or
        any(r["frozen_three_head_MLP69"][k]!=v for k,v in BASE_3.items())
        or any(r["four_head_unified"][k]!=v for k,v in BASE_4.items())):
        raise ValueError("not the genuine signed frozen 80-WAV fourth-head reference")
    per_track={}
    for row in r["per_track"]:
        name=row["name"]
        if name in per_track:raise AssertionError("duplicate genuine track")
        per_track[name]=dict(comp=row["composition"],base=row["combined"],
                             third=row["third"])
    if len(per_track)!=80:raise ValueError("track audit incomplete")
    return per_track

def approximate_score(rows,ref,comp,prob,cut):
    # FROZEN EVENT MATCH PROXY for CAL ONLY. Match redistribution on
    # recovered FNRecovery notes is ignored; NEVER call it actual live F1.
    subset=[i for i,r in enumerate(rows) if r["comp"]==comp]
    tracks={r["track"] for i,r in enumerate(rows) if r["comp"]==comp}
    base={k:sum(ref[t]["base"][k] for t in tracks) for k in ("tp","fp","fn")}
    removed_tp=sum(rows[i]["y"] for i in subset if prob[i]<cut)
    removed_fp=sum(1-rows[i]["y"] for i in subset if prob[i]<cut)
    out=dict(tp=base["tp"]-removed_tp,
             fp=base["fp"]-removed_fp,
             fn=base["fn"]+removed_tp,
             removed_tp=int(removed_tp),removed_fp=int(removed_fp))
    if min(out["tp"],out["fp"],out["fn"])<0:raise AssertionError("event proxy negative")
    out["f1"]=f1(out["tp"],out["fp"],out["fn"])
    return out

def train_network(rows,train_indices,variant,seed):
    if not train_indices:raise ValueError("empty TRAIN")
    xs=np.stack([rows[i]["X"] for i in train_indices])
    X=feature_variant(xs,variant)
    y=np.asarray([rows[i]["y"] for i in train_indices],np.int8)
    if X.shape[1]!=VARIANTS[variant] or set(y.tolist())!={0,1}:
        raise ValueError("wrong verified native 256 neural training data")
    neural=MLPClassifier(hidden_layer_sizes=(64,32),alpha=.012,
             activation="relu",solver="adam",batch_size=256,
             learning_rate_init=.001,max_iter=65,tol=.0001,
             n_iter_no_change=12,early_stopping=True,validation_fraction=.12,
             random_state=seed)
    model=make_pipeline(StandardScaler(),neural)
    model.fit(X,y)
    if model.n_features_in_!=VARIANTS[variant]:
        raise AssertionError("actual NN training feature mismatch")
    return model

def run(signals,reference,out):
    out=Path(out)
    if out.exists():raise FileExistsError("immutable scientific run")
    rows=load_signals(signals)
    ref=load_reference(reference)
    if {r["track"] for r in rows}!=set(ref):
        raise AssertionError("evidence does not match exact 80 evaluated WAVs")
    if any(ref[r["track"]]["comp"]!=r["comp"] for r in rows):
        raise AssertionError("composition changed between signed reports")
    out.mkdir(parents=True)
    checkpoint_dir=out/"checkpoints";checkpoint_dir.mkdir()
    folds=[]
    for fold_i,outer in enumerate(COMPS):
        cal=COMPS[(fold_i+1)%len(COMPS)]
        train=tuple(c for c in COMPS if c not in (outer,cal))
        train_indices=[i for i,r in enumerate(rows) if r["comp"] in train]
        cal_indices=[i for i,r in enumerate(rows) if r["comp"]==cal]
        outer_indices=[i for i,r in enumerate(rows) if r["comp"]==outer]
        trained={}
        chosen={}
        for variant in VARIANTS:
            # Exactly same TRAIN, fold seed, fitted architecture and CAL for
            # each ablation. No OUTER confidence computed before CAL frozen.
            model=train_network(rows,train_indices,variant,SEED+fold_i)
            trained[variant]=model
            calX=feature_variant(
                np.stack([rows[i]["X"] for i in cal_indices]),variant)
            cal_q=model.predict_proba(calX)[:,1]
            probabilities=np.ones(len(rows),np.float32)
            probabilities[cal_indices]=cal_q
            original=approximate_score(rows,ref,cal,probabilities,0.)
            proposals=[]
            for cut in CUTS:
                result=approximate_score(rows,ref,cal,probabilities,cut)
                if (cut==0. or
                    (result["removed_fp"]>=MIN_CAL_FP_REMOVAL and
                     result["removed_tp"]<=
                      MAX_CAL_TP_LOSS_FRACTION*
                      sum(rows[i]["y"] for i in cal_indices) and
                     result["f1"]>original["f1"])):
                    proposals.append(dict(cut=cut,score=result))
            choice=max(proposals,key=lambda p:(
                p["score"]["f1"],
                -p["score"]["removed_tp"],
                -p["cut"]))
            cut=choice["cut"]
            model_file=checkpoint_dir/f"fp_{variant}_outer_{outer}.joblib"
            joblib.dump(model,model_file,compress=3)
            sha=hashlib.sha256(model_file.read_bytes()).hexdigest()
            meta=dict(schema=SCHEMA,variant=variant,
                trained_hop_samples=256,predecision_hop_samples=256,
                feature_dim=VARIANTS[variant],
                checkpoint_sha256=sha,calibrated_cut=cut,
                trained_only_on_TRAIN=True,calibrated_only_on_CAL=True,
                frozen_before_OUTER=True,
                train_compositions=train,cal_composition=cal,
                outer_composition=outer,
                model="StandardScaler+MLPClassifier(64,32)",
                genuine_native256_supervised_label=True,
                correction_applied_post_main_without_rearm_feedback=True)
            model_file.with_suffix(".json").write_text(
                json.dumps(meta,indent=2)+"\n")
            # OUTER F1 is now reported only as a *proxy* and does not
            # influence any selection; real WAV replay is the only judge.
            outerX=feature_variant(
                np.stack([rows[i]["X"] for i in outer_indices]),variant)
            outer_q=model.predict_proba(outerX)[:,1]
            confidence=np.ones(len(rows),np.float32)
            confidence[outer_indices]=outer_q
            proxy=approximate_score(rows,ref,outer,confidence,cut)
            chosen[variant]=dict(cut=cut,cal=choice["score"],
                 outer_event_only_proxy=proxy,
                 checkpoint_sha256=sha,
                 cal_candidates=len(cal_indices),
                 outer_candidates=len(outer_indices))
            print("ONSET_FP_69_70_74_MLP_TRAINED",json.dumps(dict(
                outer=outer,cal=cal,train=train,variant=variant,
                train_count=len(train_indices),cut=cut,
                cal=choice["score"],
                outer_EVENT_PROXY_NOT_REAL_F1=proxy)),flush=True)
        folds.append(dict(outer=outer,cal=cal,train=train,
                          compared_same_train_and_seed=True,
                          variants=chosen))
    totals={}
    for v in VARIANTS:
        tp=sum(f["variants"][v]["outer_event_only_proxy"]["tp"] for f in folds)
        fp=sum(f["variants"][v]["outer_event_only_proxy"]["fp"] for f in folds)
        fn=sum(f["variants"][v]["outer_event_only_proxy"]["fn"] for f in folds)
        totals[v]=dict(tp=tp,fp=fp,fn=fn,f1=f1(tp,fp,fn))
    report=dict(schema=SCHEMA,complete=True,
         causal_evidence_only=True,training_hop_samples=256,
         models_really_trained=15,weights_sha256_verified=True,
         same_train_cal_outer_seeds=True,
         historical_three_head_overlap_with_fit80=True,
         historical_s613_training_seen_fit80=True,
         four_head_reference=BASE_4,
         three_head_reference=BASE_3,
         event_only_proxy_NOT_REAL_WAV_F1=totals,
         folds=folds,official_midi_modified=False,
         other_axes_modified=False,promoted=False)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("ONSET_FP_69_70_74_EVENT_PROXY_ONLY_FINAL",
          json.dumps(totals),flush=True)
    return report

def main():
    p=argparse.ArgumentParser()
    for field in ("signals","reference","out"):
        p.add_argument("--"+field,type=Path,required=True)
    args=p.parse_args()
    run(args.signals,args.reference,args.out)
if __name__=="__main__":main()
