"""S662 — reproducible 80% NOTE Offset feasibility audit, and held-song residual head.

Baseline: frozen S425 note/pitch/onset/offsets on GuitarSet. Test the maximum
coverage of existing offset proposals, then train a *new* per-note regression
head ONLY on other composition groups. Crucial: oracle numbers are diagnostics,
NOT runtime scores. S425 upstream model training is not fully nested;
everything here is developmental and NOT strict Live.

Original 59309 S80 K0..K6 and all 33475 MIDI pitches/onsets remain untouched.
"""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from scripts.s425_songheld_select_S422_S420_offsets_no_K_regression import (
    read, labels_for_other_songs_only, make_features, need, SR,
    N_EVENTS,N_NOTES,N_GT,N_MATCHED,
)
from scripts.s402_k_input_first_supervised_midi_pitch_network import (
    composition_key,composition_fold,
)
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha

TOL=int(round(.050*SR))
BASE_HITS=16543
TARGET_80=int(np.ceil(.80*N_MATCHED))
ALPHAS=(.5,1.)
CLIP_MS=1000.
LONG_BOUNDARIES=(0,100,500,1000,5000,10**8)


def evidence(p):
    k=p["original_S80_K"].astype(np.int8)
    ids=p["original_S80_row_id"].astype(str)
    rows=p["source_row_id"].astype(str)
    counts=collections.Counter(rows)
    need(len(ids)==N_EVENTS and len(set(ids))==N_EVENTS and
         len(rows)==N_NOTES and np.isin(k,np.arange(7)).all() and
         sum(int(counts[x])==int(v) for x,v in zip(ids,k))==N_EVENTS,
         "S662 exact-K identity or cardinality corrupted")
    on=p["onset_samples"].astype(np.int64)
    old=p["S420_offset_samples"].astype(np.int64)
    surv=p["S422_offset_samples"].astype(np.int64)
    base=p["S425_offset_samples"].astype(np.int64)
    need(np.all(old>on) and np.all(surv>on) and np.all(base>on),
         "A MIDI note has an invalid closed interval")
    return on,old,surv,base


def candidates(on,old,surv,base):
    """Cross-model predictions and physically feasible interpolations.
    Oracle only: NEVER expose truth to selection or model features.
    """
    out=dict(S420=old,S422=surv,S425=base)
    for frac in (.25,.50,.75):
        mixed=np.rint(frac*old+(1-frac)*surv).astype(np.int64)
        out["interp_420_"+str(frac)]=mixed
    for frac in (.25,.50,.75):
        mixed=np.rint(frac*base+(1-frac)*surv).astype(np.int64)
        out["interp_425_"+str(frac)]=mixed
    for scale in (.75,.9,1.1,1.25,1.5):
        offset=on+np.rint((base-on)*scale).astype(np.int64)
        out["scale_425_"+str(scale)]=offset
    return {k:np.maximum(v,on+1) for k,v in out.items()}


def stats(pred,target,mask):
    diff=(pred[mask]-target[mask])/SR*1000.
    return dict(
        correct=int((np.abs(diff)<=50.).sum()),
        accuracy=float((np.abs(diff)<=50.).mean()),
        mae_ms=float(np.abs(diff).mean()),
        signed_median_ms=float(np.median(diff))
    )


def compute_oracles(cands,target,matched):
    near={name:(np.abs(value[matched]-target[matched])<=TOL)
          for name,value in cands.items()}
    base_names=("S420","S422","S425")
    raw=np.any(np.stack([near[n] for n in base_names]),axis=0)
    extended=np.any(np.stack(list(near.values())),axis=0)
    return dict(
      real_three_candidates=dict(hits=int(raw.sum()),
          accuracy=float(raw.mean())),
      diagnostic_interpolation_grid=dict(hits=int(extended.sum()),
          accuracy=float(extended.mean()),
          caution="ORACLE ONLY: selects by true offset; cannot run at inference"),
      baseline=S425_ORIG(near["S425"]),
      target_80=TARGET_80,
      missing_vs_three_oracle=max(0,TARGET_80-int(raw.sum())),
      missing_vs_grid_oracle=max(0,TARGET_80-int(extended.sum())),
    )

def S425_ORIG(mask):
    return dict(hits=int(mask.sum()),accuracy=float(mask.mean()))


def build_features(p,on,old,surv,base):
    tracks=p["track_member"].astype(str)
    pitches=p["MIDI"].astype(np.int16)
    source_ids=p["source_row_id"].astype(str)
    ids=p["original_S80_row_id"].astype(str)
    K=p["original_S80_K"].astype(np.int8)
    lut={key:int(value) for key,value in zip(ids,K)}
    event_k=np.array([lut[x] for x in source_ids],np.int8)
    X=make_features(tracks,pitches,on,old,surv,event_k)
    dur=np.maximum(base-on,1)/SR*1000
    d0=np.maximum(old-on,1)/SR*1000
    d1=np.maximum(surv-on,1)/SR*1000
    features=np.column_stack((X,
        np.log1p(dur),
        np.log1p(d0),np.log1p(d1),
        np.log1p(np.abs(dur-d0)),
        np.log1p(np.abs(dur-d1)),
        (dur-d0)/1000.,
        (dur-d1)/1000.,
        np.sign(dur-d0),
        np.sign(dur-d1),
    )).astype(np.float32)
    need(np.isfinite(features).all(),"S662 features nonfinite")
    return features


def fit_residual(X,target,on,base,train,seed):
    """Train from other songs only. No future GT enters held inference."""
    residual_ms=np.clip((target[train]-base[train])/SR*1000.,
                       -CLIP_MS,CLIP_MS)
    model=HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=.045,
        max_iter=115,
        max_leaf_nodes=15,
        min_samples_leaf=35,
        l2_regularization=12.,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(X[train],residual_ms)
    return model


def corrected(base,on,resid_ms,alpha):
    offset=base+np.rint(alpha*np.clip(resid_ms,-CLIP_MS,CLIP_MS)*
                         SR/1000.).astype(np.int64)
    return np.maximum(offset,on+1)


def stratify(target,on,matched,emissions):
    gt_dur=(target-on)/SR*1000
    out={}
    for left,right in zip(LONG_BOUNDARIES[:-1],LONG_BOUNDARIES[1:]):
        group=matched&(gt_dur>=left)&(gt_dur<right)
        if not group.any():continue
        out[f"{left}-{right}ms"]={
            name:dict(n=int(group.sum()),correct=int(
                (np.abs(v[group]-target[group])<=TOL).sum()))
            for name,v in emissions.items()
        }
    return out


def run(frozen,annotations,out):
    need(not out.exists(),"S662 evidence immutable; new output dir required")
    p=read(frozen)
    on,old,surv,base=evidence(p)
    target,gt=labels_for_other_songs_only(p,annotations)
    matched=target>0
    need(int(matched.sum())==N_MATCHED and
         int((np.abs(base[matched]-target[matched])<=TOL).sum())==BASE_HITS,
         "Frozen S425 16543/25395 baseline mismatch")
    cans=candidates(on,old,surv,base)
    oracle=compute_oracles(cans,target,matched)
    X=build_features(p,on,old,surv,base)
    songs=p["track_member"].astype(str)
    comp=np.asarray([composition_key(name) for name in songs])
    fold=np.asarray([composition_fold(c) for c in comp])
    need(len(np.unique(fold))==4,"Expected GuitarSet four composition folds")
    estimates={a:base.copy() for a in ALPHAS}
    fold_rows=[]
    for f in range(4):
        held=(fold==f)
        train=(fold!=f)&matched
        need(not(set(comp[train])&set(comp[held])) and
             train.sum()>10000 and held.sum()>4000,
             "Composition leakage or insufficient supervision")
        predictor=fit_residual(X,target,on,base,train,662+f)
        residual=predictor.predict(X[held])
        for a in ALPHAS:
            estimates[a][held]=corrected(base[held],on[held],
                                         residual,a)
        fold_rows.append(dict(
           fold=int(f),
           train_notes=int(train.sum()),
           held_notes=int(held.sum()),
           held_true_matched=int((held&matched).sum()),
           delta_correct_by_alpha={
              str(a):int(
                 (np.abs(estimates[a][held&matched]-
                    target[held&matched])<=TOL).sum()-
                 (np.abs(base[held&matched]-
                    target[held&matched])<=TOL).sum())
              for a in ALPHAS},
           trained_using_held_labels=False,
        ))
        print("S662 FOLD",json.dumps(fold_rows[-1]),flush=True)
    scores={
        "S420":stats(old,target,matched),
        "S422":stats(surv,target,matched),
        "S425":stats(base,target,matched),
    }
    emission=dict(S425=base)
    for a in ALPHAS:
        name=f"S662_residual_{a}"
        scores[name]=stats(estimates[a],target,matched)
        emission[name]=estimates[a]
    # No selection or deployment based on performance over held fold scores.
    oracle["baseline_raw_correct"]=scores["S425"]["correct"]
    oracle["target_80_additional"]=TARGET_80-scores["S425"]["correct"]
    strata=stratify(target,on,matched,emission)
    out.mkdir(parents=True)
    report=dict(
       status="S662_HELD_COMPOSITION_80_OFFSET_FEASIBILITY_RESEARCH",
       baselines_correct=dict(S420=16140,S422=16340,S425=16543),
       original_ExactK_59309_events_UNCHANGED=True,
       pitch_onset_33475_notes_UNCHANGED=True,
       no_live_integration=True,
       original_upstream_S80_overlap_with_GuitarSet=True,
       upstream_S425_not_fully_nested=True,
       goal_80_percent_achieved_on_this_dataset=bool(
          max(scores[name]["accuracy"] for name in emission)>=.80),
       oracle_NOT_an_achieved_result=True,
       selected_for_promotion=None,
       target_50ms_oracle=oracle,
       offset_metrics=scores,
       by_duration=strata,
       composition_folds=fold_rows,
       reference_artifact_sha256=sha(frozen),
       annotations_sha256=sha(annotations),
    )
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S662 ORACLE",json.dumps(oracle),flush=True)
    print("S662 SCORES",json.dumps(scores),flush=True)
    print("S662 FINAL",json.dumps(dict(
       baseline_acc=scores["S425"]["accuracy"],
       best_experimental=max(scores.values(),key=lambda x:x["accuracy"])["accuracy"],
       target=.8,no_promotion=True)),flush=True)


def self_test():
    on=np.array([0,0,0],np.int64)
    old=np.array([10,30,50],np.int64)
    surv=np.array([20,40,60],np.int64)
    base=np.array([10,40,50],np.int64)
    c=candidates(on,old,surv,base)
    assert set(("S420","S422","S425")).issubset(c)
    assert all((v>on).all() for v in c.values())
    print("S662 SELFTEST PASS: physical candidate grid, positive intervals")

if __name__=="__main__":
    q=argparse.ArgumentParser()
    q.add_argument("--frozen",type=Path)
    q.add_argument("--annotations",type=Path)
    q.add_argument("--out",type=Path)
    q.add_argument("--self-test",action="store_true")
    args=q.parse_args()
    if args.self_test:self_test()
    else:
        need(args.frozen and args.annotations and args.out,
             "Required --frozen --annotations --out")
        run(args.frozen,args.annotations,args.out)
