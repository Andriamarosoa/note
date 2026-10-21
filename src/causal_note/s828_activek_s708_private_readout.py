"""S828: *private Active Exact-K* fitted readout atop ORIGINAL frozen S708 at 256.

This does NOT fine-tune the shared S708 network. The unmodified original
S708 temporal/harmonic model runs freshly at each incoming PCM256, as proven
in S827; ONLY a private count readout and its CAL-chosen selector are fit.
Inputs: causal S708 288D acoustic evidence and PRIVATE Active-K count state.
No H0 MIDI, S795 stale posterior or any other axis head output is a feature.
No changes to Onset, Exact-K, Offset, Full Note, Frame Pitch or LIVE output.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import hashlib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from causal_note.s827_s708_activek_only import S827S708ActiveKOnly
from causal_note.s814_activek_teacher import active_score

SCHEMA="S828_S708_FROZEN_288D_PRIVATE_COUNT_READOUT_MIDI256_V1"
# Source S708 feature positions: 192D S704 temporal acoustics +96D S706 sharp.
# All sampled at CURRENT frame, each computed from current and past PCM only.
DIM_SELECT=tuple([*range(0,12),*range(48,60),*range(126,141),
                  *range(192,216),*range(248,264)])
MODES=("s708_private","learned_private","confidence_router")
SHIFTS=(0,-1,1)
MODEL_ARGS=dict(max_iter=115,max_depth=5,max_leaf_nodes=25,
                min_samples_leaf=120,learning_rate=.075,
                l2_regularization=2.0,max_bins=63,early_stopping=False)


def summarize_one(x,history,source_count,source_model_count,confidence):
    arr=np.asarray(x)
    if arr.shape!=(49,288) or not np.isfinite(arr).all():
        raise ValueError("required freshly recalculated S708 [49,288] at 256")
    sel=arr[:,DIM_SELECT]
    # Identical feature dimensions and rule each time; no access to future.
    vals=np.concatenate([
        sel.mean(axis=0),sel.std(axis=0),
        np.sort(sel,axis=0)[-3:].mean(axis=0),
        np.array([float(source_count),float(source_model_count),
                  float(confidence),float(history[-1]),
                  float(history[-2]),float(history[-4])],np.float32)
    ]).astype(np.float32)
    if not np.isfinite(vals).all():raise AssertionError("nonfinite readout")
    return vals


class S828Stream(S827S708ActiveKOnly):
    """Private S708 stream: no ON/OFF events emitted, own native256 vectors."""
    def __init__(self,native_heads,original_s708_model,frozen_policy):
        super().__init__(native_heads,original_s708_model,frozen_policy,
                         capture_features=True)
        self.readout_rows=[]
        self.s708_baselines=[]
        self.count_history=deque([0,0,0,0],maxlen=4)

    def _step(self,end,f):
        pred=super()._step(end,f)
        x=self.feature_debug.pop()
        row=summarize_one(x,self.count_history,pred.count,
                          pred.model_count,pred.model_confidence)
        self.readout_rows.append(row)
        self.s708_baselines.append(pred.count)
        self.count_history.append(pred.count)
        return pred

    def collected(self):
        x=np.stack(self.readout_rows)
        baseline=np.asarray(self.s708_baselines,dtype=np.uint8)
        if len(x)!=self.model_calls or len(baseline)!=len(x):
            raise AssertionError("S708 private model calls not at every 256")
        return x,baseline


def validate(rows,kind):
    rows=sorted(rows,key=lambda r:r["name"])
    if not rows or len(set(r["name"] for r in rows))!=len(rows):
        raise ValueError(kind+" rows must be distinct")
    size=None
    for r in rows:
        x=np.asarray(r["X"]);k=np.asarray(r["k"]);b=np.asarray(r["baseline"])
        if (x.ndim!=2 or len(x)==0 or x.shape[0]!=len(k) or
            len(k)!=len(b) or not np.isfinite(x).all() or
            not np.isin(k,range(7)).all() or not np.isin(b,range(7)).all() or
            not isinstance(r["composition"],str) or not r["composition"]):
            raise ValueError("bad native256 readout inputs")
        size=x.shape[1] if size is None else size
        if x.shape[1]!=size:raise ValueError("inconsistent 256 feature ABI")
    return rows


def stratified_train(rows,cap=2100,seed=828):
    xs,ys,ws=[],[],[]
    if cap<70:raise ValueError("cap must admit K0-K6")
    for r in validate(rows,"TRAIN"):
        y=np.asarray(r["k"])
        stable=int.from_bytes(hashlib.sha256(r["name"].encode()).digest()[:4],"big")
        rng=np.random.default_rng(seed+stable)
        picks=[];weights=[]
        for count in range(7):
            ids=np.flatnonzero(y==count)
            if len(ids):
                n=min(len(ids),cap//7)
                sampled=np.sort(rng.choice(ids,size=n,replace=False))
                picks.append(sampled)
                weights.append(np.full(n,len(ids)/n,np.float64))
        selection=np.concatenate(picks)
        weight=np.concatenate(weights)
        order=np.argsort(selection)
        selection=selection[order];weight=weight[order]
        natural=np.bincount(y,minlength=7)
        restored=np.bincount(y[selection],weights=weight,minlength=7)
        if not np.allclose(natural,restored,rtol=0,atol=1e-8):
            raise AssertionError("TRAIN class priors not restored")
        xs.append(np.asarray(r["X"][selection],np.float32))
        ys.append(np.asarray(y[selection],np.uint8))
        ws.append(weight)
    w=np.concatenate(ws);w/=w.mean()
    return np.concatenate(xs),np.concatenate(ys),w


def proposed(p,baseline,mode,shift):
    probs=np.asarray(p)
    old=np.asarray(baseline,dtype=np.int16)
    learned=np.argmax(probs,axis=1).astype(np.int16)
    if mode=="s708_private":s=old
    elif mode=="learned_private":s=learned
    elif mode=="confidence_router":
        s=np.where(probs.max(axis=1)>.72,learned,old)
    else:raise ValueError("unsupported mode")
    if shift not in SHIFTS:raise ValueError("unsupported shift")
    return np.clip(s+shift,0,6).astype(np.uint8)


@dataclass
class FittedActiveKOnly:
    model:HistGradientBoostingClassifier
    mode:str
    shift:int
    metadata:dict

    def predict(self,x,baseline):
        features=np.asarray(x)
        old=np.asarray(baseline)
        if features.ndim!=2 or old.shape!=(len(features),):
            raise ValueError("private readout requires aligned MIDI256 features")
        with threadpool_limits(limits=1):
            partial=self.model.predict_proba(features)
        scores=np.zeros((len(features),7),np.float32)
        scores[:,np.asarray(self.model.classes_,dtype=int)]=partial
        result=proposed(scores,old,self.mode,self.shift)
        result.setflags(write=False)
        return result


def fit_private(train,cal,*,seed=828):
    tr=validate(train,"TRAIN");ca=validate(cal,"CAL")
    tc={x["composition"] for x in tr}
    cc={x["composition"] for x in ca}
    if tc&cc or {x["name"] for x in tr}&{x["name"] for x in ca}:
        raise ValueError("TRAIN and inner CAL must be composition-disjoint")
    x,y,w=stratified_train(tr,seed=seed)
    if len(np.unique(y))<2:raise ValueError("private teacher requires >1 K class")
    model=HistGradientBoostingClassifier(**MODEL_ARGS,random_state=seed)
    with threadpool_limits(limits=1):model.fit(x,y,sample_weight=w)
    score={key:[0,0,0] for key in ((mode,shift) for mode in MODES for shift in SHIFTS)}
    for r in ca:
        with threadpool_limits(limits=1):part=model.predict_proba(r["X"])
        p=np.zeros((len(part),7),np.float32)
        p[:,np.asarray(model.classes_,dtype=int)]=part
        for key,counts in score.items():
            sc=active_score(proposed(p,r["baseline"],*key),r["k"])
            counts[0]+=sc["correct"];counts[1]+=sc["frames"]
            counts[2]+=sc["overcount"]
    best=max(score,key=lambda mode:(
        score[mode][0]/max(1,score[mode][1]),
        -score[mode][2]/max(1,score[mode][1]),
        -MODES.index(mode[0]),-SHIFTS.index(mode[1])))
    proof=dict(schema=SCHEMA,calibration=sorted(cc),train=sorted(tc),
        frozen_s708_original_weights=True,only_private_readout_fit=True,
        training_hop_samples=256,decision_hop_samples=256,
        added_lookahead_samples=0,added_hold_samples=0,
        selected_calibration_mode=best[0],selected_shift=best[1],
        calibration_scores={str(k):v for k,v in score.items()},
        no_other_axis_model_trained=True,not_public_midi=True)
    return FittedActiveKOnly(model,best[0],best[1],proof)
