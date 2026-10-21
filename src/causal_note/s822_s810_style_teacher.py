"""S822 re-train *S810-style* independent Active Exact-K at native MIDI256.

Two controlled supervised heads, SAME 23 predeclared acoustics, SAME frames,
same 48/16/16 composition-disjoint split, SAME LogisticRegression hyperparams:
  RAW23 = pitchwise mean of 23 acoustic coordinates
  AGG61 = exact shape and operation of S810 count_features on those coordinates.
The actual S810 input was 23 old S795 channels at 1024; our 23 are selected
from S813 unsupervised 222-dim PCM256. This is deliberately NOT bit-exact
historical S810 feature reproduction. No S708 event/posterior as inputs,
no peer task model, no hold/lookahead, no old thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from types import MappingProxyType

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

SCHEMA="S822_S810_STYLE_23_VS_61_SUPERVISED_ACTIVE_K_NATIVE256_V1"
SOURCE_COLS=(0,1,2,8,9,10,16,17,18,24,25,26,32,33,34,
             126,134,142,150,158,166,174,182)
MODES=("raw23","agg61")
PARAMS=dict(C=.5,max_iter=230,solver="lbfgs",class_weight=None,
            random_state=822)
PITCHES=49
HOP=256

def original_23(X):
    x=np.asarray(X)
    if (x.ndim!=3 or x.shape[1:]!=(49,222) or not len(x)
        or not np.isfinite(x).all()):
        raise ValueError("original finite PCM MIDI256 acoustic [T,49,222] required")
    # Advanced selection only materializes 23 coordinates from original 222.
    return np.asarray(x[:,:,SOURCE_COLS],dtype=np.float32)

def count_features_61(a):
    """Identical S810 count_features formula, on NEW acoustic23 observations.

    23 channel means + 23 maxima + top6 of channel1 + top5 channel0 +
    energy(sum channel12)/6 + 3 coarse register channel1 maxima = 61.
    """
    a=np.asarray(a,dtype=np.float32)
    if a.ndim!=3 or a.shape[1:]!=(49,23) or not np.isfinite(a).all():
        raise ValueError("23 acoustic channels per each of 49 pitches required")
    summary=np.concatenate((
        a.mean(axis=1),a.max(axis=1),
        np.sort(a[:,:,1],axis=1)[:,-6:],
        np.sort(a[:,:,0],axis=1)[:,-5:],
        a[:,:,12].sum(axis=1)[:,None]/6.,
        np.stack((a[:,:8,1].max(axis=1),
                  a[:,8:29,1].max(axis=1),
                  a[:,29:,1].max(axis=1)),axis=1)),axis=1)
    if summary.shape!=(len(a),61):
        raise AssertionError("S810 count_features summary shape drift")
    return np.ascontiguousarray(summary,dtype=np.float32)

def feats(X,mode):
    if mode not in MODES:raise ValueError("undeclared ablation mode")
    a=original_23(X)
    if mode=="raw23":
        return np.ascontiguousarray(a.mean(axis=1),dtype=np.float32)
    return count_features_61(a)

def sample_active(rows,mode,*,cap=1440,seed=822):
    """Per recording K1..6 subsampling, weights restore K1..6 natural priors."""
    if mode not in MODES or cap<24:raise ValueError("invalid sampling declaration")
    xs=[];ys=[];ws=[];proof=[]
    for r in sorted(rows,key=lambda x:x["name"]):
        X=np.asarray(r["X"])
        truth=np.asarray(r["k"],dtype=np.uint8)
        if truth.shape!=(len(X),) or not np.isin(truth,range(7)).all():
            raise ValueError("misaligned native256 annotation")
        name=r["name"]
        stamp=int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],"big")
        rng=np.random.default_rng(seed+stamp)
        groups=[];weights=[]
        for label in range(1,7):
            population=np.flatnonzero(truth==label)
            n=min(len(population),cap//6)
            if not n:continue
            indices=np.sort(rng.choice(population,n,replace=False))
            groups.append(indices)
            weights.append(np.full(n,len(population)/n,dtype=np.float64))
        if not groups:continue
        ii=np.concatenate(groups);ww=np.concatenate(weights)
        order=np.argsort(ii,kind="stable")
        ii,ww=ii[order],ww[order]
        freq=np.bincount(truth,minlength=7).astype(np.float64)
        restored=np.bincount(truth[ii],weights=ww,minlength=7)
        if not np.allclose(freq[1:],restored[1:],atol=1e-8,rtol=0):
            raise AssertionError("K1..6 training natural prior not restored")
        # Slice sparse samples before transforming costly float16 PCM tensors.
        xs.append(feats(X[ii],mode))
        ys.append(truth[ii]);ws.append(ww)
        proof.append(dict(name=name,selected_frames=len(ii),
            active_truth_frames=int(np.count_nonzero(truth)),
            class_prior_verified=True))
    if not xs:raise ValueError("empty training")
    x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws)
    if len(np.unique(y))<2:raise ValueError("class diversity absent")
    w=w/w.mean()
    return x,y,w,proof

@dataclass
class Teacher:
    model:object
    mode:str
    metadata:dict

    @property
    def heads(self):
        return MappingProxyType({self.mode:self.model})
    def predict(self,X):
        with threadpool_limits(limits=1):
            output=self.model.predict(feats(X,self.mode))
        out=np.asarray(output,dtype=np.uint8)
        if not np.isin(out,np.arange(1,7)).all():
            raise AssertionError("active-only logistic output outside K1..6")
        out.setflags(write=False)
        return out

def fit_teacher(train,mode,*,seed=822,cap=1440):
    if not train or len({r["name"] for r in train})!=len(train):
        raise ValueError("need distinct composition training")
    x,y,w,audit=sample_active(train,mode,seed=seed,cap=cap)
    model=make_pipeline(StandardScaler(),LogisticRegression(**PARAMS))
    with threadpool_limits(limits=1):
        model.fit(x,y,logisticregression__sample_weight=w)
    return Teacher(model,mode,dict(schema=SCHEMA,train_compositions=sorted({
        r["composition"] for r in train}),train_names=[r["name"] for r in train],
        fitted_feature_dim=x.shape[1],s810_formula_equal=True,
        s810_original_channels_equal=False,
        training_frames=len(y),sampling=audit,
        private_activeK_head=True,foreign_axis_weight_used=False,
        labels_used_during_inference=False,hop_samples=HOP,
        additional_future_samples=0,additional_hold_samples=0))

def choose(cal,teachers):
    if set(teachers)!=set(MODES):
        raise ValueError("must compare both ablations before CAL selection")
    train_comps=[set(x.metadata["train_compositions"]) for x in teachers.values()]
    cal_comps={r["composition"] for r in cal}
    if len(cal_comps)!=1 or any(cal_comps&c for c in train_comps):
        raise ValueError("CAL and TRAIN compositions overlapped")
    from causal_note.s814_activek_teacher import active_score
    counts={m:{"correct":0,"frames":0,"overcount":0,"undercount":0}
            for m in MODES}
    for row in cal:
        for mode,model in teachers.items():
            z=active_score(model.predict(row["X"]),row["k"])
            for key in counts[mode]:counts[mode][key]+=z[key]
    select=max(MODES,key=lambda m:(counts[m]["correct"]/max(counts[m]["frames"],1),
                  -counts[m]["overcount"],-MODES.index(m)))
    return select,counts
