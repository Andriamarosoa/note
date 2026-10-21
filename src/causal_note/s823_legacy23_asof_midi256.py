"""S823 diagnostic: real S795 23-channel OOF features causally carried to MIDI256.

S795's original inputs include S708 posteriors, count estimates, and already
emitted MIDI. They were observed ONLY at 1024-sample frame ends, so this
as-of replay does NOT convert them into 256-sample audio observations.
No future availability, no MIDI mutation, NEVER deploy or promote.
"""
from __future__ import annotations
import hashlib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from causal_note.s810_trained_six_axis_shadow import count_features as old61

SCHEMA="S823_S795_SIGNED_LEGACY_ASOF1024_TO_MIDI256_SHADOW_V1"
MODES=("raw23","agg61")
HOP=256
LEGACY_HOP=1024

def asof_indices(decision_ends,metric_ends):
    d=np.asarray(decision_ends,dtype=np.int64)
    e=np.asarray(metric_ends,dtype=np.int64)
    if (not len(d) or not len(e)
        or np.any(np.diff(d)<=0) or np.any(np.diff(e)<=0)
        or np.any(d%LEGACY_HOP) or np.any(e%HOP)
        or d[0]<LEGACY_HOP or e[0]!=HOP
        or not np.isin(np.diff(e),[HOP]).all()):
        raise ValueError("S795 original 1024 ends and MIDI256 original ends required")
    i=np.searchsorted(d,e,side="right")-1
    valid=i>=0
    if np.any(d[np.maximum(i,0)][valid]>e[valid]):
        raise AssertionError("noncausal feature time supplied")
    return i.astype(np.int32)

def feature_at(original,indices,mode):
    x=np.asarray(original)
    ii=np.asarray(indices,dtype=np.int64)
    if mode not in MODES or x.ndim!=3 or x.shape[1:]!=(49,23):
        raise ValueError("original S795 49x23 and declared modes only")
    if not np.isfinite(x).all() or np.any(ii>=len(x)) or np.any(ii< -1):
        raise ValueError("invalid original acoustic+S708 tensors or time indices")
    a=np.zeros((len(ii),49,23),dtype=np.float32)
    valid=ii>=0
    a[valid]=x[ii[valid]]
    if mode=="raw23":
        out=a.mean(axis=1)
    else:
        out=old61(a)
    return np.ascontiguousarray(out,dtype=np.float32)

def balanced_signed_sample(rows,mode,*,cap=840,seed=823):
    """K0..K6 labels training-only; inverse frame selection weighting."""
    if cap<28:raise ValueError("at least 4 sampled frames per class")
    features=[];labels=[];weights=[];proof=[]
    for row in sorted(rows,key=lambda x:x["name"]):
        k=np.asarray(row["k"],dtype=np.uint8)
        ix=np.asarray(row["asof"],dtype=np.int32)
        if len(k)!=len(ix) or not np.isin(k,range(7)).all():
            raise ValueError("time/labels mismatch")
        s=int.from_bytes(hashlib.sha256(row["name"].encode()).digest()[:4],"big")
        rng=np.random.default_rng(seed+s)
        selected=[];sample_weight=[]
        for c in range(7):
            poss=np.flatnonzero(k==c)
            n=min(len(poss),cap//7)
            if not n:continue
            take=np.sort(rng.choice(poss,n,replace=False))
            selected.append(take)
            sample_weight.append(np.full(n,len(poss)/n,np.float64))
        if not selected:continue
        j=np.concatenate(selected);w=np.concatenate(sample_weight)
        order=np.argsort(j,kind="stable");j=j[order];w=w[order]
        if not np.allclose(np.bincount(k[j],weights=w,minlength=7),
                           np.bincount(k,minlength=7),rtol=0,atol=1e-8):
            raise AssertionError("class prior altered by subsampling")
        features.append(feature_at(row["X"],ix[j],mode))
        labels.append(k[j]);weights.append(w)
        proof.append(dict(name=row["name"],natural_frames=len(k),
                          subsampled=len(j),asof_legacy_only=True))
    if not features:raise AssertionError("no training examples")
    a=np.concatenate(features);y=np.concatenate(labels);w=np.concatenate(weights)
    if len(np.unique(y))<2:raise ValueError("binary-only labels")
    w/=w.mean()
    return a,y,w,proof

class Teacher:
    def __init__(self,model,mode,meta):
        self.model,self.mode,self.metadata=model,mode,meta
    def predict(self,row,*,batch=2048):
        pred=[]
        for start in range(0,len(row["asof"]),batch):
            xx=feature_at(row["X"],row["asof"][start:start+batch],self.mode)
            with threadpool_limits(limits=1):
                pred.append(self.model.predict(xx).astype(np.uint8))
        answer=np.concatenate(pred)
        if not np.isin(answer,np.arange(7)).all():
            raise AssertionError("prediction not in K0..K6")
        answer.setflags(write=False)
        return answer

def fit_teacher(rows,mode,*,cap=840,seed=823):
    if mode not in MODES or not rows or len({r["name"] for r in rows})!=len(rows):
        raise ValueError("distinct records and mode")
    x,y,w,proof=balanced_signed_sample(rows,mode,cap=cap,seed=seed)
    model=make_pipeline(StandardScaler(),LogisticRegression(
        C=.5,max_iter=230,solver="lbfgs",class_weight=None,random_state=823))
    with threadpool_limits(limits=1):
        model.fit(x,y,logisticregression__sample_weight=w)
    metadata=dict(schema=SCHEMA,mode=mode,feature_dim=x.shape[1],
        train_compositions=sorted({r["composition"] for r in rows}),
        trained_K0_to_K6=True,features_reused_from_S800_S795=True,
        depends_on_original_S708_posteriors_and_MIDI=True,
        source_feature_period_samples=1024,decision_period_samples=256,
        reextracted_PCM_native256=False,peer_axis_independence=False,
        added_lookahead_samples=0,added_hold_samples=0,
        original_MIDI_modified=False,production_LIVE_modified=False,
        training_rows=len(y),source_proof=proof,promotion=False)
    return Teacher(model,mode,metadata)

def calibrate(rows,models):
    from causal_note.s814_activek_teacher import active_score
    if set(models)!=set(MODES) or len({r["composition"] for r in rows})!=1:
        raise ValueError("all modes + one calibration composition")
    if any({r["composition"] for r in rows} & set(x.metadata["train_compositions"])
            for x in models.values()):
        raise ValueError("CAL overlaps TRAIN")
    totals={m:dict(correct=0,frames=0,undercount=0,overcount=0) for m in MODES}
    for r in rows:
        for m,teacher in models.items():
            v=active_score(teacher.predict(r),r["k"])
            for key in totals[m]:totals[m][key]+=v[key]
    best=max(MODES,key=lambda m:(totals[m]["correct"]/max(totals[m]["frames"],1),
           -totals[m]["overcount"],-MODES.index(m)))
    return best,totals
