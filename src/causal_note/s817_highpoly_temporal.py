"""S817 Active Exact-K private high-polyphony causal temporal specialist.

No imported FramePitch/Onset/Offset/ExactK weights or MIDI emissions. The
already independently trained S815 Active-K teacher is frozen. Two new
specialists learn high-polyphonic acoustic evidence K>=4 and K4/K5/K6.
Each current decision depends ONLY on the current and past MIDI256 features.
No future lookahead, post-onset persistence, buffering or retiming.
This is shadow-only research; physical runtime and LIVE remain unverified.
"""
from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
import hashlib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from causal_note.s814_activek_teacher import summarize, active_score

SCHEMA="S817_ACTIVEK_PRIVATE_CAUSAL_HIGH_POLYPHONY_V1"
LAGS=(1,4,16)
# A deterministic subset of global spectral morphology at each native 256 hop.
CONTEXT_INDICES=tuple(sorted(set(
    list(range(0,88,4))+list(range(1,88,4))
    +list(range(88,220,11))+[90,112,134,156,178,200,219])))
DIM=len(CONTEXT_INDICES)*(1+len(LAGS))
SELECTORS=("base","override","max_high")
THRESHOLDS=(.50,.70,.90)
PARAMS=dict(loss="log_loss",max_iter=60,max_leaf_nodes=17,max_depth=5,
            min_samples_leaf=55,learning_rate=.09,l2_regularization=2.,
            max_bins=63,early_stopping=False,categorical_features=None,
            class_weight=None)


class PrivateTemporalContext:
    """Re-usable state; arbitrary streamed partitions equal full-track pass."""
    def __init__(self):
        self.past=np.empty((0,len(CONTEXT_INDICES)),np.float32)

    def push(self,X):
        raw=np.asarray(X)
        if raw.ndim!=3 or raw.shape[1:]!=(49,222):
            raise ValueError("native fresh MIDI256 [T,49,222] required")
        if not np.isfinite(raw).all():
            raise ValueError("nonfinite acoustic observations")
        if not len(raw):
            return np.empty((0,DIM),np.float32)
        current=summarize(raw)[:,CONTEXT_INDICES]
        hist=np.concatenate((self.past,current),axis=0)
        offset=len(self.past)
        source_positions=np.arange(len(current),dtype=np.int64)+offset
        diffs=[]
        for lag in LAGS:
            history=hist[np.maximum(0,source_positions-lag)]
            diffs.append(current-history)
        self.past=hist[-LAGS[-1]:].copy()
        return np.ascontiguousarray(np.concatenate((current,*diffs),axis=1),
                                    dtype=np.float32)


def temporal(X):
    return PrivateTemporalContext().push(X)


def validate(rows,name):
    rows=sorted(list(rows),key=lambda r:r["name"])
    if not rows or len({r["name"] for r in rows})!=len(rows):
        raise ValueError(name+" distinct recordings required")
    for row in rows:
        x,k=np.asarray(row["X"]),np.asarray(row["k"])
        if x.ndim!=3 or x.shape[1:]!=(49,222) or not len(x) or k.shape!=(len(x),):
            raise ValueError("fresh MIDI256 acoustic and K0..6 label shape required")
        if not np.isin(k,range(7)).all() or not row.get("composition"):
            raise ValueError("invalid original K/composition")
    return rows


def stratified(rows,head,*,cap=600,seed=817):
    """Training data stratification with inverse-inclusion population weights.

    'gate' has true classes K<4 and K>=4, 'high' classes K4/K5/K6 only.
    """
    if head not in ("gate","high") or cap<12:
        raise ValueError("invalid specialist sample scheme")
    xs,ys,ws,proof=[],[],[],[]
    for row in validate(rows,"training"):
        raw=np.asarray(row["k"],np.int16)
        labels=(raw>=4).astype(np.uint8) if head=="gate" else raw
        allowed=(0,1) if head=="gate" else (4,5,6)
        selected,weights=[],[]
        rng=np.random.default_rng(seed+int.from_bytes(hashlib.sha256(
            (head+"/"+row["name"]).encode()).digest()[:4],"big"))
        context=temporal(row["X"])
        for label in allowed:
            subset=np.flatnonzero(labels==label)
            n=min(len(subset),cap//len(allowed))
            if n:
                sampled=np.sort(rng.choice(subset,n,replace=False))
                selected.append(sampled)
                weights.append(np.full(n,len(subset)/n,dtype=np.float64))
        if not selected:
            continue
        indices=np.concatenate(selected)
        importance=np.concatenate(weights)
        order=np.argsort(indices,kind="stable")
        indices,importance=indices[order],importance[order]
        xs.append(context[indices])
        ys.append(labels[indices])
        ws.append(importance)
        observed=np.bincount(labels,minlength=7 if head=="high" else 2)
        restored=np.bincount(labels[indices],weights=importance,
                             minlength=7 if head=="high" else 2)
        if head=="high":
            check=(4,5,6)
        else:
            check=(0,1)
        if any(not np.isclose(observed[i],restored[i],atol=1e-8,rtol=0)
               for i in check):
            raise AssertionError("population prior distorted")
        proof.append(dict(name=row["name"],natural={str(i):int(observed[i]) for i in check},
                          selected=len(indices)))
    if not xs:
        raise ValueError("missing training data")
    X,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws)
    if len(np.unique(y))<2:
        raise ValueError("insufficient independent classes to fit specialist")
    w/=w.mean()
    return X,y,w,proof


def proposals(X,gate,high):
    ctx=temporal(X)
    with threadpool_limits(limits=1):
        g=gate.predict_proba(ctx)
        h=high.predict_proba(ctx)
    if 1 not in gate.classes_:
        raise AssertionError("high-K gate missing true positive class")
    confidence=g[:,list(gate.classes_).index(1)].astype(np.float32)
    proposed=high.classes_[np.argmax(h,axis=1)].astype(np.uint8)
    if np.any((proposed<4)|(proposed>6)):
        raise AssertionError("specialist inferred a non-high class")
    return confidence,proposed


def select(base,confidence,proposal,selector,threshold):
    b=np.asarray(base,np.uint8)
    c=np.asarray(confidence)
    p=np.asarray(proposal,np.uint8)
    if b.ndim!=1 or b.shape!=c.shape or b.shape!=p.shape:
        raise ValueError("misaligned private teacher K decisions")
    if selector not in SELECTORS or threshold not in THRESHOLDS:
        raise ValueError("undeclared policy")
    if selector=="base":
        prediction=b.copy()
    elif selector=="override":
        prediction=np.where(c>=threshold,p,b)
    else:
        prediction=np.where(c>=threshold,np.maximum(b,p),b)
    return np.asarray(prediction,np.uint8)


@dataclass
class HighPolyTeacher:
    gate: HistGradientBoostingClassifier
    high: HistGradientBoostingClassifier
    selector: str
    threshold: float
    metadata: dict

    def __post_init__(self):
        if self.gate is self.high or self.selector not in SELECTORS or self.threshold not in THRESHOLDS:
            raise ValueError("independent trained high-polyphony heads required")

    @property
    def heads(self):
        return MappingProxyType({"high_gate":self.gate,"high_cardinality":self.high})

    def predict(self,X,private_activek_base):
        original=np.asarray(private_activek_base,np.uint8)
        if original.shape!=(len(X),):
            raise ValueError("base counts must be private Active-K axis only")
        confidence,high=proposals(X,self.gate,self.high)
        result=select(original,confidence,high,self.selector,self.threshold)
        result.setflags(write=False)
        return result


def fit_teacher(train_rows,cal_rows,base_teacher,*,seed=817):
    train=validate(train_rows,"train")
    cal=validate(cal_rows,"innercal")
    tc={r["composition"] for r in train}
    cc={r["composition"] for r in cal}
    if tc&cc or {r["name"] for r in train}&{r["name"] for r in cal}:
        raise ValueError("train vs cal must be composition-disjoint")
    gateX,gateY,gateW,gateProof=stratified(train,"gate",cap=600,seed=seed)
    highX,highY,highW,highProof=stratified(train,"high",cap=600,seed=seed+1)
    gate=HistGradientBoostingClassifier(**PARAMS,random_state=seed)
    high=HistGradientBoostingClassifier(**PARAMS,random_state=seed+1)
    with threadpool_limits(limits=1):
        gate.fit(gateX,gateY,sample_weight=gateW)
        high.fit(highX,highY,sample_weight=highW)
    choices={(selector,threshold):[0,0,0,0] for selector in SELECTORS
             for threshold in THRESHOLDS}
    for row in cal:
        base=base_teacher.predict(row["X"])
        conf,proposed=proposals(row["X"],gate,high)
        for (selector,threshold),tally in choices.items():
            predicted=select(base,conf,proposed,selector,threshold)
            m=active_score(predicted,row["k"])
            tally[0]+=m["correct"]
            tally[1]+=m["frames"]
            tally[2]+=m["overcount"]
            tally[3]+=m["undercount"]
    if len({v[1] for v in choices.values()})!=1:
        raise AssertionError("calibration population changed per policy")
    best=max(choices,key=lambda key:(
        choices[key][0]/max(choices[key][1],1),
        -choices[key][2]/max(choices[key][1],1),
        -SELECTORS.index(key[0]),-THRESHOLDS.index(key[1])))
    metadata=dict(schema=SCHEMA,train_compositions=sorted(tc),
        calibration_compositions=sorted(cc),heads=("high_gate","high_cardinality"),
        private_parameters=True,context_lags_samples=[x*256 for x in LAGS],
        feature_hop_samples=256,added_lookahead_samples=0,post_onset_hold_samples=0,
        trained_high_gate=True,trained_high_cardinality=True,
        gate_training=gateProof,high_training=highProof,
        selected=dict(selector=best[0],threshold=best[1]),
        calibration_results=[dict(selector=k[0],threshold=k[1],correct=v[0],
            frames=v[1],overcount=v[2],undercount=v[3]) for k,v in choices.items()],
        no_peer_characteristic_model_or_state=True)
    return HighPolyTeacher(gate,high,best[0],best[1],metadata)
