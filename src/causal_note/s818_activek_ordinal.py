"""S818 — six independently fitted causal ordinal Active-K heads at MIDI256.

The six heads separately estimate P(K>=1), ... P(K>=6) from received
acoustics. Their weights are private to Active Exact-K, with no inputs from
the other five task axes. Dynamic per-current-frame ordinal aggregation
never waits for onset, frames, or release. FIT80 remains explored research;
no MIDI decoder, S708, S815, or LIVE runtime is modified by this module.
"""
from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
import hashlib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from causal_note.s814_activek_teacher import summarize, active_score
from causal_note.s817_highpoly_temporal import temporal

SCHEMA = "S818_ACTIVE_K_SIX_INDEPENDENT_ORDINAL_CAUSAL_MIDI256_V1"
LEVELS = (1, 2, 3, 4, 5, 6)
THRESHOLDS = (.30, .45, .60, .75)
MODES = ("crossing", "expected")
EXPECTED_SHIFTS = (-.35, 0.0, .35)
FEATURES = 220 + temporal(np.zeros((1,49,222),dtype=np.float16)).shape[1]
MODEL_ARGS = dict(loss="log_loss", max_iter=55, max_leaf_nodes=19,
                  max_depth=5, min_samples_leaf=40, l2_regularization=2.0,
                  learning_rate=.09, max_bins=63, early_stopping=False,
                  categorical_features=None, class_weight=None)

def audio_features(X):
    """Received PCM feature vectors for current and earlier MIDI256 hops only."""
    x=np.asarray(X)
    if x.ndim!=3 or x.shape[1:]!=(49,222) or not len(x) or not np.isfinite(x).all():
        raise ValueError("need nonempty finite fresh MIDI256 [T,49,222] acoustics")
    return np.ascontiguousarray(np.concatenate(
        (summarize(x), temporal(x)), axis=1),dtype=np.float32)


def training_samples(rows,level,*,cap_per_class=130,seed=818):
    """Binary rare-class stratification; inverse inclusion restores priors."""
    if level not in LEVELS or cap_per_class < 4:
        raise ValueError("invalid K level or cap")
    Xs,Ys,Ws,proof=[],[],[],[]
    for row in sorted(rows,key=lambda r:r["name"]):
        k=np.asarray(row["k"])
        audio=np.asarray(row["X"])
        if k.shape!=(len(audio),):
            raise ValueError("misaligned count target")
        y=(k>=level).astype(np.uint8)
        selected=[];weights=[]
        salt=int.from_bytes(hashlib.sha256(
            f"{level}/{row['name']}".encode()).digest()[:4],"big")
        rng=np.random.default_rng(seed+salt)
        for c in (0,1):
            at=np.flatnonzero(y==c)
            sample_n=min(len(at),cap_per_class)
            if sample_n:
                chosen=np.sort(rng.choice(at,size=sample_n,replace=False))
                selected.append(chosen)
                weights.append(np.full(sample_n,len(at)/sample_n,np.float64))
        if not selected: raise AssertionError("empty recording")
        idx=np.concatenate(selected)
        w=np.concatenate(weights)
        order=np.argsort(idx)
        idx,w=idx[order],w[order]
        restored=np.bincount(y[idx],weights=w,minlength=2)
        natural=np.bincount(y,minlength=2)
        if not np.allclose(restored,natural,rtol=0,atol=1e-8):
            raise AssertionError("K threshold prevalence distortion")
        Xs.append(audio_features(audio)[idx])
        Ys.append(y[idx]);Ws.append(w)
        proof.append(dict(record=row["name"],original_positive=int(natural[1]),
                          sampled_positive=int(np.count_nonzero(y[idx])),
                          original_frames=len(y),sampled_frames=len(idx)))
    X=np.concatenate(Xs);Y=np.concatenate(Ys);W=np.concatenate(Ws)
    if len(np.unique(Y))!=2:
        raise ValueError(f"cannot train K>={level}: only one class")
    W /= W.mean()
    return X,Y,W,proof


def ordinal_probabilities(X,heads):
    observed=audio_features(X)
    logits=[]
    with threadpool_limits(limits=1):
        for head in heads:
            if tuple(head.classes_)!=(0,1):
                raise AssertionError("ordinal model missing positive/negative class")
            logits.append(head.predict_proba(observed)[:,1])
    # Enforce P(K>=j+1) <= P(K>=j), physically coherent ordinal survival.
    cumulative=np.minimum.accumulate(np.stack(logits,axis=1),axis=1)
    if cumulative.shape!=(len(X),6):
        raise AssertionError("six independent ordinal outputs absent")
    return cumulative.astype(np.float32)


def decide(proba,mode,parameter):
    p=np.asarray(proba)
    if p.ndim!=2 or p.shape[1]!=6 or not np.isfinite(p).all():
        raise ValueError("required six finite K>=j probabilities")
    if mode=="crossing" and parameter in THRESHOLDS:
        k=(p>=parameter).sum(axis=1)
    elif mode=="expected" and parameter in EXPECTED_SHIFTS:
        k=np.rint(p.sum(axis=1)+parameter)
    else:
        raise ValueError("unknown fixed ordinal selection policy")
    return np.clip(k,0,6).astype(np.uint8)


@dataclass
class OrdinalTeacher:
    models: tuple
    mode: str
    parameter: float
    metadata: dict

    def __post_init__(self):
        if len(self.models)!=6 or len({id(m) for m in self.models})!=6:
            raise ValueError("six separately fitted heads required")
        if (self.mode,self.parameter) not in policy_grid():
            raise ValueError("undeclared private calibration selection")

    @property
    def heads(self):
        return MappingProxyType({f"k_ge_{k}":self.models[k-1] for k in LEVELS})

    def predict_proba(self,X):
        z=ordinal_probabilities(X,self.models)
        z.setflags(write=False)
        return z

    def predict(self,X):
        value=decide(self.predict_proba(X),self.mode,self.parameter)
        value.setflags(write=False)
        return value


def policy_grid():
    return tuple([("crossing",t) for t in THRESHOLDS]
                 + [("expected",b) for b in EXPECTED_SHIFTS])


def fit_teacher(train_rows,cal_rows,*,seed=818,cap=130,model_iter=55):
    train=sorted(list(train_rows),key=lambda r:r["name"])
    cal=sorted(list(cal_rows),key=lambda r:r["name"])
    if not train or not cal or len({r["name"] for r in train+cal})!=len(train+cal):
        raise ValueError("distinct disjoint recordings required")
    trc={r["composition"] for r in train}
    cac={r["composition"] for r in cal}
    if trc & cac:
        raise ValueError("train/calibration must be composition-disjoint")
    for row in train+cal:
        X=np.asarray(row["X"]); k=np.asarray(row["k"])
        if X.ndim!=3 or X.shape[1:]!=(49,222) or k.shape!=(len(X),):
            raise ValueError("unaligned original PCM256 and K labels")
        if not np.isin(k,range(7)).all():
            raise ValueError("target must be original capped K0..6")
    models=[];sampling=[]
    for level in LEVELS:
        x,y,w,audit=training_samples(train,level,cap_per_class=cap,seed=seed)
        params=dict(MODEL_ARGS,max_iter=model_iter)
        head=HistGradientBoostingClassifier(**params,random_state=seed+level)
        with threadpool_limits(limits=1):
            head.fit(x,y,sample_weight=w)
        models.append(head)
        sampling.append(dict(level=level,recordings=audit,
                             training_rows=len(y),natural_positive=float(np.dot(y,w)/w.sum())))
    choices={(mode,p):[0,0,0,0] for mode,p in policy_grid()}
    for row in cal:
        probabilities=ordinal_probabilities(row["X"],tuple(models))
        for selection,aggregates in choices.items():
            v=active_score(decide(probabilities,*selection),row["k"])
            for i,name in enumerate(("correct","frames","overcount","undercount")):
                aggregates[i]+=v[name]
    frames={v[1] for v in choices.values()}
    if len(frames)!=1 or not next(iter(frames)):
        raise AssertionError("fixed inner calibration Active-K denominator")
    best=max(choices,key=lambda k:(
        choices[k][0]/choices[k][1],
        -choices[k][2]/choices[k][1],
        -policy_grid().index(k)))
    proof=dict(schema=SCHEMA,train_compositions=sorted(trc),
               calibration_compositions=sorted(cac),
               train_tracks=len(train),calibration_tracks=len(cal),
               trained_private_heads=[f"k_ge_{k}" for k in LEVELS],
               independent_model_object_ids_verified=len({id(x) for x in models})==6,
               class_prior_sampling=sampling,
               selected=dict(mode=best[0],parameter=best[1]),
               calibration_results=[dict(mode=m,parameter=p,correct=a[0],
                   frames=a[1],overcount=a[2],undercount=a[3])
                   for (m,p),a in choices.items()],
               future_lookahead_samples=0,post_onset_hold_samples=0,
               feature_hop_samples=256,only_private_activek_audio_inputs=True,
               other_five_task_heads_untouched=True)
    return OrdinalTeacher(tuple(models),best[0],best[1],proof)
