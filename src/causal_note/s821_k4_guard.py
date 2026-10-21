"""S821 private K4->K5/K6 false-addition verifier, strict causal MIDI256.

Two NEW gradient-boosted binary heads on acoustic pitchwise evidence:
  p5=P(true K >= 5 | true K>=3), p6=P(true K>=6 | true K>=4).
Original S815 and S817 are FROZEN, used only for private Active-K
candidate proposals. No peer-axis output, weight/state, future PCM, hold,
retiming or MIDI emission. Composition-disjoint 48 TRAIN / 16 CAL / 16
TEST; calibration selects one predeclared rule including BASE.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
import hashlib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from causal_note.s814_activek_teacher import active_score

SCHEMA="S821_ACTIVEK_PRIVATE_PITCHWISE_K4_FALSE_ADDITION_GUARD_V1"
THRESHOLDS=(.40,.55,.70,.85)
MODES=("base","confirm5","verify5","verify56")
MODEL_PARAMS=dict(loss="log_loss",max_iter=64,learning_rate=.085,
                  max_leaf_nodes=18,max_depth=5,min_samples_leaf=45,
                  l2_regularization=2.,max_bins=63,early_stopping=False,
                  class_weight=None,categorical_features=None)

def _column_signature(X):
    """Causal native-256 acoustic signature preserving pitch *structure*.

    Scores are deterministic transforms of S813 per-pitch acoustic readings,
    not trained embeddings from other task axes. 49 pitch bins, 3 resolution
    groups, f0-like and second/third-harmonic-like channels. Parent partial
    competition at semitone offsets 12/19/24 is a *hypothesis*, not a veto.
    """
    a=np.asarray(X)
    if a.ndim!=3 or a.shape[1:]!=(49,222) or not len(a) or not np.isfinite(a).all():
        raise ValueError("fresh nonempty causal PCM MIDI256 [T,49,222] required")
    f=np.asarray(a[:,:,[126,134,142]],dtype=np.float32).mean(axis=2)
    h2=np.asarray(a[:,:,[127,135,143]],dtype=np.float32).mean(axis=2)
    h3=np.asarray(a[:,:,[128,136,144]],dtype=np.float32).mean(axis=2)
    low=np.zeros_like(f)
    for dist in (12,19,24):
        low[:,dist:]=np.maximum(low[:,dist:],f[:,:-dist])
    residual=f-.7*low
    # Crucially preserve pitchwise morphology rather than 49-pitch means.
    ranks=np.sort(f,axis=1)[:,-14:]
    res_rank=np.sort(residual,axis=1)[:,-14:]
    h2rank=np.sort(h2,axis=1)[:,-8:]
    contrast=np.sort(f-h2,axis=1)[:,-10:]
    # Full pitch patterns remain visible to the independently learned head.
    bands=np.column_stack((
        f[:,::2],residual[:,::2],h2[:,::3],
        ranks,res_rank,h2rank,contrast,
        f.mean(axis=1),f.std(axis=1),
        h2.mean(axis=1),residual.mean(axis=1),
        *[np.count_nonzero(residual>v,axis=1) for v in (.0,.25,.5,.8,1.2)],
        *[np.count_nonzero(f>v,axis=1) for v in (.25,.5,.8,1.2)]
    ))
    return np.ascontiguousarray(bands,np.float32)

def sample_binary(rows, level, *, cap=160, seed=821):
    if level not in (5,6) or cap<20:
        raise ValueError("K5/K6 specialist only")
    samples=[];labels=[];weights=[];audit=[]
    for row in sorted(rows,key=lambda r:r["name"]):
        k=np.asarray(row["k"])
        a=np.asarray(row["X"])
        if a.ndim!=3 or a.shape[1:]!=(49,222) or k.shape!=(len(a),):
            raise ValueError("unaligned native256 labels")
        # Restrict learning to the relevant high K frontier only.
        permitted=np.flatnonzero(k >= (3 if level==5 else 4))
        binary=(k[permitted]>=level).astype(np.uint8)
        seed_local=int.from_bytes(hashlib.sha256(
            (str(level)+"/"+row["name"]).encode()).digest()[:4],"big")
        rng=np.random.default_rng(seed+seed_local)
        indices=[];ws=[]
        for truth in (0,1):
            part=permitted[binary==truth]
            size=min(len(part),cap)
            if not size:continue
            idx=np.sort(rng.choice(part,size,replace=False))
            indices.append(idx)
            ws.append(np.full(size,len(part)/size,np.float64))
        if not indices:continue
        ix=np.concatenate(indices)
        w=np.concatenate(ws)
        order=np.argsort(ix,kind="stable")
        ix,w=ix[order],w[order]
        positives=int(np.count_nonzero(k[permitted]>=level))
        if abs(sum(w[k[ix]>=level])-positives)>1e-8:
            raise AssertionError("inverse sampling did not preserve K class prior")
        samples.append(_column_signature(a)[ix])
        labels.append((k[ix]>=level).astype(np.uint8))
        weights.append(w)
        audit.append(dict(name=row["name"],natural_positive=positives,
                          eligible_frames=len(permitted),selected_frames=len(ix)))
    if not samples:raise ValueError("no high-K audio training data")
    X,y,w=np.concatenate(samples),np.concatenate(labels),np.concatenate(weights)
    if len(np.unique(y))<2:raise ValueError("no binary classes for high-K")
    w/=w.mean()
    return X,y,w,audit

def probabilities(X,model5,model6):
    x=_column_signature(X)
    with threadpool_limits(limits=1):
        p5=model5.predict_proba(x)[:,list(model5.classes_).index(1)]
        p6=model6.predict_proba(x)[:,list(model6.classes_).index(1)]
    return np.asarray(p5,np.float32),np.asarray(p6,np.float32)

def policy_grid():
    return (("base",0.),)+tuple((m,float(t)) for m in MODES[1:] for t in THRESHOLDS)

def apply_policy(base,proposal,high_gate,p5,p6,mode,threshold):
    b=np.asarray(base,dtype=np.uint8)
    h=np.asarray(proposal,dtype=np.uint8)
    gate=np.asarray(high_gate)
    q5,q6=np.asarray(p5),np.asarray(p6)
    if not(b.shape==h.shape==gate.shape==q5.shape==q6.shape) or b.ndim!=1:
        raise ValueError("aligned individual audio-frame predictions mandatory")
    if (mode,float(threshold)) not in policy_grid():
        raise ValueError("undeclared policy cannot inspect outer truth")
    output=b.copy()
    if mode=="base":return output
    # HARD REGRESSION LIMIT: never change any old K != 4.
    eligible=b==4
    if mode=="confirm5":
        eligible &= (h>=5)&(gate>=.70)&(q5>=threshold)
        output[eligible]=5
    elif mode=="verify5":
        eligible &= q5>=threshold
        output[eligible]=5
    elif mode=="verify56":
        eligible &= q5>=threshold
        output[eligible]=5
        six=eligible&(q6>=threshold)&(h==6)&(gate>=.70)
        output[six]=6
    if np.any(output[b!=4]!=b[b!=4]):
        raise AssertionError("non-K4 Active-K predictions changed")
    return output

@dataclass
class K4GuardTeacher:
    five:HistGradientBoostingClassifier
    six:HistGradientBoostingClassifier
    mode:str
    threshold:float
    metadata:dict

    @property
    def heads(self):
        if self.five is self.six:raise ValueError("two private heads must be distinct")
        return MappingProxyType({"k_ge_5":self.five,"k_ge_6":self.six})

    def predict(self,X,old,proposal,high_gate):
        p5,p6=probabilities(X,self.five,self.six)
        y=apply_policy(old,proposal,high_gate,p5,p6,self.mode,self.threshold)
        y.setflags(write=False)
        return y

def fit_teacher(train,cal,base_model,high_model,*,seed=821):
    train=sorted(train,key=lambda r:r["name"]);cal=sorted(cal,key=lambda r:r["name"])
    if (not train or not cal or
        len({r["name"] for r in train+cal})!=len(train+cal)
        or {r["composition"] for r in train}&{r["composition"] for r in cal}):
        raise ValueError("train/cal composition leakage or duplicate audio")
    models=[];audit={}
    for level in (5,6):
        X,y,w,proof=sample_binary(train,level,seed=seed)
        model=HistGradientBoostingClassifier(**MODEL_PARAMS,random_state=seed+level)
        with threadpool_limits(limits=1):
            model.fit(X,y,sample_weight=w)
        models.append(model)
        audit[str(level)]=dict(training_rows=len(y),source_tracks=len(proof),
                               sampling=proof)
    choices={p:[0,0,0,0] for p in policy_grid()}
    from causal_note.s817_highpoly_temporal import proposals as high_proposals
    for row in cal:
        base=base_model.predict(row["X"])
        gate,hi=high_proposals(row["X"],high_model.gate,high_model.high)
        p5,p6=probabilities(row["X"],*models)
        for sel,r in choices.items():
            outcome=apply_policy(base,hi,gate,p5,p6,*sel)
            score=active_score(outcome,row["k"])
            for i,key in enumerate(("correct","frames","overcount","undercount")):
                r[i]+=score[key]
    if len(set(v[1] for v in choices.values()))!=1:raise AssertionError("CAL denominators differ")
    # Largest original Active-K CAL score only; if tied, select simplest BASE.
    ordered=policy_grid()
    selected=max(ordered,key=lambda p:(
        choices[p][0]/max(1,choices[p][1]),
        -choices[p][2]/max(1,choices[p][1]),
        -ordered.index(p)))
    metadata=dict(schema=SCHEMA,train_compositions=sorted({r["composition"] for r in train}),
        calibration_compositions=sorted({r["composition"] for r in cal}),
        private_heads=("k_ge_5","k_ge_6"),no_peer_axis_model=True,
        train_only_sampling=audit,selected=dict(mode=selected[0],threshold=selected[1]),
        calibration_results=[dict(mode=m,threshold=t,correct=v[0],frames=v[1],
              overcount=v[2],undercount=v[3]) for (m,t),v in choices.items()],
        native_hop_samples=256,added_lookahead_samples=0,added_hold_samples=0,
        original_base_K_other_than_4_unchanged=True)
    return K4GuardTeacher(*models,selected[0],selected[1],metadata)
