"""S825: native PCM256 independent Active-K with strictly OOF-private pitch maps.

Never consume a trained peer axis, S708 posterior, previous emitted MIDI, or
ground-truth pitch/count in inference. Within each outer fold, cross-fit a
private note-activity model on only two of the three TRAIN compositions to
generate unbiased K-learner training observations on the third. A final
private pitch model is fit on all three TRAIN compositions for inner CAL and
OUTER evaluation; it is not the Frame Pitch characteristic.

Three predeclared count loops/ablations:
 pitch_stats; pitch_harmonic; pitch_temporal.
All current-hop acoustic inputs, previous-hop only temporal difference.
No lookahead, no post-onset hold, no retiming, no shared axis weights.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from causal_note.s813_framepitch_teacher import sample_training_rows
from causal_note.s814_activek_teacher import active_score

SCHEMA="S825_OOF_PRIVATE_PITCH_ACTIVEK_NATIVE256_V1"
MODES=("pitch_stats","pitch_harmonic","pitch_temporal")
PITCH_PARAMS=dict(loss="log_loss",learning_rate=.105,max_iter=36,
    max_leaf_nodes=13,max_depth=4,min_samples_leaf=55,
    l2_regularization=1.0,max_bins=63,early_stopping=False,
    categorical_features=None,class_weight=None)
MAX_PITCH_POINTS=650
MAX_K_POINTS=560
N_PITCH=49
DIM=222

def validate_record(r):
    x=np.asarray(r["X"])
    k=np.asarray(r["k"])
    y=np.asarray(r["y"])
    if x.ndim!=3 or x.shape[1:]!=(N_PITCH,DIM) or not len(x) or (
        k.shape!=(len(x),) or y.shape!=(len(x),N_PITCH) or
        not np.isin(k,range(7)).all() or not np.isin(y,(0,1)).all()):
        raise ValueError("unaligned original [T,49,222] PCM256 and FIT labels")
    if not isinstance(r.get("composition"),str) or not r["composition"]:
        raise ValueError("composition is missing")
    return r

def pitch_train(rows,seed=825,max_points=MAX_PITCH_POINTS):
    rows=sorted((validate_record(r) for r in rows),key=lambda r:r["name"])
    if len({r["name"] for r in rows})!=len(rows):
        raise ValueError("duplicate audio in pitch head training")
    inp=[dict(name=r["name"],composition=r["composition"],X=r["X"],y=r["y"])
         for r in rows]
    x,y,w,proof=sample_training_rows(inp,"general",seed=seed,
                                      max_points=max_points)
    teacher=HistGradientBoostingClassifier(**PITCH_PARAMS,random_state=seed)
    with threadpool_limits(limits=1):
        teacher.fit(x,y,sample_weight=w)
    if list(teacher.classes_)!=[0,1]:
        raise AssertionError("pitch learner lacks binary classes")
    return teacher,proof

def pitch_proba(model,X,block=96):
    x=np.asarray(X)
    if x.ndim!=3 or x.shape[1:]!=(49,222) or not np.isfinite(x).all():
        raise ValueError("fresh original PCM256 required")
    out=np.empty((len(x),N_PITCH),dtype=np.float32)
    with threadpool_limits(limits=1):
        pos=list(model.classes_).index(1)
        for start in range(0,len(x),block):
            end=min(start+block,len(x))
            part=np.asarray(x[start:end],dtype=np.float32).reshape(-1,DIM)
            out[start:end]=model.predict_proba(part)[:,pos].reshape(-1,N_PITCH)
    out.setflags(write=False)
    return out

def cardinality_features(p,X,mode):
    """Pitch map is THIS architecture's private inferred audio-only output.

    Frame t uses p[t], X[t], and p[t-1] from same recording; no lookahead.
    """
    p=np.asarray(p,dtype=np.float32)
    x=np.asarray(X)
    if (mode not in MODES or p.ndim!=2 or p.shape!=(len(x),N_PITCH)
        or x.ndim!=3 or x.shape[1:]!=(N_PITCH,DIM)
        or not np.isfinite(p).all() or not np.isfinite(x).all()):
        raise ValueError("independent aligned [T,49] pitch and PCM256 needed")
    sortedp=np.sort(p,axis=1)
    base=np.column_stack((
       sortedp[:,-12:],
       p.mean(axis=1),p.std(axis=1),p.sum(axis=1),
       p[:,:8].sum(axis=1),p[:,8:29].sum(axis=1),p[:,29:].sum(axis=1),
       -np.sum(p*np.log(np.clip(p,1e-6,1)),axis=1)/49.,
       *[np.count_nonzero(p>t,axis=1) for t in (.15,.25,.35,.5,.65,.8)],
       *[np.sum(np.maximum(p-t,0),axis=1) for t in (.25,.5,.75)]))
    if mode=="pitch_stats":
        return np.ascontiguousarray(base,dtype=np.float32)
    # Per-pitch competition, not arbitrary averaged physical amplitudes.
    parents=np.zeros_like(p)
    for dist in (12,19,24):
        parents[:,dist:]=np.maximum(parents[:,dist:],p[:,:-dist])
    residual=np.maximum(p-.65*parents,0.)
    fundamental=np.asarray(x[:,:,(126,134,142)],dtype=np.float32).mean(axis=2)
    second=np.asarray(x[:,:,(127,135,143)],dtype=np.float32).mean(axis=2)
    # These are DSP channel summaries, NOT physical measured F0 vs H2.
    harmonic=np.column_stack((
        np.sort(residual,axis=1)[:,-10:],residual.sum(axis=1),
        np.sort(p*residual,axis=1)[:,-6:],
        np.sort(fundamental-second,axis=1)[:,-8:],
        np.sum(p*(fundamental-second),axis=1)))
    if mode=="pitch_harmonic":
        return np.ascontiguousarray(np.column_stack((base,harmonic)),np.float32)
    previous=np.vstack((p[:1],p[:-1]))
    diff=p-previous
    temporal=np.column_stack((
        np.sort(np.maximum(diff,0),axis=1)[:,-8:],
        np.sort(np.maximum(-diff,0),axis=1)[:,-8:],
        diff.sum(axis=1),np.maximum(diff,0).sum(axis=1),
        np.maximum(-diff,0).sum(axis=1)))
    return np.ascontiguousarray(np.column_stack((base,harmonic,temporal)),np.float32)

def sample_count_features(rows,oof,mode,seed=825,max_points=MAX_K_POINTS):
    if max_points<28:raise ValueError("at least 4 examples per K required")
    xs=[];ys=[];weights=[];proof=[]
    for r in sorted(rows,key=lambda z:z["name"]):
        validate_record(r)
        if r["name"] not in oof:raise AssertionError("missing OOF pitch predictions")
        p=oof[r["name"]]
        if p.shape!=(len(r["k"]),49):raise AssertionError("OOF shape mismatch")
        y=np.asarray(r["k"],np.uint8)
        seed_local=int.from_bytes(hashlib.sha256(r["name"].encode()).digest()[:4],"big")
        rng=np.random.default_rng(seed+seed_local)
        pieces=[];weights_local=[]
        for k in range(7):
            pop=np.flatnonzero(y==k)
            n=min(len(pop),max_points//7)
            if not n:continue
            ix=np.sort(rng.choice(pop,n,replace=False))
            pieces.append(ix);weights_local.append(np.full(n,len(pop)/n,np.float64))
        inds=np.concatenate(pieces);w=np.concatenate(weights_local)
        order=np.argsort(inds,kind="stable");inds,w=inds[order],w[order]
        restored=np.bincount(y[inds],weights=w,minlength=7)
        if not np.allclose(restored,np.bincount(y,minlength=7),rtol=0,atol=1e-8):
            raise AssertionError("K training natural prior changed")
        xs.append(cardinality_features(p[inds],r["X"][inds],mode)
                  if mode!="pitch_temporal" else
                  cardinality_features(p,r["X"],mode)[inds])
        # For temporal features, use p[t-1] of original recording, not
        # p[previous sampled row]; sampling preserves original time context.
        ys.append(y[inds]);weights.append(w)
        proof.append(dict(name=r["name"],sampled=len(inds),
              observed_frames=len(y),out_of_fold_pitch=True))
    x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(weights)
    if set(np.unique(y))!=set(range(7)):
        raise AssertionError("the head must see GT K0..6 in TRAIN")
    w/=w.mean()
    return x,y,w,proof

@dataclass
class Teacher:
    pitch:object
    models:dict
    selector:str
    metadata:dict

    def predict_all(self,X):
        proba=pitch_proba(self.pitch,X)
        result={}
        for mode in MODES:
            feat=cardinality_features(proba,X,mode)
            with threadpool_limits(limits=1):
                pred=self.models[mode].predict(feat).astype(np.uint8)
            pred.setflags(write=False)
            result[mode]=pred
        return result
    def predict(self,X):
        return self.predict_all(X)[self.selector]

def fit_teacher(train,cal,*,seed=825,pitch_max=MAX_PITCH_POINTS,
                count_max=MAX_K_POINTS):
    train=sorted((validate_record(r) for r in train),key=lambda r:r["name"])
    cal=sorted((validate_record(r) for r in cal),key=lambda r:r["name"])
    train_comp={r["composition"] for r in train}
    cal_comp={r["composition"] for r in cal}
    if (not train or not cal or len(train_comp)!=3 or len(cal_comp)!=1
        or train_comp&cal_comp or
        len({r["name"] for r in train+cal})!=len(train+cal)):
        raise ValueError("strict 3-composition TRAIN and independent 1-composition CAL required")
    oof={};inner_proofs={}
    for i,withheld in enumerate(sorted(train_comp)):
        part=[r for r in train if r["composition"]!=withheld]
        omitted=[r for r in train if r["composition"]==withheld]
        p,proof=pitch_train(part,seed=seed+i,max_points=pitch_max)
        # Explicit no-self/no-same-composition OOF guard.
        if any(r["composition"]==withheld for r in part):
            raise AssertionError("OOF pitch head trained on its evaluation composition")
        for r in omitted:
            oof[r["name"]]=pitch_proba(p,r["X"])
        inner_proofs[withheld]=dict(trained_compositions=sorted({r["composition"] for r in part}),
              withheld=withheld,withheld_names=[r["name"] for r in omitted],
              train_samples=proof["samples"],oof_rows=len(omitted))
    if len(oof)!=len(train):raise AssertionError("every TRAIN row requires OOF pitch")
    models={};train_proof={}
    for i,mode in enumerate(MODES):
        x,y,w,proof=sample_count_features(train,oof,mode,seed=seed,
                                          max_points=count_max)
        count=make_pipeline(StandardScaler(),LogisticRegression(
            C=.5,max_iter=170,solver="lbfgs",class_weight=None,random_state=seed+i))
        with threadpool_limits(limits=1):
            count.fit(x,y,logisticregression__sample_weight=w)
        models[mode]=count
        train_proof[mode]=dict(samples=len(y),features=x.shape[1],source=proof)
    if len({id(m) for m in models.values()})!=3:
        raise AssertionError("all count heads must own distinct models")
    final_pitch,final_proof=pitch_train(train,seed=seed+10,
                                       max_points=pitch_max)
    choices={m:dict(correct=0,frames=0,under=0,over=0) for m in MODES}
    temp=Teacher(final_pitch,models,MODES[0],{})
    for r in cal:
        predictions=temp.predict_all(r["X"])
        for m,z in predictions.items():
            v=active_score(z,r["k"])
            choices[m]["correct"]+=v["correct"]
            choices[m]["frames"]+=v["frames"]
            choices[m]["under"]+=v["undercount"]
            choices[m]["over"]+=v["overcount"]
    select=max(MODES,key=lambda m:(choices[m]["correct"]/max(1,choices[m]["frames"]),
        -choices[m]["over"],-MODES.index(m)))
    metadata=dict(schema=SCHEMA,train_compositions=sorted(train_comp),
        calibration_compositions=sorted(cal_comp),
        crossfit_oof_composition_proofs=inner_proofs,
        private_pitch_trained_from_pcm_only=True,
        private_pitch_final_training=final_proof["samples"],
        private_count_training=train_proof,trained_private_heads=4,
        selected_from_inner_cal=select,inner_calibration_scores=choices,
        feature_hop_samples=256,decision_hop_samples=256,
        lookahead_samples=0,hold_samples=0,
        S708_posteriors_or_MIDI_used=False,other_axis_checkpoint_used=False,
        deploy_live=False,promotion=False)
    return Teacher(final_pitch,models,select,metadata)
