"""S732: learn *independent-source onset likelihood* from OLD+NEW causal scene.

Not another feature injection into the signed S708 head. A distinct
discriminator is trained using actual guitar FIT annotations (after the
S708 event stream and physical scene have already been generated).

Two equal-budget HGB models:
  - baseline control observes only S708 acoustic/head/K evidence,
  - auditory model also observes S731 counterfactual owner explanation,
    newly unexplained partials, parent ownership, and S729 phase novelty.

Use ONLY FIT labels in .fit; no truth labels in .transform at inference.
All candidate NoteOn emissions are governed by the original joint
neural K0..6, original S708 pitch head and S707 NoteOff policy.

The phase channels are causal from S706's measured phase coherences;
they are not estimates of independent source phase trajectories.
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from causal_note.s729_phase_overlap_evidence import phase_track
from causal_note.s731_auditory_scene import scene_scores

N_FEATURES=21
CONTROL=(0,1,2,9,14,15,17,18,19,20)
SCENE=tuple(range(N_FEATURES))


def observation_matrix(x,post,k,baseline,ends):
    """[T,49,21] physically interpretable features, all online causal."""
    xx=np.asarray(x,np.float32)
    pp=np.asarray(post,np.float32)
    kk=np.asarray(k,np.float32)
    if (xx.ndim!=3 or xx.shape[1:]!=(49,288)
        or pp.shape!=(len(xx),49,3) or kk.shape!=(len(xx),7)
        or not np.isfinite(xx).all() or not np.isfinite(pp).all()
        or not np.isfinite(kk).all()):
        raise ValueError("S732 signed S708 x/post/K input ABI required")
    scene=scene_scores(xx,pp,kk,baseline,ends)
    phase=phase_track(xx)
    amp=xx[:,:,192:216].reshape(len(xx),49,3,8)
    level=np.maximum(amp.mean(axis=(2,3)),0.)
    prev=np.concatenate((level[:1],level[:-1]),0)
    harmonic=xx[:,:,216:240].reshape(len(xx),49,3,8)
    phase_mean=phase[:,:,0:8].mean(-1)
    phase_on=phase[:,:,8:16].mean(-1)
    phase_oct=phase[:,:,24:27].mean(-1)
    entropy=-(np.clip(kk,1e-8,1.)*np.log(np.clip(kk,1e-8,1.))).sum(-1)
    features=np.empty((len(xx),49,N_FEATURES),np.float32)
    features[:,:,0:3]=pp
    features[:,:,3]=np.stack([z.old_ownership for z in scene],0)
    features[:,:,4]=np.stack([z.novelty for z in scene],0)
    features[:,:,5]=np.stack([z.explained for z in scene],0)
    features[:,:,6]=np.stack([z.novelty_shared for z in scene],0)
    features[:,:,7]=np.stack([z.novelty_unshared for z in scene],0)
    features[:,:,8]=np.stack([z.supports for z in scene],0)/8.
    features[:,:,9]=np.array([z.count_room for z in scene],np.float32)[:,None]
    features[:,:,10]=phase_mean
    features[:,:,11]=phase_on
    features[:,:,12]=phase_oct
    features[:,:,13]=np.maximum(0.,pp[:,:,0] -
                        np.concatenate((pp[:1,:,0],pp[:-1,:,0]),0))
    features[:,:,14]=level
    features[:,:,15]=harmonic.mean(axis=(2,3))
    features[:,:,16]=np.log1p(level)
    features[:,:,17]=np.maximum(0.,level-prev)
    features[:,:,18]=entropy[:,None]
    features[:,:,19]=np.linspace(0.,1.,49,dtype=np.float32)[None,:]
    features[:,:,20]=kk[:,2:].sum(-1)[:,None]
    if not np.isfinite(features).all():
        raise AssertionError("S732 observation features must remain finite")
    return features


def sampled_fit_rows(features,onset_truth,*,seed,max_rows=4000):
    """FIT-only independent events vs difficult harmonics, maintain priors."""
    f=np.asarray(features,np.float32)
    y=np.asarray(onset_truth).astype(bool)
    if f.ndim!=3 or f.shape[1:]!=(49,N_FEATURES) or y.shape!=f.shape[:2]:
        raise ValueError("S732 FIT only expected [T,49] onset labels")
    rng=np.random.default_rng(seed)
    # All positive starts, and hard negatives near acoustic or model onset:
    # never mistake high-frequency harmonic energy alone for a true note.
    hard=(f[:,:,0]>=.14)|(f[:,:,1]>=.5)|(f[:,:,3]>.15)|(f[:,:,4]>.10)
    probability=np.where(y,1.,np.where(hard,.35,.025)).astype(np.float32)
    chosen=rng.random(y.shape)<probability
    indices=np.flatnonzero(chosen.ravel())
    if len(indices)>max_rows:
        true_indices=np.flatnonzero(y.ravel())
        keep_true=np.intersect1d(indices,true_indices,assume_unique=True)
        rest=np.setdiff1d(indices,keep_true,assume_unique=True)
        need=max(0,max_rows-len(keep_true))
        sampled=rng.choice(rest,size=min(need,len(rest)),replace=False)
        indices=np.sort(np.concatenate((keep_true,sampled)))
    rate=probability.ravel()[indices]
    rowweight=np.minimum(20.,1./rate)
    return f.reshape(-1,N_FEATURES)[indices],y.ravel()[indices].astype(np.uint8),rowweight


def fit_discriminator(features,labels,weights,*,with_scene,seed=732):
    matrix=np.asarray(features,np.float32)
    columns=SCENE if with_scene else CONTROL
    model=HistGradientBoostingClassifier(
        max_iter=70,max_leaf_nodes=15,max_depth=4,
        learning_rate=.065,l2_regularization=10.,
        min_samples_leaf=100,early_stopping=False,random_state=seed)
    model.fit(matrix[:,columns],np.asarray(labels,np.uint8),
              sample_weight=np.asarray(weights,np.float64))
    return model


def apply_evidence(post,k,evidence,likelihood,*,gain=.2,penalty=0.,
                   use_novelty=True):
    """Cannot manufacture notes: original selector still needs real onset.

    Candidate boost requires posterior independent-source onset > neural
    onset AND K predicts spare simultaneous source capacity. Optional
    negative evidence only applies when the prior higher-harmonic source
    really exists (owner memory), never suppresses all new notes by pitch.
    """
    p=np.asarray(post,np.float32)
    q=np.asarray(likelihood,np.float32)
    if (p.shape!=(49,3) or q.shape!=(49,) or not
        np.isfinite(q).all() or gain<0 or gain>1 or
        penalty<0 or penalty>1):
        raise ValueError("S732 real-time calibrated onset evidence ABI")
    out=p.copy()
    owner=evidence.old_ownership if use_novelty else 0.
    k_room=evidence.count_room
    increase=np.maximum(q-p[:,0],0.)
    reduction=np.maximum(p[:,0]-q,0.)
    out[:,0]=np.clip(p[:,0]+gain*k_room*increase-
             penalty*owner*reduction,0.,1.)
    return out


def predict_track_causal(model,features,*,with_scene):
    matrix=np.asarray(features,np.float32)
    if matrix.ndim!=3 or matrix.shape[1:]!=(49,N_FEATURES):
        raise ValueError("S732 incoming causal observation shape wrong")
    cols=SCENE if with_scene else CONTROL
    return model.predict_proba(matrix.reshape(-1,N_FEATURES)[:,cols])[:,1].reshape(len(matrix),49).astype(np.float32)
