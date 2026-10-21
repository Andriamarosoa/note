"""S815 Active Exact-K, independent teacher with THREE privately trained heads.

Direct acoustic count, continuous cardinality, and own independently fitted
pitch-occupancy expert. Never imports trained Frame Pitch weights or receives
outputs from other axes. Calibrate on held-back FIT composition only; never use
evaluation composition for policy selection. Causal PCM-derived 256-hop inputs.
"""
from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from causal_note.s814_activek_teacher import (
    summarize, sample_frames, active_score, MODEL_ARGS
)
from causal_note.s813_framepitch_teacher import sample_training_rows

SCHEMA = "S815_THREE_PRIVATE_HEAD_ACTIVEK_FRESH_MIDI256_V1"
SELECTORS = ("direct","regression","pitch_sum","median","confidence_direct_pitch")
SHIFTS = (0,-1,1)
PITCH_PARAMS = dict(loss="log_loss", learning_rate=.1, max_iter=64,
    max_leaf_nodes=15, max_depth=4, min_samples_leaf=80,
    l2_regularization=1., max_bins=63, early_stopping=False,
    categorical_features=None, class_weight=None)

def _validate(rows, name):
    rows = sorted(list(rows), key=lambda r:r["name"])
    if not rows or len({r["name"] for r in rows}) != len(rows):
        raise ValueError(name + " distinct recordings required")
    for r in rows:
        x=np.asarray(r["X"])
        y=np.asarray(r["y"])
        k=np.asarray(r["k"])
        if (x.ndim!=3 or x.shape[1:]!=(49,222) or not len(x)
            or y.shape!=(len(x),49) or k.shape!=(len(x),)):
            raise ValueError("fresh MIDI256 [T,49,222] and private count/pitch training labels required")
        if not np.isin(y,(0,1)).all() or not np.isin(k,range(7)).all():
            raise ValueError("invalid labels")
        if not isinstance(r["composition"],str) or not r["composition"]:
            raise ValueError("invalid composition")
    return rows

def _count_outputs(x, direct, regressor, pitch):
    features=summarize(x)
    with threadpool_limits(limits=1):
        cls=direct.predict_proba(features)
        reg=regressor.predict(features)
        # Model trained from acoustic labels for this teacher ONLY.
        pitches=pitch.predict_proba(np.asarray(x,dtype=np.float32).reshape(-1,222))[:,list(pitch.classes_).index(1)]
    p=np.zeros((len(x),7),np.float32)
    p[:,np.asarray(direct.classes_,int)]=cls
    pitch_count=pitches.reshape(len(x),49).sum(axis=1)
    return p,reg,pitch_count

def _select(p,reg,pitch,selector,shift):
    kd=np.argmax(p,axis=1).astype(np.int16)
    kr=np.rint(reg).astype(np.int16)
    kp=np.rint(pitch).astype(np.int16)
    if selector=="direct":
        z=kd
    elif selector=="regression":
        z=kr
    elif selector=="pitch_sum":
        z=kp
    elif selector=="median":
        z=np.sort(np.stack((kd,kr,kp),axis=-1),axis=-1)[:,1]
    elif selector=="confidence_direct_pitch":
        z=np.where(p.max(axis=1)>=.65,kd,kp)
    else:
        raise ValueError("undeclared selector")
    if shift not in SHIFTS:
        raise ValueError("undeclared shift")
    return np.clip(z+shift,0,6).astype(np.uint8)

@dataclass
class ActiveKTeacher:
    direct: HistGradientBoostingClassifier
    regression: HistGradientBoostingRegressor
    pitch_occupancy: HistGradientBoostingClassifier
    selector: str
    shift: int
    metadata: dict

    def __post_init__(self):
        if (len({id(self.direct),id(self.regression),id(self.pitch_occupancy)})!=3
            or self.selector not in SELECTORS or self.shift not in SHIFTS):
            raise ValueError("all 3 heads must own their models")

    @property
    def heads(self):
        return MappingProxyType({"direct":self.direct,"regression":self.regression,
                                 "pitch_occupancy":self.pitch_occupancy})

    def predict(self,X):
        x=np.asarray(X)
        if x.ndim!=3 or x.shape[1:]!=(49,222) or not len(x) or not np.isfinite(x).all():
            raise ValueError("nonfinite or non-native MIDI256 acoustics")
        a,b,c=_count_outputs(x,self.direct,self.regression,self.pitch_occupancy)
        result=_select(a,b,c,self.selector,self.shift)
        result.setflags(write=False)
        return result

def fit_teacher(train_rows,cal_rows,*,seed=815,cap=1400,pitch_cap=2500):
    train=_validate(train_rows,"training")
    cal=_validate(cal_rows,"calibration")
    tc={r["composition"] for r in train}
    cc={r["composition"] for r in cal}
    if tc&cc or {r["name"] for r in train}&{r["name"] for r in cal}:
        raise ValueError("composition-disjoint calibration required")
    x,y,weights,provenance=sample_frames(train,cap=cap,seed=seed)
    direct=HistGradientBoostingClassifier(**MODEL_ARGS,random_state=seed)
    reg=HistGradientBoostingRegressor(**MODEL_ARGS,random_state=seed+1)
    # Private per-note occupancy training; NOT another characteristic's
    # checkpoint. The label is used ONLY here, never during inference.
    occ_rows=[dict(name=r["name"],composition=r["composition"],X=r["X"],y=r["y"])
              for r in train]
    xp,yp,wp,pitch_proof=sample_training_rows(
        occ_rows, "general",max_points=pitch_cap,seed=seed+2)
    pitch=HistGradientBoostingClassifier(**PITCH_PARAMS,random_state=seed+2)
    with threadpool_limits(limits=1):
        direct.fit(x,y,sample_weight=weights)
        reg.fit(x,y.astype(np.float32),sample_weight=weights)
        pitch.fit(xp,yp,sample_weight=wp)
    choices={(s,shift):[0,0,0] for s in SELECTORS for shift in SHIFTS}
    for row in cal:
        p,r,q=_count_outputs(row["X"],direct,reg,pitch)
        for s,shift in choices:
            result=active_score(_select(p,r,q,s,shift),row["k"])
            c=choices[s,shift]
            c[0]+=result["correct"];c[1]+=result["frames"];c[2]+=result["overcount"]
    # Maximize ActiveK accuracy, tie-break toward least overcount.
    # Never use the test composition's labels for choosing selectors.
    best=max(choices, key=lambda key:(
        choices[key][0]/max(choices[key][1],1),
        -choices[key][2]/max(choices[key][1],1),
        -SELECTORS.index(key[0]),-SHIFTS.index(key[1])))
    proof=dict(schema=SCHEMA, trained_heads=["direct","regression","pitch_occupancy"],
      private_weights=True, no_peer_axis_checkpoint=True,
      supervised_inference_inputs=False, train_compositions=sorted(tc),
      calibration_compositions=sorted(cc), selector=best[0],shift=best[1],
      train_recordings=len(train), calibration_recordings=len(cal),
      acoustic_feature_hop_samples=256,
      added_lookahead_samples=0,post_onset_hold_samples=0,
      sample_count=provenance, sample_pitch=pitch_proof,
      calibration_results=[dict(selector=s,shift=shift,correct=v[0],
        frames=v[1],overcount=v[2]) for (s,shift),v in choices.items()])
    return ActiveKTeacher(direct,reg,pitch,best[0],best[1],proof)
