"""S824 PREDECLARED re-training ablation of genuinely archived S795 features.

S708-family feature columns are not new acoustic observations. Source S795
was sampled at 1024 samples, projected as-of to an Active-K 256-sample
decision grid. No production promotion, six-axes remain immutable.
"""
from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from causal_note.s823_legacy23_asof_midi256 import (
    balanced_signed_sample,feature_at,fit_teacher as fit_original_s823
)
from causal_note.s814_activek_teacher import active_score

SCHEMA="S824_S795_PREDECLARED_FAMILY_ABLATION_SAME_MIDI256_V1"
SEED=823
CAP=840
PRED=tuple(list(range(12))+[13,14])  # original S708 posteriors / count / harmonic competition
MIDI=(12,15)                          # emitted source occupancy and count
ACOUSTIC=tuple(range(16,23))          # six S708-native acoustic slots + pitch coordinate
if (set(PRED)|set(MIDI)|set(ACOUSTIC)!=set(range(23))
    or set(PRED)&set(MIDI) or set(PRED)&set(ACOUSTIC)
    or set(MIDI)&set(ACOUSTIC)):
    raise AssertionError("S795 exact 23-column ABI family partition changed")

# Pre-registered fixed set, no selecting masks or new thresholds on outer data.
MASKS={
  "full23":tuple(range(23)),
  "without_midi":PRED+ACOUSTIC,
  "without_s708":MIDI+ACOUSTIC,
  "acoustic_only":ACOUSTIC,
  "without_acoustic":PRED+MIDI,
  "s708_only":PRED,
  "midi_only":MIDI,
  "s708_plus_acoustic":PRED+ACOUSTIC,
  "midi_plus_acoustic":MIDI+ACOUSTIC,
  "without_native_post":tuple(k for k in range(23) if k not in (7,8,9)),
  "without_primary_post":tuple(k for k in range(23) if k not in (0,1,2,3,4,5,6,10,11,13,14)),
}
# Intentional mathematical duplicates are retained as controls but trained
# exactly once: WITHOUT MIDI == S708_PLUS_ACOUSTIC, etc.
UNIQUE_MODES=(
  "full23","without_midi","without_s708","acoustic_only",
  "without_acoustic","s708_only","midi_only",
  "without_native_post","without_primary_post",
)
ALIASES={"s708_plus_acoustic":"without_midi",
         "midi_plus_acoustic":"without_s708"}
MODES=tuple(MASKS)
assert len(MODES)==11 and len(UNIQUE_MODES)==9 and len(ALIASES)==2
for mode,alias in ALIASES.items():
    assert set(MASKS[mode])==set(MASKS[alias])

def masked_matrix(x,mode):
    a=np.asarray(x,dtype=np.float32)
    if mode not in MASKS or a.ndim!=2 or a.shape[1]!=23:
        raise ValueError("expected raw genuine S795 [T,23] and declared mask")
    z=np.zeros_like(a,dtype=np.float32)
    keys=list(MASKS[mode])
    z[:,keys]=a[:,keys]
    return np.ascontiguousarray(z)
@dataclass
class Teacher:
    model:object
    mode:str
    metadata:dict
    def predict(self,row,*,batch=2048):
        predictions=[]
        for start in range(0,len(row["asof"]),batch):
            full=feature_at(row["X"],row["asof"][start:start+batch],"raw23")
            current=masked_matrix(full,self.mode)
            with threadpool_limits(limits=1):
                predictions.append(self.model.predict(current).astype(np.uint8))
        p=np.concatenate(predictions)
        if not np.isin(p,np.arange(7)).all():
            raise AssertionError("S824 predictions not in K0..6")
        p.setflags(write=False)
        return p

def train_all(train,*,seed=SEED,cap=CAP):
    if len(train)!=48 or len({r["name"] for r in train})!=48 or len({r["composition"] for r in train})!=3:
        raise ValueError("48 original distinct TRAIN tracks, 3 compositions required")
    x,y,w,proof=balanced_signed_sample(train,"raw23",cap=cap,seed=seed)
    teachers={}
    for mode in UNIQUE_MODES:
        if mode=="full23":
            # The scientific control reuses the *exact S823 training routine*,
            # not a duplicated approximation with subtly different numerics.
            control=fit_original_s823(train,"raw23",cap=cap,seed=seed)
            model=control.model
        else:
            z=masked_matrix(x,mode)
            model=make_pipeline(StandardScaler(),LogisticRegression(
                C=.5,max_iter=230,solver="lbfgs",class_weight=None,random_state=823))
            with threadpool_limits(limits=1):
                model.fit(z,y,logisticregression__sample_weight=w)
        meta=dict(schema=SCHEMA,mode=mode,nonzero_columns=list(MASKS[mode]),
            expected_training_rows=len(y),train_compositions=sorted({r["composition"] for r in train}),
            weights_are_owned=True,model_S708_loaded=False,
            original_S708_influenced_features=bool(set(MASKS[mode])&set(PRED)),
            original_source_MIDI_influenced_features=bool(set(MASKS[mode])&set(MIDI)),
            original_S708_acoustic_frontend_influenced_features=bool(set(MASKS[mode])&set(ACOUSTIC)),
            peer_axis_independence=False if bool(set(MASKS[mode])&set(PRED+MIDI)) else None,
            inference_uses_truth=False,source_observation_hop=1024,decision_hop=256,
            added_lookahead_samples=0,added_hold_samples=0,
            calibration_never_trained=True,original_S708_unchanged=True,
            production_LIVE_unchanged=True,promotion=False)
        teachers[mode]=Teacher(model,mode,meta)
    for alias,canonical in ALIASES.items():
        base=teachers[canonical]
        # Reuse exact same prediction for mathematically identical selected
        # columns. Do not pretend these aliases are independent architectures.
        teachers[alias]=Teacher(base.model,alias,dict(base.metadata,mode=alias,
                      exact_alias_of=canonical))
    if set(teachers)!=set(MODES):
        raise AssertionError("ablation mask declarations missing")
    return teachers

def choose_on_cal(cal,teachers):
    if len(cal)!=16 or len({r["composition"] for r in cal})!=1:
        raise ValueError("16 separate CAL tracks required")
    composition=cal[0]["composition"]
    if any(composition in teacher.metadata["train_compositions"]
            for teacher in teachers.values()):
        raise ValueError("CAL composition touched TRAIN")
    counts={k:dict(correct=0,frames=0,undercount=0,overcount=0)
            for k in MODES}
    for row in cal:
        for mode,teacher in teachers.items():
            result=active_score(teacher.predict(row),row["k"])
            for key in counts[mode]:counts[mode][key]+=result[key]
    choice=max(UNIQUE_MODES,key=lambda mode:(counts[mode]["correct"]/
         max(1,counts[mode]["frames"]),-counts[mode]["overcount"],
         -UNIQUE_MODES.index(mode)))
    return choice,counts
