"""S830: three independent, learned musical evidence heads owned by Exact-K.

Only S829's *own* causal PCM->spectral extractor is reused. The model owns
private Onset/Sustain/Release evidence, selectors, state and three learned
classifiers; it never imports or uses another task's prediction, checkpoint or
state. S809's principle (multiple proposals -> private per-frame selection)
is reconstructed at a fresh 256 sample cadence, not replayed at 1024.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from causal_note.s829_exactk_s809_native256 import (
    S829AcousticFrontEnd, HOP, PITCHES, SR, score_exact
)

SCHEMA = "S830_EXACTK_THREE_PRIVATE_MUSICAL_EVIDENCE_HEADS_PCM256_V1"
DOMAINS = ("onset","sustain","release")
# 72 PCM-only pitch-specific harmonic/temporal dimensions, plus log pitch.
FEATURES = 73
ARGUMENTS = dict(max_iter=48,max_depth=4,max_leaf_nodes=15,
                 min_samples_leaf=65,learning_rate=.09,
                 l2_regularization=2.0,max_bins=63,early_stopping=False)
MODES = ("s829", "sustain", "joint", "router", "blend")
THRESHOLDS = (.30,.40,.50,.60,.70)
SHIFTS = (-1,0,1)


def pitch_features(frames):
    """Pitch evidence derived from PCM only, never S708/S795 model outputs."""
    f=np.asarray(frames,np.float32)
    if f.ndim!=3 or f.shape[1:]!=(PITCHES,72) or not np.isfinite(f).all():
        raise ValueError("per-pitch PCM [T,49,72] required")
    position=np.broadcast_to(
        (np.arange(40,89,dtype=np.float32)/127.)[None,:,None],
        (*f.shape[:2],1))
    return np.concatenate((f,position),axis=-1)


def truth_pitches_to_three(pitches):
    """Training labels ONLY, never a feature or LIVE input."""
    y=np.asarray(pitches,bool)
    if y.ndim!=2 or y.shape[1]!=PITCHES or len(y)==0:
        raise ValueError("pitch occupancy truth must be [T,49]")
    prior=np.concatenate((np.zeros((1,PITCHES),bool),y[:-1]),axis=0)
    return dict(onset=y & ~prior, sustain=y,
                release=~y & prior)


class S830PCMAcoustics:
    """An Exact-K-owned right-aligned PCM256 frontend and private state."""

    def __init__(self):
        self.frontend=S829AcousticFrontEnd()
        self.last_end=0
        self.calls=0

    def push(self,pcm,*,start_sample):
        results=[]
        for end,pitch,raw_k in self.frontend.push(pcm,start_sample=start_sample):
            if end!=self.last_end+HOP:
                raise AssertionError("S830 missing/noncontiguous MIDI256 decision")
            if pitch.shape!=(49,72):
                raise AssertionError("wrong private acoustic feature shape")
            self.last_end=end;self.calls+=1
            results.append((end,pitch.copy(),raw_k))
        return tuple(results)


@dataclass
class LearnedEvidence:
    """Three separate fitted classifiers, no shared model or mutable state."""
    onset:HistGradientBoostingClassifier
    sustain:HistGradientBoostingClassifier
    release:HistGradientBoostingClassifier
    meta:dict

    def posterior(self,frames,batch_frames=256):
        f=np.asarray(frames)
        if f.ndim!=3 or f.shape[1:]!=(PITCHES,72):
            raise ValueError("S830 expects raw private [T,49,72] evidence")
        pred=np.zeros((*f.shape[:2],3),np.float32)
        for first in range(0,len(f),batch_frames):
            part=pitch_features(np.asarray(f[first:first+batch_frames],np.float32))
            flat=part.reshape(-1,FEATURES)
            for c,domain in enumerate(DOMAINS):
                model=getattr(self,domain)
                if not np.array_equal(model.classes_,[0,1]):
                    raise AssertionError("head must learn both classes")
                with threadpool_limits(limits=1):
                    p=model.predict_proba(flat)[:,1]
                pred[first:first+len(part),:,c]=p.reshape(-1,PITCHES)
        return pred


def train_evidence(rows,*,seed=830,positive_per_track=160,negative_per_track=240):
    """Fit independent heads from TRAIN compositions only."""
    records=sorted(rows,key=lambda r:r["name"])
    if not records:raise ValueError("no TRAIN rows")
    trained={}
    for domain_index,domain in enumerate(DOMAINS):
        xs,ys,ws=[],[],[]
        positive_total=0
        for row in records:
            features=np.asarray(row["pitch_features"],np.float32)
            label=truth_pitches_to_three(row["pitch_target"])[domain].reshape(-1)
            if features.ndim!=3 or features.shape[1:]!=(PITCHES,72) or len(features)!=len(row["pitch_target"]):
                raise ValueError("S830 training ABI mismatch")
            stable=int.from_bytes(hashlib.sha256((row["name"]+domain).encode()).digest()[:4],"big")
            rng=np.random.default_rng(seed+stable)
            pos=np.flatnonzero(label)
            neg=np.flatnonzero(~label)
            selected=[];weights=[]
            for ids,limit in ((pos,positive_per_track),(neg,negative_per_track)):
                if len(ids):
                    n=min(len(ids),limit)
                    chosen=np.sort(rng.choice(ids,size=n,replace=False))
                    selected.append(chosen)
                    weights.append(np.full(n,len(ids)/n,np.float64))
            ix=np.concatenate(selected);w=np.concatenate(weights)
            ixorder=np.argsort(ix)
            ix=ix[ixorder];w=w[ixorder]
            pos_count=int(np.count_nonzero(label[ix]))
            positive_total+=pos_count
            pitch_idx=ix%PITCHES
            frame_idx=ix//PITCHES
            rows_x=features[frame_idx,pitch_idx]
            full_x=np.column_stack((rows_x, (pitch_idx+40)/127.)).astype(np.float32)
            xs.append(full_x);ys.append(label[ix].astype(np.uint8));ws.append(w)
        X=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws)
        if len(np.unique(y))<2:raise ValueError("no positives for "+domain)
        w=w/w.mean()
        model=HistGradientBoostingClassifier(**ARGUMENTS,random_state=seed+domain_index*100)
        with threadpool_limits(limits=1):model.fit(X,y,sample_weight=w)
        trained[domain]=model
    if len({id(trained[x]) for x in DOMAINS})!=3:
        raise AssertionError("S830 heads must not share learned weights")
    return LearnedEvidence(trained["onset"],trained["sustain"],trained["release"],
        dict(schema=SCHEMA,training_compositions=sorted(
             {r["composition"] for r in records}),
             private_fitted_heads=3,shared_model_weights=False,
             shared_axis_posteriors=False,source_checkpoint=None,
             native_hop_samples=HOP,added_future_samples=0,
             added_hold_samples=0))


def counts_from_evidence(posterior,thr):
    """Two causal competing K proposals, pitch and note-state based."""
    p=np.asarray(posterior,np.float32)
    if p.ndim!=3 or p.shape[1:]!=(PITCHES,3) or not np.isfinite(p).all():
        raise ValueError("must supply three own learned musical probabilities")
    on,sustain,release=p.transpose(2,0,1)
    direct=np.minimum(np.count_nonzero(sustain>=thr,axis=1),6).astype(np.uint8)
    # Private note state and no future samples. The current attack can
    # initiate immediately; current release can stop immediately.
    held=np.zeros(PITCHES,bool)
    state=np.empty(len(p),np.uint8)
    for i in range(len(p)):
        start=((on[i]>=.45)&(sustain[i]>=.15))|(sustain[i]>=min(.95,thr+.10))
        end=((release[i]>=.55)&(sustain[i]<.55))|(sustain[i]<max(.10,thr-.22))
        held=(held&~end)|start
        state[i]=min(6,int(held.sum()))
    return direct,state


def selected(base,pitch,held,confidence,mode,shift=0):
    """Dynamic fragment selector; CAL chooses its policy, not OUTER truth."""
    a=np.asarray(base,dtype=np.int16)
    b=np.asarray(pitch,dtype=np.int16)
    c=np.asarray(held,dtype=np.int16)
    confidence=np.asarray(confidence)
    if a.shape!=b.shape or b.shape!=c.shape or confidence.shape!=a.shape:
        raise ValueError("private selector requires aligned per-hop outputs")
    if mode=="s829":v=a
    elif mode=="sustain":v=b
    elif mode=="joint":v=c
    elif mode=="blend":v=np.rint((a+b+c)/3.).astype(np.int16)
    elif mode=="router":
        # Local fragment choice, never other axis or GT.
        v=np.where(confidence>=.75,b,np.where(confidence>=.55,c,a))
    else:raise ValueError("unknown private selector")
    return np.clip(v+int(shift),0,6).astype(np.uint8)


def choose_selector(cal,*,modes=MODES,thresholds=THRESHOLDS):
    """CAL-only choice over all private head combinations."""
    totals={}
    for mode in modes:
        for thr in thresholds:
            for shift in SHIFTS:
                correct=under=over=frames=0
                for row in cal:
                    direct,state=counts_from_evidence(row["evidence"],thr)
                    pred=selected(row["base"],direct,state,row["confidence"],mode,shift)
                    s=score_exact(pred,row["k"])
                    correct+=s["correct"];under+=s["undercount"]
                    over+=s["overcount"];frames+=s["frames"]
                totals[(mode,thr,shift)]=(correct,frames,under,over)
    return max(totals,key=lambda key:(
        totals[key][0]/max(1,totals[key][1]),
        -totals[key][3],-totals[key][2],
        -MODES.index(key[0]),-THRESHOLDS.index(key[1]),-SHIFTS.index(key[2]))),totals


def apply(fitted_k,fitted_pitch,x,raw_k,raw_pitch,selector):
    """Pure inference: never labels, files, other axes or future samples."""
    mode,thr,shift=selector
    baseline=fitted_k.predict(np.asarray(x,np.float32),np.asarray(raw_k))
    post=fitted_pitch.posterior(raw_pitch)
    direct,state=counts_from_evidence(post,thr)
    confidence=np.mean(np.maximum(post[:,:,1],1-post[:,:,1]),axis=1)
    pred=selected(baseline,direct,state,confidence,mode,shift)
    return pred,baseline,dict(direct=direct,state=state,post=post)
