"""S683: train a causal acoustic onset residual head, then enforce four locks.

A new per-pitch tree head sees ONLY current/past native harmonic NOTE features,
current/past frozen NOTE onset/sustain probabilities and frozen K logits.
It is not given future PCM, annotated timing, current GT, or true K in inference.
It only offers a gated improvement to onset probabilities; frozen K, pitch,
native sustain/release and S675 NoteOff checkpoint are immutable.

TRAIN: GuitarSet 5 composition groups for learning; 4 unseen TRAIN
composition groups for selecting ONE blending policy. DEV 3 different
compositions for external retrospective diagnostics; TEST never opened.
No production integration or promotion absent non-regression on Exact-K,
Onset+Pitch F1, S675 Offset±50ms and full-note F1.
"""
from __future__ import annotations
import argparse,hashlib,json,tempfile,zipfile,time
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from causal_note.strict_live import HOP,SR,Selection,load_heads,feature_track
from scripts.s613_prepare_strict_live import download,digest,labels
from scripts.s613_train_strict_live import decode,event_score,real_notes
from scripts.s672_causal_offset_timing_grid import (
    MODEL,ROOT,raw_guitar_audio,native_posteriors,
    note_instances,native_rhythm_clock,hazard_predictions,choose_offset
)
from scripts.s682_onset_four_lock_guard import (
    native_metrics,aggregate,full_aggregate,safe_full,score_release
)

FEATURE_CHOICES=np.array([0,1,2,8,9,10,16,17,18,24,25,26,
    54,55,62,63,70,71,78,79,84,85,90,91,96,97,104,105,
    112,113,120,121],np.int64)
VARIANTS=("native","max_0.70","max_0.95","max_1.20",
          "blend_0.30","blend_0.55","blend_0.75")
TRAIN_POSITIVE_MS=40
MAX_EXAMPLES_PER_FILE=14000
MAX_NEGATIVE_POS_RATIO=7


def onset_feature_rows(features,post,k,frame,pitch):
    """Pure causal features: current and historical frames only.

    frame and pitch must be 1D integer arrays of equal length;
    source NOTE feature dim is the frozen 126D native ABI.
    """
    frame=np.asarray(frame,dtype=np.int64)
    pitch=np.asarray(pitch,dtype=np.int64)
    if frame.shape!=pitch.shape or np.any(pitch<0) or np.any(pitch>48):
        raise ValueError("Pitch/frame dimensions outside NOTE ABI")
    if len(frame)==0:
        return np.empty((0,75),np.float32)
    now=features[frame,pitch]
    selected=now[:,FEATURE_CHOICES]
    old8=features[np.maximum(0,frame-8),pitch][:,FEATURE_CHOICES[:12]]
    now12=now[:,FEATURE_CHOICES[:12]]
    t=post[frame,pitch]
    lag2=post[np.maximum(0,frame-2),pitch]
    lag8=post[np.maximum(0,frame-8),pitch]
    lag24=post[np.maximum(0,frame-24),pitch]
    octave=post[frame,np.minimum(48,pitch+12),:2]
    low=post[frame,np.maximum(0,pitch-12),:2]
    krow=k[frame]
    globalpeak=np.max(post[frame,:,0],axis=1)[:,None]
    x=np.column_stack((selected,now12-old8,
                       t,lag2,lag8,lag24,t-lag2,t-lag8,
                       octave,low,krow,globalpeak,
                       pitch.astype(np.float32)/127.))
    if not np.isfinite(x).all():
        raise ValueError("Non-finite learned onset features")
    return x.astype(np.float32)


def training_rows(spec,root,archive,heads,seed):
    audio,_=raw_guitar_audio(spec,archive)
    truth=real_notes(labels(spec,root)["notes"],len(audio)/SR)
    ends,x=feature_track(audio,block_size=4096)
    p,k=native_posteriors(x,heads)
    n=len(ends)
    positive=np.zeros((n,49),bool)
    # Past-target supervision ONLY on TRAIN, never inference inputs.
    for note in truth:
        pitch=int(note["pitch"])-40
        start=float(note["time"])
        i=int(np.searchsorted(ends,round(start*SR),side="left"))
        j=int(np.searchsorted(ends,round((start+TRAIN_POSITIVE_MS/1000.)*SR),
                              side="right"))
        if 0<=pitch<49 and i<n:
            positive[i:min(n,max(i+1,j)),pitch]=True
    if not np.any(positive):
        raise AssertionError("Unexpected empty real guitar onset labels")
    true_idx=np.flatnonzero(positive)
    rng=np.random.default_rng(seed)
    hard=np.flatnonzero(((p[:,:,0]>.12) |
                        ((p[:,:,1]>.50)&(p[:,:,0]>.05))) & ~positive)
    other=np.flatnonzero(~positive)
    maxneg=min(max(250,len(true_idx)*MAX_NEGATIVE_POS_RATIO),
               MAX_EXAMPLES_PER_FILE-len(true_idx))
    take_hard=min(len(hard),int(maxneg*.75))
    hardpick=rng.choice(hard,size=take_hard,replace=False) if take_hard else np.empty(0,int)
    left=maxneg-take_hard
    negpick=rng.choice(other,size=min(len(other),max(left,0)),
                       replace=False) if left>0 else np.empty(0,int)
    indices=np.unique(np.r_[true_idx,hardpick,negpick])
    frame=indices//49;pitch=indices%49
    X=onset_feature_rows(x,p,k,frame,pitch)
    y=positive.ravel()[indices].astype(np.int8)
    return (X,y),dict(name=spec["name"],compositions=spec["composition"],
                      rows=int(len(y)),positives=int(y.sum()),
                      negatives=int((1-y).sum()))


def train(examples):
    X=np.vstack([x for x,_ in examples])
    y=np.concatenate([y for _,y in examples])
    if y.sum()<600 or (1-y).sum()<2500:
        raise AssertionError("Onset model lacks real labelled training rows")
    model=HistGradientBoostingClassifier(
        loss="log_loss",max_iter=125,learning_rate=.06,
        max_leaf_nodes=18,min_samples_leaf=50,l2_regularization=17,
        early_stopping=False,random_state=683)
    model.fit(X,y,sample_weight=np.where(y>0,3.0,1.0))
    return model,dict(rows=int(len(y)),positive=int(y.sum()),
                      negative=int((1-y).sum()),
                      feature_dims=int(X.shape[1]))


def new_onset_probabilities(feat,p,k,model):
    n=len(p);result=np.empty((n,49),np.float32)
    # Streaming after training would require just 49 current-pitch samples.
    # This offline batching calls the same past-only feature extractor.
    for start in range(0,n,256):
        stop=min(n,start+256)
        frame=np.repeat(np.arange(start,stop),49)
        pitch=np.tile(np.arange(49),stop-start)
        chunk=onset_feature_rows(feat,p,k,frame,pitch)
        result[start:stop]=model.predict_proba(chunk)[:,1].reshape(stop-start,49)
    assert np.isfinite(result).all()
    return result


def fuse(original,new,policy):
    if policy=="native":return original
    blended=original.copy()
    if policy.startswith("max_"):
        factor=float(policy[4:])
        blended[:,:,0]=np.maximum(original[:,:,0],np.minimum(1.,new*factor))
    elif policy.startswith("blend_"):
        alpha=float(policy[6:])
        blended[:,:,0]=(1-alpha)*original[:,:,0]+alpha*new
    else:
        raise ValueError("Unknown S683 onset-only policy")
    # Sustain / NoteOff & original K logits MUST remain byte-identical.
    if not np.array_equal(blended[:,:,1:],original[:,:,1:]):
        raise AssertionError("S683 altered frozen sustain/release head")
    return blended


def eval_one(spec,root,archive,heads,model,policies,offset_model=None):
    audio,sha=raw_guitar_audio(spec,archive)
    truth=real_notes(labels(spec,root)["notes"],len(audio)/SR)
    ends,feat=feature_track(audio,block_size=4096)
    p,k=native_posteriors(feat,heads)
    before=digest(p[:,:,1:].tobytes()+k.tobytes())
    new=new_onset_probabilities(feat,p,k,model)
    selector=Selection(**heads.metadata["selection"])
    results={};details={}
    for name in policies:
        chosen=fuse(p,new,name)
        events=decode(ends,chosen,k,selector,len(audio))
        results[name]=native_metrics(events,truth,ends)
        if offset_model is not None:
            onset,pairs=event_score(events,truth,tolerance=.080)
            item=dict(truth=truth,ends=ends,feats=feat,post=p,k=k,
                      events=events,onset=onset,pairs=pairs,
                      notes=note_instances(events,ends),
                      clock=native_rhythm_clock(events,ends))
            details[name]=score_release(item,offset_model)
        assert digest(chosen[:,:,1:].tobytes()+k.tobytes())==before
    return dict(name=spec["name"],composition=spec["composition"],
                pcm_sha256=sha,results=results,offsets=details)


def choose_onset(held,policies):
    all_scores={p:aggregate([r["results"] for r in held],p) for p in policies}
    base=all_scores["native"]
    guarded=[]
    for policy in policies:
        if policy=="native":continue
        v=all_scores[policy]
        countgood=sum(
            aggregate([r["results"] for r in held
                       if r["composition"]==comp],policy)["onset_F1"] >=
            aggregate([r["results"] for r in held
                       if r["composition"]==comp],"native")["onset_F1"]
            for comp in sorted({r["composition"] for r in held}))
        if (v["onset_F1"]>=base["onset_F1"]+.002 and
            v["K_frames_correct"]>=base["K_frames_correct"] and
            v["tp"]>=base["tp"] and countgood>=2):
            guarded.append(policy)
    selected=(max(guarded,key=lambda n:(all_scores[n]["onset_F1"],
                   all_scores[n]["K_frames_correct"])) if guarded else "native")
    return selected,all_scores,guarded


def run(out,checkpoint):
    if out.exists():raise FileExistsError("Evidence folder is immutable")
    start=time.perf_counter()
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    proto=json.loads((ROOT/"protocol.json").read_text())
    assert digest(MODEL.read_bytes())==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    trains=[s for s in proto["sources"]
            if s["corpus"]=="GuitarSet" and s["split"]=="train"]
    dev=[s for s in proto["sources"]
         if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    comps=sorted({r["composition"] for r in trains})
    assert len(trains)==36 and len(dev)==12 and len(comps)==9
    calcomp=set(comps[:5])
    training=[r for r in trains if r["composition"] in calcomp]
    held=[r for r in trains if r["composition"] not in calcomp]
    heads=load_heads(MODEL,expected_sha256=frozen["model_sha256"])
    from causal_note.s672_live_offset import S672TrialNoteOff
    frozen_release=S672TrialNoteOff.load(checkpoint,heads)
    assert frozen_release.gate==.60 and frozen_release.persistence==1
    report=dict(schema="S683_CAUSAL_LEARNED_ONSET_FOUR_LOCK_AUDIT",
        frozen_native_sha256=frozen["model_sha256"],
        frozen_s675_sha256=digest(checkpoint.read_bytes()),
        frozen_note_pitch_classes=True,frozen_K_head=True,
        frozen_sustain_release=True,original_LIVE_untouched=True,
        first_party_model_only_onset_probability_fused=True,
        inference_causal_past_only=True,
        no_DEV_or_TEST_GT_used_for_training=True,
        held_compositions=sorted(set(comps)-calcomp),
        training_compositions=sorted(calcomp),
        TEST_unopened=True,DEV_previously_exposed=True)
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);(root/"data").mkdir()
        for key,file in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            source=proto["guitar_archives"][key]
            download(source["url"],root/"data"/file,
                     source["size"],source["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as archive:
            examples=[];counts=[]
            for i,s in enumerate(training,1):
                sample,info=training_rows(s,root,archive,heads,seed=683+i)
                examples.append(sample);counts.append(info)
                print("S683 TRAIN",i,"/",len(training),s["name"],flush=True)
            model,meta=train(examples)
            report["trained_head"]=meta
            report["train_sample_counts"]=counts
            del examples
            heldrows=[]
            for i,s in enumerate(held,1):
                heldrows.append(eval_one(s,root,archive,heads,model,
                                         VARIANTS,offset_model=None))
                print("S683 HELD",i,"/",len(held),s["name"],flush=True)
            selected,held_stats,qualified=choose_onset(heldrows,VARIANTS)
            report["TRAIN_held_selection"]=selected
            report["TRAIN_held_scores"]=held_stats
            report["TRAIN_held_qualified"]=qualified
            print("S683 HELD SELECTION",json.dumps(dict(selected=selected,
                native=held_stats["native"],
                selected_onset=held_stats[selected])),flush=True)
            # Freeze chosen policy now; S675 offsets are scored only for
            # original versus this frozen choice, using original classifier.
            selected_policies=("native",) if selected=="native" else ("native",selected)
            heldoffset=[]
            for i,s in enumerate(held,1):
                heldoffset.append(eval_one(s,root,archive,heads,model,
                                          selected_policies,
                                          offset_model=frozen_release.model))
                print("S683 HELD S675",i,"/",len(held),s["name"],flush=True)
            offset_scores={p:full_aggregate([r["offsets"] for r in heldoffset],p)
                           for p in selected_policies}
            held_safe=(selected!="native" and
                       safe_full(offset_scores["native"],offset_scores[selected]))
            report["TRAIN_held_S675_release"]=offset_scores
            report["TRAIN_held_offset_safe"]=bool(held_safe)
            devrows=[]
            for i,s in enumerate(dev,1):
                devrows.append(eval_one(s,root,archive,heads,model,
                                        selected_policies,
                                        offset_model=frozen_release.model))
                print("S683 DEV",i,"/",len(dev),s["name"],flush=True)
            dev_native={p:aggregate([r["results"] for r in devrows],p)
                        for p in selected_policies}
            dev_note={p:full_aggregate([r["offsets"] for r in devrows],p)
                      for p in selected_policies}
            report["DEV_onset_and_K"]=dev_native
            report["DEV_S675_offset_full_note"]=dev_note
            report["DEV_by_recording"]=[
                dict(name=r["name"],composition=r["composition"],
                     native=r["results"],offset=r["offsets"])
                for r in devrows
            ]
    report["DEV_all_guards_pass"] = bool(
        selected!="native" and held_safe and
        dev_native[selected]["onset_F1"]>dev_native["native"]["onset_F1"] and
        dev_native[selected]["K_frames_correct"]>=
            dev_native["native"]["K_frames_correct"] and
        safe_full(dev_note["native"],dev_note[selected]))
    report["promotion"]="NOT_ELIGIBLE_RETROSPECTIVE_DEV_AND_NO_LIVE_INTEGRATION"
    report["elapsed_seconds"]=round(time.perf_counter()-start,2)
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S683 FINAL",json.dumps(dict(
        selection=selected,
        native_onset=dev_native["native"]["onset_F1"],
        chosen_onset=dev_native[selected]["onset_F1"],
        original_exactK=dev_native["native"]["decoded_exact_k_accuracy"],
        chosen_exactK=dev_native[selected]["decoded_exact_k_accuracy"],
        original_offset=dev_note["native"],
        chosen_offset=dev_note[selected],
        no_regression=report["DEV_all_guards_pass"])),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--checkpoint",required=True,type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.out,args.checkpoint)
