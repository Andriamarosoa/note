"""S838: learned selection PER 256-sample fragment, never per song.

Risk gate on the two K-count axes only. Every event uses current causal S708
posteriors and a PRIVATE expert bank. 3 comps TRAIN, 1 comp CAL, 1 comp OUTER;
CAL fixes the confidence threshold, OUTER unseen by fitting/selection.
No cross-axis decision dependence and no wait. Shadow only; no MIDI output.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from causal_note.s834_native256_decoders import ActiveKDecoder
from causal_note.strict_live import load_heads, HOP
from causal_note.s712_s708_live import load_s708_checkpoint
from scripts.s809_fit80_independence_replay import ground_truth
from scripts.s813_prepare_native256 import save_json, digest_file
from scripts.s837_six_axis_policy_loops256 import (
    COMPS, SOURCE_SCHEMA, source_from_real_audio,
    choices, original_for_axis,
)
import zipfile

SCHEMA="S838_PRIVATE_COUNT_EXPERT_PER_FRAGMENT_PUREPCM256_V1"
AXES=("exact_k","active_exact_k")
THRESHOLDS=(.35,.50,.65,.80,.90,.95)


def fragment_features(src):
    """No other head output, no future; fixed feature ABI from current fragment."""
    p=np.stack([f.post for f in src.snapshots],axis=0)
    k=np.stack([f.k for f in src.snapshots],axis=0)
    on,sus,off=p.transpose(2,0,1)
    top_s=np.sort(sus,axis=1)[:,-6:]
    top_o=np.sort(on,axis=1)[:,-3:]
    levels=[(sus>=v).sum(axis=1)[:,None]
            for v in (.15,.25,.35,.45,.55,.65,.75)]
    features=np.concatenate((
        k, top_s, top_o, *levels,
        on.sum(axis=1)[:,None],sus.sum(axis=1)[:,None],
        off.sum(axis=1)[:,None], k.max(axis=1)[:,None],
        sus[:,:12].sum(axis=1)[:,None],
        sus[:,12:29].sum(axis=1)[:,None],
        sus[:,29:].sum(axis=1)[:,None],
        ),axis=1).astype(np.float32)
    if features.shape[1]!=30 or not np.isfinite(features).all():
        raise AssertionError("S838 fixed private K gate ABI mismatch")
    return features


def expert_predictions(src,axis):
    """All alternatives run at every hop with own decoder states, no lookahead."""
    fragments=src.snapshots
    if axis=="exact_k":
        source=np.array([int(f.k.argmax()) for f in fragments],np.int16)
        candidates=[source]
        names=["reference"]
        sus=np.stack([f.post[:,1] for f in fragments],axis=0)
        for t in (.25,.40,.55,.70):
            names.append("sustain_%.2f"%t)
            candidates.append(np.minimum(6,np.count_nonzero(sus>=t,axis=1)).
                              astype(np.int16))
    elif axis=="active_exact_k":
        names=[];candidates=[]
        for name,params in choices(axis):
            obj=ActiveKDecoder(**params)
            names.append(name)
            candidates.append(np.fromiter(
                (obj.step(fr).count for fr in fragments),dtype=np.int16,
                count=len(fragments)))
    else:
        raise ValueError(axis)
    all_pred=np.stack(candidates)
    expected=original_for_axis(axis,src)
    _,k=ground_truth(src.truth,src.frame_ends)
    valid=k>0 if axis=="active_exact_k" else np.ones(len(k),bool)
    if int((all_pred[0,valid]==k[valid]).sum())!=int(expected["correct"]):
        raise AssertionError("S838 reference baseline changed "+src.name+axis)
    return names,all_pred,k,valid


def train_labels(predictors,truth,valid):
    """Expert index may switch ONLY when base errs and alternative is correct."""
    if predictors.shape[1]!=len(truth):
        raise ValueError("candidate clock mismatches truth")
    target=np.zeros(len(truth),dtype=np.int32)
    eligible=valid & (predictors[0]!=truth)
    for idx in range(1,len(predictors)):
        select=eligible&(target==0)&(predictors[idx]==truth)
        target[select]=idx
    return target


def gate_probs(model,inputs,n_experts):
    """Map class posterior to fixed expert indexes, without mutable pointers."""
    out=np.zeros((len(inputs),n_experts),np.float32)
    p=model.predict_proba(inputs)
    for col,cls in enumerate(model.classes_):
        out[:,int(cls)]=p[:,col]
    return out


def select_predictions(experts,prob,threshold):
    """Default reference unless a learned alternative clears CAL cutoff."""
    alt=prob[:,1:]
    best=alt.argmax(axis=1)+1
    confidence=alt.max(axis=1)
    best=np.where(confidence>=threshold,best,0)
    return experts[best,np.arange(experts.shape[1])],best


def frame_audit(old,new,truth,valid,ends,*,comp):
    was=valid&(old==truth)
    now=valid&(new==truth)
    recovery=now&~was
    loss=was&~now
    changed=valid&(old!=new)
    by_k={}
    for k in range(7):
        mask=valid&(truth==k)
        by_k[str(k)]=dict(frames=int(mask.sum()),
            old_correct=int((was&mask).sum()),
            new_correct=int((now&mask).sum()),
            recovered=int((recovery&mask).sum()),
            lost=int((loss&mask).sum()),
            new_over=int((mask&(new>k)).sum()),
            new_under=int((mask&(new<k)).sum()))
    return dict(composition=comp,valid_frames=int(valid.sum()),
        total_changed=int(changed.sum()),
        recovered=int(recovery.sum()),lost=int(loss.sum()),
        both_wrong_changed=int((changed&~was&~now).sum()),
        changed_to_higher=int((changed&(new>old)).sum()),
        changed_to_lower=int((changed&(new<old)).sum()),
        per_true_k=by_k,
        lost_examples=[dict(sample=int(ends[i]),truth=int(truth[i]),
                            before=int(old[i]),after=int(new[i]))
                       for i in np.flatnonzero(loss)[:20]],
        recovered_examples=[dict(sample=int(ends[i]),truth=int(truth[i]),
                                 before=int(old[i]),after=int(new[i]))
                            for i in np.flatnonzero(recovery)[:20]])


def run(*,audio_zip,traces,manifest_path,source_checkpoint,source_report,
        native_model,out,tracks_per_composition=1):
    if out.exists():raise FileExistsError("S838 evidence is immutable")
    manifest=json.loads(Path(manifest_path).read_text())
    if (manifest.get("schema")!=SOURCE_SCHEMA or not manifest.get("complete") or
        len(manifest.get("fit_files",[]))!=80 or
        any(manifest.get(z) for z in ("CAL_opened","HELD_opened","TEST_opened"))):
        raise ValueError("authentic FIT80 source required, CAL/HELD/TEST forbidden")
    if not 1<=tracks_per_composition<=16:raise ValueError("1..16 tracks")
    chosen=[r for c in COMPS for r in
        sorted((r for r in manifest["fit_files"] if r["composition"]==c),
               key=lambda r:r["index"])[:tracks_per_composition]]
    net,policy,sha=load_s708_checkpoint(source_checkpoint,source_report)
    native=load_heads(native_model)
    out.mkdir(parents=True)
    report=dict(schema=SCHEMA,series="S838",complete=False,
                training_source_S708_original_hop=1024,decision_hop_samples=HOP,
                added_wait_samples=0,future_audio_samples=0,
                protected_midi_untouched=True,
                safe_to_promote=False,
                FIT80_previously_explored=True,external_blind_proof=False,
                axes={},source_model_sha256=sha,tracks=[])
    sources=[]
    with zipfile.ZipFile(audio_zip) as archive:
        for info in chosen:
            s=source_from_real_audio(info,archive,Path(traces),native,net,policy)
            sources.append(s)
            report["tracks"].append(dict(name=s.name,composition=s.composition,
                                         frames=len(s.snapshots),wav_sha256=s.sha256))
            print("S838_TRUE_PCM256_CAPTURE",s.name,len(s.snapshots),flush=True)
    for axis in AXES:
        cached={}
        for s in sources:
            names,experts,truth,valid=expert_predictions(s,axis)
            x=fragment_features(s)
            cached[s.name]=dict(x=x,pred=experts,truth=truth,valid=valid,
                                 labels=train_labels(experts,truth,valid),
                                 source=s)
        fold_reports={}
        base_correct=trial_correct=frames=0
        for i,outer in enumerate(COMPS):
            cal=COMPS[(i+1)%len(COMPS)]
            train=[c for c in COMPS if c not in (outer,cal)]
            train_set=[d for d in cached.values()
                       if d["source"].composition in train]
            x=np.concatenate([d["x"][d["valid"]] for d in train_set])
            y=np.concatenate([d["labels"][d["valid"]] for d in train_set])
            if len(np.unique(y))<=1:
                tree=None
            else:
                tree=DecisionTreeClassifier(max_depth=4,min_samples_leaf=90,
                                            random_state=838,
                                            class_weight="balanced")
                tree.fit(x,y)
            cal_set=[d for d in cached.values()
                     if d["source"].composition==cal]
            def metric_at(t,d):
                if tree is None:
                    pred=d["pred"][0].copy()
                    chosen=np.zeros(len(pred),np.int16)
                else:
                    prob=gate_probs(tree,d["x"],len(d["pred"]))
                    pred,chosen=select_predictions(d["pred"],prob,t)
                good=d["valid"]
                return int((pred[good]==d["truth"][good]).sum()),                       int(good.sum()),pred,chosen
            cal_grid=[]
            for threshold in THRESHOLDS:
                correct=total=0
                for d in cal_set:
                    a,b,_,_=metric_at(threshold,d)
                    correct+=a;total+=b
                cal_grid.append((correct/max(1,total),threshold))
            # Select solely on CAL. Candidate gate can be turned off if it
            # does not improve the reference on CAL (ties -> reference).
            cal_ref=sum(int((d["pred"][0,d["valid"]]==d["truth"][d["valid"]]).sum())
                        for d in cal_set)
            cal_total=sum(int(d["valid"].sum()) for d in cal_set)
            best=max(cal_grid,key=lambda q:(q[0],q[1]))
            enabled=bool(tree is not None and best[0]>cal_ref/max(1,cal_total))
            fold_audits=[]
            for d in cached.values():
                if d["source"].composition!=outer:continue
                old=d["pred"][0]
                if enabled:
                    good,total,new,selected=metric_at(best[1],d)
                else:
                    new=old.copy()
                    selected=np.zeros(len(new),np.int16)
                    good=int((new[d["valid"]]==d["truth"][d["valid"]]).sum())
                    total=int(d["valid"].sum())
                base=int((old[d["valid"]]==d["truth"][d["valid"]]).sum())
                base_correct+=base;trial_correct+=good;frames+=total
                audit=frame_audit(old,new,d["truth"],d["valid"],
                                  d["source"].frame_ends,comp=outer)
                audit["source"]=d["source"].name
                audit["selected_alternative_fragments"]=int(
                    (selected[d["valid"]]>0).sum())
                fold_audits.append(audit)
            fold_reports[outer]=dict(train=train,cal=cal,outer=outer,
                tree_leaves=None if tree is None else int(tree.get_n_leaves()),
                label_hist=dict(Counter(y.tolist())),
                cal_reference=cal_ref/max(1,cal_total),
                cal_best=best[0],cal_threshold=best[1],
                enable_gate=enabled,
                held_out_audits=fold_audits)
        result=dict(reference_correct=base_correct,candidate_correct=trial_correct,
                    total_frames=frames,reference_score=base_correct/frames,
                    candidate_score=trial_correct/frames,
                    improves_global=trial_correct>base_correct,
                    reaches_80_percent=trial_correct/frames>.80,
                    safe_to_promote=False,
                    trained_fragments_only_on_other_compositions=True,
                    per_fragment_selection=True,folds=fold_reports)
        report["axes"][axis]=result
        save_json(out/"report.json",report)
        print("S838_FRAGMENT_GATE",axis,round(result["reference_score"],6),
              round(result["candidate_score"],6),
              "recovered_net",trial_correct-base_correct,
              "target80",result["reaches_80_percent"],flush=True)
    report["complete"]=True
    report["frames"]=sum(len(s.snapshots) for s in sources)
    report["tracks_count"]=len(sources)
    report["safe_to_promote"]=False
    save_json(out/"report.json",report)
    print("S838_FINAL",json.dumps({k:dict(
        before=v["reference_score"],after=v["candidate_score"],
        improved=v["improves_global"],above80=v["reaches_80_percent"])
        for k,v in report["axes"].items()},sort_keys=True))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("audio-zip","traces","manifest-path","source-checkpoint",
                 "source-report","native-model","out"):
        p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--tracks-per-composition",type=int,default=1)
    a=p.parse_args()
    run(audio_zip=a.audio_zip,traces=a.traces,manifest_path=a.manifest_path,
        source_checkpoint=a.source_checkpoint,source_report=a.source_report,
        native_model=a.native_model,out=a.out,
        tracks_per_composition=a.tracks_per_composition)
