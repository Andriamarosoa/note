"""S755 FIT-only K2..K6 specialist, 12 true full-MIDI CAL32 loops.

Train four separately calibrated K-head specialists from native causal NOTE
posteriors and sharp harmonic features, then blend each class distribution
with the signed S708 K head per 4-hop fragment. Inference sees only PCM
already received; the original six scores are evaluated on the same
emitted causal MIDI stream, not on detached K logits.

No alternative backend, TEST or HELD access, student distillation, LIVE
deployment, or automatic promotion. S708/S707/S80 remain unchanged.
"""
from __future__ import annotations
import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path
import tempfile
import zipfile

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from causal_note.strict_live import SR
from causal_note.s708_cardinality_selection import GuidedPolicy
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s704_train_joint_loops import targets
from scripts.s704_joint_metrics import SCORES,metric,aggregate
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import (
    FIT_COMPS,CAL_COMPS,load_model)
from scripts.s733_train_onset_only import baseline_input
from scripts.s731_auditory_scene_calibration import emit

ARCHITECTURES={
    "depth3":dict(max_depth=3,max_iter=95,learning_rate=.08),
    "depth5":dict(max_depth=5,max_iter=120,learning_rate=.06),
}
# Separate rare-class weights: S755's pooled >=5 strategy
# never selected K6 on CAL32 among 189 true K6 decisions.
WEIGHTS={
    "rare":(1.,2.5,12.,30.),
    "rare_strong":(1.,3.,20.,65.),
}
BLENDS=(.15,.35,.55)
GRID={f"{depth}_{weight}_blend{int(blend*100)}":
      (depth,weight,blend) for depth,weight,blend in
      product(ARCHITECTURES,WEIGHTS,BLENDS)}
assert len(GRID)==12


def causal_k_features(x,post,k,native):
    """Per-decision current and PAST audio/posteriors. No oracle or duration.

    Input ABI matches S708: [T,49,288] causal sharp spectrum,
    [T,49,3] onset/sustain/release channels and [T,7] K posterior.
    """
    x=np.asarray(x,np.float32)
    p=np.asarray(post,np.float32)
    n=np.asarray(native,np.float32)
    k=np.asarray(k,np.float32)
    t=len(k)
    if (t==0 or x.shape!=(t,49,288) or p.shape!=(t,49,3)
        or n.shape!=(t,49,3) or k.shape!=(t,7)
        or not all(np.isfinite(v).all() for v in (x,p,n,k))):
        raise ValueError("S755 signed NOTE acoustic K decision ABI mismatch")
    spectra=x[:,:,192:216].reshape(t,49,3,8)
    acoustic=np.maximum(spectra.mean(axis=(2,3)),0)
    top_acoustic=np.sort(acoustic,axis=1)[:,-8:]
    p_top=np.sort(p,axis=1)[:,-8:,:].reshape(t,24)
    native_top=np.sort(n,axis=1)[:,-8:,:].reshape(t,24)
    counts=np.stack([
        (p[:,:,channel]>threshold).sum(axis=1)
        for channel in range(3) for threshold in (.10,.30,.50,.70)
    ],axis=-1).astype(np.float32)/49.
    prev=p[np.maximum(np.arange(t)-1,0)]
    rises=np.maximum(p-prev,0)
    rise_top=np.sort(rises[:,:,0],axis=1)[:,-8:]
    history=np.concatenate([
        k[np.maximum(np.arange(t)-lag,0)]
        for lag in (1,4,16)],axis=-1)
    result=np.concatenate((k,history,p_top,native_top,
        top_acoustic,counts,rise_top),axis=-1).astype(np.float32)
    if not np.isfinite(result).all() or result.shape[1]!=104:
        # 7+21+24+24+8+12+8 = 104 scalar causal inputs.
        raise AssertionError("S755 K-head feature schema changed")
    return result



def coherent_k_features(x,post,k,native):
    """Pitch-coherent per-fragment K features, no label or future PCM.

    S755 separately sorted each channel by pitch, losing the association
    between a candidate note's onset, sustain, release and harmonics.
    A single stable pitch ranking preserves those joint observations.
    """
    original=causal_k_features(x,post,k,native)
    p=np.asarray(post,np.float32)
    n=np.asarray(native,np.float32)
    t=len(original)
    spectra=np.asarray(x,np.float32)[:,:,192:216].reshape(t,49,3,8)
    strength=np.maximum(spectra.mean(axis=(2,3)),0)
    strength=strength/(1.+strength)
    score=.45*p[:,:,0]+.35*p[:,:,1]+.20*strength
    order=np.argsort(-score,axis=1,kind="stable")[:,:8]
    def gather(array):
        return np.take_along_axis(array,order[:,:,None],axis=1)
    aligned_p=gather(p).reshape(t,24)
    aligned_n=gather(n).reshape(t,24)
    harmonics=np.take_along_axis(
        spectra[:,:,2,:],order[:,:,None],axis=1).reshape(t,64)
    previous=p[np.maximum(np.arange(t)-1,0)]
    rise=np.maximum(p[:,:,0]-previous[:,:,0],0)
    aligned_rise=np.take_along_axis(rise,order,axis=1)
    pairs=[]
    for shift in (12,19):
        for signal in (p[:,:,0],p[:,:,1]):
            interaction=signal[:,:49-shift]*signal[:,shift:]
            pairs.extend((
                interaction.max(axis=1),
                interaction.mean(axis=1),
                (interaction>.15).sum(axis=1).astype(np.float32)/49.,
                (interaction>.35).sum(axis=1).astype(np.float32)/49.))
    features=np.concatenate((
        original,aligned_p,aligned_n,harmonics,aligned_rise,
        np.stack(pairs,axis=-1)),axis=-1).astype(np.float32)
    if features.shape!=(t,240) or not np.isfinite(features).all():
        raise AssertionError("S755 pitch-coherent 240D K feature ABI mismatch")
    return features


def calibrated_k(base,estimated,blend):
    """Blend the original and independent FIT-trained 7-class K heads."""
    b=np.asarray(base,np.float32)
    q=np.asarray(estimated,np.float32)
    if (b.shape!=q.shape or b.ndim!=2 or b.shape[1]!=7
        or not np.isfinite(q).all() or np.any(q<0)
        or not 0<=blend<=1):
        raise ValueError("S755 invalid K specialist output")
    candidate=(1-blend)*b+blend*q
    candidate/=candidate.sum(-1,keepdims=True).clip(min=1e-8)
    return candidate.astype(np.float32)


def report_write(out,report):
    (out/"report.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n")


def evaluate(args):
    if args.out.exists():
        raise FileExistsError("S755 experiment output must be new and immutable")
    args.out.mkdir(parents=True)
    signed=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    if sha!=signed["checkpoint_sha256"]:
        raise AssertionError("S755 immutable signed S708 checkpoint mismatch")
    release=signed["frozen_candidate"]["choice"]["release"]
    if release!=dict(gate=.8,persistence=1,grace=24):
        raise AssertionError("S755 signed S707 release policy mismatch")
    policy=GuidedPolicy(**signed["frozen_candidate"]["choice"]["policy"])
    torch.set_num_threads(2)
    net=load_model(args.s708_dir/"s708_selected.pt")[0]
    report=dict(schema="S755_PITCH_COHERENT_K2_K6_SPECIALISTS_TWELVE_FULL_MIDI_CAL32",
                signed_s708_sha256=sha,source="S708_frozen",
                complete=False,promoted=False,selected="S708_frozen",
                TEST_opened=False,HELD_opened=False,LIVE_changed=False,
                FIT=dict(),CAL={},loops=12,training="FIT80 only",
                split="FIT80 CAL32 compositions-disjoint",
                no_truth_at_inference=True,reference_parity=None,
                grid={name:dict(model=arch,weight=weight,blend=blend)
                      for name,(arch,weight,blend) in GRID.items()})
    report_write(args.out,report)
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        proto,heads,selection,offset=setup(root,args.offset_checkpoint)
        for name,file in (("GuitarSet_audio","GuitarSet_audio.zip"),
                          ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            a=proto["guitar_archives"][name]
            download(a["url"],root/"data"/file,a["size"],a["sha256"])
        originals=[s for s in proto["sources"]
                   if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as audio,\
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as annotations:
            expanded=expanded_train_specs(originals,audio,annotations)
            fit=[s for s in expanded if s["composition"] in FIT_COMPS]
            cal=[s for s in expanded if s["composition"] in CAL_COMPS]
            if (len(fit),len(cal))!=(80,32) or set(FIT_COMPS)&set(CAL_COMPS):
                raise AssertionError("S755 FIT/CAL split differs from signed project protocol")
            xs=[];ys=[];counts=Counter()
            for idx,spec in enumerate(fit):
                item,post,k,events,_,_,_=baseline_input(
                    spec,audio,root,heads,selection,offset,net,policy,release)
                features=coherent_k_features(item["x"],post,k,item["native_post"])
                # FIT annotations only; model inputs were finalized before
                # reading the supervision.
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                _,target=targets(item["decision_ends"],truth)
                if len(target)!=len(features):
                    raise AssertionError("S755 FIT K-target alignment mismatch")
                xs.append(features);ys.append(target)
                counts.update(int(v) for v in target)
                print("S755 FIT",idx+1,80,"frames",len(target),flush=True)
            X=np.concatenate(xs)
            y=np.concatenate(ys)
            if not set(range(7)).issubset(set(int(v) for v in np.unique(y))):
                raise AssertionError("S755 FIT missing one or more K0–K6 classes")
            report["FIT"]=dict(decisions=len(y),class_counts=dict(counts),
                compositions=list(FIT_COMPS),features=X.shape[1])
            report_write(args.out,report)
            trained={}
            for architecture,weight_profile in product(ARCHITECTURES,WEIGHTS):
                cfg=ARCHITECTURES[architecture]
                w0,wpoly,w5,w6=WEIGHTS[weight_profile]
                sample_weight=np.select(
                    (y==6,y==5,y>=2),(w6,w5,wpoly),
                    default=w0).astype(np.float32)
                clf=HistGradientBoostingClassifier(
                    random_state=755,max_leaf_nodes=31,
                    l2_regularization=.5,early_stopping=False,**cfg)
                clf.fit(X,y,sample_weight=sample_weight)
                if not np.array_equal(clf.classes_,np.arange(7)):
                    raise AssertionError("S755 specialist class coverage mismatch")
                key=(architecture,weight_profile)
                trained[key]=clf
                joblib.dump(dict(
                    model=clf,training="FIT80_only",
                    source_sha256=sha,features=X.shape[1],
                    classes=clf.classes_.tolist(),
                    class_weights=WEIGHTS[weight_profile]),
                    args.out/f"s755_{architecture}_{weight_profile}.joblib")
                print("S755 SPECIALIST TRAINED",*key,flush=True)
            all_rows={name:[] for name in ("S708_frozen",*GRID)}
            lost={name:Counter() for name in GRID}
            head_score={name:Counter() for name in ("S708_frozen",*GRID)}
            for index,spec in enumerate(cal):
                item,post,k,original,_,_,_=baseline_input(
                    spec,audio,root,heads,selection,offset,net,policy,release)
                feats=coherent_k_features(item["x"],post,k,item["native_post"])
                candidates={"S708_frozen":original}
                K={"S708_frozen":k}
                predictions={
                    family:trained[family].predict_proba(feats).astype(np.float32)
                    for family in trained}
                for name,(arch,weight,blend) in GRID.items():
                    adjusted=calibrated_k(k,predictions[(arch,weight)],blend)
                    K[name]=adjusted
                    candidates[name]=emit(item,post,adjusted,policy,offset,release)
                # Labels appear strictly AFTER every variant has emitted its
                # complete NoteOn/NoteOff MIDI stream on the same PCM.
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                _,truth_k=targets(item["decision_ends"],truth)
                observed={
                    name:metric(events,truth,item["ends"])
                    for name,events in candidates.items()
                }
                base=observed["S708_frozen"]
                for name,scores in observed.items():
                    all_rows[name].append(scores)
                    pred=np.argmax(K[name],axis=1)
                    head_score[name].update(
                        total=len(truth_k),
                        correct=int(np.count_nonzero(pred==truth_k)),
                        k5_total=int(np.count_nonzero(truth_k==5)),
                        k5_correct=int(np.count_nonzero((truth_k==5)&(pred==5))),
                        k6_total=int(np.count_nonzero(truth_k==6)),
                        k6_correct=int(np.count_nonzero((truth_k==6)&(pred==6))))
                for name in GRID:
                    trial=observed[name]
                    lost[name].update(
                        correct_offsets=len(
                            set(base["correct_offset_truth_ids"])-
                            set(trial["correct_offset_truth_ids"])),
                        long_onsets=len(
                            set(base["long_onset_truth_ids"])-
                            set(trial["long_onset_truth_ids"])),
                        onset_tp=max(0,int(base["tp"])-int(trial["tp"])))
                report["CAL_recordings_done"]=index+1
                report_write(args.out,report)
                print("S755 CAL",index+1,32,spec["name"],flush=True)
    summarized={name:aggregate(s) for name,s in all_rows.items()}
    baseline=summarized["S708_frozen"]
    reference=signed["frozen_candidate"]["CAL"]
    parity={name:float(baseline[name]-reference[name]) for name in SCORES}
    parity_valid=all(abs(d)<=1e-8 for d in parity.values())
    report["reference_parity"]=dict(valid=parity_valid,delta=parity)
    report["reference_scores"]={name:float(baseline[name]) for name in SCORES}
    report["reference_head"]=dict(head_score["S708_frozen"])
    safe=[]
    for name in GRID:
        score=summarized[name]
        loss=dict(lost[name])
        passed=bool(parity_valid and all(
            score[key]>=baseline[key]-1e-12 for key in SCORES)
            and score["tp"]>=baseline["tp"]
            and all(v==0 for v in loss.values()))
        delta={key:float(score[key]-baseline[key]) for key in SCORES}
        report["CAL"][name]=dict(
            scores={key:float(score[key]) for key in SCORES},
            delta=delta,correct_source_offsets_lost=loss["correct_offsets"],
            long_onsets_lost=loss["long_onsets"],onset_tp_lost=loss["onset_tp"],
            safe=passed,six_gt80=bool(passed and all(
                score[key]>.80 for key in SCORES)),
            head=dict(head_score[name]))
        if passed and any(d>1e-12 for d in delta.values()):
            safe.append(name)
    report["CAL_candidate"]=max(safe,key=lambda name:(
        min(report["CAL"][name]["scores"].values()),
        sum(report["CAL"][name]["delta"].values())),default=None)
    report["complete"]=parity_valid
    report_write(args.out,report)
    if not parity_valid:
        raise AssertionError("S755 changed signed S708 six-score CAL32 reference")
    print("S755_COMPLETE",json.dumps({
        "reference":report["reference_scores"],
        "K6_head_reference":report["reference_head"]["k6_correct"],
        "CAL_candidate":report["CAL_candidate"],
        "safe":safe,
        "best_head_k6":max(
            ((name,metrics["head"]["k6_correct"])
             for name,metrics in report["CAL"].items()),
            key=lambda trial: (trial[1],trial[0])),
        "best_active":max(
            ((name,metrics["scores"]["active_exact_k"])
             for name,metrics in report["CAL"].items()),
            key=lambda trial: (trial[1],trial[0]))}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--s708-dir",type=Path,required=True)
    p.add_argument("--offset-checkpoint",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    evaluate(p.parse_args())
