"""S745: FIT80 supervised multihead fundamental birth, twelve real MIDI CAL32 loops.

Separate specialist, never distillation. Two FIT-only HGB heads are fitted
to causal S708 acoustics and explicit missing-Onset labels, with
composition-disjoint evaluation. H5 candidate probabilities feed the
original S743 32 possible per-fragment dynamic subsets. Frozen signed S708,
six complete-MIDI metrics and original S707 NoteOff remain unchanged.
No TEST, no LIVE promotion without strict paired no-regression.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
import tempfile
import zipfile
import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from causal_note.strict_live import SR
from causal_note.s708_cardinality_selection import GuidedPolicy
from causal_note.s707_causal_release import prepare_hazards,render_release
from causal_note.s731_auditory_scene import scene_scores
from causal_note.s743_fundamental_router import (
    MultiHeadBirthPolicy,candidate_heads,decode_s743)
from causal_note.s744_birth_specialist import (
    frozen_source_activity,birth_features,select_fit_examples,
    boost_fit_weights)
from causal_note.s745_calibrated_birth_router import (
    CalibratedBirthPolicy,fit_prior_correction,corrected_specialist_track,
    decode_s745)
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s704_joint_metrics import SCORES,metric,aggregate
from scripts.s704_train_joint_loops import targets
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import FIT_COMPS,CAL_COMPS,load_model
from scripts.s735_audit_32_onset_combinations import load_s734
from scripts.s741_train_exactk_fragments import one_k_track

# Two fully FIT-trained specialist depths x two probability thresholds
# x two causal K-tail confidence levels = twelve REAL event-level loops.
MODELS={"depth3":dict(max_depth=3,max_iter=130,learning_rate=.07),
        "depth5":dict(max_depth=5,max_iter=160,learning_rate=.05)}
CHOICES={
    f"{model}_risk{int(thr*1000)}_tail{int(tail*100)}":
    (model,CalibratedBirthPolicy(
        min_probability=thr,k_tail=tail,
        min_acoustic=.13,min_sustain=.35,
        min_rise=.015,mask_delta=.04,
        mode="adaptive",max_new=1))
    for model,thr,tail in product(
        tuple(MODELS),(.004,.016,.055),(.25,.45))
}
assert len(CHOICES)==12


def save(path,x):
    path.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")


def track_sources(spec,audio,root,heads,selection,offset,network,policy,release,h5):
    tr=one_k_track(spec,audio,root,heads,selection,offset,
                   network,policy,release,h5)
    item=tr["item"]
    scene=scene_scores(item["x"],tr["post"],tr["k"],
                       tr["baseline"],item["decision_ends"])
    h=candidate_heads(tr["post"],item["native_post"],
                      tr["context"],item["x"],scene)
    state=frozen_source_activity(
        tr["post"],tr["k"],item["decision_ends"],policy)
    acoustic=birth_features(tr["post"],item["native_post"],
                tr["context"],item["x"],tr["k"],state)
    tr.update(physical_heads=h,source_active=state,birth_input=acoustic)
    return tr


def run(args):
    if args.out.exists():
        raise FileExistsError("Immutable FIT model and CAL report requested")
    args.out.mkdir(parents=True)
    reference=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    if sha!=reference["checkpoint_sha256"]:
        raise AssertionError("S708 signed source mismatch")
    h5report=json.loads(args.s734_report.read_text())
    if h5report["base_sha256"]!=sha or not h5report["complete"]:
        raise AssertionError("S734 source mismatch")
    torch.set_num_threads(2)
    network=load_model(args.s708_dir/"s708_selected.pt")[0]
    h5=load_s734(args.s734_checkpoint,sha)
    policy=GuidedPolicy(**reference["frozen_candidate"]["choice"]["policy"])
    release=reference["frozen_candidate"]["choice"]["release"]
    if release!=dict(gate=.8,persistence=1,grace=24):
        raise AssertionError("Frozen S707 NoteOff policy changed")
    report=dict(schema="S745_REAL_FIT_SUPERVISED_ACTIVE_K_HEAD_CAL32",
        base_signed_sha256=sha,training="FIT80 only",
        target="additional_acoustically_confirmed_NoteOn",
        loops=len(CHOICES),completed=False,promoted=False,
        LIVE_changed=False,TEST_opened=False,
        selected="S708_frozen",FIT={},CAL={},param_grid={
            name:dict(model=family,policy=asdict(choice))
            for name,(family,choice) in CHOICES.items()})
    save(args.out/"report.json",report)
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary);(root/"data").mkdir()
        proto,heads,selection,offset=setup(root,args.offset_checkpoint)
        for kind,name in (("GuitarSet_audio","GuitarSet_audio.zip"),
                          ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            data=proto["guitar_archives"][kind]
            download(data["url"],root/"data"/name,data["size"],data["sha256"])
        train=[x for x in proto["sources"] if
               x["corpus"]=="GuitarSet" and x["split"]=="train"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as audio, \
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as annotations:
            expanded=expanded_train_specs(train,audio,annotations)
            groups={
                "FIT":[x for x in expanded if x["composition"] in FIT_COMPS],
                "CAL":[x for x in expanded if x["composition"] in CAL_COMPS]}
            if [len(groups[z]) for z in ("FIT","CAL")]!=[80,32]:
                raise AssertionError("S745 composition-disjoint source split")
            samples=[];labels_fit=[];stats=[]
            for i,spec in enumerate(groups["FIT"]):
                track=track_sources(spec,audio,root,heads,selection,offset,
                                    network,policy,release,h5)
                # Causal acoustic features and original emitted state were
                # frozen BEFORE the FIT labels are opened.
                truth=real_notes(labels(spec,root)["notes"],
                                 track["item"]["samples"]/SR)
                active_labels,_=targets(
                    track["item"]["decision_ends"],truth)
                X,y,count=select_fit_examples(
                    track["birth_input"],active_labels[:,:,0],
                    track["source_active"],track["context"],
                    track["physical_heads"],seed=744+i,limit=2000)
                samples.append(X);labels_fit.append(y)
                stats.append(count)
                print("S745 FIT",i+1,80,"positives",count["positives"],
                      "all_positives",count["all_true_onsets"],flush=True)
            X=np.concatenate(samples,axis=0)
            y=np.concatenate(labels_fit)
            if not set(np.unique(y))=={0,1}:
                raise AssertionError("FIT needs both true and false attacks")
            report["FIT"]=dict(
                rows=int(len(y)),positives=int(y.sum()),
                true_positive_candidates=sum(
                    z["true_positive_candidates"] for z in stats),
                all_true_onsets=sum(z["all_true_onsets"] for z in stats),
                split_compositions=sorted({
                    z["composition"] for z in groups["FIT"]}))
            save(args.out/"report.json",report)
            factor,correction=fit_prior_correction(
                natural_positive=sum(z["true_positive_candidates"] for z in stats),
                total_eligible=sum(z["natural_candidate_population"] for z in stats),
                sampled_positive=int(y.sum()),sampled_total=len(y),
                positive_weight=4.)
            report["FIT"]["population_prior_correction"]=correction
            save(args.out/"report.json",report)
            models={}
            for name,hyper in MODELS.items():
                estimator=HistGradientBoostingClassifier(
                    random_state=744 if name=="depth3" else 745,
                    l2_regularization=.25,early_stopping=False,
                    max_bins=255,**hyper)
                estimator.fit(X,y,sample_weight=boost_fit_weights(y,4.))
                modelpath=args.out/f"s745_{name}.joblib"
                joblib.dump(dict(
                    base_signed_sha256=sha,
                    model=estimator,
                    split="FIT80 only",
                    features=X.shape[-1],
                    population_prior_correction=correction),modelpath)
                models[name]=estimator
                print("S745 TRAINED",name,"rows",len(y),flush=True)
            # No CAL data participates in model fit.
            names=["S708_frozen",*CHOICES]
            records={name:[] for name in names}
            lost={name:Counter() for name in CHOICES}
            detail={name:Counter() for name in CHOICES}
            hist={name:Counter() for name in CHOICES}
            for i,spec in enumerate(groups["CAL"]):
                tr=track_sources(spec,audio,root,heads,selection,offset,
                                 network,policy,release,h5)
                item=tr["item"]
                augmented={name:corrected_specialist_track(
                    models[name],tr["birth_input"],factor)
                    for name in models}
                events={"S708_frozen":tr["baseline"]}
                for name,(family,parameters) in CHOICES.items():
                    source,diagnostic=decode_s745(
                        item["decision_ends"],tr["post"],tr["k"],
                        tr["physical_heads"],tr["context"],
                        augmented[family],policy,parameters,
                        item["samples"])
                    final,skip=render_release(
                        prepare_hazards(item,source,offset),
                        item["ends"],item["samples"],**release)
                    events[name]=final
                    detail[name].update(dict(
                        added_births=diagnostic["added_births"],
                        blocked_K=diagnostic["blocked_K"],
                        blocked_acoustic=diagnostic["blocked_acoustic"],
                        skipped_polyphony=int(skip)))
                    hist[name].update(diagnostic["masks"])
                # Now (and only now) CAL annotation is used to score
                # the already-fixed MIDI candidate event streams.
                truth=real_notes(labels(spec,root)["notes"],
                                 item["samples"]/SR)
                measured={name:metric(ev,truth,item["ends"])
                          for name,ev in events.items()}
                baseline=measured["S708_frozen"]
                for name in names:
                    records[name].append(measured[name])
                for name in CHOICES:
                    now=measured[name]
                    lost[name]["correct_offsets"]+=len(
                        set(baseline["correct_offset_truth_ids"])-
                        set(now["correct_offset_truth_ids"]))
                    lost[name]["long_onsets"]+=len(
                        set(baseline["long_onset_truth_ids"])-
                        set(now["long_onset_truth_ids"]))
                print("S745 REAL CAL",i+1,32,spec["name"],
                      "reference_activeK",round(baseline["active_exact_k"],5),
                      "best_trial_activeK",round(max(measured[z]["active_exact_k"]
                                                     for z in CHOICES),5),flush=True)
                report["CAL_recordings_done"]=i+1
                save(args.out/"report.json",report)
            aggregated={k:aggregate(v) for k,v in records.items()}
            original=reference["frozen_candidate"]["CAL"]
            base=aggregated["S708_frozen"]
            for metric_name in SCORES:
                if abs(base[metric_name]-original[metric_name])>1e-8:
                    raise AssertionError("Signed S708 baseline changed "+metric_name)
            eligible=[]
            for name,(family,parameters) in CHOICES.items():
                x=aggregated[name]
                safe=bool(all(x[k]>=base[k]-1e-12 for k in SCORES)
                       and lost[name]["correct_offsets"]==0
                       and lost[name]["long_onsets"]==0
                       and x["tp"]>=base["tp"])
                result=dict(
                    scores={k:float(x[k]) for k in SCORES},
                    delta={k:float(x[k]-base[k]) for k in SCORES},
                    safe=safe,
                    added_births=detail[name]["added_births"],
                    offset_losses=lost[name]["correct_offsets"],
                    long_onset_losses=lost[name]["long_onsets"],
                    mask_histogram=dict(hist[name]),
                    six_gt80=bool(safe and all(x[k]>.80 for k in SCORES)))
                report["CAL"][name]=result
                if safe and any(result["delta"][k]>1e-10 for k in SCORES):
                    eligible.append((name,result))
                print("S745 LOOP",json.dumps(dict(
                    name=name,added_births=result["added_births"],
                    active_exact_k=result["scores"]["active_exact_k"],
                    safe=safe,offset_losses=result["offset_losses"])),flush=True)
            report["CAL_candidate"]=max(eligible,key=lambda t:(
                t[1]["six_gt80"],min(t[1]["delta"].values()),
                t[1]["delta"]["active_exact_k"]))[0] if eligible else None
            report["reference_scores"]={k:float(base[k]) for k in SCORES}
            report["completed"]=True
            report["promoted"]=False
            report["selected"]="S708_frozen"
            save(args.out/"report.json",report)
            print("S745_FINAL",json.dumps(dict(
                loops=len(CHOICES),trained_models=list(MODELS),
                eligible_CAL_candidate=report["CAL_candidate"],
                best_observed_activeK=max(v["scores"]["active_exact_k"]
                                          for v in report["CAL"].values()),
                six_gt80=any(v["six_gt80"] for v in report["CAL"].values()),
                promoted=False,LIVE_changed=False)),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s708-dir",type=Path,required=True)
    parser.add_argument("--s734-checkpoint",type=Path,required=True)
    parser.add_argument("--s734-report",type=Path,required=True)
    parser.add_argument("--offset-checkpoint",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        run(args)
