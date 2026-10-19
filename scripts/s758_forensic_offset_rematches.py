"""S758: exact-source event forensic for the three missing correct Offsets.

For *one frozen S757 candidate* (depth5_rare_guard160ms), reconstruct
real GuitarSet CAL32, keep S708 events byte-identical, then inspect
post-hoc Hungarian matching of old and new NoteOn/NoteOff to truth.
All annotations remain forbidden from decisions; no model training or
score optimization, no HELD/TEST or LIVE modifications.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile
import zipfile

import joblib
import numpy as np
import torch

from causal_note.strict_live import SR
from causal_note.s708_cardinality_selection import GuidedPolicy
from causal_note.s746_instance_guard import InstanceGate,merge_events
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import real_notes,event_score
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s704_joint_metrics import SCORES,metric,aggregate
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import (
    FIT_COMPS,CAL_COMPS,load_model)
from scripts.s733_train_onset_only import baseline_input
from scripts.s731_auditory_scene_calibration import emit
from scripts.s755_pitch_coherent_k_fit_cal32 import (
    coherent_k_features,calibrated_k)
from scripts.s757_source_onset_collision_guard import select_novel_births

EXPERT=("depth5","rare")
GUARD=.16
RISK=.55
BLEND=.55
EXPECTED=dict(onset_f1=.7431763361685921,
              exact_k=.6225456042563973,
              active_exact_k=.5655756511676626,
              offset_accuracy=.771556166265888,
              full_note_f1=.5702678684778469,
              frame_pitch_f1=.766179302861793)


def get_intervals(events):
    """Predictor positions in event_score's ordered NoteOn list."""
    ons=[];intervals={};active={}
    for e in events:
        pitch=int(e["pitch"])
        if e["type"]=="note_on":
            if pitch in active:
                raise AssertionError("S758 source overlap")
            i=len(ons)
            ons.append(e)
            active[pitch]=i
        elif e["type"]=="note_off":
            if pitch not in active:
                raise AssertionError("S758 source orphan")
            intervals[active.pop(pitch)]=e
        else:
            raise ValueError("S758 bad MIDI type")
    if active:
        raise AssertionError("S758 unclosed MIDI prediction")
    return ons,intervals


def signature(event):
    return (event["type"],int(event["pitch"]),
            int(event["decision_sample"]),int(event["available_sample"]),
            event["reason"])


def inspect_one(recording,reference,trial,truth,ends):
    """Read truth only after both entire full MIDI streams are finalized."""
    src_on,src_off=get_intervals(reference)
    new_on,new_off=get_intervals(trial)
    _,old_pairs=event_score(reference,truth)
    _,new_pairs=event_score(trial,truth)
    old_match={j:i for i,j in old_pairs}
    new_match={j:i for i,j in new_pairs}
    signed=[signature(e) for e in reference]
    frozen=[signature(e) for e in trial if not e.get("_s746_extra")
            and not str(e.get("reason","")).startswith("s746_")]
    if signed!=frozen:
        raise AssertionError("S758 lost signed S708 event metadata")
    score_source=metric(reference,truth,ends)
    score_trial=metric(trial,truth,ends)
    lost=sorted(set(score_source["correct_offset_truth_ids"])-
                set(score_trial["correct_offset_truth_ids"]))
    cases=[]
    new_native=Counter()
    for j in lost:
        n=truth[j]
        old_idx=old_match[j]
        new_idx=new_match.get(j)
        original_on=src_on[old_idx]
        original_off=src_off.get(old_idx)
        latest=new_on[new_idx] if new_idx is not None else None
        latest_off=new_off.get(new_idx) if new_idx is not None else None
        same_native=(latest is not None and signature(latest)==signature(original_on))
        cause=("unmatched" if latest is None else
               "reassigned_to_supplemental" if latest.get("_s746_extra") else
               "reassigned_to_other_native" if not same_native else
               "same_onset_termination_changed")
        near=[dict(kind=e["type"],pitch=int(e["pitch"]),
                   when=float(e["time"]),reason=e["reason"],
                   is_extra=bool(e.get("_s746_extra")),
                   signed=signature(e) in signed)
              for e in trial if int(e["pitch"])==int(n["pitch"])
              and abs(float(e["time"])-float(n["time"]))<=.6]
        cases.append(dict(
            recording=recording,truth_id=int(j),
            truth=dict(pitch=int(n["pitch"]),onset=float(n["time"]),
                       release=float(n["time"]+n["duration"])),
            source_on=dict(pitch=int(original_on["pitch"]),
                when=float(original_on["time"]),reason=original_on["reason"]),
            source_off=dict(when=float(original_off["time"]),
                            reason=original_off["reason"]) if original_off else None,
            trial_on=dict(when=float(latest["time"]),reason=latest["reason"],
                is_extra=bool(latest.get("_s746_extra"))) if latest else None,
            trial_off=dict(when=float(latest_off["time"]),
                reason=latest_off["reason"]) if latest_off else None,
            cause=cause,nearby_same_pitch_events=near))
        new_native[cause]+=1
    return cases,dict(new_native),score_source,score_trial


def run(args):
    if args.out.exists():
        raise FileExistsError("S758 immutable output cannot be overwritten")
    args.out.mkdir(parents=True)
    frozen=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    if sha!=frozen["checkpoint_sha256"]:
        raise AssertionError("S758 changed native signed S708 SHA")
    release=frozen["frozen_candidate"]["choice"]["release"]
    if release!=dict(gate=.8,persistence=1,grace=24):
        raise AssertionError("S758 changed original S707 release")
    policy=GuidedPolicy(**frozen["frozen_candidate"]["choice"]["policy"])
    package=joblib.load(args.specialist_dir/"s755_depth5_rare.joblib")
    if (package["source_sha256"]!=sha or
        package["training"]!="FIT80_only" or package["features"]!=240):
        raise AssertionError("S758 specialist not fitted only on FIT80")
    model=package["model"]
    torch.set_num_threads(2)
    network=load_model(args.s708_dir/"s708_selected.pt")[0]
    report=dict(schema="S758_GUITARSET_CAL32_SIGNED_OFFSET_LOSSES_FORENSICS",
        complete=False,promoted=False,source_sha256=sha,
        candidate="S757_depth5_rare_guard160ms",lookahead=False,
        trained_on_CAL=False,HELD_opened=False,TEST_opened=False,
        LIVE_changed=False,recordings_done=0,lost_cases=[],
        case_causes={},source_parity=None,candidate_parity=None,
        tracks=[])
    def save():
        (args.out/"report.json").write_text(
            json.dumps(report,indent=2,sort_keys=True)+"\n")
    save()
    losses=Counter();total_source=[];total_trial=[]
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        proto,heads,selection,offset=setup(root,args.offset_checkpoint)
        for name,filename in (
            ("GuitarSet_audio","GuitarSet_audio.zip"),
            ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            z=proto["guitar_archives"][name]
            download(z["url"],root/"data"/filename,z["size"],z["sha256"])
        originals=[s for s in proto["sources"]
                   if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as audio,\
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            specs=[s for s in expanded_train_specs(originals,audio,ann)
                   if s["composition"] in CAL_COMPS]
            if len(specs)!=32 or set(FIT_COMPS)&set(CAL_COMPS):
                raise AssertionError("S758 expected original disjoint CAL32")
            for i,spec in enumerate(specs):
                item,post,k,original,_,_,_=baseline_input(
                    spec,audio,root,heads,selection,offset,network,policy,release)
                features=coherent_k_features(item["x"],post,k,item["native_post"])
                probs=model.predict_proba(features).astype(np.float32)
                fused=calibrated_k(k,probs,BLEND)
                proposed=emit(item,post,fused,policy,offset,release)
                pairs,confidence,indices,diag=select_novel_births(
                    original,proposed,item["decision_ends"],post,k,fused,
                    source_guard_seconds=GUARD)
                trial,details=merge_events(
                    original,pairs,confidence,indices,fused,
                    InstanceGate(risk=RISK,k_tail=.10))
                # Forensic annotations are read AFTER both full MIDI
                # streams exist. They must never change trial decisions.
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                cases,causes,old,new=inspect_one(
                    spec["name"],original,trial,truth,item["ends"])
                report["lost_cases"].extend(cases)
                losses.update(causes)
                total_source.append(old);total_trial.append(new)
                report["tracks"].append(dict(
                    name=spec["name"],lost=len(cases),
                    accepted=int(details["accepted"]),
                    preempted=int(details["preempted"]),
                    causes=causes))
                report["recordings_done"]=i+1
                report["case_causes"]=dict(losses)
                save()
                print("S758 FORENSIC",i+1,32,
                      spec["name"],"lost",len(cases),
                      "accepted",details["accepted"],flush=True)
    score1=aggregate(total_source);score2=aggregate(total_trial)
    expected=frozen["frozen_candidate"]["CAL"]
    source_delta={s:float(score1[s]-expected[s]) for s in SCORES}
    trial_delta={s:float(score2[s]-EXPECTED[s]) for s in SCORES}
    report["source_parity"]={
        "valid":all(abs(z)<1e-8 for z in source_delta.values()),
        "delta":source_delta}
    report["candidate_parity"]={
        "valid":all(abs(z)<1e-8 for z in trial_delta.values()),
        "delta":trial_delta}
    report["source_scores"]={s:float(score1[s]) for s in SCORES}
    report["candidate_scores"]={s:float(score2[s]) for s in SCORES}
    report["complete"]=report["source_parity"]["valid"] and report[
        "candidate_parity"]["valid"]
    save()
    if not report["complete"]:
        raise AssertionError("S758 source or S757 candidate signature drift")
    print("S758_THREE_OFFSET_LOSSES",json.dumps(dict(
        count=len(report["lost_cases"]),causes=dict(losses),
        cases=report["lost_cases"],
        source_parity=report["source_parity"]["valid"],
        candidate_parity=report["candidate_parity"]["valid"])),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--s708-dir",type=Path,required=True)
    p.add_argument("--offset-checkpoint",type=Path,required=True)
    p.add_argument("--specialist-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    run(p.parse_args())
