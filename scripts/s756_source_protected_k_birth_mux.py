"""S756: S755 poly-K specialists propose births, never change S708 notes.

The FIT80-only pretrained S755 K models operate causally. Proposed K
distributions are decoded to *candidate* MIDI independently. Only truly new
NoteOn/NoteOff pairs enter a strict availability-ordered S746 merger. Every
original S708 NoteOn, NoteOff, decision and availability timestamp remains
identical and has absolute priority. Acceptance at NoteOn uses only current
onset/sustain and K probabilities, never the future NoteOff or CAL truth.

Four specialists x three causal onset-confidence thresholds = 12 true
GuitarSet CAL32 full-MIDI evaluations on the same signed S708/S707 sources.
This script is a non-promoted research branch; not a LIVE deployment.
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

from causal_note.strict_live import SR
from causal_note.s708_cardinality_selection import GuidedPolicy
from causal_note.s746_instance_guard import InstanceGate, merge_events
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s704_joint_metrics import SCORES,metric,aggregate
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import (
    FIT_COMPS,CAL_COMPS,load_model)
from scripts.s733_train_onset_only import baseline_input
from scripts.s731_auditory_scene_calibration import emit
from scripts.s755_pitch_coherent_k_fit_cal32 import (
    ARCHITECTURES,WEIGHTS,coherent_k_features,calibrated_k)

FAMILIES=tuple(product(ARCHITECTURES,WEIGHTS))
RISK=(.40,.55,.70)
BLEND=.55
GRID={
    f"{arch}_{weights}_risk{int(risk*100)}":(arch,weights,risk)
    for (arch,weights),risk in product(FAMILIES,RISK)
}
assert len(GRID)==12


def pairs_from_events(events):
    """Extract causally closed candidate notes without allowing rereleases."""
    active={};pairs=[]
    for e in events:
        pitch=int(e["pitch"])
        if e["type"]=="note_on":
            if pitch in active:
                raise AssertionError("S756 candidate duplicate active pitch")
            active[pitch]=e
        elif e["type"]=="note_off":
            on=active.pop(pitch,None)
            if on is None:
                raise AssertionError("S756 candidate orphan NoteOff")
            pairs.append((on,e))
        else:
            raise ValueError("S756 unexpected MIDI event")
    if active:
        raise AssertionError("S756 candidate stream not closed")
    return pairs


def select_novel_births(reference,candidate,ends,post,native_k,fused_k):
    """Source-only, causal novelty filtering and per-onset acoustic scores.

    Candidate NoteOff is treated as an opaque future closure passed to
    the mux; its time, duration and reason CANNOT affect NoteOn ranking.
    """
    ends=np.asarray(ends,np.int64)
    original=tuple(
        (e["type"],e["pitch"],e["decision_sample"],
         e["available_sample"],e["reason"]) for e in reference)
    sorted_ref=sorted(enumerate(reference),
                      key=lambda pair:(int(pair[1]["available_sample"]),pair[0]))
    pairs=pairs_from_events(candidate)
    pairs.sort(key=lambda pair:(int(pair[0]["available_sample"]),
                                int(pair[0]["pitch"])))
    original_active=set()
    last_source_on=np.full(49,-10**12,dtype=np.int64)
    cursor=0
    admitted=[];values=[];indices=[]
    diag=Counter(proposed=len(pairs))
    for on,off in pairs:
        now=int(on["available_sample"]);pitch=int(on["pitch"]);ix=pitch-40
        if not 0<=ix<49:
            raise ValueError("S756 pitch is outside source ABI")
        while cursor<len(sorted_ref) and (
                int(sorted_ref[cursor][1]["available_sample"])<=now):
            ev=sorted_ref[cursor][1]
            p=int(ev["pitch"])
            if ev["type"]=="note_on":
                original_active.add(p)
                last_source_on[p-40]=int(ev["available_sample"])
            elif ev["type"]=="note_off":
                original_active.remove(p)
            cursor+=1
        if pitch in original_active:
            diag["blocked_source_active"]+=1
            continue
        # Avoid duplicating a just-finished signed note; this uses
        # only PAST source NoteOns, not future original NoteOff timing.
        if now-last_source_on[ix]<.085*SR:
            diag["blocked_source_recent"]+=1
            continue
        frame=int(np.searchsorted(ends,int(on["decision_sample"])))
        if frame>=len(ends) or int(ends[frame])!=int(on["decision_sample"]):
            raise AssertionError("S756 candidate NoteOn is not at causal decision")
        if int(off["available_sample"])<now:
            raise AssertionError("S756 NoteOff predates its onset")
        active=min(6,len(original_active))
        if active>=6:
            diag["blocked_source_budget"]+=1
            continue
        k_tail=float(fused_k[frame,active+1:].sum())
        orig_tail=float(native_k[frame,active+1:].sum())
        onset=float(post[frame,ix,0]);sus=float(post[frame,ix,1])
        if k_tail<=orig_tail+1e-3 or sus<.25 or onset<.035:
            diag["blocked_unsupported"]+=1
            continue
        # The calibrated confidence is NOT a true likelihood of a
        # complete instance; it is an exploratory causal ranking.
        confidence=float(np.clip(
            .35*onset+.30*sus+.35*k_tail,0,1))
        admitted.append((on,off))
        values.append(confidence)
        indices.append((frame,ix))
    new_signature=tuple(
        (e["type"],e["pitch"],e["decision_sample"],
         e["available_sample"],e["reason"]) for e in reference)
    if new_signature!=original:
        raise AssertionError("S756 modified a signed original S708 note")
    diag["novel_candidates"]=len(admitted)
    return admitted,np.asarray(values,np.float32),indices,dict(diag)


def score_variants(reference,post,k,features,trained,policy,item,release,offset):
    """Emit each candidate stream using only frozen K models and causal PCM."""
    out={"S708_frozen":reference};diagnostics={}
    for (arch,weights),clf in trained.items():
        specialist=clf.predict_proba(features).astype(np.float32)
        candidate_k=calibrated_k(k,specialist,BLEND)
        # Re-decoded stream is only a candidate generator; its
        # baseline-affecting notes NEVER replace original S708 events.
        trial=emit(item,post,candidate_k,policy,offset,release)
        pairs,confidence,indices,diag=select_novel_births(
            reference,trial,item["decision_ends"],post,k,candidate_k)
        for risk in RISK:
            key=f"{arch}_{weights}_risk{int(risk*100)}"
            guarded,detail=merge_events(
                reference,pairs,confidence,indices,candidate_k,
                InstanceGate(risk=risk,k_tail=.10))
            out[key]=guarded
            diagnostics[key]=dict(
                candidate=dict(diag),gate=dict(detail))
    if len(out)!=13:
        raise AssertionError("S756 expected frozen source plus 12 full MIDI streams")
    return out,diagnostics


def save(out,report):
    (out/"report.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n")


def evaluate(args):
    if args.out.exists():
        raise FileExistsError("S756 needs a new immutable output folder")
    args.out.mkdir(parents=True)
    original=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    if sha!=original["checkpoint_sha256"]:
        raise AssertionError("S756 original S708 SHA mismatch")
    release=original["frozen_candidate"]["choice"]["release"]
    if release!=dict(gate=.8,persistence=1,grace=24):
        raise AssertionError("S756 original S707 policy drift")
    policy=GuidedPolicy(**original["frozen_candidate"]["choice"]["policy"])
    trained={}
    for family in FAMILIES:
        arch,weights=family
        p=args.specialist_dir/f"s755_{arch}_{weights}.joblib"
        bundle=joblib.load(p)
        if (bundle["source_sha256"]!=sha or
            bundle["training"]!="FIT80_only" or
            bundle["features"]!=240 or
            bundle["classes"]!=list(range(7))):
            raise AssertionError("S756 invalid FIT80 specialist provenance")
        trained[family]=bundle["model"]
    torch.set_num_threads(2)
    network=load_model(args.s708_dir/"s708_selected.pt")[0]
    report=dict(
        schema="S756_SOURCE_PROTECTED_POLY_K_BIRTHS_FIT80_FROZEN_CAL32",
        source_sha256=sha,source="S708_frozen",k_heads="S755_FIT80_frozen",
        complete=False,promoted=False,selected="S708_frozen",
        original_release_unchanged=True,
        LIVE_changed=False,HELD_opened=False,TEST_opened=False,
        CAL_recordings_done=0,loops=12,
        no_truth_at_inference=True,reference_parity=None,
        FIT_provenance="S755 signed FIT80 trained models",
        parameter_grid={
            name:dict(architecture=cfg[0],class_weight=cfg[1],
                      onset_confidence=cfg[2],blend=BLEND)
            for name,cfg in GRID.items()},
        CAL={},CAL_candidate=None)
    save(args.out,report)
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        protocol,heads,selection,offset=setup(root,args.offset_checkpoint)
        for name,path in (("GuitarSet_audio","GuitarSet_audio.zip"),
                          ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            x=protocol["guitar_archives"][name]
            download(x["url"],root/"data"/path,x["size"],x["sha256"])
        originals=[s for s in protocol["sources"]
                   if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as audio,\
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            expanded=expanded_train_specs(originals,audio,ann)
            cal=[s for s in expanded if s["composition"] in CAL_COMPS]
            if len(cal)!=32 or set(FIT_COMPS)&set(CAL_COMPS):
                raise AssertionError("S756 CAL32 split does not match S708")
            scores={name:[] for name in ("S708_frozen",*GRID)}
            losses={name:Counter() for name in GRID}
            events={name:Counter() for name in GRID}
            for idx,spec in enumerate(cal):
                item,post,k,baseline,_,_,_=baseline_input(
                    spec,audio,root,heads,selection,offset,
                    network,policy,release)
                f=coherent_k_features(item["x"],post,k,item["native_post"])
                alternatives,diagnostics=score_variants(
                    baseline,post,k,f,trained,policy,item,release,offset)
                # Only now inspect JAMS labels: no test can influence
                # source-aligned birth decisions or offset scheduling.
                truth=real_notes(labels(spec,root)["notes"],
                                 item["samples"]/SR)
                measured={name:metric(stream,truth,item["ends"])
                          for name,stream in alternatives.items()}
                src=measured["S708_frozen"]
                for name,result in measured.items():
                    scores[name].append(result)
                for name in GRID:
                    trial=measured[name]
                    losses[name].update(dict(
                        correct_offsets=len(
                            set(src["correct_offset_truth_ids"])-
                            set(trial["correct_offset_truth_ids"])),
                        long_onsets=len(
                            set(src["long_onset_truth_ids"])-
                            set(trial["long_onset_truth_ids"])),
                        onset_tp=max(0,src["tp"]-trial["tp"])))
                    events[name].update(diagnostics[name]["gate"])
                report["CAL_recordings_done"]=idx+1
                save(args.out,report)
                print("S756 CAL",idx+1,32,spec["name"],
                      "candidate_births",max(
                          diag["candidate"]["novel_candidates"]
                          for diag in diagnostics.values()),flush=True)
    aggregate_scores={name:aggregate(rows) for name,rows in scores.items()}
    baseline=aggregate_scores["S708_frozen"]
    signed=original["frozen_candidate"]["CAL"]
    diff={name:float(baseline[name]-signed[name]) for name in SCORES}
    parity=all(abs(value)<=1e-8 for value in diff.values())
    report["reference_parity"]=dict(valid=parity,diff=diff)
    report["reference_scores"]={name:float(baseline[name]) for name in SCORES}
    eligible=[]
    for name in GRID:
        s=aggregate_scores[name]
        regression={key:float(s[key]-baseline[key]) for key in SCORES}
        loss=dict(losses[name])
        safe=bool(parity and all(x>=-1e-12 for x in regression.values())
             and s["tp"]>=baseline["tp"] and all(value==0
                                                 for value in loss.values()))
        report["CAL"][name]=dict(
            scores={key:float(s[key]) for key in SCORES},
            delta=regression,
            lost=loss,
            source_events_preserved=True,
            causal_ranking_not_event_probability=True,
            event_counts=dict(events[name]),
            safe=safe,
            six_gt80=bool(safe and all(s[key]>.8 for key in SCORES)))
        if safe and any(x>1e-12 for x in regression.values()):
            eligible.append(name)
    report["CAL_candidate"]=max(eligible,key=lambda key:(
        min(report["CAL"][key]["scores"].values()),
        sum(report["CAL"][key]["delta"].values())),default=None)
    report["complete"]=parity
    save(args.out,report)
    if not parity:
        raise AssertionError("S756 S708 source baseline parity failed")
    print("S756_COMPLETE",json.dumps(dict(
        reference=report["reference_scores"],candidate=report["CAL_candidate"],
        safe=eligible,
        most_additional=max(
            ((name,info["event_counts"].get("accepted",0))
             for name,info in report["CAL"].items()),
            key=lambda row:(row[1],row[0])),
        best_active=max(
            ((name,info["scores"]["active_exact_k"])
             for name,info in report["CAL"].items()),
            key=lambda row:(row[1],row[0])))),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--s708-dir",type=Path,required=True)
    p.add_argument("--offset-checkpoint",type=Path,required=True)
    p.add_argument("--specialist-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    evaluate(p.parse_args())
