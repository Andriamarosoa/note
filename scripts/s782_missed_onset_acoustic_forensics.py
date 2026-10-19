"""S782: post-hoc acoustic evidence taxonomy for missed GuitarSet CAL32 attacks.

This is a diagnostic, NOT a trained proposal generator. S708 MIDI emissions
are computed before any labels are read. Later 1-3 decision frames are used
ONLY for descriptive analysis, not for real-time inference or promotion.
CAL is historically exposed and cannot be used for model selection.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import torch

from causal_note.strict_live import SR
from causal_note.s708_cardinality_selection import GuidedPolicy
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import real_notes,event_score
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import CAL_COMPS,load_model
from scripts.s733_train_onset_only import baseline_input

ONSET_THRESHOLDS=(.03,.10,.30,.50)
RISE_THRESHOLDS=(.10,.25,.50,1.00)

def classify_pitch(pitch):
    if pitch<52:return "E2_B2"
    if pitch<64:return "C3_B3"
    if pitch<77:return "C4_E5"
    return "F5_E6"

def true_polyphony(note,truth):
    t=float(note["time"])
    return min(6,sum(float(n["time"])<=t<float(n["time"])+float(n["duration"]) for n in truth))

def overlap_harmonic(note,truth):
    p=int(note["pitch"])
    t=float(note["time"])
    return any(n is not note and abs(p-int(n["pitch"])) in (12,19,24)
         and float(n["time"])<=t<float(n["time"])+float(n["duration"])
         for n in truth)

def acoustic_evidence(note,ends,post,spectra):
    """Reference-aligned descriptive evidence: never returned to decoder."""
    p=int(note["pitch"])-40
    assert 0<=p<49
    clock=np.asarray(ends,dtype=np.int64)
    at=int(np.searchsorted(clock,float(note["time"])*SR,side="left"))
    if at>=len(clock):
        return None
    # Diagnostic follows at most three observed *causal* decisions after attack.
    last=min(len(clock),at+3)
    onset=np.asarray(post[at:last,p,0],dtype=np.float32)
    sustain=np.asarray(post[at:last,p,1],dtype=np.float32)
    energy=np.asarray(spectra[at:last,p],dtype=np.float32)
    before=float(spectra[at-1,p]) if at else float(spectra[at,p])
    rises=np.maximum(energy-np.r_[before,energy[:-1]],0.)
    return dict(onset=float(onset.max()),sustain=float(sustain.max()),
                spectral_energy=float(energy.max()),rise=float(rises.max()),
                onset_first=float(onset[0]),rise_first=float(rises[0]))

def bins(note):
    k=true_polyphony(note["note"],note["truth"])
    return dict(poly=("K1" if k==1 else "K2_K4" if k<5 else "K5_K6"),
                register=classify_pitch(int(note["note"]["pitch"])),
                texture=note["texture"],composition=note["composition"],
                overlap=("harmonic_neighbor" if overlap_harmonic(note["note"],note["truth"])
                         else "no_harmonic_neighbor"))

def summarize(records):
    found=[r for r in records if r["matched"]]
    missed=[r for r in records if not r["matched"]]
    if not records: return dict(total=0,matched=0,missed=0)
    return dict(total=len(records),matched=len(found),missed=len(missed),
      recall=len(found)/len(records),
      acoustic_diagnostic_coverage=sum(r["evidence"] is not None for r in missed),
      missed_onset_above={str(t):sum((r["evidence"] or {}).get("onset",-1)>=t for r in missed)
                          for t in ONSET_THRESHOLDS},
      missed_rise_above={str(t):sum((r["evidence"] or {}).get("rise",-1)>=t for r in missed)
                         for t in RISE_THRESHOLDS},
      missed_weak_both=sum(r["evidence"] is not None and r["evidence"]["onset"]<.03
                           and r["evidence"]["rise"]<.25 for r in missed),
      missed_strong_both=sum(r["evidence"] is not None and r["evidence"]["onset"]>=.1
                             and r["evidence"]["rise"]>=.25 for r in missed),
      matched_strong_both=sum(r["evidence"] is not None and r["evidence"]["onset"]>=.1
                              and r["evidence"]["rise"]>=.25 for r in found))

def by_group(records,field):
    return {str(k):summarize([r for r in records if r["tags"][field]==k])
            for k in sorted({r["tags"][field] for r in records})}

def evaluate(args):
    if args.out.exists():raise FileExistsError("S782 immutable output")
    args.out.mkdir(parents=True)
    signed=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    if sha!=signed["checkpoint_sha256"]:
        raise AssertionError("S782 signed S708 checkpoint mismatch")
    policy=GuidedPolicy(**signed["frozen_candidate"]["choice"]["policy"])
    release=signed["frozen_candidate"]["choice"]["release"]
    model=load_model(args.s708_dir/"s708_selected.pt")[0]
    torch.set_num_threads(2)
    report=dict(schema="S782_CAUSAL_MISSED_ONSET_DIAGNOSTIC_CAL32",
        complete=False,source="S708_frozen",promoted=False,LIVE_changed=False,
        TEST_opened=False,HELD_opened=False,source_MIDI_immutable=True,
        labels_used_only_after_prediction=True,
        diagnostic_uses_three_post_attack_frames=True,
        diagnostic_not_realtime_prediction=True,cal_recordings_done=0,results={})
    def save():
        (args.out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    save()
    detail=[]
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/"data").mkdir()
        protocol,heads,selection,offset=setup(root,args.offset_checkpoint)
        for key,name in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            remote=protocol["guitar_archives"][key]
            download(remote["url"],root/"data"/name,remote["size"],remote["sha256"])
        original=[s for s in protocol["sources"]
                  if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as wav,\
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            specs=[s for s in expanded_train_specs(original,wav,ann)
                   if s["composition"] in CAL_COMPS]
            if len(specs)!=32:raise AssertionError("S782 CAL32 split drift")
            for i,spec in enumerate(specs):
                item,post,k,source,proposals,ctx,eligible=baseline_input(
                    spec,wav,root,heads,selection,offset,model,policy,release)
                spectra=np.asarray(item["x"][:,:,192:216],np.float32).reshape(
                    len(post),49,3,8).mean(axis=(2,3))
                # The causal stream above is final before reading annotations.
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                _,pairs=event_score(source,truth,tolerance=.080)
                matched={j for _,j in pairs}
                texture=spec["name"].rsplit("_",1)[-1]
                for j,note in enumerate(truth):
                    if int(note["pitch"])<40 or int(note["pitch"])>88:continue
                    e=acoustic_evidence(note,item["decision_ends"],post,spectra)
                    x=dict(note=note,truth=truth,matched=(j in matched),
                           evidence=e,texture=texture,composition=spec["composition"])
                    detail.append(dict(matched=x["matched"],evidence=e,tags=bins(x)))
                report["cal_recordings_done"]=i+1;save()
                print("S782 CAL",i+1,32,spec["name"],"notes",len(truth),
                      "missed",len(truth)-len(matched),flush=True)
    report["results"]=dict(all=summarize(detail),
        by_poly=by_group(detail,"poly"),by_register=by_group(detail,"register"),
        by_texture=by_group(detail,"texture"),by_composition=by_group(detail,"composition"),
        by_harmonic_overlap=by_group(detail,"overlap"),
        calibration_warning="CAL32 has prior repeated use; descriptive only")
    report["complete"]=True
    save()
    print("S782_MISSED_ONSET_TAXONOMY",json.dumps(report["results"]),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--s708-dir",type=Path,required=True)
    p.add_argument("--offset-checkpoint",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    evaluate(p.parse_args())
