"""S784: FIT-only selection of causal spectral+posterior attack rescue policies.

A genuinely independent acoustic proposal source over all 49 MIDI pitches.
Candidates use current/past S708 acoustic observations only; native S708
MIDI source is immutable. FIT80 tunes 48 fixed policies, CAL32 tests the
frozen policy diagnostically. Shadow onset != final 256-hop MIDI onset.
"""
from __future__ import annotations
import argparse
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
from scripts.s706_train_sharp_temporal_loops import FIT_COMPS,CAL_COMPS,load_model
from scripts.s733_train_onset_only import baseline_input
from scripts.s768_independent_onset_lattice_fit80 import onset_score,combine

PITCHES=49
BLOCK=1024
ON_THRESHOLDS=(.10,.20,.35,.50)
RISE_THRESHOLDS=(.08,.18,.30)
HARMONIC_PENALTIES=(0.,.4)
LOW_FACTORS=(.6,1.)
COOLDOWN_BLOCKS=3
RETRIGGER_BLOCKS=4

def policy_key(p):
    return f"on{p[0]:.2f}_rise{p[1]:.2f}_harm{p[2]:.1f}_low{p[3]:.1f}"

def policy_grid():
    return [(on,rise,harm,low) for on in ON_THRESHOLDS
            for rise in RISE_THRESHOLDS
            for harm in HARMONIC_PENALTIES for low in LOW_FACTORS]

def acoustic_grid(item,post):
    """Source-independent two-view causal spectral onset evidence."""
    n=len(post)
    x=np.asarray(item["x"],np.float32)
    if x.shape!=(n,49,288) or np.asarray(post).shape!=(n,49,3):
        raise ValueError("S784 expected S708 acoustic 49x288 and 49x3 ABI")
    spectrum=x[:,:,192:216].reshape(n,49,3,8).mean(axis=(2,3))
    prev=spectrum[np.maximum(0,np.arange(n)-1)]
    rise=np.maximum(spectrum-prev,0.)
    # An overtone at p can be explained by a simultaneous lower fundamental
    # at p-12 or p-19; retain no-lookahead competitors for a soft veto.
    lower=np.zeros_like(rise)
    lower[:,12:]=rise[:,:-12]
    lower[:,19:]=np.maximum(lower[:,19:],rise[:,:-19])
    posterior=np.asarray(post[:,:,0],np.float32)
    if not np.isfinite(rise).all() or not np.isfinite(posterior).all():
        raise ValueError("S784 non-finite acoustic stream")
    return dict(onset=posterior,rise=rise,lower_rise=lower)

def propose(evidence,ends,source,policy):
    """Only incoming/current snapshots allowed to affect current decision."""
    on_thr,rise_thr,harm,low_factor=policy
    ends=np.asarray(ends,np.int64)
    on=np.asarray(evidence["onset"],np.float32)
    rise=np.asarray(evidence["rise"],np.float32)
    lower=np.asarray(evidence["lower_rise"],np.float32)
    if any(a.shape!=(len(ends),PITCHES) for a in (on,rise,lower)):
        raise ValueError("S784 expected pitch- and clock-aligned cues")
    last_native=np.full(PITCHES,-10**12,dtype=np.int64)
    last_birth=np.full(PITCHES,-10**12,dtype=np.int64)
    armed=np.ones(PITCHES,bool)
    out=[]
    current=0
    events=sorted((e for e in source if e["type"]=="note_on"),
                  key=lambda e:int(e["available_sample"]))
    thresholds=np.full(PITCHES,on_thr,np.float32)
    thresholds[:12]*=low_factor
    for i,now64 in enumerate(ends):
        now=int(now64)
        while current<len(events) and int(events[current]["available_sample"])<=now:
            e=events[current]
            pitch=int(e["pitch"])-40
            if 0<=pitch<PITCHES:
                last_native[pitch]=int(e["available_sample"])
            current+=1
        contrast=np.maximum(rise[i]-harm*lower[i],0.)
        armed |= ((on[i] < thresholds*.45) | (contrast < rise_thr*.5))
        available=np.flatnonzero((on[i]>=thresholds)&(contrast>=rise_thr)
              & armed & (now-last_birth>=RETRIGGER_BLOCKS*BLOCK)
              & (now-last_native>=COOLDOWN_BLOCKS*BLOCK))
        for ix in available:
            out.append(dict(type="note_on",pitch=int(ix+40),velocity=96,
                       time=now/SR,decision_sample=now,
                       available_sample=now,reason="s784_acoustic_shadow"))
            last_birth[ix]=now
            armed[ix]=False
    return out

def by_group(records):
    return {c:combine([r["score"] for r in records if r["composition"]==c])
            for c in sorted({r["composition"] for r in records})}

def choose_policies(groups,baseline,policies):
    """FIT-only group robust selection. Can legitimately choose no rescue."""
    base=combine([r["score"] for r in baseline])
    base_comp=by_group(baseline)
    summary={key:combine([r["score"] for r in rows]) for key,rows in policies.items()}
    held={key:by_group(rows) for key,rows in policies.items()}
    feasible=[key for key in policies
        if summary[key]["f1"]>base["f1"]
        and summary[key]["TP"]>base["TP"]
        and all(held[key][c]["f1"]>=base_comp[c]["f1"]-.002
                for c in groups)]
    best=max(feasible,key=lambda k:(min(held[k][c]["f1"]-base_comp[c]["f1"]
                                          for c in groups),
                                     summary[k]["f1"],
                                     summary[k]["precision"]),default=None)
    return best,base,base_comp,summary,held,feasible

def write(path,report):
    path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")

def evaluate(args):
    if args.out.exists():raise FileExistsError("S784 immutable result directory")
    args.out.mkdir(parents=True)
    signed=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    if sha!=signed["checkpoint_sha256"]:
        raise AssertionError("S784 signed S708 model mismatch")
    frozen=signed["frozen_candidate"]["choice"]
    policy=GuidedPolicy(**frozen["policy"])
    network=load_model(args.s708_dir/"s708_selected.pt")[0]
    torch.set_num_threads(2)
    report=dict(schema="S784_CAUSAL_SPECTRAL_ONSET_RESCUE_FIT80",
        complete=False,source="S708_frozen",promoted=False,LIVE_changed=False,
        final_MIDI_changed=False,ONSET_only_not_MIDI=True,
        causal_current_past_only=True,split_protected=True,
        fit_recordings_done=0,cal_recordings_done=0,
        TEST_opened=False,HELD_opened=False,no_CAL_training_or_selection=True,
        checkpoint_sha256=sha,FIT={},CAL={},selected_policy=None)
    save=lambda:write(args.out/"report.json",report)
    save()
    options=policy_grid()
    keys=[policy_key(p) for p in options]
    fit_baseline=[];fit_trial={key:[] for key in keys}
    cal_baseline=[];cal_frozen=[]
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
            specs=expanded_train_specs(original,wav,ann)
            fit_specs=[s for s in specs if s["composition"] in FIT_COMPS]
            cal_specs=[s for s in specs if s["composition"] in CAL_COMPS]
            if len(fit_specs)!=80 or len(cal_specs)!=32 or set(FIT_COMPS)&set(CAL_COMPS):
                raise AssertionError("S784 composition split drift")
            for i,spec in enumerate(fit_specs):
                item,post,k,source,acoustic,ctx,eligible=baseline_input(
                    spec,wav,root,heads,selection,offset,network,policy,frozen["release"])
                evidence=acoustic_grid(item,post)
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                fit_baseline.append(dict(composition=spec["composition"],
                                   score=onset_score(source,[],truth)))
                for cfg,key in zip(options,keys):
                    added=propose(evidence,item["decision_ends"],source,cfg)
                    fit_trial[key].append(dict(composition=spec["composition"],
                                   score=onset_score(source,added,truth)))
                report["fit_recordings_done"]=i+1;save()
                print("S784 FIT",i+1,80,spec["name"],flush=True)
            best,base,basecomp,allmetrics,bycomp,eligible=choose_policies(
                FIT_COMPS,fit_baseline,fit_trial)
            lookup=dict(zip(keys,options))
            report["FIT"]=dict(baseline=base,by_composition_baseline=basecomp,
                               policies=allmetrics,by_composition=bycomp,
                               eligible=eligible,selected=best,
                               risk_objective="maximin_nonregression_per_composition",
                               frozen_grid_size=len(options))
            report["selected_policy"]=best
            save()
            for i,spec in enumerate(cal_specs):
                item,post,k,source,acoustic,ctx,eligible_source=baseline_input(
                    spec,wav,root,heads,selection,offset,network,policy,frozen["release"])
                evidence=acoustic_grid(item,post)
                additional=(propose(evidence,item["decision_ends"],source,lookup[best])
                            if best is not None else [])
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                cal_baseline.append(dict(composition=spec["composition"],
                                         score=onset_score(source,[],truth)))
                cal_frozen.append(dict(composition=spec["composition"],
                                       score=onset_score(source,additional,truth),
                                       extra_emitted=len(additional)))
                report["cal_recordings_done"]=i+1;save()
                print("S784 CAL",i+1,32,spec["name"],
                      "acoustic_proposals",len(additional),flush=True)
    original=combine([r["score"] for r in cal_baseline])
    frozen_result=combine([r["score"] for r in cal_frozen])
    report["CAL"]=dict(baseline=original,frozen=frozen_result,
        frozen_extra_emitted=sum(r["extra_emitted"] for r in cal_frozen),
        delta_f1=frozen_result["f1"]-original["f1"],
        by_composition_baseline=by_group(cal_baseline),
        by_composition_frozen=by_group(cal_frozen))
    report["complete"]=True
    save()
    print("S784_FROZEN_ONSET",json.dumps(dict(selected=best,
        fit_base=base["f1"],fit_selected=(allmetrics[best]["f1"] if best else base["f1"]),
        cal_base=original["f1"],cal_selected=frozen_result["f1"],
        delta_f1=report["CAL"]["delta_f1"],extra_emitted=report["CAL"]["frozen_extra_emitted"])),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--s708-dir",type=Path,required=True)
    p.add_argument("--offset-checkpoint",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    evaluate(p.parse_args())
