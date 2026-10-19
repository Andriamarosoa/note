"""S794 Active Exact-K source diagnosis, true MIDI256 counts and zero-delay.

Not a model-training script. Reconstruct signed S708 GuitarSet FIT80/CAL32,
compute all six metrics from identical MIDI streams and classify K1..K6
undercount/overcount along with causal acoustic-cue opportunities.

The S793 promotion gate stays mandatory for any future candidate. No
note timing or inference is changed here.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile
import zipfile
import numpy as np
import torch

from causal_note.s708_cardinality_selection import GuidedPolicy
from causal_note.strict_live import SR
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s704_joint_metrics import SCORES,aggregate,metric
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import (
    FIT_COMPS,CAL_COMPS,load_model)
from scripts.s733_train_onset_only import baseline_input
from scripts.s794_activek_failure_taxonomy import diagnostic,summarize

SCHEMA="S794_ZERO_DELAY_REAL_MIDI256_ACTIVEK_ROOTCAUSE"

def save(path,report):
    path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")

def scan(spec,archive,root,heads,selection,offset,network,policy,release):
    item,post,k,events,*_=baseline_input(
        spec,archive,root,heads,selection,offset,network,policy,release)
    # No GT labels until AFTER the full original MIDI has been generated.
    truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
    measured=metric(events,truth,item["ends"])
    diag=diagnostic(dict(item=item,post=post,events=events,truth=truth))
    for axis in SCORES:
        if not np.isclose(measured[axis],diag["scores"][axis],
                atol=1e-12,rtol=0):
            raise AssertionError("S794 disconnected MIDI metric "+axis)
    return measured,diag

def evaluate(args):
    if args.out.exists():raise FileExistsError("S794 immutable experiment")
    args.out.mkdir(parents=True)
    signed=json.loads((args.s708/"report.json").read_text())
    sha=digest((args.s708/"s708_selected.pt").read_bytes())
    if sha!=signed["checkpoint_sha256"] or signed["schema"]!="S708_K_GUIDED_S706_CAL_LOOPS":
        raise AssertionError("S794 signed S708 protocol identity mismatch")
    choice=signed["frozen_candidate"]["choice"]
    policy=GuidedPolicy(**choice["policy"])
    if choice["release"]!=dict(gate=.8,persistence=1,grace=24):
        raise AssertionError("S794 frozen S707 release changed")
    torch.set_num_threads(2)
    network=load_model(args.s708/"s708_selected.pt")[0]
    report=dict(schema=SCHEMA,complete=False,promoted=False,
       LIVE_changed=False,final_MIDI_changed=False,TEST_opened=False,
       HELD_opened=False,source_sha256=sha,score_clock_samples=256,
       extra_lookahead_samples=0,extra_buffer_samples=0,
       extra_confirmation_samples=0,extra_hold_samples=0,
       fitted_specialists=0,source_identity_confirmed=False,
       fit_done=0,cal_done=0,FIT={},CAL={},
       promotion_allowed=False,real_microphone_latency_measured=False)
    def checkpoint():save(args.out/"report.json",report)
    checkpoint()
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        proto,heads,selection,offset=setup(root,args.offset_checkpoint)
        for key,name in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            source=proto["guitar_archives"][key]
            download(source["url"],root/"data"/name,source["size"],source["sha256"])
        originals=[s for s in proto["sources"]
                   if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as audio,\
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            specs=expanded_train_specs(originals,audio,ann)
            fit=[s for s in specs if s["composition"] in FIT_COMPS]
            cal=[s for s in specs if s["composition"] in CAL_COMPS]
            if len(fit)!=80 or len(cal)!=32 or set(FIT_COMPS)&set(CAL_COMPS):
                raise AssertionError("FIT/CAL composition protocol drift")
            for name,batch in (("FIT",fit),("CAL",cal)):
                rows=[];diagnostics=[]
                for i,spec in enumerate(batch):
                    measured,details=scan(spec,audio,root,heads,selection,
                        offset,network,policy,choice["release"])
                    rows.append(dict(name=spec["name"],
                        composition=spec["composition"],
                        texture=spec["name"].rsplit("_",1)[-1],metric=measured))
                    diagnostics.append(dict(name=spec["name"],
                        composition=spec["composition"],
                        texture=spec["name"].rsplit("_",1)[-1],
                        details=details))
                    report[f"{name.lower()}_done"]=i+1
                    checkpoint()
                    print("S794_"+name+"_MIDI256",i+1,"/",len(batch),
                          spec["name"],flush=True)
                aggregated=aggregate([r["metric"] for r in rows])
                analyzed=summarize([r["details"] for r in diagnostics])
                if not np.isclose(aggregated["active_exact_k"],analyzed["active_exact_k"],
                                 atol=1e-12,rtol=0):
                    raise AssertionError("S794 combined 256hop active K divergence")
                comps=sorted(set(row["composition"] for row in rows))
                bycomp={}
                textures=sorted(set(row["texture"] for row in rows))
                bytexture={}
                for key in comps:
                    bycomp[key]=dict(metrics=aggregate(
                        [r["metric"] for r in rows if r["composition"]==key]),
                        audit=summarize([r["details"] for r in diagnostics
                                         if r["composition"]==key]))
                for key in textures:
                    bytexture[key]=dict(metrics=aggregate(
                        [r["metric"] for r in rows if r["texture"]==key]),
                        audit=summarize([r["details"] for r in diagnostics
                                         if r["texture"]==key]))
                report[name]=dict(metrics=aggregated,
                    audit=analyzed,compositions=comps,by_composition=bycomp,
                    by_texture=bytexture)
                checkpoint()
                print("S794_"+name+"_SUMMARY",json.dumps(dict(
                  active_exact_k=aggregated["active_exact_k"],
                  active_undercount=analyzed["under_active_frames"],
                  active_overcount=analyzed["over_active_frames"],
                  by_true_k=analyzed["by_true_k"])),flush=True)
    signedcal=signed["frozen_candidate"]["CAL"]
    for key in ("tp","fp","fn","frames","k_correct","active_frames",
                "active_k_correct","frame_pitch_tp","frame_pitch_fp",
                "frame_pitch_fn","offset_eligible","offset_correct"):
        a=report["CAL"]["metrics"][key];b=signedcal[key]
        if a!=b:raise AssertionError(f"S794 signed MIDI S708 count {key}: {a} != {b}")
    for axis in SCORES:
        if not np.isclose(report["CAL"]["metrics"][axis],signedcal[axis],
                         atol=1e-12,rtol=0):
            raise AssertionError("S794 six-source-metric signed CAL parity "+axis)
    report["source_identity_confirmed"]=True
    report["complete"]=True
    checkpoint()
    print("S794_REAL_MIDI256_RESULT",json.dumps(dict(
        source=report["CAL"]["metrics"]["active_exact_k"],
        by_k=report["CAL"]["audit"]["by_true_k"],
        active_under=report["CAL"]["audit"]["under_active_frames"],
        active_over=report["CAL"]["audit"]["over_active_frames"],
        cue_candidates=report["CAL"]["audit"]["expert_opportunity"],
        proposed_model_count=0,extra_latency=0)),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--s708",type=Path,required=True)
    p.add_argument("--offset-checkpoint",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    evaluate(p.parse_args())
