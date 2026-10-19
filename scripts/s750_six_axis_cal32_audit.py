"""S750 actual S708 CAL32 six-axis error anatomy.

No training, no tuning or model replacement. This runs signed S708
full-MIDI on the original 32 CAL composition-disjoint GuitarSet files,
then opens labels solely for independent error diagnosis. Measures
true K0..6, under/overcount, register and harmonic error indicators.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile
import zipfile
import numpy as np
import torch

from causal_note.strict_live import SR
from causal_note.s708_cardinality_selection import GuidedPolicy
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s704_joint_metrics import SCORES,metric,aggregate
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import CAL_COMPS,FIT_COMPS,load_model
from scripts.s733_train_onset_only import baseline_input
from scripts.s748_six_axis_failure_audit import axis_audit

def evaluate(args, *, decision_audit=None,
             schema="S750_REAL_S708_GUITARSET_CAL32_K0_K6_SIX_AXIS_ROOTCAUSE"):
    if args.out.exists():
        raise FileExistsError("S750 requires a new immutable audit folder")
    args.out.mkdir(parents=True)
    frozen=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    if sha!=frozen["checkpoint_sha256"]:
        raise AssertionError("S750 signed S708 checkpoint changed")
    choice=frozen["frozen_candidate"]["choice"]
    release=choice["release"]
    if release!=dict(gate=.8,persistence=1,grace=24):
        raise AssertionError("S750 signed S708 S707 release changed")
    network=load_model(args.s708_dir/"s708_selected.pt")[0]
    policy=GuidedPolicy(**choice["policy"])
    torch.set_num_threads(2)
    report=dict(schema=schema,
        signed_sha256=sha,complete=False,training=False,promoted=False,
        LIVE_changed=False,TEST_opened=False,HELD_opened=False,
        recordings=[],split="CAL32 BN2 Jazz2 composition-disjoint",
        reference_parity=None)
    (args.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    rows=[]
    confusion=np.zeros((7,7),np.int64)
    per_true={str(k):Counter() for k in range(7)}
    by_register={key:Counter() for key in ("low_E2_E3","middle_F3_A4","high_Asharp4_E6")}
    harmonics=Counter()
    release_reasons=Counter()
    totals=Counter()
    decision_records=[]
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        proto,heads,selection,offset=setup(root,args.offset_checkpoint)
        for source,name in (("GuitarSet_audio","GuitarSet_audio.zip"),
                            ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            data=proto["guitar_archives"][source]
            download(data["url"],root/"data"/name,
                     data["size"],data["sha256"])
        train=[s for s in proto["sources"] if
               s["corpus"]=="GuitarSet" and s["split"]=="train"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as audio,\
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as annotations:
            recordings=[s for s in expanded_train_specs(train,audio,annotations)
                        if s["composition"] in CAL_COMPS]
            if len(recordings)!=32:
                raise AssertionError("Expected exactly CAL32 annotated GuitarSet")
            if set(CAL_COMPS)&set(FIT_COMPS):
                raise AssertionError("S750 CAL composition leakage")
            for i,spec in enumerate(recordings):
                # MIDI emitted using frozen signed S708 source without CAL truth.
                item,post,k,events,_,_,_=baseline_input(
                    spec,audio,root,heads,selection,offset,
                    network,policy,release)
                if not all(e["available_sample"]>=e["decision_sample"]
                           for e in events):
                    raise AssertionError("S750 native event backdated")
                # Only at this point CAL annotations may be used in audit.
                truth=real_notes(labels(spec,root)["notes"],
                                 item["samples"]/SR)
                detail=axis_audit(events,truth,item["ends"])
                scores=metric(events,truth,item["ends"])
                if decision_audit is not None:
                    diagnosis=decision_audit(
                        item["decision_ends"],k,events,truth)
                    decision_records.append(diagnosis)
                rows.append(scores)
                confusion+=np.asarray(detail["confusion_k0_k6"],np.int64)
                for k,v in detail["per_true_k"].items():
                    per_true[k].update({
                        name:int(v[name]) for name in
                        ("true_frames","exact","under","over")})
                for register,reg in detail["register"].items():
                    by_register[register].update({
                        key:int(reg[key]) for key in
                        ("onset_tp","onset_fp","onset_fn",
                         "frame_pitch_tp","frame_pitch_fp","frame_pitch_fn")})
                harmonics.update(detail["harmonic_false_alarms_near_miss"])
                release_reasons.update(detail["offset_reasons"])
                totals.update({key:detail[key] for key in (
                    "k_gt6_frames","overcount_active_frames",
                    "undercount_active_frames","missing_first_onsets")})
                report["recordings"].append(dict(
                    name=spec["name"],composition=spec["composition"],
                    six_metrics=detail["source_metrics"],
                    per_true_k=detail["per_true_k"],
                    register=detail["register"],
                    harmonic_false_alarms=detail["harmonic_false_alarms_near_miss"],
                    offset_reasons=detail["offset_reasons"],
                    k_gt6_frames=detail["k_gt6_frames"],
                    overcount_active_frames=detail["overcount_active_frames"],
                    undercount_active_frames=detail["undercount_active_frames"],
                    **({"khead_vs_decoder":diagnosis}
                       if decision_audit is not None else {})))
                report["recordings_done"]=i+1
                (args.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
                print("S750 AUDIT",i+1,32,spec["name"],
                      "activeK",round(scores["active_exact_k"],4),
                      "under_active",detail["undercount_active_frames"],
                      flush=True)
    overall=aggregate(rows)
    signed=frozen["frozen_candidate"]["CAL"]
    parity={name:dict(source=float(signed[name]),observed=float(overall[name]),
                      delta=float(overall[name]-signed[name]))
            for name in SCORES}
    okay=all(abs(v["delta"])<1e-8 for v in parity.values())
    report["reference_parity"]=dict(valid=okay,six=parity)
    report["metrics"]={name:float(overall[name]) for name in SCORES}
    report["true_k0_k6_confusion"]=confusion.tolist()
    report["per_true_k"]={
        k:{**dict(v),
           "exact_rate":v["exact"]/max(1,v["true_frames"]),
           "under_rate":v["under"]/max(1,v["true_frames"]),
           "over_rate":v["over"]/max(1,v["true_frames"])}
        for k,v in per_true.items()}
    report["register"]={k:dict(v) for k,v in by_register.items()}
    report["harmonic_false_alarm_near_miss"]=dict(harmonics)
    report["offset_reasons"]=dict(release_reasons)
    report["totals"]=dict(totals)
    if decision_audit is not None:
        from scripts.s753_khead_decoder_anatomy import aggregate_decision_audit
        report["khead_vs_decoder"]=aggregate_decision_audit(decision_records)
    report["complete"]=okay
    (args.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    if not okay:
        raise AssertionError("S750 S708 source metric parity violated; no audit promotion")
    print("S750_SIX_AXIS_ROOTCAUSE",json.dumps({
        "source":report["metrics"],"per_true_k":report["per_true_k"],
        "register":report["register"],
        "harmonics":report["harmonic_false_alarm_near_miss"],
        "offset_reasons":report["offset_reasons"],
        "totals":report["totals"],"parity":okay}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--s708-dir",type=Path,required=True)
    p.add_argument("--offset-checkpoint",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    evaluate(p.parse_args())
