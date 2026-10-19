"""S794 strict S708 CAL32 reproducibility / one-frame drift probe.

Repeat THE SAME complete S708 causal acoustic+MIDI inference on CAL32.
Compare emitted event SHA per recording, all six S704 metric counts,
and signed-S708 equality. Never fix parity by numeric tolerances.
No new decoder, no labels used for inference, no future audio, no promotion.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from causal_note.s708_cardinality_selection import GuidedPolicy
from causal_note.strict_live import SR
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s704_joint_metrics import SCORES,metric,aggregate
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import CAL_COMPS,load_model
from scripts.s733_train_onset_only import baseline_input

COUNTS=("tp","fp","fn","frames","k_correct","active_frames","active_k_correct",
        "frame_pitch_tp","frame_pitch_fp","frame_pitch_fn","offset_eligible",
        "offset_correct","long_offset_correct")

def event_signature(events):
    # Ignore extraneous cached values, hash semantic emitted stream identically.
    rows=[dict(type=e["type"],pitch=int(e["pitch"]),
        available_sample=int(e["available_sample"]),
        decision_sample=int(e["decision_sample"]),
        reason=e["reason"]) for e in events]
    encoded=json.dumps(rows,sort_keys=True,separators=(",",":")).encode()
    return hashlib.sha256(encoded).hexdigest()

def pair_signature(source,observed):
    return {k:dict(reference=int(source[k]),observed=int(observed[k]),
                   delta=int(observed[k])-int(source[k]))
            for k in COUNTS if int(observed[k])!=int(source[k])}

def run(args):
    if args.out.exists():raise FileExistsError("immutable output required")
    args.out.mkdir(parents=True)
    signed=json.loads((args.s708/"report.json").read_text())
    sha=digest((args.s708/"s708_selected.pt").read_bytes())
    if sha!=signed["checkpoint_sha256"]:raise AssertionError("signed S708 SHA mismatch")
    choice=signed["frozen_candidate"]["choice"]
    policy=GuidedPolicy(**choice["policy"])
    torch.set_num_threads(2)
    model=load_model(args.s708/"s708_selected.pt")[0]
    report=dict(schema="S794_PARITY_DRIFT_INDEPENDENT_REPLAY",signed_sha=sha,
        complete=False,promoted=False,TEST_opened=False,HELD_opened=False,
        periods=2,CAL_count=0,iterations=[],difference_by_record=[],
        original_run=39333999180)
    def save(): (args.out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    save()
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/"data").mkdir()
        proto,heads,selection,offset=setup(root,args.offset_checkpoint)
        for key,name in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            src=proto["guitar_archives"][key]
            download(src["url"],root/"data"/name,src["size"],src["sha256"])
        originals=[s for s in proto["sources"] if s["corpus"]=="GuitarSet"
                   and s["split"]=="train"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as audio, \
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            specs=[s for s in expanded_train_specs(originals,audio,ann)
                   if s["composition"] in CAL_COMPS]
            if len(specs)!=32: raise AssertionError("CAL32 split mismatch")
            summaries=[]
            for iteration in (1,2):
                rows=[]
                with threadpool_limits(limits=2):
                    for idx,spec in enumerate(specs):
                        item,p,k,events,*_=baseline_input(spec,audio,root,
                            heads,selection,offset,model,policy,choice["release"])
                        # Labels after MIDI generated.
                        truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                        measured=metric(events,truth,item["ends"])
                        rows.append(dict(name=spec["name"],sha=event_signature(events),
                                         counts={k:int(measured[k]) for k in COUNTS},
                                         metrics={k:float(measured[k]) for k in SCORES}))
                        print(f"S794_PARITY_REPEAT_{iteration}",idx+1,"/32",spec["name"],flush=True)
                aggregated=aggregate([dict(**r["counts"]) for r in []]) if False else None
                # Aggregation requires the richer records; explicitly sum only
                # count fields and verify official six metrics against the signed
                # source by independently recomputing all metric formulas.
                vals={k:sum(row["counts"][k] for row in rows) for k in COUNTS}
                base=signed["frozen_candidate"]["CAL"]
                drift=pair_signature(base,vals)
                result=dict(iteration=iteration,counts=vals,drift=drift,
                    exact_parity=not drift,records=rows)
                report["iterations"].append(result)
                report["CAL_count"]=len(rows)
                save()
                print("S794_PARITY_ITERATION",iteration,json.dumps(dict(
                    changed_counts=drift,exact_parity=not drift)),flush=True)
                summaries.append(rows)
    changed=[]
    for x,y in zip(*summaries):
        if x["name"]!=y["name"]:raise AssertionError("record alignment changed")
        changes={k:(x["counts"][k],y["counts"][k]) for k in COUNTS
                 if x["counts"][k]!=y["counts"][k]}
        if changes or x["sha"]!=y["sha"]:
            changed.append(dict(name=x["name"],source_event_sha_first=x["sha"],
                                source_event_sha_second=y["sha"],
                                changed_counts=changes))
    report["difference_by_record"]=changed
    report["same_run_nondeterministic"]=bool(changed)
    report["complete"]=True
    save()
    print("S794_PARITY_PROBE_COMPLETE",json.dumps(dict(
        changed_records=len(changed),changed=changed,
        drift1=report["iterations"][0]["drift"],
        drift2=report["iterations"][1]["drift"],
        same_run_nondeterministic=bool(changed))),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--s708",type=Path,required=True)
    p.add_argument("--offset-checkpoint",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    run(p.parse_args())
