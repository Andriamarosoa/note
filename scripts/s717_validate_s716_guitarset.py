"""S717: independent real GuitarSet HELD/DEV check of S716 causal live offset.

Frozen S708 predictions are computed with source-only past, then emitted
online one 4-hop block at a time. Both offline S707 and online S716 share
the exact SAME S708 source selector stream, original S675 75D hazard model,
CAL-frozen gate .80 / grace 24 / persistence 1, corpus/splits and metrics.
Only the source-only online NoteOff state/fallback differs.

HELD+DEV have prior experimental exposure: diagnostic, NOT a fresh test or
basis for selecting alternative thresholds. Does not change official LIVE.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from causal_note.strict_live import SR
from causal_note.s708_cardinality_selection import GuidedSelector, GuidedPolicy
from causal_note.s716_s707_online import S716S707OnlineRelease
from causal_note.s707_causal_release import prepare_hazards, render_release
from scripts.s613_prepare_strict_live import download,digest,labels
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s704_joint_metrics import metric,aggregate,SCORES
from scripts.s705_train_temporal_loops import setup,load_model
from scripts.s708_count_guided_joint_loops import source,evaluate_policy


def online(item, policy, release_model, release_config, model_name="s708_selected.pt"):
    selector=GuidedSelector(policy)
    specialist=S716S707OnlineRelease(release_model,**release_config)
    events=[]
    p,k=item["predictions"][model_name]
    if (len(p)!=len(item["decision_ends"]) or
            len(item["native_features"])!=len(p)):
        raise AssertionError("S708 time or native ABI wrong")
    for i,end in enumerate(item["decision_ends"]):
        fused=policy.blend*p[i]+(1-policy.blend)*item["native_post"][i]
        raw=selector.step(fused,k[i],int(end))
        events.extend(specialist.step(int(end),raw,item["native_features"][i],
                    item["native_post"][i],item["native_k"][i],int(end)))
    selector.finish(item["samples"])
    events.extend(specialist.finish(item["samples"]))
    return events,specialist.diagnostics()


def run(args):
    if args.out.exists():
        raise FileExistsError("S717 immutable output already exists")
    args.out.mkdir(parents=True)
    start=time.perf_counter()
    frozen=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    if sha!=frozen["checkpoint_sha256"] or frozen["source_family"]!=706:
        raise ValueError("Must use exact S708 selected 706 checkpoint")
    choice=frozen["frozen_candidate"]["choice"]
    policy=GuidedPolicy(**choice["policy"])
    config=choice["release"]
    if config!=dict(gate=.8,persistence=1,grace=24):
        raise ValueError("S717 forbidden unfrozen offset config")
    report=dict(schema="S717_ONLINE_S707_REAL_GUITAR_EMITTED_NOTE_AUDIT",
                promoted=False,official_live_changed=False,TEST_opened=False,
                HELD_previously_exposed=True,DEV_previously_exposed=True,
                onset_model_sha256=sha,policy=asdict(policy),release=config,
                comparison="same frozen S708 source, online causal vs S707 offline")
    def save():
        report["elapsed_s"]=round(time.perf_counter()-start,2)
        (args.out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    save()
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as work:
        root=Path(work);(root/"data").mkdir()
        protocol,heads,selection,offset=setup(root,args.offset_checkpoint)
        for name,filename in (("GuitarSet_audio","GuitarSet_audio.zip"),
                              ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            spec=protocol["guitar_archives"][name]
            download(spec["url"],root/"data"/filename,spec["size"],spec["sha256"])
        originals=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        dev=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
        model,bundle=load_model(args.s708_dir/"s708_selected.pt")
        models={"s708_selected.pt":model}
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as audio, zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            expanded=expanded_train_specs(originals,audio,ann)
            held=[s for s in expanded if s["composition"] in frozen["held_compositions"]]
            assert len(held)==32 and len(dev)==12
            for phase,specs in (("HELD",held),("DEV",dev)):
                raw=[]
                file=args.out/(phase.lower()+"_emissions.jsonl")
                with file.open("x") as output:
                    for i,spec in enumerate(specs):
                        item=source(spec,audio,heads,selection,offset,models,706)
                        src=evaluate_policy(item,"s708_selected.pt",policy)
                        offline,skipped=render_release(
                            prepare_hazards(item,src,offset),item["ends"],item["samples"],
                            **config)
                        online_events,stats=online(item,policy,offset,config)
                        # Neither model sees annotations until both event lists
                        # and their source SHA are already complete.
                        truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                        baseline=metric(offline,truth,item["ends"])
                        result=metric(online_events,truth,item["ends"])
                        for name,events in (("offline_s707",offline),("online_s716",online_events)):
                            if len({e["pitch"] for e in events if e["type"]=="note_on"})>49:
                                raise AssertionError("MIDI pitch range corrupt")
                            if any(e["available_sample"]<e["decision_sample"] for e in events):
                                raise AssertionError("Backdated event")
                        raw.append(dict(name=spec["name"],baseline=baseline,online=result,
                            skipped_offline=skipped,online_diagnostics=stats))
                        output.write(json.dumps(dict(name=spec["name"],
                            audio_sha256=item["sha256"],samples=item["samples"],
                            events=dict(offline_s707=offline,online_s716=online_events),
                            truth=truth),sort_keys=True)+"\n")
                        output.flush()
                        print("S717",phase,i+1,len(specs),spec["name"],
                              json.dumps(dict(online={k:result[k] for k in SCORES},
                                  offline={k:baseline[k] for k in SCORES})),flush=True)
                current=aggregate([row["baseline"] for row in raw])
                experimental=aggregate([row["online"] for row in raw])
                for key in SCORES:
                    if abs(current[key]-frozen[phase]["candidate"][key])>1e-8:
                        raise AssertionError(f"{phase} frozen S708 offline mismatch {key}")
                paired=dict(recovered=sum(len(set(r["online"]["matched_truth_ids"])-
                    set(r["baseline"]["matched_truth_ids"])) for r in raw),
                    lost=sum(len(set(r["baseline"]["matched_truth_ids"])-
                    set(r["online"]["matched_truth_ids"])) for r in raw),
                    lost_long=sum(len(set(r["baseline"]["long_onset_truth_ids"])-
                    set(r["online"]["long_onset_truth_ids"])) for r in raw),
                    lost_correct_offsets=sum(len(set(r["baseline"]["correct_offset_truth_ids"])-
                    set(r["online"]["correct_offset_truth_ids"])) for r in raw))
                report[phase]=dict(tracks=len(specs),
                    offline={k:current[k] for k in SCORES},
                    online={k:experimental[k] for k in SCORES},
                    paired=paired,nonregression=bool(
                        all(experimental[k]>=current[k] for k in SCORES) and
                        paired["lost_long"]==0 and paired["lost_correct_offsets"]==0))
                save()
                print("S717",phase,"FINAL",json.dumps(report[phase]),flush=True)
    report["promoted"]=False
    report["s716_matches_s708_offline_or_better"]=bool(
        report["HELD"]["nonregression"] and report["DEV"]["nonregression"])
    save()
    print("S717 FINAL",json.dumps(dict(shadow_pass=report["s716_matches_s708_offline_or_better"],
        promoted=False,held=report["HELD"],dev=report["DEV"])),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s708-dir",type=Path,required=True)
    ap.add_argument("--offset-checkpoint",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    with threadpool_limits(limits=2):
        run(ap.parse_args())
