"""S715: CAL-only shadow Exact-K policy loop over the frozen real S708 heads.

The original S708 model, native features, K distribution, and causal S707
research release are unchanged. Only K budget/guide thresholds vary. All
candidates emit a real NoteOn/NoteOff stream before labels are opened.
HELD and DEV are reused/exposed research diagnostics, NOT prospective test.
NO automatic promotion or LIVE wiring.
"""
import argparse
from dataclasses import asdict, replace
import itertools
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from causal_note.strict_live import SR
from causal_note.s708_cardinality_selection import GuidedPolicy
from causal_note.s707_causal_release import prepare_hazards, render_release
from scripts.s613_prepare_strict_live import download, digest, labels
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s704_joint_metrics import aggregate, metric, SCORES, target_rank
from scripts.s705_train_temporal_loops import setup, load_model
from scripts.s708_count_guided_joint_loops import source, evaluate_policy

MODEL_NAME = "s708_selected.pt"


def variants(original):
    policies = {"S708_frozen": original}
    for cc, gc, ws in itertools.product((.55,.70,.85,1.10),(.50,.70),(.45,.55)):
        trial = replace(original,count_confidence=cc,guide_confidence=gc,weak_sustain=ws)
        if trial == original:
            continue
        key = f"kconf{cc:.2f}_guide{gc:.2f}_sustain{ws:.2f}"
        policies[key] = trial
    return policies


def emitted_for(item, model_name, policy, offset, release):
    """One single causal-note event stream; never oracle K or future labels."""
    events=evaluate_policy(item,model_name,policy)
    plan=prepare_hazards(item,events,offset)
    emitted, skipped=render_release(plan,item["ends"],item["samples"],**release)
    return emitted, skipped


def save(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")


def phase_results(specs,archive,root,heads,selection,offset,models,
                  policies,release, *, phase):
    rows={name:[] for name in policies}
    paired={name:dict(lost_long=0,lost_correct_offsets=0) for name in policies}
    notes=[]
    for i,spec in enumerate(specs):
        # Raw audio and predictions FIRST, labels only after every variant emits.
        item=source(spec,archive,heads,selection,offset,models,706)
        emitted={}
        for name,policy in policies.items():
            emitted[name],_ = emitted_for(item,MODEL_NAME,policy,offset,release)
        truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
        measured={name:metric(events,truth,item["ends"]) for name,events in emitted.items()}
        baseline=measured["S708_frozen"]
        original_long=set(baseline["long_onset_truth_ids"])
        original_offset=set(baseline["correct_offset_truth_ids"])
        for name,result in measured.items():
            rows[name].append(result)
            paired[name]["lost_long"]+=len(original_long-set(result["long_onset_truth_ids"]))
            paired[name]["lost_correct_offsets"]+=len(original_offset-set(result["correct_offset_truth_ids"]))
        notes.append(dict(name=spec["name"],composition=spec["composition"],
                          audio_sha256=item["sha256"],events_sha256={
                              name:digest(json.dumps(events,sort_keys=True).encode())
                              for name,events in emitted.items()}))
        print("S715",phase,i+1,len(specs),spec["name"],flush=True)
    return {name:aggregate(values) for name,values in rows.items()},paired,notes


def run(args):
    if args.out.exists():
        raise FileExistsError("Cannot overwrite S715 experiment")
    args.out.mkdir(parents=True)
    original=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/MODEL_NAME).read_bytes())
    assert original["checkpoint_sha256"]==sha
    assert original["source_family"]==706
    assert original["frozen_candidate"]["choice"]["model_file"]=="s706_epoch10.pt"
    frozen=GuidedPolicy(**original["frozen_candidate"]["choice"]["policy"])
    release=original["frozen_candidate"]["choice"]["release"]
    assert release==dict(gate=.80,persistence=1,grace=24)
    options=variants(frozen)
    print("S715 FROZEN",json.dumps(dict(sha256=sha,selected_policy=asdict(frozen),
                        shadow_options=len(options))),flush=True)
    report=dict(schema="S715_CAL_ONLY_K_BUDGET_AND_HARMONIC_RISK_SHADOW",
                promoted=False,official_LIVE_changed=False,TEST_opened=False,
                reused_CAL=True,HELD_previously_exposed=True,DEV_previously_exposed=True,
                s708_checkpoint_sha256=sha,baseline_policy=asdict(frozen),
                frozen_s707_release=release,
                candidates={name:asdict(p) for name,p in options.items()},
                selection_rule="six scores >= original on CAL, zero losses of formerly correct offsets and long onsets; then maximize rank",
                primary_goal="Correct polyphonic undercount without trading off Offset, Onset, active-K, full notes or pitch F1")
    save(args.out/"report.json",report)
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/"data").mkdir()
        protocol,heads,selection,offset=setup(root,args.offset_checkpoint)
        for name,filename in (("GuitarSet_audio","GuitarSet_audio.zip"),
                               ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            src=protocol["guitar_archives"][name]
            download(src["url"],root/"data"/filename,src["size"],src["sha256"])
        train=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        dev=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as audio, zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            expanded=expanded_train_specs(train,audio,ann)
            cal=[s for s in expanded if s["composition"] in original["cal_compositions"]]
            held=[s for s in expanded if s["composition"] in original["held_compositions"]]
            assert (len(cal),len(held),len(dev))==(32,32,12)
            model,bundle=load_model(args.s708_dir/MODEL_NAME)
            models={MODEL_NAME:model}
            cal_results,cal_paired,cal_sources=phase_results(cal,audio,root,heads,selection,offset,models,
                                           options,release,phase="CAL")
            baseline=cal_results["S708_frozen"]
            eligible=[name for name in options if all(cal_results[name][key]>=baseline[key] for key in SCORES)
                  and cal_paired[name]["lost_long"]==0
                  and cal_paired[name]["lost_correct_offsets"]==0]
            if "S708_frozen" not in eligible:
                raise AssertionError("Original S708 CAL must satisfy itself")
            best=max(eligible,key=lambda name:target_rank(cal_results[name]))
            report["CAL"]=dict(scores=cal_results,paired=cal_paired,
                               chosen=best,eligible=eligible,tracks=cal_sources)
            report["selected_shadow_policy"]=dict(name=best,parameters=asdict(options[best]))
            save(args.out/"report.json",report)
            print("S715 CAL FROZEN",json.dumps(dict(selected=best,qualified=len(eligible),
                baseline={k:baseline[k] for k in SCORES},
                shadow={k:cal_results[best][k] for k in SCORES})),flush=True)
            # HELD and DEV are explicitly NOT consulted for candidate selection.
            for phase,specs in (("HELD",held),("DEV",dev)):
                policies={"S708_frozen":options["S708_frozen"]}
                if best!="S708_frozen": policies[best]=options[best]
                scores,paired,sources=phase_results(specs,audio,root,heads,selection,offset,models,
                                        policies,release,phase=phase)
                # Guard: do not accidentally compare to a different frozen S708 run.
                orig=original[phase]["candidate"]
                for key in SCORES:
                    if abs(scores["S708_frozen"][key]-orig[key])>1e-8:
                        raise AssertionError(f"S715 {phase} S708 baseline mismatch: {key}")
                baseline=scores["S708_frozen"]
                shadow=scores[best]
                protected=all(shadow[k]>=baseline[k] for k in SCORES) and (
                    paired[best]["lost_long"]==0 and paired[best]["lost_correct_offsets"]==0)
                report[phase]=dict(scores=scores,paired=paired,protected=protected,tracks=sources)
                save(args.out/"report.json",report)
                print("S715",phase,"FROZEN_VALIDATION",json.dumps(dict(
                    selected=best,protects_six_metrics=protected,
                    baseline={k:baseline[k] for k in SCORES},
                    shadow={k:shadow[k] for k in SCORES})),flush=True)
    report["shadow_passes"]=bool(report["HELD"]["protected"] and report["DEV"]["protected"]
                                 and best!="S708_frozen")
    report["promoted"]=False
    report["official_LIVE_changed"]=False
    save(args.out/"report.json",report)
    print("S715 FINAL",json.dumps(dict(selected=best,
        shadow_passes=report["shadow_passes"],promoted=False,
        held=report["HELD"]["protected"],dev=report["DEV"]["protected"])),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s708-dir",type=Path,required=True)
    ap.add_argument("--offset-checkpoint",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    with threadpool_limits(limits=2):
        run(ap.parse_args())
