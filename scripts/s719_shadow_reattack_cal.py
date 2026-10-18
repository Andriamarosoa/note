"""S719: controlled CAL-only retrigger selection ablation on real S708+S707.

No trained weight is changed. S708 policy remains an exact immutable baseline.
Only a new SHADOW selector may rearm on a fresh strong posterior rise after
a minimum same-pitch gap. CAL-only selection requires all six scores to
not regress AND zero formerly good long onsets / offsets lost. HELD/DEV are
historically exposed diagnostic (not new independent external test).
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from causal_note.strict_live import SR
from causal_note.s708_cardinality_selection import GuidedPolicy
from causal_note.s719_shadow_reattack import decode_shadow
from causal_note.s707_causal_release import prepare_hazards,render_release
from scripts.s613_prepare_strict_live import download,digest,labels
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s705_train_temporal_loops import setup,load_model
from scripts.s708_count_guided_joint_loops import source,evaluate_policy
from scripts.s704_joint_metrics import aggregate,metric,SCORES,target_rank

MODEL="s708_selected.pt"


def variants():
    choices={"S708_frozen":None}
    for rise in (.10,.20,.30):
        for gap in (90,150):
            choices[f"reattack_rise{rise:.2f}_gap{gap}ms"]=dict(rise=rise,min_gap_ms=gap)
    return choices


def select_cal_only(scores,paired):
    baseline=scores["S708_frozen"]
    qualified=[name for name in scores if
       all(scores[name][k]>=baseline[k]-1e-12 for k in SCORES)
       and paired[name]["lost_long"]==0
       and paired[name]["lost_correct_offsets"]==0]
    if "S708_frozen" not in qualified:
        raise AssertionError("A frozen baseline must qualify against itself")
    return max(qualified,key=lambda name:target_rank(scores[name])),qualified


def propose_events(item,policy,parameters):
    if parameters is None:
        return evaluate_policy(item,MODEL,policy),dict(reattack_proposals=0,actual_retriggers=0)
    p,k=item["predictions"][MODEL]
    post=policy.blend*p+(1-policy.blend)*item["native_post"]
    return decode_shadow(item["decision_ends"],post,k,policy,item["samples"],**parameters)


def score_one_phase(specs,audio,root,heads,selection,offset,models,policy,release,choices,phase):
    scores={n:[] for n in choices}
    paired={n:dict(lost_long=0,lost_correct_offsets=0) for n in choices}
    totals={n:dict(reattack_proposals=0,actual_retriggers=0) for n in choices}
    per_track=[]
    for i,spec in enumerate(specs):
        item=source(spec,audio,heads,selection,offset,models,706)
        all_events={}
        for name,params in choices.items():
            source_events,stats=propose_events(item,policy,params)
            plan=prepare_hazards(item,source_events,offset)
            all_events[name],_ = render_release(plan,item["ends"],item["samples"],**release)
            if params is not None:
                totals[name]["reattack_proposals"]+=stats["rearm_proposals"]
                totals[name]["actual_retriggers"]+=stats["actual_retriggers"]
        truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
        measured={n:metric(e,truth,item["ends"]) for n,e in all_events.items()}
        baseline=measured["S708_frozen"]
        old_offsets=set(baseline["correct_offset_truth_ids"])
        old_long=set(baseline["long_onset_truth_ids"])
        for name,result in measured.items():
            scores[name].append(result)
            paired[name]["lost_long"]+=len(old_long-set(result["long_onset_truth_ids"]))
            paired[name]["lost_correct_offsets"]+=len(old_offsets-set(result["correct_offset_truth_ids"]))
        per_track.append(dict(name=spec["name"],scores={
            n:{k:m[k] for k in SCORES} for n,m in measured.items()}))
        print("S719",phase,i+1,len(specs),spec["name"],flush=True)
    return ({name:aggregate(rows) for name,rows in scores.items()},
             paired,totals,per_track)


def run(args):
    if args.out.exists():raise FileExistsError("Refuse overwriting S719")
    args.out.mkdir(parents=True)
    frozen=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/MODEL).read_bytes())
    assert sha==frozen["checkpoint_sha256"] and frozen["source_family"]==706
    choice=frozen["frozen_candidate"]["choice"]
    assert choice["model_file"]=="s706_epoch10.pt"
    policy=GuidedPolicy(**choice["policy"])
    release=choice["release"]
    assert release==dict(gate=.8,persistence=1,grace=24)
    opts=variants()
    report=dict(schema="S719_ONSET_RETRIGGER_SHADOW_CAL_ONLY",promoted=False,
        official_live_changed=False,TEST_opened=False,
        HELD_previously_exposed=True,DEV_previously_exposed=True,
        source_sha256=sha,baseline_policy=asdict(policy),release=release,
        candidates=opts,all_six_scores_must_protect=True,
        recovered_noteoffs_must_not_be_lost=True,
        no_label_selection_beyond_CAL=True)
    def save():
        (args.out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    save()
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);(root/"data").mkdir()
        protocol,heads,selection,offset=setup(root,args.offset_checkpoint)
        for category,filename in (("GuitarSet_audio","GuitarSet_audio.zip"),
                                  ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            origin=protocol["guitar_archives"][category]
            download(origin["url"],root/"data"/filename,origin["size"],origin["sha256"])
        train=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        dev=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
        model,_=load_model(args.s708_dir/MODEL);models={MODEL:model}
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as audio,zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            expanded=expanded_train_specs(train,audio,ann)
            cal=[s for s in expanded if s["composition"] in frozen["cal_compositions"]]
            held=[s for s in expanded if s["composition"] in frozen["held_compositions"]]
            assert (len(cal),len(held),len(dev))==(32,32,12)
            scores,paired,totals,tracks=score_one_phase(cal,audio,root,heads,selection,offset,
                                                 models,policy,release,opts,"CAL")
            baseline=scores["S708_frozen"]
            best,qualified=select_cal_only(scores,paired)
            report["CAL"]=dict(scores=scores,paired=paired,
                               selected=best,qualified=qualified,
                               statistics=totals,tracks=tracks)
            report["frozen_selection"]=dict(name=best,params=opts[best])
            save()
            print("S719 CAL FROZEN",json.dumps(dict(selected=best,
                 eligible=len(qualified),baseline={k:baseline[k] for k in SCORES},
                 selected_stats=totals[best])),flush=True)
            for phase,specs in (("HELD",held),("DEV",dev)):
                choices={"S708_frozen":None}
                if best!="S708_frozen":choices[best]=opts[best]
                measures,guards,stats,tracks=score_one_phase(specs,audio,root,heads,
                                    selection,offset,models,policy,release,choices,phase)
                old=measures["S708_frozen"]
                for k in SCORES:
                    if abs(old[k]-frozen[phase]["candidate"][k])>1e-8:
                        raise AssertionError(f"S719 {phase} frozen baseline mismatch: {k}")
                trial=measures[best]
                safe=bool(all(trial[k]>=old[k]-1e-12 for k in SCORES)
                          and guards[best]["lost_long"]==0
                          and guards[best]["lost_correct_offsets"]==0)
                report[phase]=dict(baseline={k:old[k] for k in SCORES},
                   selected={k:trial[k] for k in SCORES},safe=safe,
                   paired=guards,statistics=stats,tracks=tracks)
                save()
                print("S719",phase,"FROZEN",json.dumps(dict(
                   safe=safe,baseline=report[phase]["baseline"],
                   trial=report[phase]["selected"],stats=stats[best])),flush=True)
    report["shadow_passes"]=bool(best!="S708_frozen" and
                       report["HELD"]["safe"] and report["DEV"]["safe"])
    report["promoted"]=False
    save()
    print("S719 FINAL",json.dumps(dict(selected=best,
         qualifies=report["shadow_passes"],promoted=False)),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s708-dir",type=Path,required=True)
    ap.add_argument("--offset-checkpoint",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    with threadpool_limits(limits=2):
        run(ap.parse_args())
