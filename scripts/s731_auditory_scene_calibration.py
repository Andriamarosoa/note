"""S731 real guitar causal auditory scene 'old plus new' all-pitch experiment.

This is NOT 32D fine-tuning of S708. It compares competing acoustic
explanations at decision-time using S708's real emitted source memory:
  old-source H0: historical harmonic profile / parent broadband common rise
  new-source H1: independently rising harmonic components after explaining H0
S708 joint K0..6 is a prior/guard; real acoustic novelty can modify only
Onset posteriors. Sustain/Release/Exact-K probabilities, S707 release,
neural checkpoint and LIVE stay frozen.

FIXED CAL GRID: learned source explanation X onset gain (.15/.35/.65)
and old-harmonic explain penalty (0/.05/.10). Select only if the six
aggregate metrics AND previously correct events all hold. Historical
HELD/DEV diagnostic replay only AFTER freeze, TEST unopened.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import zipfile
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from causal_note.strict_live import SR
from causal_note.s705_temporal_heads import predict_track
from causal_note.s708_cardinality_selection import GuidedPolicy,decode_guided
from causal_note.s707_causal_release import prepare_hazards,render_release
from causal_note.s731_auditory_scene import scene_scores,reweight_onsets
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s706_train_sharp_temporal_loops import record,load_model
from scripts.s705_train_temporal_loops import setup
from scripts.s704_joint_metrics import SCORES,metric,aggregate,target_rank
from scripts.s728_real_octave_overlap_audit import overlap_results

REF="S708_frozen"
GAINS=(.15,.35,.65)
PENALTIES=(0.,.05,.10)
INTERVALS=(12,19)
CHOICES={f"novel_{str(g).replace('.','p')}_explain_{str(h).replace('.','p')}":(g,h)
         for g in GAINS for h in PENALTIES}


def emit(item,p,k,policy,offset,release):
    src=decode_guided(item["decision_ends"],p,k,policy,item["samples"])
    events,_=render_release(prepare_hazards(item,src,offset),
                           item["ends"],item["samples"],**release)
    return events


def guard(old,new,losses):
    return (all(new[m]>=old[m]-1e-12 for m in SCORES)
            and new["tp"]>=old["tp"]
            and losses["lost_correct_offsets"]==0
            and losses["lost_long"]==0)


def evaluate(specs,phase,archive,root,heads,selection,offset,model,
             policy,release,choices):
    scores={REF:[]}
    paired={key:dict(lost_correct_offsets=0,lost_long=0) for key in choices}
    overlap={name:{str(d):Counter() for d in INTERVALS}
             for name in [REF,*choices]}
    for name in choices:scores[name]=[]
    tracks=[]
    for n,spec in enumerate(specs):
        item=record(spec,archive,heads,selection,offset)
        x=item["x"]
        item["native_features"]=x[:,:,:126].copy()
        probabilities,k=predict_track(model,x)
        post=(policy.blend*probabilities+
             (1-policy.blend)*item["native_post"])
        baseline=emit(item,post,k,policy,offset,release)
        physical=scene_scores(x,post,k,baseline,item["decision_ends"])
        emitted={REF:baseline}
        for key,(gain,penalty) in choices.items():
            pp=np.stack([reweight_onsets(row,feature,gain=gain,
                                         explain=penalty,require_k=True)
                         for row,feature in zip(post,physical)])
            emitted[key]=emit(item,pp,k,policy,offset,release)
        # All models have finalized their complete streaming causal notes.
        # Truth may now be opened for diagnostics, never for post-processing.
        truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
        measured={name:metric(v,truth,item["ends"]) for name,v in emitted.items()}
        old=measured[REF]
        oldoff=set(old["correct_offset_truth_ids"])
        oldlong=set(old["long_onset_truth_ids"])
        track=dict(name=spec["name"],sha256=item["sha256"],
            evidence=dict(mean_independent_novelty=float(np.mean(
                    [f.novelty.mean() for f in physical])),
                K_room_mean=float(np.mean([f.count_room for f in physical]))),
            metrics={name:{m:row[m] for m in SCORES}
                     for name,row in measured.items()})
        for name,row in measured.items():
            scores[name].append(row)
            for d in INTERVALS:
                o=overlap_results(emitted[name],truth,d)
                for key in ("observed_real_targets","correct_real_onsets",
                            "missed_real_onsets","spurious_harmonic_child_onsets"):
                    overlap[name][str(d)][key]+=o[key]
            if name!=REF:
                paired[name]["lost_correct_offsets"]+=len(
                    oldoff-set(row["correct_offset_truth_ids"]))
                paired[name]["lost_long"]+=len(
                    oldlong-set(row["long_onset_truth_ids"]))
        tracks.append(track)
        print("S731 REAL SCENE",phase,n+1,len(specs),spec["name"],flush=True)
    return ({name:aggregate(rows) for name,rows in scores.items()},
             paired,{name:{d:dict(c) for d,c in row.items()}
                         for name,row in overlap.items()},tracks)


def run(args):
    if args.out.exists():raise FileExistsError("S731 immutable trial already exists")
    args.out.mkdir(parents=True)
    historic=json.loads((args.s708_dir/"report.json").read_text())
    original_sha=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    if (original_sha!=historic["checkpoint_sha256"] or
        historic["source_family"]!=706):
        raise AssertionError("S731 requires exactly signed original S708 model")
    policy=GuidedPolicy(**historic["frozen_candidate"]["choice"]["policy"])
    release=historic["frozen_candidate"]["choice"]["release"]
    assert release==dict(gate=.8,persistence=1,grace=24)
    model=load_model(args.s708_dir/"s708_selected.pt")[0]
    report=dict(schema="S731_CAUSAL_OLD_PLUS_NEW_ALL_NOTE_SCENE_COMPETITION",
        promoted=False,LIVE_changed=False,TEST_opened=False,
        CAL_only_selection=True,HELD_DEV_previously_exposed=True,
        original_s708_sha256=original_sha,original_k_and_offset_frozen=True,
        aud_scene_inference="past emitted S708 active sources, partial ownership up to 32, old spectrum explained away, new coherent excitation",
        calibrated_grid={name:dict(novelty=g,explain=h) for name,(g,h) in CHOICES.items()},
        release=release,policy=asdict(policy),
        condition="All six S708 metrics nonregressive; no formerly correct offset or long onset lost; no release model changes",
        research_only=True)
    def save():
        (args.out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    save()
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);(root/"data").mkdir()
        proto,heads,selection,offset=setup(root,args.offset_checkpoint)
        for kind,file in (("GuitarSet_audio","GuitarSet_audio.zip"),
                          ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            src=proto["guitar_archives"][kind]
            download(src["url"],root/"data"/file,src["size"],src["sha256"])
        train=[s for s in proto["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        dev=[s for s in proto["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as archive,zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as annotations:
            full=expanded_train_specs(train,archive,annotations)
            cal=[s for s in full if s["composition"] in historic["cal_compositions"]]
            held=[s for s in full if s["composition"] in historic["held_compositions"]]
            assert (len(cal),len(held),len(dev))==(32,32,12)
            a,paired,overlap,tracks=evaluate(cal,"CAL",archive,root,heads,
                       selection,offset,model,policy,release,CHOICES)
            old=a[REF]
            eligible=[REF]+[name for name in CHOICES
                                if guard(old,a[name],paired[name])]
            winner=max(eligible,key=lambda name:target_rank(a[name]))
            report["CAL"]=dict(scores=a,paired=paired,overlap=overlap,
                               tracks=tracks,eligible=eligible,selected=winner)
            report["frozen_choice"]=dict(selected=winner,params=CHOICES.get(winner))
            save()
            print("S731 CAL FROZEN",json.dumps(dict(selected=winner,
                 eligible=eligible,scores={name:{m:row[m] for m in SCORES}
                              for name,row in a.items()},
                 octave=overlap)),flush=True)
            choice={} if winner==REF else {winner:CHOICES[winner]}
            for phase,specs in (("HELD",held),("DEV",dev)):
                row,paired,overlap,tracks=evaluate(specs,phase,archive,root,
                            heads,selection,offset,model,policy,release,choice)
                for m in SCORES:
                    if abs(row[REF][m]-historic[phase]["candidate"][m])>1e-8:
                        raise AssertionError("Signed S708 replay mismatch "+phase+" "+m)
                safe=(True if winner==REF else
                      guard(row[REF],row[winner],paired[winner]))
                report[phase]=dict(baseline={m:row[REF][m] for m in SCORES},
                     candidate={m:row[winner][m] for m in SCORES},
                     safe=safe,overlap=overlap,tracks=tracks)
                save()
                print("S731",phase,"VALIDATED",
                      json.dumps(dict(selected=winner,safe=safe)),flush=True)
    report["shadow_qualifies"]=bool(winner!=REF and
                            report["HELD"]["safe"] and report["DEV"]["safe"])
    report["promoted"]=False
    save()
    print("S731 FINAL",json.dumps(dict(selected=winner,
        qualifies=report["shadow_qualifies"],promoted=False)),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s708-dir",type=Path,required=True)
    ap.add_argument("--offset-checkpoint",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    with threadpool_limits(limits=2):run(ap.parse_args())
