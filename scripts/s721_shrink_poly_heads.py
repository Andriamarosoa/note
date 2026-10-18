"""S721: head-specific causal blending of independently FIT-trained S720 and S708.

Why: S720 polyphonic FIT head gains active K and frame-pitch F1 but worsens
Onset/Offset/complete-note F1. Frozen S708 reference ALWAYS included. No new
training, no changes to the baseline model, Live, or datasets. Only CAL may
choose; HELD/DEV already exposed and diagnostics only.

Each blended distribution is evaluated through the same actual S708 guided
NoteOn/Off source and fixed S707 causal hazard replay as the frozen baseline.
There is no target-label injection into head outputs or selection.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from causal_note.strict_live import SR
from causal_note.s708_cardinality_selection import GuidedPolicy
from causal_note.s707_causal_release import prepare_hazards,render_release
from scripts.s613_prepare_strict_live import download,digest,labels
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s704_joint_metrics import metric,aggregate,target_rank,SCORES
from scripts.s705_train_temporal_loops import setup,load_model
from scripts.s708_count_guided_joint_loops import source,evaluate_policy

FROZEN="s708_selected.pt"
POLY="poly_bass_epoch1.pt"


def trials():
    """Frozen CAL experiment grid specified before any HELD/DEV reading."""
    candidates={"S708_frozen":dict(onset=0.,sustain=0.,release=0.,k=0.)}
    for alpha in (.10,.25,.50):
        candidates[f"all_{alpha:.2f}"]=dict(onset=alpha,sustain=alpha,release=alpha,k=alpha)
        candidates[f"onset_{alpha:.2f}"]=dict(onset=alpha,sustain=0.,release=0.,k=0.)
        candidates[f"sustain_{alpha:.2f}"]=dict(onset=0.,sustain=alpha,release=0.,k=0.)
        candidates[f"k_{alpha:.2f}"]=dict(onset=0.,sustain=0.,release=0.,k=alpha)
    candidates["sustain_k_0.25"]=dict(onset=0.,sustain=.25,release=0.,k=.25)
    candidates["onset_sustain_0.25"]=dict(onset=.25,sustain=.25,release=0.,k=0.)
    return candidates


def blend_heads(original,weighted,weights):
    """Causal elementwise convex combination; no pitch/temporal hindsight."""
    op,ok=original;wp,wk=weighted
    if (op.shape!=wp.shape or ok.shape!=wk.shape or op.shape[-2:]!=(49,3)
        or ok.shape[-1]!=7 or op.shape[:-2]!=ok.shape[:-1]):
        raise ValueError("S721 temporal head frozen ABI mismatch")
    output=np.empty_like(op)
    for j,key in enumerate(("onset","sustain","release")):
        alpha=float(weights[key])
        if not 0<=alpha<=1:raise ValueError("Invalid S721 head blend")
        output[...,j]=(1-alpha)*op[...,j]+alpha*wp[...,j]
    alpha=float(weights["k"])
    if not 0<=alpha<=1:raise ValueError("Invalid S721 K blend")
    count=(1-alpha)*ok+alpha*wk
    if not np.isfinite(output).all() or not np.isfinite(count).all():
        raise ValueError("Non-finite S721 probability")
    return output,count


def six_metric_guard(base,trial,paired):
    return (all(trial[s]>=base[s]-1e-12 for s in SCORES)
            and paired["lost_long"]==0 and paired["lost_correct_offsets"]==0
            and trial["tp"]>=base["tp"])


def score_phase(specs,archive,root,heads,selection,offset,models,
                policy,release,grid,phase):
    rows={name:[] for name in grid}
    losses={name:dict(lost_long=0,lost_correct_offsets=0) for name in grid}
    source_sha=[]
    for ix,spec in enumerate(specs):
        item=source(spec,archive,heads,selection,offset,models,706)
        original=item["predictions"][FROZEN]
        weighted=item["predictions"][POLY]
        for key,weights in grid.items():
            item["predictions"][key]=blend_heads(original,weighted,weights)
        emitted={}
        for key in grid:
            src=evaluate_policy(item,key,policy)
            emitted[key],_=render_release(
                prepare_hazards(item,src,offset),
                item["ends"],item["samples"],**release)
        # Open annotations only after the full causal emitted note streams.
        truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
        measures={name:metric(ev,truth,item["ends"])
                  for name,ev in emitted.items()}
        base=measures["S708_frozen"]
        base_offsets=set(base["correct_offset_truth_ids"])
        base_long=set(base["long_onset_truth_ids"])
        for name,row in measures.items():
            rows[name].append(row)
            losses[name]["lost_long"]+=len(base_long-set(row["long_onset_truth_ids"]))
            losses[name]["lost_correct_offsets"]+=len(base_offsets-set(row["correct_offset_truth_ids"]))
        source_sha.append(dict(name=spec["name"],audio_sha256=item["sha256"],
                               counts={name:dict(tp=measures[name]["tp"],
                                                 fp=measures[name]["fp"],
                                                 fn=measures[name]["fn"]) for name in grid}))
        print("S721",phase,ix+1,len(specs),spec["name"],flush=True)
    return {key:aggregate(score) for key,score in rows.items()},losses,source_sha


def run(args):
    if args.out.exists():raise FileExistsError("S721 refuses to overwrite result directory")
    args.out.mkdir(parents=True)
    frozen=json.loads((args.s708_dir/"report.json").read_text())
    if digest((args.s708_dir/FROZEN).read_bytes())!=frozen["checkpoint_sha256"]:
        raise ValueError("Not the original selected S708 checkpoint")
    training=json.loads((args.s720_dir/"report.json").read_text())
    if (training["base_checkpoint_sha256"]!=frozen["checkpoint_sha256"]
        or training["predeclared_training"]["poly_bass_epoch1"]["poly_factor"]!=3.):
        raise ValueError("Not frozen S720 FIT polyphonic checkpoint")
    if digest((args.s720_dir/POLY).read_bytes())!=training["train"]["poly_bass_epoch1_sha256"]:
        raise ValueError("S720 weighted model SHA mismatch")
    choice=frozen["frozen_candidate"]["choice"]
    policy=GuidedPolicy(**choice["policy"]);release=choice["release"]
    assert release==dict(gate=.80,persistence=1,grace=24)
    grid=trials()
    report=dict(schema="S721_CAL_ONLY_CAUSAL_SPECIALIST_HEAD_BLENDING",
         official_live_changed=False,promoted=False,TEST_opened=False,
         CAL_selection_only=True,HELD_DEV_previously_exposed=True,
         s708_sha256=frozen["checkpoint_sha256"],
         s720_weighted_sha256=training["train"]["poly_bass_epoch1_sha256"],
         selected_policy=asdict(policy),release=release,
         candidates=grid,requirements="six event metrics nonregressive; no lost correct offset or long onset")
    def save():
        (args.out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    save()
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/"data").mkdir()
        protocol,heads,selection,offset=setup(root,args.offset_checkpoint)
        for cat,file in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            bundle=protocol["guitar_archives"][cat]
            download(bundle["url"],root/"data"/file,bundle["size"],bundle["sha256"])
        train=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        dev=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
        models={FROZEN:load_model(args.s708_dir/FROZEN)[0],
                POLY:load_model(args.s720_dir/POLY)[0]}
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as audio,zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            expanded=expanded_train_specs(train,audio,ann)
            cal=[s for s in expanded if s["composition"] in frozen["cal_compositions"]]
            held=[s for s in expanded if s["composition"] in frozen["held_compositions"]]
            assert (len(cal),len(held),len(dev))==(32,32,12)
            scores,losses,sources=score_phase(cal,audio,root,heads,selection,offset,
                                   models,policy,release,grid,"CAL")
            old=scores["S708_frozen"]
            eligible=[n for n in grid if six_metric_guard(old,scores[n],losses[n])]
            assert "S708_frozen" in eligible
            best=max(eligible,key=lambda n:target_rank(scores[n]))
            report["CAL"]=dict(scores=scores,paired=losses,sources=sources,
                                eligible=eligible,selected=best)
            report["frozen_selection"]=dict(name=best,blend=grid[best])
            save()
            print("S721 CAL FROZEN",json.dumps(dict(selected=best,eligible=eligible,
              six_scores={n:{k:scores[n][k] for k in SCORES} for n in scores})),flush=True)
            for name,specs in (("HELD",held),("DEV",dev)):
                options={"S708_frozen":grid["S708_frozen"]}
                if best!="S708_frozen":options[best]=grid[best]
                check,paired,tracks=score_phase(specs,audio,root,heads,selection,offset,
                                                models,policy,release,options,name)
                for metric in SCORES:
                    if abs(check["S708_frozen"][metric]-frozen[name]["candidate"][metric])>1e-8:
                        raise AssertionError(f"S721 {name} original S708 benchmark mismatch {metric}")
                safe=six_metric_guard(check["S708_frozen"],check[best],paired[best])
                report[name]=dict(baseline={k:check["S708_frozen"][k] for k in SCORES},
                    trial={k:check[best][k] for k in SCORES},
                    paired=paired,validated=safe,tracks=tracks)
                save()
                print("S721",name,"GUARDED",json.dumps(dict(selected=best,
                        safe=safe,baseline=report[name]["baseline"],
                        trial=report[name]["trial"])),flush=True)
    report["shadow_qualifies"]=bool(best!="S708_frozen" and
                                  report["HELD"]["validated"] and report["DEV"]["validated"])
    report["promoted"]=False
    save()
    print("S721 FINAL",json.dumps(dict(selected=best,
       shadow_qualifies=report["shadow_qualifies"],promoted=False)),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s708-dir",type=Path,required=True)
    ap.add_argument("--s720-dir",type=Path,required=True)
    ap.add_argument("--offset-checkpoint",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    with threadpool_limits(limits=2):
        run(ap.parse_args())
