"""S770: causal joint ONSET shadow, 65 FIT80 OOF combinations, frozen CAL32.
S708 signed MIDI, S707 offsets and microphone live remain untouched.
CAL32 is a repeatedly used diagnostic split, not an untouched test.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
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
from scripts.s704_joint_metrics import SCORES,metric,aggregate
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import FIT_COMPS,CAL_COMPS,load_model
from scripts.s733_train_onset_only import baseline_input
from scripts.s768_independent_onset_lattice_fit80 import (
    feature_grid,missing_onset_targets,sample_training,
    fit as fit_birth,onset_probabilities,proposals,combine)
from scripts.s769_source_onset_precision_fit80 import (
    source_rows,fit as fit_native_quality,scores as quality_scores)

QUALITY=(0.0,0.2,0.3,0.4,0.5)
BIRTH=(None,0.6,0.7,0.8,0.9)
RISE=(0.0,0.025,0.10)
GRID=tuple((q,b,r) for q in QUALITY for b in BIRTH
           for r in ((0.,) if b is None else RISE))
SEED=770

def label_source(events,truth):
    _,matches=event_score(events,truth,tolerance=.080)
    hit={int(i) for i,_ in matches}
    return np.array([int(i in hit) for i in range(len(events))],np.int8)

def rise_only(events,ends,features,floor):
    """Causal spectral rise at this pitch, using only already available samples."""
    if floor<=0:
        return list(events)
    ends=np.asarray(ends,np.int64)
    feat=np.asarray(features,np.float32)
    if feat.shape!=(len(ends),49,87):
        raise ValueError("S770 immutable causal feature ABI mismatch")
    kept=[]
    for e in events:
        i=int(np.searchsorted(ends,int(e["available_sample"]),side="left"))
        p=int(e["pitch"])-40
        if i>=len(ends) or int(ends[i])!=int(e["available_sample"]) or not 0<=p<49:
            raise AssertionError("S770 proposed onset misaligned with causal clock")
        if float(feat[i,p,63:71].mean())>=floor:
            kept.append(e)
    return kept

def shadow(source_events,native_ons,quality,births,ends,features,choice):
    """Independent hypotheses ONLY, original signed source unmodified."""
    q,b,r=choice
    if choice not in GRID:
        raise ValueError("S770 unregistered trial")
    if len(native_ons)!=len(quality):
        raise ValueError("S770 native event / quality mismatch")
    retained=[e for e,prob in zip(native_ons,quality) if float(prob)>=q]
    extras=[] if b is None else proposals(births,ends,source_events,b)
    return retained+rise_only(extras,ends,features,r)

def stats(prediction,truth):
    d,_=event_score(prediction,truth,tolerance=.080)
    return {k:int(d[k]) for k in ("TP","FP","FN")}

def report_scores(source,ons,quality,births,ends,features,truth):
    return {str(c):stats(shadow(source,ons,quality,births,ends,features,c),truth)
            for c in GRID}

def all_group_fit(rows):
    xb=np.concatenate([r["train_X"] for r in rows])
    yb=np.concatenate([r["train_y"] for r in rows])
    gb=np.concatenate([np.repeat(r["composition"],len(r["train_y"])) for r in rows])
    xq=np.concatenate([r["source_X"] for r in rows])
    yq=np.concatenate([r["source_y"] for r in rows])
    gq=np.concatenate([np.repeat(r["composition"],len(r["source_y"])) for r in rows])
    groups=sorted({r["composition"] for r in rows})
    if (len(groups)!=5 or set(gb)!=set(groups) or set(gq)!=set(groups)
        or set(np.unique(yb))!={0,1} or set(np.unique(yq))!={0,1}):
        raise AssertionError("S770 missing independent FIT groups or both classes")
    oof=defaultdict(list);folds={}
    for held in groups:
        mb=fit_birth(xb[gb!=held],yb[gb!=held])
        mq=fit_native_quality(xq[gq!=held],yq[gq!=held])
        if mb is None or mq is None:
            raise AssertionError("S770 composition holdout collapsed")
        subset=[r for r in rows if r["composition"]==held]
        for rec in subset:
            feat=np.load(rec["features"],mmap_mode="r")
            birth=onset_probabilities(mb,feat)
            quality=quality_scores(mq,rec["source_X"])
            out=report_scores(rec["source"],rec["source_ons"],quality,birth,
                              rec["ends"],feat,rec["truth"])
            for c in GRID:
                oof[str(c)].append(out[str(c)])
        folds[held]=dict(held_recordings=len(subset),
                         birth_train_rows=int((gb!=held).sum()),
                         native_train_rows=int((gq!=held).sum()))
        print("S770 OOF",held,len(subset),flush=True)
    summaries={str(c):combine(oof[str(c)]) for c in GRID}
    selected=max(GRID,key=lambda c:(summaries[str(c)]["f1"],
                                    summaries[str(c)]["precision"],
                                    c[1] is None,-c[0]))
    return (fit_birth(xb,yb),fit_native_quality(xq,yq),selected,summaries,
            folds,dict(birth_train_rows=len(yb),birth_positives=int(yb.sum()),
                       native_train_rows=len(yq),native_correct=int(yq.sum())))

def write(path,data):
    path.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n")

def evaluate(args):
    if args.out.exists():
        raise FileExistsError("S770 immutable output must not exist")
    args.out.mkdir(parents=True)
    signed=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    if sha!=signed["checkpoint_sha256"]:
        raise AssertionError("S770 signed S708 checkpoint hash mismatch")
    frozen=signed["frozen_candidate"]["choice"]
    if frozen["release"]!=dict(gate=.8,persistence=1,grace=24):
        raise AssertionError("S770 signed S707 release mismatch")
    net=load_model(args.s708_dir/"s708_selected.pt")[0]
    policy=GuidedPolicy(**frozen["policy"])
    torch.set_num_threads(2)
    report=dict(schema="S770_GROUPED_JOINT_ONSET_SHADOW",complete=False,
                selected="S708_frozen",promoted=False,source_midi_immutable=True,
                LIVE_changed=False,TEST_opened=False,HELD_opened=False,
                ONSET_only_not_MIDI=True,no_future_audio=True,no_truth_at_inference=True,
                fit_recordings_done=0,cal_recordings_done=0,
                GRID=[str(c) for c in GRID],FIT={},CAL={},source_reference_parity=None)
    write(args.out/"report.json",report)
    train_rows=[];signed_rows=[];cal_trials=defaultdict(list)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/"data").mkdir();cache=root/"features";cache.mkdir()
        protocol,heads,selection,offset=setup(root,args.offset_checkpoint)
        for key,fn in (("GuitarSet_audio","GuitarSet_audio.zip"),
                       ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            remote=protocol["guitar_archives"][key]
            download(remote["url"],root/"data"/fn,remote["size"],remote["sha256"])
        train=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as wav,\
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            expanded=expanded_train_specs(train,wav,ann)
            fit_specs=[s for s in expanded if s["composition"] in FIT_COMPS]
            cal_specs=[s for s in expanded if s["composition"] in CAL_COMPS]
            if len(fit_specs)!=80 or len(cal_specs)!=32 or set(FIT_COMPS)&set(CAL_COMPS):
                raise AssertionError("S770 FIT/CAL partition changed")
            for i,spec in enumerate(fit_specs):
                item,post,k,source,_,ctx,_=baseline_input(
                    spec,wav,root,heads,selection,offset,net,policy,frozen["release"])
                feat=feature_grid(item,post,k,ctx)
                native_ons,native_X=source_rows(source,item["decision_ends"],feat)
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                _,matches=event_score(source,truth,tolerance=.080)
                targets=missing_onset_targets(item["decision_ends"],truth,
                                               [j for _,j in matches])
                bx,by=sample_training(feat,targets,post,ctx,SEED+i)
                saved=cache/f"{i:03d}.npy";np.save(saved,feat.astype(np.float16))
                train_rows.append(dict(composition=spec["composition"],
                    train_X=bx,train_y=by,source_X=native_X,
                    source_y=label_source(native_ons,truth),source_ons=native_ons,
                    source=source,ends=item["decision_ends"],features=saved,truth=truth))
                report["fit_recordings_done"]=i+1
                write(args.out/"report.json",report)
                print("S770 FIT",i+1,80,spec["name"],flush=True)
            mb,mq,chosen,fit_summary,folds,counts=all_group_fit(train_rows)
            report["FIT"]=dict(folds=folds,selected_frozen=str(chosen),
                               onset_oof=fit_summary,counts=counts)
            write(args.out/"report.json",report)
            for i,spec in enumerate(cal_specs):
                item,post,k,source,_,ctx,_=baseline_input(
                    spec,wav,root,heads,selection,offset,net,policy,frozen["release"])
                feat=feature_grid(item,post,k,ctx)
                ons,native_X=source_rows(source,item["decision_ends"],feat)
                birth=onset_probabilities(mb,feat)
                quality=quality_scores(mq,native_X)
                # Construct all streams before reading CAL truth.
                streams={str(c):shadow(source,ons,quality,birth,
                                       item["decision_ends"],feat,c) for c in GRID}
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                signed_rows.append(metric(source,truth,item["ends"]))
                for key,events in streams.items():
                    cal_trials[key].append(stats(events,truth))
                report["cal_recordings_done"]=i+1
                write(args.out/"report.json",report)
                print("S770 CAL",i+1,32,spec["name"],flush=True)
    official=aggregate(signed_rows)
    drift={k:float(official[k]-signed["frozen_candidate"]["CAL"][k]) for k in SCORES}
    exact=all(abs(d)<1e-8 for d in drift.values())
    report["source_reference_parity"]=dict(valid=exact,delta=drift)
    report["source"]={k:float(official[k]) for k in SCORES}
    scored={k:combine(v) for k,v in cal_trials.items()}
    report["CAL"]=dict(native_onset=dict(TP=int(official["tp"]),
        FP=int(official["fp"]),FN=int(official["fn"]),f1=float(official["onset_f1"])),
        onset_shadow_grid=scored,selected_frozen=str(chosen),
        frozen_score=scored[str(chosen)],
        frozen_risk_over_80=bool(scored[str(chosen)]["f1"]>.8))
    report["complete"]=exact
    write(args.out/"report.json",report)
    if not exact:
        raise AssertionError("S770 signed reference six-metric parity failed")
    print("S770_JOINT_ONSET_SHADOW",json.dumps({
        "source":report["source"],"selected":str(chosen),
        "FIT":report["FIT"]["counts"],
        "CAL":dict(reference=report["CAL"]["native_onset"],
                   frozen=report["CAL"]["frozen_score"],
                   exceeds_80=report["CAL"]["frozen_risk_over_80"]),
        "promoted":False}),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s708-dir",type=Path,required=True)
    parser.add_argument("--offset-checkpoint",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    evaluate(parser.parse_args())
