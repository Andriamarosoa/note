"""S792 actual 256-hop MIDI replay of S790's frozen Frame Pitch specialist.

Training and policy choice happen on FIT80 only. CAL32 has zero influence
on which 0/1/2/4 causal NOTE-OFF defer-policy gets evaluated. Replayed
events are passed to S704's six official MIDI metrics simultaneously.
The SHADOW pitch score from S790 is never used as the official result.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import tempfile
import zipfile

import joblib
import numpy as np
import torch

from causal_note.strict_live import SR
from causal_note.s708_cardinality_selection import GuidedPolicy
from scripts.s613_prepare_strict_live import download, labels, digest
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import FIT_COMPS, CAL_COMPS, load_model
from scripts.s733_train_onset_only import baseline_input
from scripts.s704_train_joint_loops import targets
from scripts.s704_joint_metrics import SCORES, aggregate, metric
from scripts.s790_independent_framepitch_expert import (
    causal_features, sample_training, fit_specialist, evaluate_policy, DIM,
)

HOLD_BLOCKS=(0,1,2,4)
REFERENCE_SHA="c81d98dbce8d096b54d71dedd88f521162ea95b19d0c059ab13baf69d5cf0588"
PITCH_THRESHOLD=.45
HARM_PENALTY=0.
SOURCE_S790_POLICY="cut0.45_harm0.00"
BLOCK_SAMPLES=1024
MAX_ACTIVE=6

def clone_later(off,time,reason):
    """An event is timestamped ONLY when the decision can really be emitted."""
    update=dict(off)
    update["available_sample"]=int(time)
    update["decision_sample"]=int(time)
    update["time"]=int(time)/SR
    update["reason"]=reason
    return update

def replay_delayed_noteoffs(events,decision_ends,expert_mask,max_hold_blocks,samples):
    """Stateful event replay. No events are backdated and no NoteOn added.

    At time t, the algorithm sees ONLY original events available at t and
    head output for decision_ends <= t. A held NoteOff is emitted only
    after a future actual time has arrived; no oracle or future lookahead.
    """
    ends=np.asarray(decision_ends,np.int64)
    mask=np.asarray(expert_mask,bool)
    if mask.shape!=(len(ends),49) or np.any(np.diff(ends)<=0) or not \
       max_hold_blocks in HOLD_BLOCKS:
        raise ValueError("invalid frozen S790 mask or S792 release policy")
    if max_hold_blocks==0:
        return [dict(e) for e in events],dict(held=0,extended=0,capacity=0,
                                           retrigger=0,timeout=0,expert_off=0,eof=0)
    if len(ends) and (ends[-1]>samples or ends[0]<0):
        raise AssertionError("future acoustic observation beyond audio length")
    original=sorted(enumerate(events),key=lambda z:(
        int(z[1]["available_sample"]),
        0 if z[1]["type"]=="note_off" else 1, z[0]))
    for _,e in original:
        if int(e["available_sample"])<int(e["decision_sample"]):
            raise AssertionError("source event backdated")
    active=set()
    pending={}
    result=[]
    stats=Counter()
    frame=0
    now_mask=np.zeros(49,bool)

    def close(pitch,at,reason):
        saved=pending.pop(pitch)
        result.append(clone_later(saved["event"],at,reason))
        active.remove(pitch)
        stats["extended"]+=int(at>saved["start"])
        stats[reason]+=1

    def advance(t):
        nonlocal frame,now_mask
        while frame<len(ends) and int(ends[frame])<=t:
            moment=int(ends[frame])
            now_mask=mask[frame]
            frame+=1
            for pitch,saved in list(pending.items()):
                if not now_mask[pitch-40]:
                    close(pitch,moment,"expert_off")
                elif moment>=saved["deadline"]:
                    close(pitch,moment,"timeout")
    cursor=0
    while cursor<len(original):
        t=int(original[cursor][1]["available_sample"])
        advance(t)
        bucket=[]
        while cursor<len(original) and int(original[cursor][1]["available_sample"])==t:
            bucket.append(original[cursor][1])
            cursor+=1
        for e in bucket:
            pitch=int(e["pitch"])
            ix=pitch-40
            if not 0<=ix<49:raise AssertionError("pitch outside MIDI40..88")
            if e["type"]=="note_off":
                if pitch not in active:
                    raise AssertionError("source NoteOff without active source pitch")
                if (e["reason"] not in ("stream_end","shutdown","stream_error")
                    and now_mask[ix] and t<samples):
                    pending[pitch]=dict(event=dict(e),start=t,
                        deadline=t+max_hold_blocks*BLOCK_SAMPLES)
                    stats["held"]+=1
                else:
                    if pitch in pending:raise AssertionError("same pitch pending twice")
                    result.append(dict(e))
                    active.remove(pitch)
            elif e["type"]=="note_on":
                if pitch in pending:
                    close(pitch,t,"retrigger")
                if pitch in active:
                    raise AssertionError("overlapping same source pitch")
                while len(active)>=MAX_ACTIVE and pending:
                    candidate=min(pending,key=lambda p:pending[p]["start"])
                    close(candidate,t,"capacity")
                if len(active)>=MAX_ACTIVE:
                    raise AssertionError("source capacity broken")
                result.append(dict(e))
                active.add(pitch)
            else:
                raise AssertionError("unexpected source event")
    # Process only real received frames; the final EOF flush is explicit.
    advance(int(samples))
    for pitch in list(pending):
        close(pitch,int(samples),"eof")
    if active:
        raise AssertionError(f"unclosed MIDI notes after S792 replay: {active}")
    nsource=Counter(e["type"] for e in events)
    nout=Counter(e["type"] for e in result)
    if nsource!=nout:
        raise AssertionError("S792 silently added/dropped MIDI events")
    if any(e["available_sample"]<e["decision_sample"] for e in result):
        raise AssertionError("backdated replay event")
    ordered=sorted(enumerate(result),key=lambda z:(z[1]["available_sample"],z[0]))
    if [i for i,_ in ordered]!=list(range(len(result))):
        raise AssertionError("S792 event output not chronological")
    return result,dict((key,int(stats[key])) for key in
        ("held","extended","capacity","retrigger","timeout","expert_off","eof"))

def pair(item,post,k,events,model):
    X,source,harm,sustain=causal_features(item,post,k,events)
    probabilities=model.predict_proba(X.reshape(-1,DIM))[:,1].reshape(source.shape)
    shadow=evaluate_policy(probabilities,source,harm,sustain,PITCH_THRESHOLD,HARM_PENALTY)
    return shadow

def build_item(spec,archive,root,heads,selection,offset,network,policy,release):
    item,post,k,events,*_=baseline_input(
        spec,archive,root,heads,selection,offset,network,policy,release)
    # Labels opened AFTER source events and causal pitch features produced.
    X,source,harm,sustain=causal_features(item,post,k,events)
    truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
    occupancy=targets(item["decision_ends"],truth)[0][:,:,1].astype(bool)
    return dict(name=spec["name"],composition=spec["composition"],
        texture=spec["name"].rsplit("_",1)[-1],
        X=X.astype(np.float32),source=source,harm=harm.astype(np.float16),
        sustain=sustain.astype(np.float16),
        post=post.astype(np.float32),
        truth=truth,occupancy=occupancy,
        events=events,decision_ends=item["decision_ends"],
        ends=item["ends"],samples=item["samples"])

def totals(rows,key):
    return aggregate([r[key] for r in rows])

def by_composition(rows,key):
    return {c:totals([r for r in rows if r["composition"]==c],key)
            for c in sorted(set(r["composition"] for r in rows))}

def by_texture(rows,key):
    return {c:totals([r for r in rows if r["texture"]==c],key)
            for c in sorted(set(r["texture"] for r in rows))}

def score_row(record,model,hold):
    prediction=pair_from_cached(record,model)
    trial,changes=replay_delayed_noteoffs(record["events"],
            record["decision_ends"],prediction,hold,record["samples"])
    baseline=metric(record["events"],record["truth"],record["ends"])
    score=metric(trial,record["truth"],record["ends"])
    return dict(name=record["name"],composition=record["composition"],
        texture=record["texture"],baseline=baseline,trial=score,changes=changes)

def pair_from_cached(record,model):
    X=record["X"].astype(np.float32)
    source=record["source"].astype(bool)
    harm=record["harm"].astype(np.float32)
    sustain=record["sustain"].astype(np.float32)
    prob=model.predict_proba(X.reshape(-1,DIM))[:,1].reshape(source.shape)
    return evaluate_policy(prob,source,harm,sustain,PITCH_THRESHOLD,HARM_PENALTY)

def good_candidate(base,trial,bybase,bytrial):
    # NO loss on the other five axes, with 0.1 pt Frame Pitch improvement.
    return (trial["frame_pitch_f1"]>=base["frame_pitch_f1"]+.001
        and all(trial[k]>=base[k]-1e-10 for k in SCORES if k!="frame_pitch_f1")
        and all(bytrial[c]["frame_pitch_f1"]>=bybase[c]["frame_pitch_f1"]-.005
                for c in bybase))

def save(path,data):
    path.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")

def run(args):
    if args.out.exists():raise FileExistsError("S792 immutable directory")
    args.out.mkdir(parents=True)
    signed=json.loads((args.s708/"report.json").read_text())
    sha=digest((args.s708/"s708_selected.pt").read_bytes())
    if sha!=signed["checkpoint_sha256"] or sha!=REFERENCE_SHA:
        raise AssertionError("S792 altered signed S708 checkpoint")
    if signed["frozen_candidate"]["choice"]["release"]!=dict(gate=.8,persistence=1,grace=24):
        raise AssertionError("S792 source S707 release ABI changed")
    s790=json.loads(args.s790_report.read_text())
    if (s790["schema"]!="S790_INDEPENDENT_CAUSAL_FRAMEPITCH_FIT80_CAL32"
        or not s790["complete"] or s790["selected_policy"]!=SOURCE_S790_POLICY
        or s790["promoted"] or not s790["no_CAL_training_or_selection"]):
        raise AssertionError("S790 head protocol changed or was selected via CAL")
    torch.set_num_threads(2)
    network=load_model(args.s708/"s708_selected.pt")[0]
    choice=signed["frozen_candidate"]["choice"]
    policy=GuidedPolicy(**choice["policy"])
    report=dict(schema="S792_FRAMEPITCH_S790_TRANSFER_REAL_MIDI256",
        complete=False,promoted=False,official_LIVE_changed=False,
        TEST_opened=False,HELD_opened=False,cal_selection_forbidden=True,
        official_evaluation=True,score_clock_samples=256,
        pitch_head_clock_samples=1024,s790_frozen_policy=SOURCE_S790_POLICY,
        fitted_on=80,validated_on=32,fit_done=0,cal_done=0,
        selected_hold=None,FIT={},CAL={})
    def checkpoint():save(args.out/"report.json",report)
    checkpoint()
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        (root/"data").mkdir()
        protocol,heads,selection,offset=setup(root,args.offset_checkpoint)
        for key,name in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            entry=protocol["guitar_archives"][key]
            download(entry["url"],root/"data"/name,entry["size"],entry["sha256"])
        originals=[x for x in protocol["sources"] if
                   x["corpus"]=="GuitarSet" and x["split"]=="train"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as audio,\
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            specs=expanded_train_specs(originals,audio,ann)
            fit_specs=[x for x in specs if x["composition"] in FIT_COMPS]
            cal_specs=[x for x in specs if x["composition"] in CAL_COMPS]
            if len(fit_specs)!=80 or len(cal_specs)!=32 or set(FIT_COMPS)&set(CAL_COMPS):
                raise AssertionError("GuitarSet split protocol changed")
            fitrows=[]
            for i,spec in enumerate(fit_specs):
                row=build_item(spec,audio,root,heads,selection,offset,
                               network,policy,choice["release"])
                # Match S790: select training samples from original float32,
                # BEFORE its float16 cache quantization for inference.
                X=row["X"]; post=row["post"]
                training_x,training_y=sample_training(X,row["occupancy"],post,790+i)
                row["X"]=X.astype(np.float16)
                row["post"]=post.astype(np.float16)
                path=root/f"fit_{i:03d}.joblib"
                joblib.dump(row,path,compress=0)
                fitrows.append(dict(cache=path,train_x=training_x,
                    train_y=training_y,composition=spec["composition"]))
                report["fit_done"]=i+1;checkpoint()
                print("S792_FIT_RECORD",i+1,"/80",spec["name"],flush=True)
            # Cross-validate entire actual emitted MIDI on composition-held-out FIT.
            groups=sorted(set(r["composition"] for r in fitrows))
            if len(groups)!=5:raise AssertionError("five OOF composition folds needed")
            variants={hold:[] for hold in HOLD_BLOCKS}
            for comp in groups:
                model=fit_specialist(fitrows,omit=comp)
                for entry in (r for r in fitrows if r["composition"]==comp):
                    record=joblib.load(entry["cache"])
                    # Compute pitch probabilities ONCE per recording, then
                    # replay all four FIT policies on identical evidence.
                    shadow=pair_from_cached(record,model)
                    baseline=metric(record["events"],record["truth"],record["ends"])
                    empty=dict((x,0) for x in
                        ("held","extended","capacity","retrigger","timeout",
                         "expert_off","eof"))
                    for hold in HOLD_BLOCKS:
                        if hold:
                            emitted,changes=replay_delayed_noteoffs(
                                record["events"],record["decision_ends"],
                                shadow,hold,record["samples"])
                            measured=metric(emitted,record["truth"],record["ends"])
                        else:
                            changes=dict(empty);measured=dict(baseline)
                        variants[hold].append(dict(
                            name=record["name"],composition=record["composition"],
                            texture=record["texture"],baseline=dict(baseline),
                            trial=measured,changes=changes))
                print("S792_MIDI256_OOF_COMP",comp,flush=True)
            rows={}
            base=totals(variants[0],"baseline")
            compbase=by_composition(variants[0],"baseline")
            for hold,records in variants.items():
                candidate=totals(records,"trial")
                bycomp=by_composition(records,"trial")
                rows[str(hold)]=dict(metrics=candidate,
                    by_composition=bycomp,
                    eligible=(hold==0 or good_candidate(base,candidate,compbase,bycomp)),
                    event_changes={
                        name:sum(int(r["changes"][name]) for r in records)
                        for name in ("held","extended","capacity","retrigger",
                                     "timeout","expert_off","eof")})
            eligible=[k for k,v in rows.items() if v["eligible"]]
            chosen=max(eligible,key=lambda k:(rows[k]["metrics"]["frame_pitch_f1"],
                                             -int(k)))
            report["selected_hold"]=int(chosen)
            report["FIT"]=dict(baseline=base,by_composition=compbase,
                variants=rows,selected_hold=int(chosen),
                fit_compositions=groups,protocol="FIT-only 5-fold OOF, no other-axis regression")
            checkpoint()
            # Fit final expert on FIT only, then apply frozen delay on CAL.
            model=fit_specialist(fitrows) if int(chosen) else None
            calrows=[]
            for i,spec in enumerate(cal_specs):
                record=build_item(spec,audio,root,heads,selection,offset,
                                  network,policy,choice["release"])
                if model is not None:
                    row=score_row(record,model,int(chosen))
                else:
                    # Zero-hold is an EXACT identity copy of real source MIDI.
                    baseline=metric(record["events"],record["truth"],record["ends"])
                    row=dict(name=record["name"],composition=record["composition"],
                        texture=record["texture"],baseline=baseline,trial=dict(baseline),
                        changes=dict((name,0) for name in
                            ("held","extended","capacity","retrigger","timeout",
                             "expert_off","eof")))
                calrows.append(row)
                report["cal_done"]=i+1;checkpoint()
                print("S792_CAL_RECORD",i+1,"/32",spec["name"],flush=True)
    calbase=totals(calrows,"baseline")
    caltrial=totals(calrows,"trial")
    signedcal=signed["frozen_candidate"]["CAL"]
    # Literal source parity: check all six official values and relevant counts
    # before ascribing *any* score to the new architecture.
    for field in ("tp","fp","fn","frames","k_correct","active_frames",
                  "active_k_correct","frame_pitch_tp","frame_pitch_fp",
                  "frame_pitch_fn","offset_eligible","offset_correct"):
        if calbase[field]!=signedcal[field]:
            raise AssertionError(f"Source S708 CAL32 parity mismatch {field}: "+
                                 f"{calbase[field]} != {signedcal[field]}")
    for axis in SCORES:
        if not np.isclose(calbase[axis],signedcal[axis],rtol=0,atol=1e-12):
            raise AssertionError(f"S708 score parity drift on {axis}")
    report["CAL"]=dict(baseline=calbase,trial=caltrial,
        by_composition_baseline=by_composition(calrows,"baseline"),
        by_composition_trial=by_composition(calrows,"trial"),
        by_texture_baseline=by_texture(calrows,"baseline"),
        by_texture_trial=by_texture(calrows,"trial"),
        changes={key:sum(row["changes"][key] for row in calrows)
                 for key in ("held","extended","capacity","retrigger",
                             "timeout","expert_off","eof")},
        official_source_parity=True,
        six_metric_no_regression=all(caltrial[k]>=calbase[k]-1e-12 for k in SCORES),
        frame_pitch_above_80=caltrial["frame_pitch_f1"]>.8,
        official_promotion_admissible=False)
    report["complete"]=True
    checkpoint()
    print("S792_RESULT",json.dumps(dict(
        chosen_hold=report["selected_hold"],
        cal_source={k:calbase[k] for k in SCORES},
        cal_trial={k:caltrial[k] for k in SCORES},
        source_parity=True,
        changes=report["CAL"]["changes"],
        six_nonregression=report["CAL"]["six_metric_no_regression"],
        cal_frame_above_80=report["CAL"]["frame_pitch_above_80"],
        promoted=False)),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--s708",type=Path,required=True)
    parser.add_argument("--s790-report",type=Path,required=True)
    parser.add_argument("--offset-checkpoint",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    run(parser.parse_args())
