"""S765: FIT80-only learned arbiter over all SIX independently decoded axes.

Acoustic inputs, six S763 private decoder outputs and source MIDI identity
are available strictly at or before a candidate Onset. FullNote's future Off
is never an acceptance feature; it closes only an accepted proposal later.
No source S708/S707 event is altered. Cross-fit thresholds BY COMPOSITION.
CAL32 only audits frozen choices. Research-only; no LIVE or TEST updates.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile
import zipfile

import joblib
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from causal_note.strict_live import SR
from causal_note.s708_cardinality_selection import GuidedPolicy
from causal_note.s746_instance_guard import InstanceGate, fit_event_labels, merge_events
from scripts.s613_prepare_strict_live import download, labels, digest
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s704_joint_metrics import SCORES, metric, aggregate
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import FIT_COMPS, CAL_COMPS, load_model
from scripts.s733_train_onset_only import baseline_input
from scripts.s760_causal_confirmation_delay import pairs_from_events
from scripts.s761_fit_dualquality_birth_routing import onset_labels
from scripts.s764_six_decoder_shadow_cal32 import replay_six

RISK=(.50,.70,.85,.95)
DIM=40
MIN_NATIVE_RECENT=.100
FAMILIES=("onset","complete")

def acoustic_birth_pairs(item, post, proposals, ctx):
    """Independent H4/novelty + sustain candidate generator (strictly causal).

    Unlike S763 FullNote it may propose a missing pitch even when the
    original 0.75 onset decoder did not fire. Every closure is determined
    later by received release/sustain state. Neither future offsets nor
    annotations enter the proposal choice.
    """
    ends=np.asarray(item["decision_ends"],np.int64)
    p=np.asarray(post,np.float32)
    q=np.asarray(proposals,np.float32)
    context=np.asarray(ctx,np.float32)
    if (p.shape!=(len(ends),49,3) or q.shape!=(len(ends),49,5)
        or context.shape!=(len(ends),49,16)):
        raise ValueError("S765 acoustic candidate ABI mismatch")
    active={}
    last_open=np.full(49,-10**12,np.int64)
    pairs=[]
    opened=closed=0
    for frame,now in enumerate(ends):
        now=int(now)
        # Local acoustic release, not another decoder's state.
        for ix,on in list(active.items()):
            age=now-int(on["available_sample"])
            if age>=2*1024 and (
                p[frame,ix,2]>=.8 or p[frame,ix,1]<.30
                or age>=96*1024):
                off=dict(type="note_off",pitch=ix+40,
                    decision_sample=now,available_sample=now,
                    time=now/SR,reason="s765_acoustic_causal_release",
                    velocity=0)
                pairs.append((on,off));del active[ix];closed+=1
        for ix in range(49):
            if ix in active or now-last_open[ix]<6*1024:
                continue
            on=float(p[frame,ix,0])
            sustain=float(p[frame,ix,1])
            novelty=float(context[frame,ix,4])
            rise=float(context[frame,ix,8])
            harmonics=float(context[frame,ix,7])
            h4=float(q[frame,ix,4])
            strong_h4=(h4>=.80 and on<.75 and novelty>.075
                       and harmonics>=.25 and sustain>.45)
            soft_novel=(on>.055 and sustain>.48 and novelty>.035
                        and rise>.018 and harmonics>=.125)
            if not (strong_h4 or soft_novel):
                continue
            # This is a proposal not a verified MIDI note.
            active[ix]=dict(type="note_on",pitch=ix+40,
                decision_sample=now,available_sample=now,
                time=now/SR,reason="s765_independent_acoustic_birth",
                velocity=96)
            last_open[ix]=now;opened+=1
    end=int(item["samples"])
    for ix,on in sorted(active.items()):
        off=dict(type="note_off",pitch=ix+40,
            decision_sample=end,available_sample=end,
            time=end/SR,reason="stream_end",velocity=0)
        pairs.append((on,off))
    return pairs,dict(acoustic_opened=opened,
                      acoustic_closed=closed,
                      acoustic_stream_end=len(active))


def candidate_rows(item,post,k,source,shadow,ctx,proposals=None):
    """Causal 40D features of NEW FullNote proposals, all six heads used.

    Candidate future release data is opaque, never inspected. Source
    events at this sample may be observed; future source events cannot.
    """
    decision_ends=np.asarray(item["decision_ends"],np.int64)
    p=np.asarray(post,np.float32)
    kk=np.asarray(k,np.float32)
    acoustic=np.asarray(ctx,np.float32)
    original=np.asarray(item["native_post"],np.float32)
    if (p.shape!=(len(decision_ends),49,3)
        or kk.shape!=(len(decision_ends),7)
        or original.shape!=p.shape
        or acoustic.shape!=(len(decision_ends),49,16)):
        raise ValueError("S765 causal input ABI changed")
    onset={(int(e["available_sample"]),int(e["pitch"]))
           for e in shadow["onset"]}
    offset={(int(e["available_sample"]),int(e["pitch"]))
            for e in shadow["offset"]}
    source_sorted=sorted(enumerate(source),key=lambda x:(
        int(x[1]["available_sample"]),x[0]))
    paired=pairs_from_events(shadow["full"])
    acoustic_info={}
    if proposals is not None:
        extra,acoustic_info=acoustic_birth_pairs(item,p,proposals,ctx)
        paired.extend(extra)
    deduplicated={}
    for on,off in paired:
        key=(int(on["available_sample"]),int(on["pitch"]))
        if key not in deduplicated or on["reason"]=="s765_independent_acoustic_birth":
            deduplicated[key]=(on,off)
    original_proposed=len(paired)
    proposed=sorted(deduplicated.values(),
                    key=lambda row:(int(row[0]["available_sample"]),
                                    int(row[0]["pitch"])))
    active=set()
    laston=np.full(49,-10**12,dtype=np.int64)
    pos=0
    features,pairs,indices=[],[],[]
    excluded=Counter()
    for on,off in proposed:
        sample=int(on["available_sample"])
        pitch=int(on["pitch"])
        ix=pitch-40
        if not 0<=ix<49:
            raise ValueError("S765 FullNote pitch outside 49 semitones")
        frame=int(np.searchsorted(decision_ends,sample))
        if frame>=len(decision_ends) or int(decision_ends[frame])!=sample:
            raise AssertionError("S765 future-stamped candidate")
        while pos<len(source_sorted) and int(source_sorted[pos][1]["available_sample"])<=sample:
            e=source_sorted[pos][1];q=int(e["pitch"])
            if e["type"]=="note_on":
                active.add(q);laston[q-40]=int(e["available_sample"])
            elif e["type"]=="note_off":
                active.discard(q)
            else:
                raise ValueError("S765 unknown source event")
            pos+=1
        # Causal source-priority filter, no access to future signed NoteOns.
        if pitch in active:
            excluded["source_active"]+=1
            continue
        elapsed=float((sample-laston[ix])/SR)
        if elapsed<MIN_NATIVE_RECENT:
            excluded["source_recent"]+=1
            continue
        if len(active)>=6:
            excluded["source_full"]+=1
            continue
        spec=item["x"][frame,ix,192:216]
        supporting=float(p[frame,ix,0]*p[frame,ix,1])
        data=np.concatenate((
            acoustic[frame,ix],
            p[frame,ix],
            original[frame,ix],
            kk[frame],
            np.array([
                len(active)/6.,
                min(max(elapsed,0.),2.)/2.,
                shadow["counts"]["exact_k"][frame]/6.,
                shadow["counts"]["active_exact_k"][frame]/6.,
                float(pitch in shadow["pitches"][frame]),
                float((sample,pitch) in onset),
                float((sample,pitch) in offset),
                supporting,
                float(spec.mean()),
                float(spec.max()),
                (ix-24.)/24.
            ],np.float32)
        )).astype(np.float32)
        if data.shape!=(DIM,) or not np.isfinite(data).all():
            raise AssertionError("S765 40D causal feature schema drift")
        features.append(data)
        indices.append((frame,ix))
        pairs.append((on,off))
    X=np.stack(features) if features else np.empty((0,DIM),np.float32)
    return pairs,X,indices,dict(excluded,proposed=original_proposed,
                                  unique=len(proposed),eligible=len(pairs),
                                  **acoustic_info)

def train_one(X,y):
    if not len(X) or len(np.unique(y))<2:
        return None
    return make_pipeline(
        StandardScaler(),LogisticRegression(
            C=.1,max_iter=600,random_state=765,
            class_weight="balanced")).fit(X,y)

def positive_probability(model,X):
    if len(X)==0:
        return np.empty(0,np.float32)
    if model is None:
        return np.zeros(len(X),np.float32)
    i=np.flatnonzero(model.classes_==1)
    if not len(i):
        return np.zeros(len(X),np.float32)
    return model.predict_proba(X)[:,int(i[0])].astype(np.float32)

def grouped_quality(X,y_on,y_complete,groups):
    """Composition-grouped out-of-fold scores on FIT80 only."""
    groups=np.asarray(groups)
    if len(X)!=len(groups) or len(y_on)!=len(X) or len(y_complete)!=len(X):
        raise ValueError("S765 FIT grouped labels mismatch")
    unique=np.unique(groups)
    prediction=np.zeros((len(X),2),np.float32)
    coverage=np.zeros(len(X),np.int32)
    if len(unique)>=2 and len(X):
        for train,test in GroupKFold(n_splits=len(unique)).split(X,y_complete,groups):
            models=(train_one(X[train],y_on[train]),
                    train_one(X[train],y_complete[train]))
            prediction[test,0]=positive_probability(models[0],X[test])
            prediction[test,1]=positive_probability(models[1],X[test])
            coverage[test]+=1
        if not np.all(coverage==1):
            raise AssertionError("S765 out-of-fold predictions are not disjoint")
    quality=np.minimum(prediction[:,0],prediction[:,1])
    stats={}
    for threshold in RISK:
        admitted=quality>=threshold
        tp=int(np.count_nonzero(admitted & (y_complete==1)))
        fp=int(np.count_nonzero(admitted & (y_complete==0)))
        stats[str(threshold)]=dict(accepted=int(admitted.sum()),
                                  correct_complete=tp,false_complete=fp,
                                  precision=float(tp/max(tp+fp,1)),
                                  recall=float(tp/max(int(y_complete.sum()),1)))
    # Fixed in protocol, not tuned using CAL labels. Favor safe precision.
    eligible=[z for z in RISK if stats[str(z)]["accepted"]>=3
              and stats[str(z)]["precision"]>=.8]
    chosen=max(eligible,key=lambda z:(stats[str(z)]["correct_complete"],
                                     stats[str(z)]["precision"]),default=max(RISK))
    return chosen,dict(folds=len(unique),coverage=int(coverage.sum()),
                       candidate_rows=len(X),positive_onset=int(y_on.sum()),
                       positive_complete=int(y_complete.sum()),
                       thresholds=stats)

def collect(spec,archive,root,heads,selection,offset,network,policy,release):
    item,post,k,source,acoustic,ctx,_=baseline_input(
        spec,archive,root,heads,selection,offset,network,policy,release)
    # SIX heads completed before reading any annotations.
    shadow=replay_six(item["decision_ends"],post,k,source,item["samples"])
    pairs,X,indices,census=candidate_rows(
        item,post,k,source,shadow,ctx,proposals=acoustic)
    return item,source,k,pairs,X,indices,census

def emit_trial(source,pairs,features,indices,k,heads,risk):
    onset=positive_probability(heads["onset"],features)
    complete=positive_probability(heads["complete"],features)
    quality=np.minimum(onset,complete)
    events,diagnostics=merge_events(source,pairs,quality,indices,k,
                                    InstanceGate(risk=risk,k_tail=0.))
    return events,dict(diagnostics,proposed=len(pairs))

def write(path,report):
    path.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")

def evaluate(args):
    if args.out.exists():
        raise FileExistsError("S765 output path must be new")
    args.out.mkdir(parents=True)
    signed=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    if sha!=signed["checkpoint_sha256"]:
        raise AssertionError("S765 signed reference SHA mismatch")
    release=signed["frozen_candidate"]["choice"]["release"]
    if release!=dict(gate=.8,persistence=1,grace=24):
        raise AssertionError("S765 S707 release changed")
    policy=GuidedPolicy(**signed["frozen_candidate"]["choice"]["policy"])
    network=load_model(args.s708_dir/"s708_selected.pt")[0]
    torch.set_num_threads(2)
    report=dict(schema="S765_FIT80_SIX_ISOLATED_HEADS_LEARNED_ARBITER_CAL32",
        complete=False,source_sha256=sha,promoted=False,selected="S708_frozen",
        LIVE_changed=False,HELD_opened=False,TEST_opened=False,
        no_labels_in_inference=True,source_events_preserved=True,
        fit_recordings_done=0,cal_recordings_done=0,
        FIT={},CAL={},reference_parity=None)
    write(args.out/"report.json",report)
    xs=[];ons=[];fulls=[];groups=[]
    source_rows=[];trial_rows={z:[] for z in RISK}
    lost={z:Counter() for z in RISK}
    counts={z:Counter() for z in RISK}
    proposals=Counter()
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        protocol,heads,selection,offset=setup(root,args.offset_checkpoint)
        for kind,name in (("GuitarSet_audio","GuitarSet_audio.zip"),
                          ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            origin=protocol["guitar_archives"][kind]
            download(origin["url"],root/"data"/name,origin["size"],origin["sha256"])
        train=[s for s in protocol["sources"]
               if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as audio, \
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            expanded=expanded_train_specs(train,audio,ann)
            fit=[s for s in expanded if s["composition"] in FIT_COMPS]
            cal=[s for s in expanded if s["composition"] in CAL_COMPS]
            if len(fit)!=80 or len(cal)!=32 or set(FIT_COMPS)&set(CAL_COMPS):
                raise AssertionError("S765 composition leakage")
            for i,spec in enumerate(fit):
                item,source,k,pairs,X,indices,census=collect(
                    spec,audio,root,heads,selection,offset,network,policy,release)
                # FIT targets opened AFTER independent candidates are complete.
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                baseline_matched=metric(source,truth,item["ends"])["matched_truth_ids"]
                y_on=onset_labels(pairs,truth,baseline_matched)
                y_full=fit_event_labels(pairs,truth,baseline_matched)
                if np.any(y_full>y_on):
                    raise AssertionError("S765 correct complete must imply correct onset")
                if len(X):
                    xs.append(X)
                    ons.append(y_on)
                    fulls.append(y_full)
                    groups.extend([spec["composition"]]*len(X))
                report["fit_recordings_done"]=i+1
                write(args.out/"report.json",report)
                print("S765 FIT",i+1,80,spec["name"],"proposals",len(X),flush=True)
            X=np.concatenate(xs) if xs else np.empty((0,DIM),np.float32)
            yon=np.concatenate(ons) if ons else np.empty(0,np.int32)
            yf=np.concatenate(fulls) if fulls else np.empty(0,np.int32)
            risk,cv=grouped_quality(X,yon,yf,groups)
            learned=dict(onset=train_one(X,yon),complete=train_one(X,yf))
            joblib.dump(dict(models=learned,source_sha256=sha,
                             training="FIT80_composition_grouped_OOF",
                             dimensions=DIM,risk=risk,
                             stats=cv),args.out/"s765_arbiter.joblib")
            report["FIT"]=dict(cv,risk_frozen=risk,compositions=len(FIT_COMPS))
            write(args.out/"report.json",report)
            for i,spec in enumerate(cal):
                item,source,k,pairs,X,indices,census=collect(
                    spec,audio,root,heads,selection,offset,network,policy,release)
                candidates={}
                diagnostics={}
                for threshold in RISK:
                    trial,info=emit_trial(source,pairs,X,indices,k,learned,threshold)
                    candidates[threshold]=trial
                    diagnostics[threshold]=info
                # Annotations opened only after all candidate MIDI outputs exist.
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                src=metric(source,truth,item["ends"])
                source_rows.append(src)
                for threshold in RISK:
                    row=metric(candidates[threshold],truth,item["ends"])
                    trial_rows[threshold].append(row)
                    lost[threshold].update({
                        "correct_offsets":len(set(src["correct_offset_truth_ids"])-
                                              set(row["correct_offset_truth_ids"])),
                        "long_onsets":len(set(src["long_onset_truth_ids"])-
                                           set(row["long_onset_truth_ids"])),
                        "onset_tp":max(0,src["tp"]-row["tp"])})
                    counts[threshold].update(diagnostics[threshold])
                proposals.update(census)
                report["cal_recordings_done"]=i+1
                write(args.out/"report.json",report)
                print("S765 CAL",i+1,32,spec["name"],
                      "eligible",len(pairs),flush=True)
    original=aggregate(source_rows)
    delta={m:float(original[m]-signed["frozen_candidate"]["CAL"][m]) for m in SCORES}
    parity=all(abs(v)<1e-8 for v in delta.values())
    report["reference_parity"]=dict(valid=parity,delta=delta)
    report["reference_scores"]={m:float(original[m]) for m in SCORES}
    report["proposal_census"]=dict(proposals)
    for threshold in RISK:
        trial=aggregate(trial_rows[threshold])
        changes={m:float(trial[m]-original[m]) for m in SCORES}
        l=dict(lost[threshold])
        safe=(parity and all(v>=-1e-12 for v in changes.values())
              and trial["tp"]>=original["tp"] and not any(l.values()))
        report["CAL"][str(threshold)]=dict(
            metrics={m:float(trial[m]) for m in SCORES},
            delta=changes,losses=l,events=dict(counts[threshold]),
            safe=safe,six_gt80=bool(safe and all(trial[m]>.80 for m in SCORES)))
    # Only the PRE-REGISTERED FIT OOF threshold could be considered,
    # and still requires six-score nonregression and a fresh final test.
    frozen=report["CAL"][str(report["FIT"]["risk_frozen"])]
    report["FIT_risk_CalSafe"]=bool(frozen["safe"])
    report["candidate_for_independent_test"]=(
        "FIT_OOF_FROZEN" if frozen["safe"] and
         any(v>0 for v in frozen["delta"].values()) else None)
    report["complete"]=parity
    write(args.out/"report.json",report)
    if not parity:
        raise AssertionError("S765 signed S708 CAL32 mismatch")
    print("S765_COMPLETE",json.dumps(dict(
        reference=report["reference_scores"],FIT=report["FIT"],
        proposal_census=report["proposal_census"],
        CAL=report["CAL"],candidate=report["candidate_for_independent_test"],
        promoted=False)),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--s708-dir",type=Path,required=True)
    p.add_argument("--offset-checkpoint",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    evaluate(p.parse_args())
