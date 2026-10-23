"""S837: six strictly independent PCM256 shadow decoder policy loops.

Runs original signed mono GuitarSet audio through unmodified S836/S708 models
ONCE, caches causal Fragment snapshots, then tunes each PRIVATE decoder
alone using other compositions. No live MIDI publication, no added hold,
no oracle at inference, and no tuning to the OUTER composition.

The historical S708 source checkpoint was trained at 1024 and the FIT80
compositions were previously explored. These are research-only CV scores,
NOT independent blind generalization or live hardware results.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import io
import itertools
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.io import wavfile

from causal_note.strict_live import HOP, SR, load_heads
from causal_note.s712_s708_live import load_s708_checkpoint
from causal_note.s834_native256_decoders import (
    AXES, Fragment, AxisDecision, Intent, OnsetDecoder, OffsetDecoder,
    ActiveKDecoder, FramePitchDecoder, FullNoteDecoder, ExactKDecoder,
)
from causal_note.s834_s809_native256_audio import S834S809Native256
from scripts.s809_fit80_independence_replay import (
    shadow_diagnostics, event, ground_truth,
)
from scripts.s613_train_strict_live import event_score
from scripts.s704_joint_metrics import metric
from scripts.s813_prepare_native256 import digest_file, save_json
from scripts.s834_s809_native256_fit80_replay import SOURCE_SCHEMA, COMPS

SERIES="S837"
SCHEMA="S837_PRIVATE_SIX_AXIS_CV_PCM256_NO_PUBLIC_MIDI_V1"


@dataclass(frozen=True)
class Source:
    name: str
    composition: str
    snapshots: tuple[Fragment, ...]
    truth: tuple[dict, ...]
    source_events: tuple[dict, ...]
    frame_ends: np.ndarray
    baseline: dict
    samples: int
    sha256: str


def choices(axis):
    """Fixed search space; neither OUTER nor result can expand the grid."""
    if axis=="onset":
        return [("reference", {})] + [
            ("on%.2f_s%.2f_r%.2f"%(on,sus,rearm),
             dict(threshold=on,sustain=sus,rearm=rearm))
            for on,sus,rearm in itertools.product(
                (.35,.45,.55,.65,.75,.85),(.20,.35,.50,.65),(.15,.25))
            if (on,sus,rearm)!=(.75,.50,.25)]
    if axis=="offset":
        return [("reference", {})] + [
            ("off%.2f_s%.2f_r%.2f"%(release,low,rearm),
             dict(release=release,low_sustain=low,rearm=rearm))
            for release,low,rearm in itertools.product(
                (.35,.50,.65,.80,.90),(.20,.35,.50,.65),(.15,.25))
            if (release,low,rearm)!=(.80,.35,.45)]
    if axis=="active_exact_k":
        return [("reference", {})] + [
            ("on%.2f_off%.2f"%(on,off),dict(on=on,off=off))
            for on,off in itertools.product((.35,.45,.55,.65),(.15,.25,.30))
            if on>off and (on,off)!=(.55,.30)]
    if axis=="frame_pitch":
        return [("reference", {})] + [
            ("s%.2f"%t,dict(sustain=t))
            for t in (.20,.30,.40,.60,.70,.80)]
    if axis=="full_note":
        return [("reference", {})] + [
            ("a%.2f_s%.2f_l%.2f"%(a,s,low),
             dict(attack=a,sustain=s,low=low))
            for a,s,low in itertools.product(
                (.40,.55,.70,.85),(.20,.40,.60),(.20,.35,.50))]
    if axis=="exact_k":
        # Confidence fallback to independent count estimate; no foreign head.
        return [("reference", {})] + [
            ("conf%.2f_s%.2f"%(confidence,threshold),
             dict(confidence=confidence,sustain=threshold))
            for confidence,threshold in itertools.product(
                (.50,.65,.80,.95),(.25,.40,.55,.70))]
    raise ValueError("unknown axis")


class PrivateExactK:
    def __init__(self, confidence, sustain):
        self.confidence,self.sustain=confidence,sustain
    def step(self, fragment):
        k=fragment.k
        native=int(np.argmax(k))
        if float(np.max(k))>=self.confidence:
            pred=native
        else:
            pred=min(6,int(np.count_nonzero(fragment.post[:,1]>=self.sustain)))
        return AxisDecision("exact_k",fragment.sample,count=pred)


CONSTRUCTORS=dict(
    exact_k=ExactKDecoder,
    active_exact_k=ActiveKDecoder,
    onset=OnsetDecoder,
    offset=OffsetDecoder,
    frame_pitch=FramePitchDecoder,
    full_note=FullNoteDecoder,
)


def new_decoder(axis, name, params):
    if axis=="exact_k" and name!="reference":
        return PrivateExactK(**params)
    return CONSTRUCTORS[axis](**params)


def score_axis(axis, items, src):
    """Same full metric as S834 SHADOW for this *one* independent axis."""
    ends=src.frame_ends
    truth=src.truth
    if axis in ("exact_k","active_exact_k","frame_pitch"):
        pitches,k=ground_truth(truth,ends)
        if axis=="exact_k":
            counts=np.fromiter((x.count for x in items),np.int16,count=len(ends))
            return dict(correct=int((counts==k).sum()),total=len(k))
        if axis=="active_exact_k":
            mask=k>0
            counts=np.fromiter((x.count for x in items),np.int16,count=len(k))
            return dict(correct=int(((counts==k)&mask).sum()),total=int(mask.sum()))
        pred=np.zeros(pitches.shape,bool)
        for t,x in enumerate(items):
            for p in x.pitches:
                pred[t,p-40]=True
        return dict(tp=int((pred&pitches).sum()),
                    fp=int((pred&~pitches).sum()),
                    fn=int((~pred&pitches).sum()))
    all_intents=[event(a) for x in items for a in x.intents]
    if axis=="onset":
        s,_=event_score(all_intents,truth,tolerance=.080)
        return {key:int(s[key]) for key in ("TP","FP","FN")}
    if axis=="offset":
        off_truth=[dict(time=n["time"]+n["duration"],pitch=n["pitch"])
                   for n in truth if n["time"]+n["duration"]<ends[-1]/SR]
        faux_on=[dict(type="note_on",pitch=x["pitch"],time=x["time"])
                 for x in all_intents]
        s,_=event_score(faux_on,off_truth,tolerance=.050)
        return {key:int(s[key]) for key in ("TP","FP","FN")}
    if axis=="full_note":
        s=metric(all_intents,truth,ends)
        return {key:int(s[key]) for key in ("tp","fp","fn")}
    raise ValueError(axis)


def evaluate(axis, name, params, src):
    d=new_decoder(axis,name,params)
    decisions=tuple(d.step(f) for f in src.snapshots)
    if axis=="full_note":
        final=d.finish(src.samples)
        decisions=decisions+(AxisDecision("full_note",int(src.frame_ends[-1]),
                                         intents=tuple(final)),)
    return score_axis(axis,decisions,src)


def audit_event_change(axis, src, baseline_label, candidate_label, candidate_params):
    """Hungarian matched GT identity and FP identity postmortem, not just F1."""
    if axis not in ("onset","offset"):
        raise ValueError("event identity only for Onset/Offset")

    def sample(name, params):
        d=new_decoder(axis,name,params)
        intent=[event(e) for f in src.snapshots for e in d.step(f).intents]
        if axis=="onset":
            truth=list(src.truth)
            predictions=intent
            tol=.080
        else:
            truth=[dict(time=t["time"]+t["duration"],pitch=t["pitch"])
                   for t in src.truth
                   if t["time"]+t["duration"]<src.frame_ends[-1]/SR]
            predictions=[dict(type="note_on",time=e["time"],pitch=e["pitch"])
                         for e in intent]
            tol=.050
        measured,pairs=event_score(predictions,truth,tolerance=tol)
        matched_pred={i for i,j in pairs}
        matched_truth={j for i,j in pairs}
        false=[predictions[i] for i in range(len(predictions))
               if i not in matched_pred]
        return measured,matched_truth,truth,false

    old,old_gt,truth,old_fp=sample(baseline_label,{})
    new,new_gt,truth,new_fp=sample(candidate_label,candidate_params)
    def ident(x):
        return (int(x["pitch"]),round(float(x["time"]),9))
    old_false=Counter(map(ident,old_fp))
    new_false=Counter(map(ident,new_fp))
    created=list((new_false-old_false).elements())
    removed=list((old_false-new_false).elements())
    recovered=sorted(new_gt-old_gt)
    lost=sorted(old_gt-new_gt)
    def register(p):
        return "grave" if p<52 else ("medium" if p<69 else "aigu")
    def explain_fp(item):
        p,t=item
        candidates=[n for n in src.truth
                    if int(n["pitch"]) in (p-12,p-19) and
                    n["time"]<=t<n["time"]+n["duration"]]
        return dict(pitch=p,time_s=t,register=register(p),
                    coincident_lower_harmonic_gt=bool(candidates),
                    harmonic_cause_proven=False)
    return dict(
        source=src.name,composition=src.composition,
        old_TP=int(old["TP"]),new_TP=int(new["TP"]),
        old_FP=int(old["FP"]),new_FP=int(new["FP"]),
        recovered_gt=int(len(recovered)),lost_gt=int(len(lost)),
        retained_gt=int(len(new_gt&old_gt)),
        new_fp_events=int(len(created)),removed_fp_events=int(len(removed)),
        new_fp_by_register=dict(Counter(register(x[0]) for x in created)),
        new_fp_harmonic_association=sum(explain_fp(e)["coincident_lower_harmonic_gt"]
                                        for e in created),
        examples_recovered=[dict(pitch=int(truth[i]["pitch"]),
                                 time_s=float(truth[i]["time"])) for i in recovered[:20]],
        examples_lost=[dict(pitch=int(truth[i]["pitch"]),
                            time_s=float(truth[i]["time"])) for i in lost[:20]],
        examples_new_fp=[explain_fp(x) for x in created[:20]],
        examples_removed_fp=[explain_fp(x) for x in removed[:20]],
        instrumentation_only=True)

def audit_count_change(axis, src, candidate_label, candidate_params):
    """Every valid PCM256 frame: recovered/lost exact count vs K0..K6.

    No selection from OUTER truth. Called only after the selection has frozen.
    """
    if axis not in ("exact_k","active_exact_k"):
        raise ValueError("count audit requires a K-axis")
    d0=new_decoder(axis,"reference",{})
    d1=new_decoder(axis,candidate_label,candidate_params)
    old=np.fromiter((d0.step(f).count for f in src.snapshots),
                    dtype=np.int16,count=len(src.snapshots))
    new=np.fromiter((d1.step(f).count for f in src.snapshots),
                    dtype=np.int16,count=len(src.snapshots))
    _,target=ground_truth(src.truth,src.frame_ends)
    valid=target>0 if axis=="active_exact_k" else np.ones(len(target),bool)
    if not valid.any():
        raise AssertionError("zero truth frames for count audit")
    changed=valid&(old!=new)
    recover=valid&(old!=target)&(new==target)
    regress=valid&(old==target)&(new!=target)
    bothwrong=valid&(old!=target)&(new!=target)
    across={}
    for k in range(7):
        mask=valid&(target==k)
        across[str(k)]=dict(frames=int(mask.sum()),
            correct_old=int((mask&(old==k)).sum()),
            correct_new=int((mask&(new==k)).sum()),
            recovered=int((mask&recover).sum()),
            lost_correct=int((mask&regress).sum()),
            candidate_over=int((mask&(new>k)).sum()),
            candidate_under=int((mask&(new<k)).sum()))
    return dict(
        name=src.name,composition=src.composition,
        valid_frames=int(valid.sum()),
        frames_changed=int(changed.sum()),
        recovered_correct=int(recover.sum()),
        lost_correct=int(regress.sum()),
        both_wrong_changed=int((bothwrong&changed).sum()),
        candidate_changed_to_higher_K=int((changed&(new>old)).sum()),
        candidate_changed_to_lower_K=int((changed&(new<old)).sum()),
        per_true_K=across,
        first_lost_examples=[
            dict(sample=int(src.frame_ends[i]),truth=int(target[i]),
                 previous=int(old[i]),candidate=int(new[i]))
            for i in np.flatnonzero(regress)[:20]],
        first_recovered_examples=[
            dict(sample=int(src.frame_ends[i]),truth=int(target[i]),
                 previous=int(old[i]),candidate=int(new[i]))
            for i in np.flatnonzero(recover)[:20]],
        source_truth_used_after_policy_locked=True)


def aggregate(axis, counts):
    result=Counter()
    for item in counts:
        result.update(item)
    return dict(result)


def score(axis, counters):
    if axis in ("exact_k","active_exact_k"):
        return counters.get("correct",0)/max(1,counters.get("total",0))
    tp,fp,fn=(counters.get(x,0) for x in
              (("TP","FP","FN") if axis in ("onset","offset")
               else ("tp","fp","fn")))
    return 2*tp/max(1,2*tp+fp+fn)


def count_differences(axis, old, new):
    """Aggregate classification change; not an identity-matching TP audit."""
    o=score(axis,old);n=score(axis,new)
    if axis in ("exact_k","active_exact_k"):
        return dict(correct_gain=new["correct"]-old["correct"],
                    incorrect_gain=old["correct"]-new["correct"],
                    metric_delta=n-o)
    keys=(("TP","FP","FN") if axis in ("onset","offset")
          else ("tp","fp","fn"))
    return dict(tp_delta=new[keys[0]]-old[keys[0]],
                fp_delta=new[keys[1]]-old[keys[1]],
                fn_delta=new[keys[2]]-old[keys[2]],
                metric_delta=n-o)


def original_for_axis(axis, src):
    b=src.baseline
    if axis in ("exact_k","active_exact_k"):
        item=b[axis]
        return dict(correct=int(item["correct"]),
                    total=int(item["frames"] if axis=="exact_k" else item["active_frames"]))
    if axis=="full_note":
        return {key:int(b[axis][key]) for key in ("tp","fp","fn")}
    if axis in ("onset","offset"):
        # S834 canonical shadow_diagnostics uses lowercase tp/fp/fn while
        # S613 event_score returns uppercase TP/FP/FN. Same counts and metric.
        return {key:int(b[axis][key] if key in b[axis] else b[axis][key.lower()])
                for key in ("TP","FP","FN")}
    return dict(b[axis])


def select_across_compositions(axis, src, all_results):
    """One prospective fold per OUTER, based ONLY on the other four comps."""
    param_map={name:params for name,params in choices(axis)}
    if len(param_map)!=len(choices(axis)):
        raise AssertionError("duplicate candidate names")
    per_outer={}
    total_base=[]
    total_trial=[]
    for outer in COMPS:
        candidates=[]
        for name in param_map:
            c=aggregate(axis,[all_results[(axis,name,s.name)] for s in src
                              if s.composition!=outer])
            candidates.append((score(axis,c),name))
        # Fixed lexicographic tie-break; reference wins ties, no OUTER peeking.
        name=max(candidates,key=lambda p:(p[0],p[1]=="reference",-list(param_map).index(p[1])))[1]
        base=aggregate(axis,[original_for_axis(axis,s) for s in src
                             if s.composition==outer])
        trial=aggregate(axis,[all_results[(axis,name,s.name)] for s in src
                              if s.composition==outer])
        total_base.append(base)
        total_trial.append(trial)
        event_audits=(
            [audit_event_change(axis,s,"reference",name,param_map[name])
             for s in src if s.composition==outer]
            if axis in ("onset","offset") else [])
        count_audits=(
            [audit_count_change(axis,s,name,param_map[name])
             for s in src if s.composition==outer]
            if axis in ("exact_k","active_exact_k") else [])
        per_outer[outer]=dict(
            count_confusion_postmortems=count_audits,
            event_identity_postmortems=event_audits,
            selected_on_other_four=name,
            candidate_parameters=param_map[name],
            reference_counts=base,candidate_counts=trial,
            reference_score=score(axis,base),candidate_score=score(axis,trial),
            failed_intervention_details=count_differences(axis,base,trial))
    original=aggregate(axis,total_base)
    candidate=aggregate(axis,total_trial)
    return dict(axis=axis,outer=per_outer,
                original_counts=original,candidate_counts=candidate,
                reference_score=score(axis,original),
                candidate_score=score(axis,candidate),
                delta=score(axis,candidate)-score(axis,original),
                requires_promotion_veto=(score(axis,candidate)<score(axis,original)),
                reaches_80_percent=(score(axis,candidate)>0.80),
                audit=count_differences(axis,original,candidate))


def source_from_real_audio(spec, audio_zip, traces, native, model, policy):
    i=int(spec["index"])
    meta_file=traces/f"fit_{i:03d}.json"
    if digest_file(meta_file)!=spec["json_sha256"]:
        raise AssertionError("S837 signed metadata SHA changed")
    meta=json.loads(meta_file.read_text())
    comp=meta["composition"]
    if comp!=spec["composition"] or meta["name"]!=spec["name"]:
        raise AssertionError("composition/source mismatch")
    rate,pcm=wavfile.read(io.BytesIO(audio_zip.read(meta["name"]+"_mic.wav")))
    if rate!=SR or pcm.dtype!=np.int16 or pcm.ndim!=1:
        raise AssertionError("must have genuine mono PCM44100 i16")
    audio=pcm[:meta["samples"]].astype(np.float32)/32768
    sha=hashlib.sha256(pcm.tobytes()).hexdigest()
    shadow=S834S809Native256(native,model,policy,sharp_engine="incremental")
    snapshots=[]
    baseline=[]
    for sample in range(0,len(audio),HOP):
        out=shadow.push(audio[sample:sample+HOP],start_sample=sample)
        if sample+HOP<=len(audio):
            if len(out)!=1:
                raise AssertionError("one decision for every 256 audio samples")
            end,decision,selector,snapshot=out[0]
            if end!=sample+HOP:
                raise AssertionError("timestamp moved or future audio observed")
            snapshots.append(snapshot)
            baseline.append(decision)
        elif out:
            raise AssertionError("invalid incomplete final hop")
    ends=np.arange(HOP,len(audio)+1,HOP,dtype=np.int64)
    if len(ends)!=len(snapshots) or shadow.model_calls!=len(ends):
        raise AssertionError("wrong PCM256 decision count")
    final=shadow.six.pipelines["full_note"].decoder.finish(meta["samples"])
    metric_base=shadow_diagnostics(baseline,final,meta["fit_truth"],ends)
    return Source(name=meta["name"],composition=comp,
                  snapshots=tuple(snapshots),truth=tuple(meta["fit_truth"]),
                  source_events=tuple(meta["source_events"]),frame_ends=ends,
                  baseline=metric_base,samples=int(meta["samples"]),sha256=sha)


def execute(*,audio_zip,traces,manifest_path,source_checkpoint,source_report,
            native_model,out,tracks_per_composition=1):
    if out.exists(): raise FileExistsError("Never overwrite signed S837 evidence")
    m=json.loads(Path(manifest_path).read_text())
    if (m.get("schema")!=SOURCE_SCHEMA or not m.get("complete") or
        len(m.get("fit_files",[]))!=80 or
        any(m.get(x) for x in ("CAL_opened","HELD_opened","TEST_opened"))):
        raise AssertionError("S837 FIT80 signed-only source required")
    if not 1<=tracks_per_composition<=16:
        raise ValueError("tracks_per_composition 1..16")
    chosen=[r for comp in COMPS for r in
        sorted((r for r in m["fit_files"] if r["composition"]==comp),
               key=lambda r:r["index"])[:tracks_per_composition]]
    if len(chosen)!=5*tracks_per_composition:
        raise AssertionError("composition coverage missing")
    native=load_heads(native_model)
    net,policy,net_hash=load_s708_checkpoint(source_checkpoint,source_report)
    out.mkdir(parents=True)
    report=dict(series=SERIES,schema=SCHEMA,complete=False,promoted=False,
                independent_axis_selection=True,decision_hop_samples=256,
                added_wait_samples=0,added_future_samples=0,
                public_midi_modified=False,source_net_sha256=net_hash,
                source_trained_originally_at_1024=True,
                fit80_previously_explored=True,
                external_held_blind=False,tracks=[],axes={})
    with zipfile.ZipFile(audio_zip) as archive:
        sources=[]
        for spec in chosen:
            src=source_from_real_audio(spec,archive,Path(traces),native,net,policy)
            sources.append(src)
            report["tracks"].append(dict(name=src.name,composition=src.composition,
                                         samples=src.samples,frames=len(src.snapshots),
                                         wav_sha256=src.sha256))
            print("S837_REAL_PCM256_SNAPSHOT",src.name,len(src.snapshots),flush=True)
    original_scores={}
    results={}
    for axis in AXES:
        grid=choices(axis)
        for label,params in grid:
            for src in sources:
                results[(axis,label,src.name)]=evaluate(axis,label,params,src)
        # Certification: direct private-decoder reference must match original
        # unchanged six-head S834 execution on every track, before tune.
        for src in sources:
            expected=original_for_axis(axis,src)
            real=results[(axis,"reference",src.name)]
            if expected!=real:
                raise AssertionError("private S837 evaluation changed S834 baseline "+
                                     axis+" "+src.name+" "+str((expected,real)))
        crossfit=select_across_compositions(axis,sources,results)
        crossfit["candidate_count"]=len(grid)
        crossfit["score_is_single_axis_shadow_not_final_midi"]=True
        report["axes"][axis]=crossfit
        save_json(out/"report.json",report)
        print("S837_CROSS_COMPOSITION_AXIS_LOOP",axis,
              round(crossfit["reference_score"],6),
              round(crossfit["candidate_score"],6),
              "delta",round(crossfit["delta"],6),
              "target80",crossfit["reaches_80_percent"],flush=True)
    report["complete"]=True
    report["tracks_count"]=len(sources)
    report["frames"]=sum(len(src.snapshots) for src in sources)
    report["all_six_above80"]=all(x["reaches_80_percent"]
                                  for x in report["axes"].values())
    report["all_six_no_regression"]=not any(x["requires_promotion_veto"]
                                            for x in report["axes"].values())
    report["safe_to_promote"]=False
    save_json(out/"report.json",report)
    print("S837_SIX_AXIS_INDEPENDENT_FINAL",
          json.dumps({k:dict(reference=round(v["reference_score"],6),
                             candidate=round(v["candidate_score"],6),
                             gain=round(v["delta"],6),
                             above_80=v["reaches_80_percent"])
                      for k,v in report["axes"].items()},sort_keys=True),flush=True)
    print("S837_VETO",json.dumps(dict(safe_to_promote=False,
                                     all_six_above80=report["all_six_above80"],
                                     all_six_no_regression=report["all_six_no_regression"])),
          flush=True)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("audio-zip","traces","manifest-path","source-checkpoint",
                 "source-report","native-model","out"):
        p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--tracks-per-composition",type=int,default=1)
    a=p.parse_args()
    execute(audio_zip=a.audio_zip,traces=a.traces,manifest_path=a.manifest_path,
            source_checkpoint=a.source_checkpoint,source_report=a.source_report,
            native_model=a.native_model,out=a.out,
            tracks_per_composition=a.tracks_per_composition)
