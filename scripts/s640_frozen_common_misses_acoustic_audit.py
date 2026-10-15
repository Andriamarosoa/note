"""S640 — frozen S638/S639 common-miss acoustic causality audit.

Replays the unchanged ORIGINAL S614 checkpoint on exact GuitarSet PCM
used in S638, enforcing each WAV SHA-256 and exact MIDI events.
No changes to S80, S614, S632, MIDI or the S635 evaluator.

For the 613 ground-truth notes missed by BOTH engines under 80ms
causal scoring, partitions failure into observable *mechanisms*:
 premature same-pitch MIDI (not eligible under causal metric), delayed
 same-pitch MIDI, inadequate onset evidence, inadequate simultaneous
 sustain, disarmed/refractory gate, polyphonic selection or unresolved
 discrepancy. These categories are diagnostic, NOT causal interventions;
 harmonic proximity is reported separately as a suspicion, NOT proof.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from causal_note.strict_live import (
    HOP, SR, CausalPitchFeatures, CausalSelection, Selection, load_heads,
)
from causal_note.s635_shared_midi_metric import (
    compare_manifest, extract_events, extract_reference,
)
from scripts.s613_prepare_strict_live import download, digest
from scripts.s616_live_dev_ablation import ROOT, MODEL
from scripts.s638_note_midi_same_guitarset import convert_same_wav, S614_SHA

WINDOW_MS=80
LOOKBACK_MS=80
LATE_MS=240
HARMONIC_OFFSETS=(12,19,24)
REGISTERS=((40,52,"bass"),(53,64,"middle"),(65,76,"high"))


def register(pitch):
    return next(label for lo,hi,label in REGISTERS if lo<=pitch<=hi)


def reproduce_native_s614(pcm, heads):
    """Native true stream: preserve output and pre-selection evidence."""
    frontend=CausalPitchFeatures()
    selection=Selection(**heads.metadata["selection"])
    selector=CausalSelection(selection)
    frames=[]
    events=[]
    for start in range(0,len(pcm),HOP):
        audio=pcm[start:start+HOP].astype(np.float32)/32768.
        for end,features in frontend.push(audio,start_sample=start):
            p,k=heads.predict(features)
            onset,sustain,release=p.T
            active_before=selector.active.copy()
            newly_low=np.where(
                (sustain<selection.active_off)|(release>.75),
                np.minimum(selector.low+1,selection.release_frames),0)
            active_after=active_before & (newly_low<selection.release_frames)
            confident=(onset>=selection.onset)&(sustain>=selection.active_on)
            arming=selector.armed | (onset<selection.onset_reset)
            next_high=np.where(confident,
                np.minimum(selector.high+1,selection.confirmation_frames),0)
            eligible=(next_high>=selection.confirmation_frames)&arming
            refractory=end-selector.last_on>=selection.refractory_frames*HOP
            ready=eligible&refractory
            budget=selection.max_notes
            if k.max()>=selection.count_confidence:
                budget=min(budget,max(int(k.argmax()),int(active_after.sum())))
            gated=ready & ~active_after & (int(active_after.sum())>=budget)
            # Report ALL co-positive same-pitch options even if a competing
            # ready candidate would win the capped capacity first.
            h=features[:,:24].reshape(49,3,8)
            fundamental=h[:,:,0].mean(axis=1)
            frame=dict(end=int(end),
                onset=np.asarray(onset,np.float32).copy(),
                sustain=np.asarray(sustain,np.float32).copy(),
                co_positive=confident.copy(),
                ready=ready.copy(),refractory_ok=refractory.copy(),
                armed=arming.copy(),gated=gated.copy(),
                fundamental=np.asarray(fundamental,np.float32).copy(),
                active_before=active_before.copy(),
                active_count=int(active_after.sum()),budget=int(budget))
            frames.append(frame)
            events.extend(selector.step(p,k,end,frontend.received))
    events.extend(selector.panic(len(pcm),"stream_end"))
    return frames,events


def preliminary_miss_classification(note,frames,outputs,selection):
    """A one-label diagnostic hierarchy with separately reported flags."""
    start=float(note["start_s"])
    pitch=int(note["pitch"])
    idx=pitch-40
    if idx<0 or idx>=49:raise ValueError("NOTE pitch index outside model")
    onset_sample=start*SR
    within=[f for f in frames if onset_sample-1e-8<=f["end"]
            <=onset_sample+WINDOW_MS*SR/1000.+1e-8]
    before=[f for f in frames if onset_sample-40*SR/1000.<=f["end"]<onset_sample]
    if not within:
        return dict(category="no_observation_window",strong=False,
             simultaneous=False,ready=False,gated=False, early=False, late=False,
             harmonic_like=False,highest_onset=None,highest_sustain=None,
             highest_onset_pre=None,relative_fundamental=None,
             max_other_pitch_onset=None,window_frames=0)
    onset=max(float(f["onset"][idx]) for f in within)
    sustain=max(float(f["sustain"][idx]) for f in within)
    simultaneous=any(bool(f["co_positive"][idx]) for f in within)
    ready=any(bool(f["ready"][idx]) for f in within)
    gated=any(bool(f["gated"][idx]) for f in within)
    pre=max((float(f["onset"][idx]) for f in before),default=0.)
    early=any(p==pitch and start-LOOKBACK_MS/1000.<=t<start-1e-12 for t,p in outputs)
    late=any(p==pitch and start+WINDOW_MS/1000.<t<=start+LATE_MS/1000. for t,p in outputs)
    higher=0.
    harmonic_like=False
    fundamental=[]
    for f in within:
        fundamental.append(float(f["fundamental"][idx]))
        candidates=[idx+d for d in HARMONIC_OFFSETS if idx+d<49]
        for other in candidates:
            higher=max(higher,float(f["onset"][other]))
            if (f["onset"][other]>=selection.onset and
                    f["sustain"][other]>=selection.active_on):
                harmonic_like=True
    rel=max(fundamental)/(1e-6+max(float(np.max(f["fundamental"])) for f in within))
    flags=dict(strong=onset>=selection.onset,
         simultaneous=simultaneous,ready=ready,gated=gated,
         early=early,late=late,harmonic_like=harmonic_like,
         highest_onset=onset,highest_sustain=sustain,
         highest_onset_pre=pre,relative_fundamental=rel,
         max_other_pitch_onset=higher,window_frames=len(within))
    if early:
        category="premature_same_pitch_output"
    elif late:
        category="delayed_same_pitch_output"
    elif not flags["strong"]:
        category="no_sufficient_onset_after_true_onset"
    elif not simultaneous:
        category="onset_present_sustain_not_simultaneous"
    elif not ready:
        unarmed=any(bool(f["co_positive"][idx]) and not bool(f["armed"][idx])
                    for f in within)
        refractory=any(bool(f["co_positive"][idx]) and not bool(f["refractory_ok"][idx])
                    for f in within)
        category=("candidate_disarmed" if unarmed else
                  "refractory_gate" if refractory else "confirmation_or_other_gate")
    elif gated:
        category="polyphony_capacity_gate"
    else:
        category="ready_but_not_matched"
    return dict(category=category,**flags)


def audit(s638,s639,protocol,heads,root,source_zip):
    scored=compare_manifest(s638)
    if scored["recordings"]!=12 or scored["overall"]["note"]["80"]["TP"]!=901:
        raise AssertionError("Expected frozen S638 12-source causal metric")
    if scored["overall"]["midi"]["80"]["TP"]!=397:
        raise AssertionError("Expected frozen original MIDI matching totals")
    if s639["matched_reference_outcomes"]["all"]["neither"]!=613:
        raise AssertionError("S639 frozen shared-miss inventory changed")
    specs={r["name"]:r for r in protocol["sources"]
            if r["split"]=="dev" and r["corpus"]=="GuitarSet"}
    cfg=Selection(**heads.metadata["selection"])
    misses=[]
    controls=[]
    sources=[]
    classified=Counter()
    flags=Counter()
    for clip in s638["clips"]:
        name=clip["clip_id"]
        if name not in specs:raise AssertionError("Wrong GuitarSet split/source")
        wav,sha,raw_sha,pcm=convert_same_wav(specs[name],source_zip,root)
        if sha!=clip["audio_sha256"]:raise AssertionError("S638 WAV SHA changed: "+name)
        emitted_by_note=[(e["type"],e["pitch"],e["available_sample"])
                         for e in clip["note"]["events"]]
        frames,events=reproduce_native_s614(pcm,heads)
        emitted_by_replay=[(e["type"],e["pitch"],e["available_sample"]) for e in events]
        if emitted_by_note!=emitted_by_replay:
            raise AssertionError("S614 replay no longer identical: "+name)
        emitted=extract_events(clip["note"]["events"],clip["duration_s"])
        outputs=[(t,pitch) for t,pitch,_ in emitted]
        references=[ref for ref in extract_reference(clip["reference"],clip["duration_s"])
                    if 40<=ref[1]<=76]
        matched_note={int(a) for a,b,delay in scored["by_clip"][name]["note"]["80"]["matches"]}
        matched_midi={int(a) for a,b,delay in scored["by_clip"][name]["midi"]["80"]["matches"]}
        both_missed={ridx for t,p,ridx in references
                     if ridx not in matched_note and ridx not in matched_midi}
        expected=set(s639["individual_truth_reference_indices"][name]["missed_by_both_reference_indices"])
        if both_missed!=expected:raise AssertionError("S639 mismatch "+name)
        for t,p,ridx in references:
            note=dict(start_s=t,pitch=p)
            record=preliminary_miss_classification(note,frames,outputs,cfg)
            group=("missed_both" if ridx in both_missed
                   else "NOTE_correct" if ridx in matched_note
                   else "MIDI_only")
            rec=dict(source=name,reference_index=ridx,time_s=float(t),
                     pitch=int(p),register=register(p),
                     role="solo" if name.endswith("_solo") else "accompaniment",
                     status=group,**record)
            if ridx in both_missed:
                misses.append(rec)
                classified[record["category"]]+=1
                flags["harmonic_competitor"]+=int(record["harmonic_like"])
                flags["onset_peak_below_threshold"]+=int(not record["strong"])
                flags["early"]+=int(record["early"])
                flags["late"]+=int(record["late"])
            else:
                controls.append(rec)
        sources.append(dict(source=name,missed_by_both=len(both_missed),
                            native_replay_exact=True,PCM_sha256=sha,
                            total_frames=len(frames),control_predictions=len(events)))
        print("S640 SOURCE",name,"missed_both",len(both_missed),
              "total",len(references),flush=True)
    if len(misses)!=613 or len(controls)!=967:
        raise AssertionError("Missed/control count drift")
    if sum(classified.values())!=613:raise AssertionError("Categories fail partition")
    by_role=defaultdict(Counter)
    by_reg=defaultdict(Counter)
    for item in misses:
        by_role[item["role"]][item["category"]]+=1
        by_reg[item["register"]][item["category"]]+=1
    # Compare control distributions, not only misses, to avoid a biased
    # acoustic diagnosis ("harmonics near notes" happens for true notes too).
    def averages(rows):
        return dict(n=len(rows),
            onset_median=float(np.median([v["highest_onset"] for v in rows])),
            sustain_median=float(np.median([v["highest_sustain"] for v in rows])),
            onset_below_threshold_pct=100*sum(not r["strong"] for r in rows)/max(1,len(rows)),
            harmonic_competitor_pct=100*sum(r["harmonic_like"] for r in rows)/max(1,len(rows)),
            ready_pct=100*sum(r["ready"] for r in rows)/max(1,len(rows)),
            early_output_pct=100*sum(r["early"] for r in rows)/max(1,len(rows)))
    return dict(status="S640_FROZEN_S638_S639_CAUSAL_ACOUSTIC_FAILURE_AUDIT",
                matched_original_NOTE_S614_events=True,verified_all_12_PCM_SHA256=True,
                original_checkpoints_unchanged=True,
                no_training=True,no_hyperparameter_selection=True,
                no_independent_test_claim=True,no_promotion=True,
                count_missed_by_both=len(misses),controls=len(controls),
                diagnosis_categories=dict(classified),
                nonexclusive_flags=dict(flags),
                acoustic_miss_summary=averages(misses),
                acoustic_control_summary=averages(controls),
                by_role={k:dict(v) for k,v in by_role.items()},
                by_register={k:dict(v) for k,v in by_reg.items()},
                by_source=sources,
                all_missed_event_diagnostics=misses)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--s638",type=Path,required=True)
    ap.add_argument("--s639",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError("Immutable S640 evidence only")
    manifest=json.loads(args.s638.read_text())
    earlier=json.loads(args.s639.read_text())
    original=json.loads((ROOT/"selection_frozen.json").read_text())
    proto=json.loads((ROOT/"protocol.json").read_text())
    assert digest(MODEL.read_bytes())==S614_SHA==original["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==original["protocol_sha256"]
    heads=load_heads(MODEL,expected_sha256=S614_SHA)
    with tempfile.TemporaryDirectory() as folder:
        cache=Path(folder);(cache/"data").mkdir()
        item=proto["guitar_archives"]["GuitarSet_audio"]
        download(item["url"],cache/"data/GuitarSet_audio.zip",
                 item["size"],item["sha256"])
        with zipfile.ZipFile(cache/"data/GuitarSet_audio.zip") as z:
            result=audit(manifest,earlier,proto,heads,cache,z)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print("S640 FINAL SUMMARY",json.dumps({key:result[key] for key in
          ("count_missed_by_both","controls","diagnosis_categories",
           "nonexclusive_flags","acoustic_miss_summary",
           "acoustic_control_summary","by_role","by_register")},indent=2),flush=True)


if __name__=="__main__":main()
