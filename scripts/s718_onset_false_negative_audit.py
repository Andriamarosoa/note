"""S718: frozen S708 onset false-negative forensic classification on real GuitarSet.

Read-only diagnostic. This does NOT adjust thresholds, labels, selected models,
or official scores. Predict first; open JAMS only after all source and S707
emitted NoteOn/Off events plus trace are frozen in memory. HELD/DEV previously
exposed, NEVER use them to select a model. Tolerance = pitch + +/-80 ms.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from causal_note.strict_live import SR
from causal_note.s708_cardinality_selection import GuidedPolicy, GuidedSelector
from causal_note.s707_causal_release import prepare_hazards, render_release
from scripts.s613_prepare_strict_live import download,digest,labels
from scripts.s613_train_strict_live import event_score,real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s705_train_temporal_loops import setup,load_model
from scripts.s708_count_guided_joint_loops import source,evaluate_policy
from scripts.s704_joint_metrics import metric,SCORES,aggregate

MODEL = "s708_selected.pt"
TOLERANCE=.080


def instrument(item, policy):
    """Replay EXACT S708 selector while observing no-label pre-decision state."""
    selector=GuidedSelector(policy)
    p,k=item["predictions"][MODEL]
    if len(p)!=len(item["decision_ends"]):
        raise AssertionError("S708 frame count mismatch")
    states=[];events=[]
    for i,end in enumerate(item["decision_ends"]):
        fused=policy.blend*p[i]+(1-policy.blend)*item["native_post"][i]
        onset,sustain,release=fused.T
        old=selector.past[0] if selector.past else np.zeros_like(fused)
        rise=(onset-old[:,0]>=policy.rise)|(sustain-old[:,1]>=2.5*policy.rise)
        arm=selector.armed|(onset<policy.onset*.4)
        state=dict(end=int(end),on=np.asarray(onset,np.float32).copy(),
                   sustain=np.asarray(sustain,np.float32).copy(),
                   release=np.asarray(release,np.float32).copy(),
                   armed=arm.copy(),active=selector.active.copy(),
                   since_last=(end-selector.last_on).astype(np.int64),
                   rising=rise.copy(),k_confidence=float(np.max(k[i])),
                   k_target=int(np.argmax(k[i])),
                   active_count=int(np.sum(selector.active)))
        these=selector.step(fused,k[i],int(end))
        state["source_on_pitches"]={
            int(e["pitch"]) for e in these if e["type"]=="note_on"}
        states.append(state)
        events.extend(these)
    events.extend(selector.finish(item["samples"]))
    canonical=evaluate_policy(item,MODEL,policy)
    actual=[(e["type"],e["pitch"],e["decision_sample"],e["reason"]) for e in events]
    expected=[(e["type"],e["pitch"],e["decision_sample"],e["reason"]) for e in canonical]
    if actual!=expected:
        raise AssertionError("S718 instrumentation changed frozen S708 source events")
    return events,states


def categorize_missing(note, states, source_matched=False):
    """Exclusive diagnostic hypotheses, not proven physical causes.

    Use same-pitch decisions within official +/-80ms timing tolerance.
    Decisions outside this window never qualify as an Onset match.
    """
    if source_matched:
        return "source_onset_removed_downstream",{}
    pitch=int(note["pitch"])
    idx=pitch-40
    t=float(note["time"])
    window=[s for s in states if abs(s["end"]/SR-t)<=TOLERANCE+1e-12]
    if not window:
        return "no_decision_in_match_window",{}
    p=next(iter(window))
    strong=lambda s: (s["on"][idx]>=.75 and s["sustain"][idx]>=.50)
    ready=lambda s: (strong(s) and bool(s["armed"][idx])
        and int(s["since_last"][idx])>=.065*SR)
    k_gate=lambda s: (s["k_confidence"]>=.50
        and s["k_target"]>s["active_count"])
    weak_base=lambda s: (s["on"][idx]>=.15 and s["sustain"][idx]>=.55
                         and s["release"][idx]<.50)
    weak_ready=lambda s: (weak_base(s) and bool(s["armed"][idx])
                and bool(s["rising"][idx])
                and s["since_last"][idx]>=.12*SR and not s["active"][idx])
    flags=dict(
       strong_joint_head_any=any(strong(s) for s in window),
       strong_selector_eligible_any=any(ready(s) for s in window),
       high_onset_alone_any=any(s["on"][idx]>=.75 for s in window),
       enough_sustain_any=any(s["sustain"][idx]>=.50 for s in window),
       armed_when_strong_any=any(strong(s) and s["armed"][idx] for s in window),
       strong_blocked_armed_any=any(strong(s) and not s["armed"][idx] for s in window),
       strong_blocked_cooldown_any=any(strong(s) and s["armed"][idx]
                        and s["since_last"][idx]<.065*SR for s in window),
       strong_ready_but_six_active_any=any(ready(s) and
                                    s["active_count"]>=6 for s in window),
       weak_acoustic_any=any(weak_base(s) for s in window),
       weak_ready_any=any(weak_ready(s) for s in window),
       weak_k_veto_any=any(weak_ready(s) and not k_gate(s) for s in window),
       weak_rise_veto_any=any(weak_base(s) and not s["rising"][idx] for s in window),
       source_emitted_in_window_any=any(pitch in s["source_on_pitches"] for s in window),
    )
    flags.update(max_onset=round(float(max(s["on"][idx] for s in window)),5),
                 max_sustain=round(float(max(s["sustain"][idx] for s in window)),5),
                 max_weak_joint=round(float(max(
                     s["on"][idx]*s["sustain"][idx] for s in window)),5),
                 frames=len(window))
    if flags["source_emitted_in_window_any"]:
        reason="source_predicted_but_match_conflict"
    elif flags["strong_selector_eligible_any"]:
        reason=("full_six_note_capacity"
                if flags["strong_ready_but_six_active_any"]
                else "eligible_strong_not_emitted")
    elif flags["strong_blocked_armed_any"]:
        reason="selector_rearming_blocked"
    elif flags["strong_blocked_cooldown_any"]:
        reason="selector_refractory_65ms"
    elif flags["weak_ready_any"] and flags["weak_k_veto_any"]:
        reason="weak_evidence_blocked_by_K_gate"
    elif flags["weak_ready_any"]:
        reason="weak_evidence_not_emitted"
    elif flags["weak_acoustic_any"] and flags["weak_rise_veto_any"]:
        reason="weak_attack_without_rising_evidence"
    elif flags["high_onset_alone_any"]:
        reason="high_onset_but_sustain_or_state_insufficient"
    elif flags["weak_acoustic_any"]:
        reason="weak_acoustic_rearmed_or_refractory"
    elif flags["max_onset"]<.15:
        reason="network_onset_below_weak_threshold"
    else:
        reason="insufficient_joint_neural_evidence"
    return reason,flags


def inspect_one(truth, source_events, emitted, states):
    _, matched_source=event_score(source_events,truth)
    _, matched_final=event_score(emitted,truth)
    source_ids={j for _,j in matched_source}
    final_ids={j for _,j in matched_final}
    reasons=Counter();flags=Counter();by_texture=Counter();by_register=Counter()
    missed=[]
    for j,note in enumerate(truth):
        if j in final_ids:continue
        reason,traits=categorize_missing(note,states,j in source_ids)
        reasons[reason]+=1
        flags.update(k for k,v in traits.items() if k.endswith("_any") and v)
        pitch=int(note["pitch"])
        reg="low_40_52" if pitch<=52 else "mid_53_64" if pitch<=64 else "high_65_88"
        by_register[reg]+=1
        # Labels are used after ALL predictions; K is descriptive only.
        at=float(note["time"])
        ktrue=sum(m["time"]<=at<m["time"]+m["duration"] for m in truth)
        missed.append(dict(truth_id=j,pitch=pitch,onset_seconds=round(at,5),
                    duration_seconds=round(float(note["duration"]),5),
                    real_K_at_onset=min(6,ktrue),reason=reason,flags=traits))
    total=len(truth)-len(final_ids)
    assert sum(reasons.values())==total
    return dict(total_truth=len(truth),true_positive=len(final_ids),
                false_negative=total,source_true_positive=len(source_ids),
                categories=dict(reasons),flags=dict(flags),
                registers=dict(by_register),missing=missed)


def run(args):
    if args.out.exists():raise FileExistsError("S718 refuses to overwrite reports")
    args.out.mkdir(parents=True)
    original=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/MODEL).read_bytes())
    assert sha==original["checkpoint_sha256"] and original["source_family"]==706
    frozen=GuidedPolicy(**original["frozen_candidate"]["choice"]["policy"])
    release=original["frozen_candidate"]["choice"]["release"]
    assert release==dict(gate=.8,persistence=1,grace=24)
    report=dict(schema="S718_FROZEN_S708_ONSET_FALSE_NEGATIVE_AUDIT",
      promoted=False,official_live_changed=False,TEST_opened=False,
      previous_HELD_DEV_exposure=True,tuned_on_HELD_DEV=False,
      source_sha256=sha,policy=asdict(frozen),source_run=39094682419,
      classification_is_diagnostic_not_causal_proof=True,
      categories_exclusive_flags_overlapping=True,phases={})
    def save():
        (args.out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    save()
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as work:
        root=Path(work);(root/"data").mkdir()
        protocol,heads,selection,offset=setup(root,args.offset_checkpoint)
        for source_name,file in (("GuitarSet_audio","GuitarSet_audio.zip"),
                                 ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            asset=protocol["guitar_archives"][source_name]
            download(asset["url"],root/"data"/file,asset["size"],asset["sha256"])
        original_sources=[s for s in protocol["sources"]
                   if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        dev=[s for s in protocol["sources"]
                if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
        model,_=load_model(args.s708_dir/MODEL)
        models={MODEL:model}
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as audio,zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            expanded=expanded_train_specs(original_sources,audio,ann)
            held=[s for s in expanded if s["composition"] in original["held_compositions"]]
            assert len(held)==32 and len(dev)==12
            for phase,specs in (("HELD",held),("DEV",dev)):
                counts=Counter();raw=[];matched=[];flags=Counter();registers=Counter()
                with (args.out/(phase.lower()+"_missing.jsonl")).open("x") as out:
                    for i,spec in enumerate(specs):
                        item=source(spec,audio,heads,selection,offset,models,706)
                        src,trace=instrument(item,frozen)
                        offline,_=render_release(prepare_hazards(item,src,offset),
                                                 item["ends"],item["samples"],**release)
                        # Ground truth opened ONLY once original events are final.
                        truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                        score=metric(offline,truth,item["ends"])
                        diagnosis=inspect_one(truth,src,offline,trace)
                        assert diagnosis["false_negative"]==score["fn"]
                        counts.update(diagnosis["categories"])
                        flags.update(diagnosis["flags"])
                        registers.update(diagnosis["registers"])
                        matched.append(score)
                        raw.append(dict(name=spec["name"],score={k:score[k] for k in SCORES},
                                   tp=score["tp"],fp=score["fp"],fn=score["fn"],
                                   source_tp=diagnosis["source_true_positive"],
                                   categories=diagnosis["categories"]))
                        out.write(json.dumps(dict(name=spec["name"],
                             audio_sha256=item["sha256"],
                             missing=diagnosis["missing"]),sort_keys=True)+"\n")
                        out.flush()
                        print("S718",phase,i+1,len(specs),spec["name"],
                            json.dumps(dict(FN=score["fn"],reasons=diagnosis["categories"])),flush=True)
                aggregate_metrics=aggregate(matched)
                for key in SCORES:
                    if abs(aggregate_metrics[key]-original[phase]["candidate"][key])>1e-8:
                        raise AssertionError(f"S718 {phase} reference {key} changed")
                assert sum(counts.values())==aggregate_metrics["fn"]
                report["phases"][phase]=dict(
                    tracks=len(specs),fn=aggregate_metrics["fn"],
                    tp=aggregate_metrics["tp"],fp=aggregate_metrics["fp"],
                    onset_f1=aggregate_metrics["onset_f1"],
                    metrics={key:aggregate_metrics[key] for key in SCORES},
                    categories=dict(counts),flags=dict(flags),
                    registers=dict(registers),per_track=raw)
                save()
                print("S718 FINAL",phase,json.dumps(dict(fn=aggregate_metrics["fn"],
                    categories=dict(counts),flags=dict(flags),
                    registers=dict(registers))),flush=True)
    print("S718 AUDIT_COMPLETE NO_PROMOTION",flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s708-dir",required=True,type=Path)
    ap.add_argument("--offset-checkpoint",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    with threadpool_limits(limits=2):
        run(ap.parse_args())
