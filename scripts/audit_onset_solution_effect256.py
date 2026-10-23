"""Explain why a candidate Onset PCM256 solution improves little or regresses.

Read-only post-mortem: compare the *same* signed OUTER notes and original
direct-acoustic events to the *selected* learned-HGB policy. Explain the F1
movement by identity, not only the global aggregate. Never tune on OUTER.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import json
from pathlib import Path

import joblib
import numpy as np

from onset256.learned_hgb_arbiter import GatePolicy, replay_precomputed
from scripts.onset_acoustic_direct256 import mask_events
from scripts.onset_event_integrity256 import counts, metric, sum_scores
from scripts.onset_learned_hgb_arbiter256 import (
    inputs, track_inputs, event_pairs, sha)

SCHEMA = "ONSET_SOLUTION_EFFECT_POSTMORTEM_PCM256_V1"
CATEGORIES = ("retained_tp", "recovered_tp", "lost_tp", "persistent_fn")
HARMONIC_INTERVALS = frozenset((12, 19, 24, 28, 31, 36))


def track_score(events, truth):
    return counts([dict(sample=s, pitch=p) for s, p in events], truth)


def matched(events, truth):
    pairs = event_pairs(events, truth)
    return {i: j for i, j in pairs}


def texture(track):
    return "solo" if track.endswith("_solo") else "comp"


def register(pitch):
    return "grave_E2_B2" if pitch < 48 else "medium_C3_B3" if pitch < 60 else "high_C4_E6"


def true_active_pitches(truth, time):
    return sorted({int(n["pitch"]) for n in truth
                   if float(n["time"]) <= time and
                   float(n["time"]) + float(n.get("duration", 0)) > time})


def near_truth(truth, pitch, time, tolerance=.080):
    same = [(i, abs(float(t["time"])-time)) for i,t in enumerate(truth)
            if int(t["pitch"]) == pitch and abs(float(t["time"])-time) <= tolerance]
    other = [t for t in truth if int(t["pitch"]) != pitch
             and abs(float(t["time"])-time) <= tolerance]
    return {
        "near_same_pitch_truth": bool(same),
        "near_other_pitch_truth": bool(other),
        "harmonic_nearby_truth": any(abs(int(t["pitch"])-pitch) in HARMONIC_INTERVALS
                                     for t in other),
    }


def reasons(baseline, direct, probabilities, enabled, policy):
    """Exact selection *explanations*, verified against causal gate output."""
    frame, pitch = np.nonzero(baseline)
    if len(frame) != len(probabilities):
        raise AssertionError("unmatched scores and HGB proposals")
    last_added = np.full(49, -10**12, np.int64)
    direct_at = [np.flatnonzero(direct[:,i]) for i in range(49)]
    states = []
    for f, p, score in zip(frame, pitch, probabilities):
        f,p = int(f),int(p)
        if direct[f,p]:
            reason = "already_selected_by_acoustic"
        elif not enabled:
            reason = "disabled_by_calibration"
        elif float(score) < policy.cut:
            reason = "gate_probability_below_cut"
        else:
            prior = direct_at[p]
            ix = np.searchsorted(prior, f, side="right")-1
            last_direct = int(prior[ix]) if ix >= 0 else -10**12
            if f-last_direct < policy.gap_hops:
                reason = "past_direct_duplicate_guard"
            elif f-last_added[p] < policy.gap_hops:
                reason = "past_gate_duplicate_guard"
            else:
                reason = "accepted_hgb_addition"
                last_added[p] = f
        states.append(reason)
    if enabled:
        replay = replay_precomputed(direct, baseline, probabilities, policy)
        got = {(int(f), int(p)) for f,p,r in zip(frame,pitch,states)
               if r=="accepted_hgb_addition"}
        actual = {(int(f),int(p)) for f,p in zip(*np.nonzero(replay & ~direct))}
        if got != actual:
            raise AssertionError("audited rejection reasons do not reproduce decoder")
    return states


def classify_truth(before_indices, after_indices, n):
    return [("retained_tp" if i in before_indices and i in after_indices else
             "lost_tp" if i in before_indices else
             "recovered_tp" if i in after_indices else "persistent_fn")
            for i in range(n)]


def summarize_rows(rows):
    if not rows:
        return dict(tracks=0, before=metric(0,0,0), after=metric(0,0,0),
                    delta_f1_points=0, recovered_tp=0, lost_tp=0,
                    added_fp=0, added_events=0)
    before = sum_scores([r["before"] for r in rows])
    after = sum_scores([r["after"] for r in rows])
    return dict(tracks=len(rows), before=before, after=after,
                delta_f1_points=100*(after["f1"]-before["f1"]),
                recovered_tp=sum(r["recovered_tp"] for r in rows),
                lost_tp=sum(r["lost_tp"] for r in rows),
                added_fp=sum(r["added_fp"] for r in rows),
                added_events=sum(r["added_events"] for r in rows),
                hgb_tp_rejected=sum(r["hgb_tp_rejected"] for r in rows),
                direct_events_removed=sum(r["direct_events_removed"] for r in rows))


def audit(experiment, gate, out):
    if out.exists():
        raise FileExistsError("immutable post-mortem output cannot be overwritten")
    replayed = json.loads((gate/"replayed/report.json").read_text())
    trained = json.loads((gate/"trained/report.json").read_text())
    if not replayed["complete"] or replayed["tracks"] != 80 or replayed["frames"] != 306536:
        raise AssertionError("requires complete 80-WAV signed gate replay")
    if not all(replayed[k] for k in ("exact_gate_parity","exact_feature_parity","exact_hgb_parity")):
        raise AssertionError("causal live/cache parity not verified")
    if (replayed["hop_samples"] != 256 or replayed["future_audio_samples"] != 0
            or replayed["added_hold_samples"] != 0):
        raise AssertionError("cannot audit a noncausal/delayed variant as PCM256")
    if any(trained["candidate"][k] != replayed["candidate"][k] for k in ("tp","fp","fn")):
        raise AssertionError("trained vs fresh audio score mismatch")
    rows, reference = inputs(experiment)
    per_track = []; per_event = []; per_note = []
    fold_deltas = []
    for fold in trained["folds"]:
        sel = fold["selected"]
        cal_ctrl = json.loads((gate/"trained"/fold["outer"]/"calibration.json").read_text())["control"]
        fold_deltas.append(dict(
            outer=fold["outer"], cal=fold["cal"],
            cal_delta_f1_points=100*(sel["score"]["f1"]-cal_ctrl["f1"]),
            outer_delta_f1_points=None, selected_enabled=sel["enabled"],
            selected_policy=sel["policy"]))
    fold_dict = {r["outer"]:r for r in fold_deltas}
    for row in rows:
        comp, name = row["composition"],row["track"]
        fold_meta = json.loads((gate/"trained"/comp/"frozen_gate.json").read_text())
        ck = gate/"trained"/comp/"hgb_arbiter.joblib"
        if sha(ck) != fold_meta["checkpoint_sha256"]:
            raise AssertionError("gate checkpoint modified")
        q, baseline, direct, _ = track_inputs(experiment, row, reference, need_features=False)
        with np.load(gate/"candidates"/("track_%03d.npz" % row["index"]),
                     allow_pickle=False) as npz:
            model = joblib.load(ck)
            prob = np.asarray(model.predict_proba(npz["X"])[:,1], np.float32)
        selected = fold_meta["selected"]
        enabled = bool(selected["enabled"])
        policy = GatePolicy(**selected["policy"]) if enabled else None
        chosen = replay_precomputed(direct, baseline, prob, policy) if enabled else direct.copy()
        old_ev = sorted((e["sample"],e["pitch"]) for e in mask_events(direct))
        new_ev = sorted((e["sample"],e["pitch"]) for e in mask_events(chosen))
        old_map = matched(old_ev, row["truth"])
        new_map = matched(new_ev, row["truth"])
        old_truth, new_truth = set(old_map.values()),set(new_map.values())
        labels = classify_truth(old_truth,new_truth,len(row["truth"]))
        control_score, gate_score = track_score(old_ev,row["truth"]),track_score(new_ev,row["truth"])
        if not set(old_ev).issubset(set(new_ev)):
            raise AssertionError("an acoustic direct event has been deleted")
        added = set(new_ev)-set(old_ev)
        if len(added) != int(chosen.sum()-direct.sum()):
            raise AssertionError("event addition accounting mismatch")
        new_tp_events = {new_ev[i] for i in new_map}
        added_fp_events = added-new_tp_events
        recovered = sum(z=="recovered_tp" for z in labels)
        lost = sum(z=="lost_tp" for z in labels)
        if gate_score["tp"]-control_score["tp"] != recovered-lost:
            raise AssertionError("truth identity accounting mismatch")
        t_summary = dict(index=row["index"], track=name, composition=comp,
            texture=texture(name), before=control_score, after=gate_score,
            delta_f1_points=100*(gate_score["f1"]-control_score["f1"]),
            recovered_tp=recovered, lost_tp=lost,
            added_events=len(added), added_fp=len(added_fp_events),
            direct_events_removed=0, hgb_tp_rejected=0,
            truth_cohorts=dict(Counter(labels)))
        event_states = reasons(baseline, direct, prob, enabled, policy)
        hgb_locs = list(zip(*np.nonzero(baseline)))
        if len(hgb_locs) != len(prob) or len(event_states) != len(prob):
            raise AssertionError("HGB candidate mismatch")
        hgb_events = [((int(f)+1)*256,int(p)+40) for f,p in hgb_locs]
        hgb_matched = matched(hgb_events,row["truth"])
        for index, ((f,p),event,state,gate_prob) in enumerate(zip(
                hgb_locs,hgb_events,event_states,prob)):
            f,p = int(f),int(p)
            truth_idx = hgb_matched.get(index)
            could_rescue = truth_idx is not None and truth_idx not in old_truth
            accepted = event in added
            if could_rescue and not accepted:
                t_summary["hgb_tp_rejected"]+=1
            label = "accepted_tp" if accepted and event in new_tp_events else (
                "accepted_fp" if accepted else
                "rejected_potential_tp" if could_rescue else "not_new_tp")
            context = near_truth(row["truth"],event[1],event[0]/44100.)
            per_event.append(dict(track=name, composition=comp, texture=texture(name),
                sample=event[0],pitch=event[1],register=register(event[1]),
                direct_probability=float(q[f,p]), gate_probability=float(gate_prob),
                selected=accepted, decision_reason=state, outcome=label,
                hgb_reference_matched_truth_index=truth_idx,
                hgb_reference_could_recover=could_rescue, **context))
        for i,note in enumerate(row["truth"]):
            t=float(note["time"])
            active=true_active_pitches(row["truth"],t)
            per_note.append(dict(track=name,composition=comp,texture=texture(name),
                truth_index=i,pitch=int(note["pitch"]), register=register(int(note["pitch"])),
                time=t, polyphony_distinct=len(active),
                cohort=labels[i]))
        per_track.append(t_summary)
        print("ONSET_EFFECT_TRACK",comp,name,
              dict(delta_f1_points=t_summary["delta_f1_points"],
                   rec=t_summary["recovered_tp"],lost=t_summary["lost_tp"],
                   fp=t_summary["added_fp"]),flush=True)
    pooled = summarize_rows(per_track)
    for name, expected in (("control",trained["control"]),("candidate",trained["candidate"])):
        actual = pooled["before" if name=="control" else "after"]
        if any(actual[k]!=expected[k] for k in ("tp","fp","fn")):
            raise AssertionError("post-mortem does not reproduce complete F1")
    for fold in fold_deltas:
        group = summarize_rows([r for r in per_track if r["composition"]==fold["outer"]])
        fold["outer_delta_f1_points"]=group["delta_f1_points"]
        fold["delta_transfer_points"]=fold["outer_delta_f1_points"]-fold["cal_delta_f1_points"]
    groups = {}
    for dim in ("composition","texture"):
        groups[dim]={v:summarize_rows([r for r in per_track if r[dim]==v])
                     for v in sorted({r[dim] for r in per_track})}
    groups["register"]={v:dict(Counter(r["cohort"] for r in per_note
                          if r["register"]==v)) for v in sorted(set(r["register"] for r in per_note))}
    groups["active_polyphony"]={str(v):dict(Counter(r["cohort"] for r in per_note
                          if r["polyphony_distinct"]==v))
                          for v in sorted(set(r["polyphony_distinct"] for r in per_note))}
    reasons_count=dict(Counter(e["decision_reason"] for e in per_event))
    fp = [e for e in per_event if e["outcome"]=="accepted_fp"]
    rescues = [e for e in per_event if e["outcome"]=="rejected_potential_tp"]
    caveats = [
        "Harmonic proximity, register and score are associations, not established physical causes.",
        "Matching is same pitch, Hungarian one-to-one, +/-80 ms; an added event may reassign truth without deleting any earlier event.",
        "CAL and OUTER compositions differ: transfer deltas measure selection robustness, not identical sample metrics.",
        "FIT80 has historical model-selection overlap; this is not independent external validation.",
        "Post-mortem uses annotations for diagnosis only; NO threshold selection or inference labels. Outermost decisions are frozen.",
    ]
    diagnoses = [
        dict(failure="extra_false_positives",n=len(fp),
             evidence="matched post-gate events outside truth under identical one-to-one protocol",
             harmonic_proximity_associations=sum(e["harmonic_nearby_truth"] for e in fp)),
        dict(failure="missed_reference_rescues",n=len(rescues),
             evidence="reference-HGB proposals matching truth that acoustic misses but gate rejects",
             by_decoder_reason=dict(Counter(e["decision_reason"] for e in rescues))),
        dict(failure="destroyed_prior_TPs_through_matching",n=pooled["lost_tp"],
             evidence="truth formerly paired with direct but reassigned in union despite immutable direct emissions"),
    ]
    disposition=("REGRESSION_REQUIRES_CAUSAL_AUDIT" if pooled["delta_f1_points"]<0 else
                 "SMALL_GAIN_REQUIRES_CAUSAL_AUDIT" if pooled["delta_f1_points"]<.25 else
                 "GAIN_REQUIRES_COUNTEREXAMPLE_AUDIT")
    report=dict(schema=SCHEMA,complete=True,hop_samples=256,
                candidate=pooled,groups=groups,folds=fold_deltas,
                gate_decision_reasons=reasons_count,root_cause_hypotheses=diagnoses,
                status=disposition,association_only=True,
                no_outer_tuning=True, unchanged_axes=True,
                factual_audio_reference_verified=True, caveats=caveats)
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    for filename,fields,content in (
       ("per_track.csv",list(per_track[0]),per_track),
       ("hgb_candidates.csv",list(per_event[0]),per_event),
       ("ground_truth_changes.csv",list(per_note[0]),per_note)):
        with (out/filename).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=fields)
            writer.writeheader();writer.writerows(content)
    print("ONSET_EFFECT_POSTMORTEM",json.dumps({
       "delta_f1_points":pooled["delta_f1_points"],
       "tp_recovered":pooled["recovered_tp"],
       "tp_lost":pooled["lost_tp"],
       "added_fp":pooled["added_fp"],
       "status":disposition}),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--experiment",required=True,type=Path)
    parser.add_argument("--gate",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    a=parser.parse_args()
    audit(a.experiment,a.gate,a.out)


if __name__=="__main__":
    main()
