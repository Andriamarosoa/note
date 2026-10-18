"""Read-only causal diagnosis of S708/S731 missed GuitarSet note attacks.

Every baseline/scene event is computed without opening annotations. Oracle-K
is explicitly label-assisted for diagnosis ONLY, never a proposed result.
The frozen S708 and S707 artifacts, LIVE, and TEST remain untouched.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from causal_note.strict_live import SR
from causal_note.s705_temporal_heads import predict_track
from causal_note.s708_cardinality_selection import GuidedPolicy
from causal_note.s731_auditory_scene import scene_scores, reweight_onsets
from scripts.s613_prepare_strict_live import download, labels, digest
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s706_train_sharp_temporal_loops import record, load_model
from scripts.s705_train_temporal_loops import setup
from scripts.s704_train_joint_loops import targets
from scripts.s704_joint_metrics import metric, aggregate, SCORES
from scripts.s731_auditory_scene_calibration import emit

REFERENCE = "S708_frozen"
SCENE = "S731_cal_frozen"
ORACLE = "oracle_K_diagnostic_NOT_DEPLOYABLE"


def emitted_activity(events, ends):
    """One active-note vector per decision, based on actually available events."""
    output = np.zeros((len(ends), 49), dtype=bool)
    ordered = sorted(events, key=lambda e: (
        int(e["available_sample"]), 0 if e["type"] == "note_off" else 1))
    state = np.zeros(49, dtype=bool)
    cursor = 0
    for i, end in enumerate(ends):
        while cursor < len(ordered) and int(ordered[cursor]["available_sample"]) <= end:
            ev = ordered[cursor]
            pitch = int(ev["pitch"]) - 40
            if 0 <= pitch < 49:
                state[pitch] = ev["type"] == "note_on"
            cursor += 1
        output[i] = state
    return output


def missed_diagnostic(note, decision_ends, post, k, active, scene):
    """A descriptive gate audit, NOT a sufficient-condition simulator."""
    pitch = int(note["pitch"]) - 40
    seconds = np.asarray(decision_ends, dtype=float) / SR
    # Only received frames from attack time to +80 ms; no future information
    # enters any real inference decision. This window is for AFTER-THE-FACT audit.
    frames = np.flatnonzero((seconds >= note["time"]) &
                            (seconds <= note["time"] + .080))
    if not len(frames):
        return dict(reason="no_received_frame_within_80ms", frames=0)
    on = post[:, pitch, 0]
    sustain = post[:, pitch, 1]
    release = post[:, pitch, 2]
    old = post[np.maximum(np.arange(len(post)) - 2, 0), pitch, :]
    rising = ((on - old[:, 0] >= .04) |
              (sustain - old[:, 1] >= .10))
    # Frozen policy values from S708. Read from instantiated GuidedPolicy
    # separately in the caller, rather than treating these masks as proof
    # that a NoteOn must have been emitted (arming/timers still matter).
    strong = (on >= .75) & (sustain >= .50)
    weak = ((on >= .15) & (sustain >= .55) &
            (release < .50) & rising)
    head_k = np.argmax(k, axis=-1)
    kconf = np.max(k, axis=-1)
    k_can_add = (kconf >= .50) & (head_k > active.sum(axis=-1))
    has_acoustic = bool(np.any(strong[frames] | weak[frames]))
    has_weak = bool(np.any(weak[frames]))
    has_k_room = bool(np.any(k_can_add[frames]))
    if not has_acoustic:
        group = "insufficient_head_acoustic_gates"
    elif has_weak and not has_k_room and not np.any(strong[frames]):
        group = "possible_K_guidance_block"
    else:
        group = "remaining_arming_timing_state_or_selection"
    return dict(
        reason=group, frames=int(len(frames)),
        max_onset=float(on[frames].max()),
        max_sustain=float(sustain[frames].max()),
        min_release=float(release[frames].min()),
        max_independent_novelty=float(max(scene[i].novelty[pitch] for i in frames)),
        max_parent_ownership=float(max(scene[i].old_ownership[pitch] for i in frames)),
        strong_gate_pass=bool(np.any(strong[frames])),
        weak_gate_pass=has_weak,
        k_room_gate_pass=has_k_room,
        k_underestimates_active_count=bool(np.all(
            head_k[frames] < np.maximum(0, [
                0 for _ in frames
            ]))) if False else None,
        head_k_at_first=int(head_k[frames[0]]),
        p_k_add_at_first=float(scene[frames[0]].count_room),
        baseline_note_active_at_first=bool(active[frames[0], pitch])
    )


def compare_group(rows):
    if not rows:
        return dict(n_tracks=0)
    totals = {key: aggregate([row["scores"][key] for row in rows])
              for key in (REFERENCE, SCENE, ORACLE)}
    misses = [miss for row in rows for miss in row["missed"]]
    why = Counter(m["reason"] for m in misses)
    return dict(
        n_tracks=len(rows),
        metrics={k: {m: float(v[m]) for m in SCORES}
                 for k, v in totals.items()},
        reference_onset_totals={x: int(totals[REFERENCE][x]) for x in ("tp", "fp", "fn")},
        missed_examples=len(misses),
        miss_gate_groups=dict(why),
        oracle_rescued_reference_misses=sum(m["oracle_K_rescued"] for m in misses),
        scene_rescued_reference_misses=sum(m["S731_rescued"] for m in misses),
        weak_acoustic_gate_present=sum(m.get("weak_gate_pass", False) for m in misses),
        old_parent_ownership_present=sum(m.get("max_parent_ownership", 0) > 0 for m in misses),
    )


def execute(args):
    if args.out.exists():
        raise FileExistsError("Refuse overwrite of an immutable audit")
    args.out.mkdir(parents=True)
    baseline_report = json.loads((args.s708_dir / "report.json").read_text())
    signature = digest((args.s708_dir / "s708_selected.pt").read_bytes())
    if signature != baseline_report["checkpoint_sha256"]:
        raise AssertionError("S708 signed checkpoint mismatch")
    policy = GuidedPolicy(**baseline_report["frozen_candidate"]["choice"]["policy"])
    release = baseline_report["frozen_candidate"]["choice"]["release"]
    assert (release == dict(gate=.8, persistence=1, grace=24))
    model = load_model(args.s708_dir / "s708_selected.pt")[0]
    torch.set_num_threads(2)
    report = dict(schema="S708_S731_MISSED_POLYPHONIC_HEAD_K_SELECTOR_AUDIT",
        original_sha256=signature, promoted=False, LIVE_changed=False,
        TEST_opened=False, oracle_K_is_label_assisted_diagnostic_only=True,
        S731_selected_on_CAL=dict(gain=.15, penalty=0.),
        groups={}, tracks=[])
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "data").mkdir()
        proto, heads, selection, offset = setup(root, args.offset_checkpoint)
        for key, filename in (("GuitarSet_audio", "GuitarSet_audio.zip"),
                              ("GuitarSet_annotations", "GuitarSet_annotations.zip")):
            entry = proto["guitar_archives"][key]
            download(entry["url"], root / "data" / filename,
                     entry["size"], entry["sha256"])
        original = [s for s in proto["sources"]
                    if s["corpus"] == "GuitarSet" and s["split"] == "train"]
        with zipfile.ZipFile(root / "data/GuitarSet_audio.zip") as aud, \\
             zipfile.ZipFile(root / "data/GuitarSet_annotations.zip") as ann:
            whole = expanded_train_specs(original, aud, ann)
            grouped = {
                "CAL": [s for s in whole
                        if s["composition"] in baseline_report["cal_compositions"]],
                "HELD": [s for s in whole
                         if s["composition"] in baseline_report["held_compositions"]],
            }
            assert (len(grouped["CAL"]), len(grouped["HELD"])) == (32, 32)
            for phase, specs in grouped.items():
                rows = []
                for j, spec in enumerate(specs):
                    item = record(spec, aud, heads, selection, offset)
                    item["native_features"] = item["x"][:, :, :126].copy()
                    probs, count = predict_track(model, item["x"])
                    post = policy.blend * probs + (1-policy.blend) * item["native_post"]
                    events0 = emit(item, post, count, policy, offset, release)
                    old_scene = scene_scores(item["x"], post, count,
                                             events0, item["decision_ends"])
                    improved_post = np.stack([
                        reweight_onsets(p, evidence, gain=.15,
                                        explain=0., require_k=True)
                        for p, evidence in zip(post, old_scene)])
                    events1 = emit(item, improved_post, count, policy, offset, release)

                    # Labels are opened ONLY AFTER real candidate emissions.
                    truth = real_notes(labels(spec, root)["notes"], item["samples"]/SR)
                    y, true_k = targets(item["decision_ends"], truth)
                    oracle_count = np.eye(7, dtype=np.float32)[true_k]
                    oracle_events = emit(item, post, oracle_count, policy, offset, release)
                    emission = {REFERENCE: events0, SCENE: events1, ORACLE: oracle_events}
                    score = {name: metric(ev, truth, item["ends"])
                             for name, ev in emission.items()}
                    # Both real runs must match S708's frozen original totals.
                    matches = {key: set(s["matched_truth_ids"]) for key, s in score.items()}
                    original_activity = emitted_activity(events0, item["decision_ends"])
                    miss = []
                    for index, note in enumerate(truth):
                        if index in matches[REFERENCE]:
                            continue
                        diagnostic = missed_diagnostic(
                            note, item["decision_ends"], post, count,
                            original_activity, old_scene)
                        diagnostic.update(dict(
                            pitch=int(note["pitch"]), onset_seconds=float(note["time"]),
                            duration=float(note["duration"]), true_note_id=int(index),
                            oracle_K_rescued=bool(index in matches[ORACLE]),
                            S731_rescued=bool(index in matches[SCENE]),
                        ))
                        miss.append(diagnostic)
                    row = dict(name=spec["name"], texture="comp" if spec["name"].endswith("_comp") else "solo",
                               sha256=item["sha256"], scores=score, missed=miss)
                    rows.append(row)
                    print("FN GATE AUDIT", phase, j+1, len(specs), spec["name"],
                          "missed", len(miss), flush=True)
                # Aggregate from real counts, not averages of per-track rates.
                for texture in ("comp", "solo"):
                    report["groups"][phase+"_"+texture] = compare_group(
                        [x for x in rows if x["texture"] == texture])
                reference = aggregate([x["scores"][REFERENCE] for x in rows])
                for met in SCORES:
                    if abs(reference[met] - baseline_report[phase]["candidate"][met]) > 1e-8:
                        raise AssertionError("Frozen S708 replay mismatch "+phase+"/"+met)
                report["tracks"].extend(rows)
                (args.out/"report.json").write_text(
                    json.dumps(report, indent=2, sort_keys=True)+"\\n")
                print("FN GATE SUMMARY", phase, json.dumps(
                    {k:v for k,v in report["groups"].items() if k.startswith(phase)}),
                    flush=True)
    print("FN GATE AUDIT COMPLETED: diagnostic only, not a promoted model", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s708-dir", required=True, type=Path)
    parser.add_argument("--offset-checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    with threadpool_limits(limits=2):
        execute(parser.parse_args())
