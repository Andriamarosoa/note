"""S740 causal action bridge: shadow-only, action-native note identity + S707 offsets.

No held-out truth, future audio, true K or oracle pitch may enter.
This module is deliberately NOT wired to LIVE without full six-score checks.
"""
from __future__ import annotations

import numpy as np
from causal_note.strict_live import SR
from causal_note.s707_causal_release import render_release
from causal_note.s740_multitarget_dynamic import ACTIONS, apply_offset_masks


def decode_pitch_actions(action_probabilities, decision_ends,
                         *, replacement_pitches=None, initial_active=(),
                         max_polyphony=6):
    """Emit chronological NoteOn/Off from five target-specific actions.

    A ChangePitch action is only valid with an independently provided MIDI
    destination at this same causal fragment.  No auto-deduced destination.
    If a note is not active, REMOVE/REATTACK/CHANGE_PITCH do nothing.
    Offs are processed before ons at a decision boundary to avoid
    accidentally rejecting a valid restrike through polyphony overflow.
    """
    scores = np.asarray(action_probabilities, dtype=np.float32)
    ends = np.asarray(decision_ends, dtype=np.int64)
    n = len(ends)
    if scores.shape != (n, 49, 5) or np.any(scores < 0) \
       or not np.isfinite(scores).all() \
       or not np.allclose(scores.sum(-1), 1, atol=1e-4):
        raise ValueError("Expected causal [fragment,49,5] action distributions")
    if ends.ndim != 1 or n == 0 or np.any(np.diff(ends) <= 0):
        raise ValueError("Expected ascending decision ends")
    if not 1 <= max_polyphony <= 6:
        raise ValueError("Physical polyphony must be 1..6")
    destination = None
    if replacement_pitches is not None:
        destination = np.asarray(replacement_pitches, dtype=np.int64)
        if destination.shape != (n, 49):
            raise ValueError("CHANGE_PITCH destinations need [fragment,49]")
    active = set(map(int, initial_active))
    if any(p < 40 or p > 88 for p in active) or len(active) > max_polyphony:
        raise ValueError("Invalid initial active note set")
    events = []
    skipped = {kind:0 for kind in ACTIONS}
    def push(kind, pitch, when, reason):
        events.append(dict(type=kind, pitch=pitch,
                           decision_sample=int(when),
                           available_sample=int(when),
                           time=float(when / SR),
                           velocity=100 if kind=="note_on" else 0,
                           reason=reason))
    for t in range(n):
        when = int(ends[t])
        requests = scores[t].argmax(axis=-1)
        outgoing = []
        incoming = []
        # Restrike and CHANGE_PITCH require an existing note.
        for col, action in enumerate(requests):
            pitch = col + 40
            if action == 0:
                continue
            if action == 1:
                incoming.append((pitch, "identity_add"))
            elif action == 2:
                if pitch in active:
                    outgoing.append((pitch, "identity_remove"))
                else:
                    skipped["REMOVE"] += 1
            elif action == 3:
                if pitch not in active or destination is None:
                    skipped["CHANGE_PITCH"] += 1
                    continue
                new_pitch = int(destination[t, col])
                if not 40 <= new_pitch <= 88 or new_pitch == pitch:
                    skipped["CHANGE_PITCH"] += 1
                    continue
                outgoing.append((pitch, "identity_change_pitch"))
                incoming.append((new_pitch, "identity_change_pitch"))
            elif action == 4:
                if pitch in active:
                    outgoing.append((pitch, "identity_reattack"))
                    incoming.append((pitch, "identity_reattack"))
                else:
                    skipped["REATTACK"] += 1
        for pitch, reason in sorted(outgoing):
            if pitch in active:
                push("note_off", pitch, when, reason)
                active.remove(pitch)
        # Assign deterministic priority, never exceed six simultaneously.
        for pitch, reason in sorted(incoming):
            if pitch in active or len(active) >= max_polyphony:
                skipped["ADD"] += 1
                continue
            push("note_on", pitch, when, reason)
            active.add(pitch)
    return dict(events=events, final_active=tuple(sorted(active)),
                skipped=skipped, no_future_audio=True)


def replay_offset_instance_masks(plan, per_instance_heads, masks,
                                 ends, samples, **release_parameters):
    """Run the actual frozen S707 causal emitter after instance edits.

    Forbids retroactive timestamps: every change becomes effective only at
    the original S707 sampled hazard frame; native fallback remains intact.
    """
    altered = apply_offset_masks(plan, per_instance_heads, masks)
    emitted, skipped = render_release(
        altered, ends, samples, **release_parameters)
    return dict(events=emitted, skipped_polyphony=skipped,
                plan=altered)
