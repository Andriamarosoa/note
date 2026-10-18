"""Experimental causal event selection; native and rescue state are isolated.

This module is not installed in StrictLiveStream. Candidate generators see
only current/past posteriors. The event classifier may accept/reject a candidate
at its actual availability time; no future peak search or backdating occurs.
"""
from __future__ import annotations

import numpy as np

from causal_note.strict_live import CausalSelection, HOP, SR


def candidate_track(ends, post, k, rescue_post, acoustic, rescue_scores, selection):
    """Union two independent selectors and attach past-only event features.

    The original selector never receives boosted probabilities. Native event
    order is retained; proposals follow native events within each frame.
    """
    native = CausalSelection(selection)
    proposal = CausalSelection(selection)
    last_candidate = np.full(49, -10**12, np.int64)
    last_native = np.full(49, -10**12, np.int64)
    rows, vectors, native_events = [], [], []
    for frame, end in enumerate(ends):
        end = int(end)
        was_active = native.active.copy()
        native_now = native.step(post[frame], k[frame], end, end)
        proposed_now = proposal.step(rescue_post[frame], k[frame], end, end)
        native_events.extend(native_now)
        ons = [e for e in native_now if e['type'] == 'note_on']
        extras = [e for e in proposed_now if e['type'] == 'note_on']
        native_pitches = {e['pitch'] for e in ons}
        proposed_pitches = {e['pitch'] for e in extras}
        for event in ons + [e for e in extras if e['pitch'] not in native_pitches]:
            pitch = int(event['pitch'])
            p = pitch - 40
            is_native = pitch in native_pitches
            row = dict(event, frame=frame, native=is_native,
                       proposal=pitch in proposed_pitches,
                       native_active=bool(native.active[p]),
                       native_active_before=bool(was_active[p]))
            extra = [float(rescue_scores[frame, p]), float(is_native),
                     float(pitch in proposed_pitches), float(was_active[p]),
                     float(was_active.sum()) / 6.,
                     min(2., max(0., (end - last_candidate[p]) / SR)),
                     min(2., max(0., (end - last_native[p]) / SR)),
                     float(was_active[p - 12]) if p >= 12 else 0.,
                     float(was_active[p - 19]) if p >= 19 else 0.,
                     float(was_active[p + 12]) if p + 12 < 49 else 0.]
            vectors.append(np.r_[acoustic[frame, p], extra].astype(np.float32))
            rows.append(row)
            last_candidate[p] = end
        for event in ons:
            last_native[event['pitch'] - 40] = end
    width = acoustic.shape[-1] + 10
    x = np.asarray(vectors, dtype=np.float32).reshape(-1, width)
    return rows, x, native_events


def render_candidates(ends, post, rows, quality, selection, total_samples,
                      *, native_threshold=None, rescue_threshold=None,
                      protect_native=False):
    """Render accepted candidates while enforcing six-note causal state.

    None native threshold keeps every native event. None rescue threshold
    disables extras. In protected mode extras cannot split a native note.
    Native proposals preempt rescued notes if the output budget is full.
    """
    if len(rows) != len(quality):
        raise ValueError('Candidate/score length mismatch')
    if native_threshold is not None and protect_native:
        raise ValueError('Protected native events cannot be thresholded')
    active = np.zeros(49, bool)
    owner = np.zeros(49, np.int8)  # 1=native, 2=rescue
    low = np.zeros(49, np.int32)
    last_on = np.full(49, -10**12, np.int64)
    strength = np.zeros(49, np.float32)
    events = []
    cursor = 0

    def emit(kind, p, end, reason):
        events.append(dict(type=kind, pitch=int(p + 40),
                           decision_sample=int(end), available_sample=int(end),
                           time=end / SR, reason=reason,
                           velocity=96 if kind == 'note_on' else 0))

    for frame, end in enumerate(ends):
        end = int(end)
        sustain, release = post[frame, :, 1], post[frame, :, 2]
        low = np.where((sustain < selection.active_off) | (release > .75),
                       np.minimum(low + 1, selection.release_frames), 0)
        for p in np.flatnonzero(active & (low >= selection.release_frames)):
            emit('note_off', p, end, 'causal_release')
            active[p] = False
            owner[p] = 0
        while cursor < len(rows) and rows[cursor]['frame'] == frame:
            row, score = rows[cursor], float(quality[cursor])
            cursor += 1
            p = row['pitch'] - 40
            if row['native']:
                if native_threshold is not None and score < native_threshold:
                    continue
            else:
                if rescue_threshold is None or score < rescue_threshold:
                    continue
                if protect_native and (row['native_active'] or active[p]):
                    continue
                if end - last_on[p] < selection.refractory_frames * HOP:
                    continue
            if not active[p] and active.sum() >= selection.max_notes:
                # Every original native event remains possible after a rescue.
                victims = np.flatnonzero(active & (owner == 2))
                if not row['native'] or not len(victims):
                    continue
                victim = min(victims, key=lambda j: (strength[j], int(j)))
                emit('note_off', victim, end, 'rescue_preempted_by_native')
                active[victim] = False
                owner[victim] = 0
            if active[p]:
                emit('note_off', p, end, 'restrike')
            emit('note_on', p, end,
                 'causal_attack' if row['native'] else 's702_rescue')
            active[p] = True
            owner[p] = 1 if row['native'] else 2
            strength[p] = score
            low[p] = 0
            last_on[p] = end
    for p in np.flatnonzero(active):
        emit('note_off', p, int(total_samples), 'stream_end')
    return events


def onset_keys(events):
    return {(int(e['pitch']), int(e['decision_sample']))
            for e in events if e['type'] == 'note_on'}


def paired_onsets(baseline, candidate, truth, scorer):
    """Count recovered/lost reference IDs, not merely a change in net TP."""
    base_score, base_pairs = scorer(baseline, truth, tolerance=.080)
    new_score, new_pairs = scorer(candidate, truth, tolerance=.080)
    base_ids = {j for _, j in base_pairs}
    new_ids = {j for _, j in new_pairs}
    old, new = onset_keys(baseline), onset_keys(candidate)
    return dict(recovered_truth_ids=sorted(new_ids - base_ids),
                lost_truth_ids=sorted(base_ids - new_ids),
                retained_truth=len(new_ids & base_ids),
                recovered=len(new_ids - base_ids), lost=len(base_ids - new_ids),
                added_events=len(new - old), removed_events=len(old - new),
                delta_tp=new_score['TP'] - base_score['TP'],
                delta_fp=new_score['FP'] - base_score['FP'])
