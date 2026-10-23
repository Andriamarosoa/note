"""Paired, post-inference audit of Onset losses; never tunes a decoder.

Ground-truth texture/polyphony and +/-80ms diagnostic neighborhoods are
evaluation information only. They are never passed to an inference head.
Counterfactuals replay complete causal streams and are not promoted policies.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from onset256.acoustic_attack import AttackPolicy, DirectAttackSelector
from onset256.acoustic_calibration import decode_cache
from onset256.feature_cache import load_features
from scripts.onset_acoustic_direct256 import baseline_mask, mask_events
from scripts.onset_event_integrity256 import digest, dump, counts, sum_scores, verify_inputs
from scripts.replay_native256 import match_onsets

SR, HOP, TOLERANCE = 44100, 256, .080
REFERENCE = (8050, 4231, 3217)
ANNOUNCED_CANDIDATE = (7585, 3068, 3682)


def decoded_trace(q, baseline, policy):
    """Observe the exact selector, including state BEFORE each decision."""
    selector = DirectAttackSelector(policy)
    output = np.zeros(q.shape, bool)
    armed = np.zeros(q.shape, bool)
    gap = np.zeros(q.shape, bool)
    eligible = np.zeros(q.shape, bool)
    for frame, scores in enumerate(q):
        end = (frame + 1) * HOP
        wide = scores.astype(np.float64)
        armed[frame] = selector.armed | (wide < policy.cut * policy.reset_ratio)
        gap[frame] = end - selector.last >= policy.gap_hops * HOP
        eligible[frame] = ((armed[frame] & (wide >= policy.cut)) |
            (policy.use_baseline & baseline[frame] & (wide >= policy.baseline_cut)))
        chosen = selector.step(scores, np.flatnonzero(baseline[frame]) + 40,
                               end_sample=end)
        output[frame, np.asarray(chosen, int) - 40] = True
    if selector.finish():
        raise AssertionError("audit selector accumulated pending events")
    np.testing.assert_array_equal(output, decode_cache(q, baseline, policy))
    return dict(output=output, armed=armed, gap=gap, eligible=eligible)


def paired_matching(before, after, truth):
    def match(events):
        values = [SimpleNamespace(sample=int(s), pitch=int(p)) for s, p in events]
        pairs = match_onsets(values, truth)
        return {j: i for i, j in pairs}, counts(maskless(events), truth)
    old, old_score = match(before)
    new, new_score = match(after)
    lost = set(old) - set(new)
    gained = set(new) - set(old)
    # Net FN increase is LOST minus RECOVERED, not the size of the lost set.
    if len(lost) - len(gained) != new_score['fn'] - old_score['fn']:
        raise AssertionError("paired ground-truth accounting failed")
    return old, new, lost, gained, old_score, new_score


def maskless(events):
    return [dict(sample=int(s), pitch=int(p)) for s, p in events]


def note_context(truth, index):
    note = truth[index]
    t = float(note['time'])
    active = [n for n in truth if float(n['time']) <= t + 1e-10
              and float(n['time']) + float(n['duration']) > t + 1e-10]
    pitches = {int(n['pitch']) for n in active}
    block = int(np.floor(t * SR / HOP))
    coattacks = sum(int(np.floor(float(n['time']) * SR / HOP)) == block for n in truth)
    earlier = [n for j, n in enumerate(truth) if j != index and
               int(n['pitch']) == int(note['pitch']) and float(n['time']) < t]
    age = None if not earlier else t - max(float(n['time']) for n in earlier)
    pitch = int(note['pitch'])
    return dict(active_distinct_pitches_at_truth_onset=len(pitches),
        active_annotations_at_truth_onset=len(active),
        simultaneous_annotations_in_pcm256_block=coattacks,
        previous_same_pitch_onset_ms=None if age is None else 1000 * age,
        register='E2-B2' if pitch < 48 else 'C3-B3' if pitch < 60 else 'C4-E6',
        polyphony='K1' if len(pitches) <= 1 else f'K{len(pitches)}')


def loss_mechanism(note, old_event, q, baseline, trace, policy):
    """Report a proven decoder obstruction, separately from acoustic causes."""
    p = int(note['pitch']) - 40
    ends = np.arange(1, len(q) + 1) * HOP / SR
    ix = np.flatnonzero(np.abs(ends - float(note['time'])) <= TOLERANCE)
    frame = int(old_event[0]) // HOP - 1
    if not len(ix):
        category = 'no_complete_hop_in_matching_window'
    elif np.any(trace['output'][ix, p]):
        category = 'one_to_one_matching_competition'
    elif not policy.enabled:
        raise AssertionError('disabled reference cannot lose a baseline match without competition')
    elif np.any(trace['eligible'][ix, p] & ~trace['gap'][ix, p]):
        category = 'refractory_blocks_eligible_proposal'
    elif np.any(q[ix, p].astype(np.float64) >= policy.cut):
        category = 'score_crosses_cut_but_not_rearmed'
    else:
        category = 'score_below_direct_cut_and_no_reference_bypass'
    same_boundary_emitted = bool(trace['output'][frame, p])
    reason_at_old_boundary = ('still_emitted_matching_reassigned' if same_boundary_emitted else
        'reference_passthrough' if not policy.enabled else
        'refractory' if trace['eligible'][frame, p] and not trace['gap'][frame, p] else
        'reference_path_disabled' if not policy.use_baseline else
        'reference_veto_cut' if float(q[frame, p]) < policy.baseline_cut else
        'not_rearmed_or_below_direct_cut')
    return dict(decoder_mechanism=category,
        original_reference_event=dict(sample=int(old_event[0]), pitch=int(old_event[1])),
        original_event_still_emitted=same_boundary_emitted,
        rejection_at_original_boundary=reason_at_old_boundary,
        q_at_original_boundary=float(q[frame, p]),
        q_max_in_matching_window=float(np.max(q[ix, p])) if len(ix) else None,
        cut=policy.cut, baseline_cut=policy.baseline_cut,
        uses_reference=policy.use_baseline, gap_hops=policy.gap_hops,
        reset_ratio=policy.reset_ratio,
        acoustic_cause_proven=False)


def summarize_notes(notes):
    counts_ = Counter(r['cohort'] for r in notes)
    lost = [r for r in notes if r['cohort'] == 'lost']
    denominator = counts_['lost'] + counts_['retained']
    return dict(truth=len(notes), cohorts=dict(counts_),
        loss_rate_among_reference_tp=len(lost) / denominator if denominator else None,
        lost_decoder_mechanisms=dict(Counter(r['decoder_mechanism'] for r in lost)))


def audit(args):
    if args.out.exists():
        raise FileExistsError('immutable audit output already exists')
    manifest = verify_inputs(args.data)
    signed = {r['index']:r for r in manifest['fit_files']}
    fresh = json.loads((args.experiment / 'replayed/report.json').read_text())
    if not (fresh['complete'] and fresh['tracks'] == 80 and fresh['frames'] == 306536
            and fresh['hop_samples'] == 256 and fresh['future_audio_samples'] == 0
            and fresh['added_hold_samples'] == 0 and fresh['exact_event_parity']
            and fresh['exact_feature_parity'] and fresh['exact_reference_parity']):
        raise ValueError('paired fresh audio replay not certified')
    if tuple(fresh['totals'][k] for k in ('tp','fp','fn')) != ANNOUNCED_CANDIDATE:
        raise ValueError('reproduction differs from announced 69.206%; audit as a separate candidate')
    reference = json.loads(args.reference.read_text())
    candidate = json.loads((args.experiment / 'trained/sealed_predictions.json').read_text())
    collection = json.loads((args.experiment / 'collection_report.json').read_text())
    collection_rows = collection['per_track']
    if (len(collection_rows) != 80 or len({r['index'] for r in collection_rows}) != 80
            or {r['index'] for r in collection_rows} != set(signed)):
        raise ValueError('audit does not contain the exact 80 signed tracks')
    records, all_notes, sources = [], [], []
    counter_scores = defaultdict(list)
    counter_transitions = defaultdict(Counter)
    for row in sorted(collection_rows, key=lambda r: r['index']):
        name = row['track']
        original = signed[row['index']]
        if (name != original['name'] or row['composition'] != original['composition']
                or row['annotation_sha256'] != original['json_sha256']):
            raise ValueError('collection track identity differs from signed manifest')
        annotation = args.data / 'traces' / f"fit_{row['index']:03d}.json"
        if digest(annotation) != row['annotation_sha256']:
            raise ValueError('signed annotation changed')
        truth = json.loads(annotation.read_text())['fit_truth']
        if truth != row['truth']:
            raise ValueError('collection ground truth differs from signed annotation')
        trained = args.experiment / 'trained' / row['composition']
        metadata = trained / 'frozen_model_and_policy.json'
        meta = json.loads(metadata.read_text())
        if (meta['trained_hop_samples'] != 256 or meta['dim'] != 195
                or meta['outer_composition'] != row['composition']
                or row['composition'] in meta['train_compositions']
                or row['composition'] == meta['cal_composition']
                or not meta['frozen_before_outer'] or meta['future_audio_samples'] != 0
                or meta['added_hold_samples'] != 0):
            raise ValueError('candidate lineage or PCM256 causality contract changed')
        freeze = json.loads((trained / 'outer_prediction_freeze.json').read_text())
        if digest(metadata) != freeze['model_and_policy_sha256']:
            raise ValueError('policy changed after OUTER freeze')
        if meta['selected']['name'] != 'reference':
            ck = trained / f"acoustic_w{meta['selected']['width']}.joblib"
            if digest(ck) != meta['checkpoint_sha256']:
                raise ValueError('checkpoint changed')
        probability = trained / f"probability_{row['index']:03d}.npy"
        if digest(probability) != freeze['probabilities'][probability.name]:
            raise ValueError('OUTER probability cache changed')
        q = np.load(probability, allow_pickle=False)
        policy = AttackPolicy(**meta['selected']['policy'])
        base = baseline_mask(row, reference)
        trace = decoded_trace(q, base, policy)
        events = sorted((e['sample'],e['pitch']) for e in mask_events(trace['output']))
        if events != [tuple(e) for e in candidate[name]]:
            raise AssertionError('audit selector differs from fresh audio events')
        old, new, lost, gained, before, after = paired_matching(reference[name], events, truth)
        feature = load_features(args.experiment / 'features', row['index'])
        track_notes = []
        for j, note in enumerate(truth):
            cohort = ('retained' if j in old and j in new else 'lost' if j in lost else
                      'recovered' if j in gained else 'persistent_fn')
            context = note_context(truth, j)
            f = min(len(q) - 1, max(0, int(np.ceil(float(note['time']) * SR / HOP)) - 1))
            p = int(note['pitch']) - 40
            record = dict(track=name, composition=row['composition'],
                texture='solo' if name.endswith('_solo') else 'comp',
                truth_index=j, pitch=int(note['pitch']), time=float(note['time']),
                duration=float(note['duration']), cohort=cohort, **context,
                first_available_boundary_sample=(f + 1) * HOP,
                direct_score_at_first_boundary=float(q[f,p]),
                score_rank_at_first_boundary=int(1 + np.sum(q[f] > q[f,p])),
                fundamental_8192_log_at_first_boundary=float(feature[f,p,16]),
                relative_fundamental_8192_at_first_boundary=float(feature[f,p,40]),
                native_onset_score_at_first_boundary=float(feature[f,p,138]))
            if cohort == 'lost':
                record.update(loss_mechanism(note, reference[name][old[j]], q, base, trace, policy))
            track_notes.append(record)
        del feature
        # Fixed before this audit; no grid search or OUTER-based policy selection.
        alternatives = {
            'restore_reference_path': replace(policy, use_baseline=True, baseline_cut=0.),
            'remove_refractory_exclusion': replace(policy, gap_hops=1),
        }
        masks = {k: decode_cache(q, base, v) for k,v in alternatives.items()}
        masks['union_all_reference_events'] = trace['output'] | base
        for kind, mask in masks.items():
            alt_events = sorted((e['sample'],e['pitch']) for e in mask_events(mask))
            _, matched, _, _, _, score = paired_matching(events, alt_events, truth)
            ids = set(matched)
            counter_scores[kind].append(score)
            counter_transitions[kind].update(dict(lost_recovered=len(lost & ids),
                candidate_tp_lost=len(set(new) - ids),
                candidate_fn_recovered=len(ids - set(new))))
            for note in track_notes:
                if note['cohort'] == 'lost':
                    note.setdefault('counterfactual_recovery', {})[kind] = note['truth_index'] in ids
        all_notes.extend(track_notes)
        records.append(dict(track=name, composition=row['composition'], texture=track_notes[0]['texture'],
            before=before, after=after, lost=len(lost), recovered=len(gained),
            selected_policy=meta['selected'], **summarize_notes(track_notes)))
        sources.append(dict(track=name, annotation_sha256=digest(annotation),
            probability_sha256=digest(probability), policy_sha256=digest(metadata)))
    before = sum_scores([r['before'] for r in records])
    after = sum_scores([r['after'] for r in records])
    if tuple(before[k] for k in ('tp','fp','fn')) != REFERENCE:
        raise AssertionError('reference counts differ')
    if any(after[k] != fresh['totals'][k] for k in ('tp','fp','fn')):
        raise AssertionError('audited counts differ from fresh replay')
    groups = {}
    for field in ('texture','composition','polyphony','register'):
        groups[field] = {value: summarize_notes([r for r in all_notes if r[field] == value])
            for value in sorted({r[field] for r in all_notes})}
    counterfactuals = {}
    for name, scores in counter_scores.items():
        total = sum_scores(scores)
        counterfactuals[name] = dict(score=total, **dict(counter_transitions[name]),
            delta={k: total[k] - after[k] for k in ('tp','fp','fn')},
            diagnostic_only=True, promoted=False, tuned_on_outer=False)
    report = dict(complete=True, schema='ONSET_PAIRED_POLYPHONY_LOSS_AUDIT256_V1',
        before=before, after=after, **summarize_notes(all_notes),
        lost_truth=sum(r['lost'] for r in records),
        recovered_truth=sum(r['recovered'] for r in records),
        net_fn_increase=after['fn']-before['fn'], groups=groups,
        counterfactuals=counterfactuals, per_track=records,
        provenance=dict(reference_sha256=digest(args.reference),
            candidate_events_sha256=digest(args.experiment/'trained/sealed_predictions.json'),
            fresh_replay_report_sha256=digest(args.experiment/'replayed/report.json'),
            files=sources),
        inference_hop_samples=256, future_audio_samples=0, added_hold_samples=0,
        physical_acoustic_cause_proven=False, acoustic_descriptors_are_associations_only=True,
        oracle_labels_used_for_evaluation_only=True, no_training_or_threshold_changes=True,
        externally_unseen_validation=False, promoted=False)
    args.out.mkdir(parents=True)
    dump(args.out/'report.json', report)
    with (args.out/'notes.jsonl').open('w') as fd:
        for note in all_notes:
            fd.write(json.dumps(note,sort_keys=True)+'\n')
    print(json.dumps({k: report[k] for k in ('before','after','cohorts','lost_truth',
        'recovered_truth','net_fn_increase','lost_decoder_mechanisms','counterfactuals')},indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', required=True, type=Path)
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--reference', type=Path, default=Path(
        'series/ONSET_SCORE_RECOVERY_HGB256/evidence/sealed_predictions.json'))
    audit(p.parse_args())


if __name__ == '__main__':
    main()
