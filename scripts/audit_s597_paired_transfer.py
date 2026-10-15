"""Offline verification of S597 frozen predictions and individual regressions.

No network, fitting or new test. Also checks TP cardinality independently
with a sorted two-pointer matcher for each pitch and tolerance.
"""
import argparse
import collections
from pathlib import Path

import numpy as np

from scripts.s592_s597_urmp_prospective import read, rows, ALL_POLICIES
from scripts.s585_attack_sustain_restrike import need, sha, save, metrics, POLICIES
from scripts.s471_independent_MAESTRO_MIDI_postfrozen_real_V88_K_audit import match_times
from scripts.s591_score_state_note_predictions import match_ledger


def maximum_count(predicted, actual, tolerance):
    """Maximum cardinality on the line; independent of Hungarian costs."""
    a = sorted(predicted); b = sorted(actual)
    i = j = count = 0
    while i < len(a) and j < len(b):
        if abs(a[i]-b[j]) <= tolerance:
            count += 1; i += 1; j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return count


def run(root, out):
    need(not out.exists(), 'S597 post-label audit output immutable')
    score = read(root / 'S597/report.json')
    freeze = read(root / 'S596/report.json')
    pred = read(root / 'S596/predictions.json')
    truth = read(root / 'S597/truth_after_freeze.json')
    scope = score['source_scope']
    need(score['freeze_sha256'] == sha(root / 'S596/report.json')
         and freeze['prediction_sha256'] == sha(root / 'S596/predictions.json')
         and all(sha(root / 'S597' / n) == h for n,h in score['audit_hashes'].items()),
         'S597 published inputs changed')
    recomputed = metrics(pred, truth, {sid:dict(composer=s['folder']) for sid,s in scope.items()})
    need(recomputed == score['scores'], 'S597 complete score replay mismatch')
    independent_checks = 0
    for name in ALL_POLICIES:
        for sid in truth:
            for ms in (30,50,80):
                tp = 0
                for pitch in range(40,89):
                    actual = [x['time'] for x in truth[sid] if pitch in x['notes']]
                    guessed = [x['time'] for x in pred[name][sid] if pitch in x['notes']]
                    tp += maximum_count(guessed, actual, ms/1000)
                need(tp == score['scores'][name]['per_song'][sid]['scores'][str(ms)]['TP'],
                     f'S597 Hungarian assignment lost a possible match: {name} {sid} {ms}')
                independent_checks += 1
    families = {}
    for family in sorted({s['family'] for s in scope.values()}):
        families[family] = {}
        ids = [sid for sid,s in scope.items() if s['family'] == family]
        for name in ALL_POLICIES:
            counts = {k:sum(score['scores'][name]['per_song'][sid]['scores']['80'][k] for sid in ids)
                      for k in ('TP','FP','FN')}
            counts['F1'] = 2*counts['TP']/max(1,2*counts['TP']+counts['FP']+counts['FN'])
            families[family][name] = counts
    ledger = read(root / 'S597/individual_note_audit.json')
    raw = rows(root / 'S595/features.jsonl')
    need(freeze['feature_sha256'] == sha(root / 'S595/features.jsonl')
         and freeze['probability_sha256'] == sha(root / 'S596/probabilities.npz'), 'S597 feature/probability SHA')
    with np.load(root / 'S596/probabilities.npz', allow_pickle=False) as z:
        gp = z['global28']; sp = z['state4']; source = z['source'].astype(str)
    need(len(raw) == len(source) and all(s == r['source_recording_sha'] for s,r in zip(source,raw)),
         'S597 input row ordering')
    bp_difference = collections.Counter()
    bp_losses = []
    for sid in truth:
        direct, _ = match_ledger(pred['BasicPitch_direct'][sid], truth[sid])
        hybrid, _ = match_ledger(pred[POLICIES[1]][sid], truth[sid])
        bp_difference['direct_BP_true_notes_missing_from_S585'] += len(direct-hybrid)
        bp_difference['S585_true_notes_missing_from_direct_BP'] += len(hybrid-direct)
        for pitch,index,t in sorted(direct-hybrid):
            actual = sorted(x['time'] for x in truth[sid] if pitch in x['notes'])
            guessed = sorted(x['time'] for x in pred['BasicPitch_direct'][sid] if pitch in x['notes'])
            pair = next(i for i,j,d in match_times(np.asarray(guessed),np.asarray(actual),.08) if j == index)
            onset = guessed[pair]
            protected = [r['original_V88_candidate_start_s'] for r in raw
                         if r['source_recording_sha'] == sid and pitch in r['old_basicpitch_birth_pitches']
                         and r['original_V88_candidate_start_s']-.0002 <= onset <= r['original_V88_candidate_start_s']+.0202]
            reason = 'no_BP_protected_V88_mapping_for_this_onset' if not protected else 'mapped_but_timing_or_matching_loss'
            bp_difference[reason] += 1
            bp_losses.append(dict(source_id=sid,folder=scope[sid]['folder'],pitch=pitch,
                truth_time=t,direct_BP_onset=onset,protected_V88_times=protected,reason=reason))
    need(bp_difference['direct_BP_true_notes_missing_from_S585']-bp_difference['S585_true_notes_missing_from_direct_BP']
         == score['scores']['BasicPitch_direct']['fullstream']['80']['TP']-score['scores'][POLICIES[1]]['fullstream']['80']['TP'],
         'S597 BP direct / S585 true-note difference reconciliation')
    details = []
    for loss in ledger:
        if loss['policy'] != POLICIES[1] or loss['change'] != 'lost':
            continue
        sid = loss['source_id']; pitch = loss['pitch']
        actual = sorted(x['time'] for x in truth[sid] if pitch in x['notes'])
        guessed = sorted(x['time'] for x in pred[POLICIES[0]][sid] if pitch in x['notes'])
        pair = next((i for i,j,d in match_times(np.asarray(guessed),np.asarray(actual),.08)
                     if j == loss['truth_index']), None)
        need(pair is not None, 'S597 supposedly lost note was not matched by S578')
        index = [i for i,r in enumerate(raw) if r['source_recording_sha'] == sid
                 and abs(r['original_V88_candidate_start_s']-guessed[pair]) < 1e-9]
        need(len(index) == 1, 'S597 matched S578 time lacks unique V88 row')
        i = index[0]; p = pitch-40
        details.append(dict(loss, folder=scope[sid]['folder'], matched_control_time=guessed[pair],
            control_probability=float(gp[i,p]), state_probabilities=sp[i,p].tolist(),
            attack_probability=float(sp[i,p,2:].sum()), BP_births=raw[i]['old_basicpitch_birth_pitches']))
    misses = read(root / 'S597/missed_note_diagnosis.json')
    reasons = collections.Counter(x['reason'] for group in misses.values() for x in group)
    need(sum(reasons.values()) == score['scores'][POLICIES[1]]['fullstream']['80']['FN'], 'S597 all FN reconciled')
    rapid_audit = []
    for r in score['same_instrument_same_pitch_restrikes_under100ms']:
        observed = {}
        for name in ALL_POLICIES:
            matched, _ = match_ledger(pred[name][r['source_id']], truth[r['source_id']])
            observed[name] = [any(p == r['pitch'] and abs(t-time) < 1e-9 for p,j,t in matched)
                              for time in (r['first'],r['second'])]
        rapid_audit.append(dict(r, both_onsets_matched_by_policy=observed))
    learned_extra_true = []
    for sid in truth:
        strict, _ = match_ledger(pred[POLICIES[1]][sid], truth[sid])
        learned, _ = match_ledger(pred[POLICIES[2]][sid], truth[sid])
        learned_extra_true.extend(dict(source_id=sid,pitch=p,truth_time=t) for p,j,t in sorted(learned-strict))
    save(out, dict(retrospective_audit_not_new_test=True, no_threshold_or_weight_change=True,
        score_recomputed_exactly=True, independent_maximum_cardinality_checks=independent_checks,
        by_family=families, strict100_lost_note_probabilities=details, strict100_FN_reasons=dict(reasons),
        direct_BP_comparison=dict(bp_difference), direct_BP_true_note_losses=bp_losses,
        genuine_same_track_rapid_pair_audit=rapid_audit,
        learned_restrike_extra_true_notes_vs_strict=learned_extra_true,
        reachable_true_notes_at_fixed_V88_sites=sum(x['maximum_TP_on_fixed_V88_sites'] for x in score['candidate_coverage'].values()),
        true_notes=score['true_notes'],
        input_sha256={str(p.relative_to(root)):sha(p) for p in (
            root/'S596/predictions.json',root/'S596/probabilities.npz',root/'S597/report.json',
            root/'S595/features.jsonl',root/'S597/truth_after_freeze.json')}))
    print('S597 OFFLINE AUDIT PASS', independent_checks, 'independent matching checks;',
          len(details), 'lost true notes;', dict(reasons))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    run(args.root,args.out)
