"""Annotation-only diagnosis after the two-window extraction is frozen."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from causal_note.guitarset import index_guitarset
from causal_note.guitarset_acoustics import load_rich_annotations
from scripts import audit_v273_internal_b_low_harmonic_strata as h
from scripts.audit_v273_internal_residual_acoustics import GROUPS, match_frequencies
from scripts.v273_residual_audit import FOLDS, require, sha256_file, write_json

ARM_NAMES = ('first', 'second', 'joint')


def note_key(note):
    return (note['slot'], note['onset_sample'], note['offset_sample'])


def second_context(owned, nearby, start):
    left, right = start + h.WINDOW, start + 2 * h.WINDOW
    taper = np.hanning(h.WINDOW) ** 2
    own_keys = {note_key(n) for n in owned}
    coverage = []
    for n in owned:
        a, b = max(0, n['onset_sample'] - left), min(h.WINDOW, n['offset_sample'] - left)
        coverage.append(float(taper[a:b].sum() / taper.sum()) if b > a else 0.)
    foreign = [n for n in nearby if note_key(n) not in own_keys and n['onset_sample'] < right and n['offset_sample'] > left]
    return {'owned_min_taper_coverage': min(coverage),
            'owned_zero_coverage': sum(v == 0 for v in coverage),
            'owned_weak_coverage': sum(v < .1 for v in coverage),
            'foreign_active': len(foreign),
            'foreign_births_in_second': sum(n['onset_sample'] >= left for n in foreign)}


def aggregate(rows):
    out = {'rows': len(rows), 'arms': {}}
    for name in ARM_NAMES:
        out['arms'][name] = {'owned_match_counts': dict(Counter(str(r['arms'][name]['matches']) for r in rows)),
                            'complete': sum(r['arms'][name]['complete'] for r in rows)}
    out['joint_improved'] = sum(r['joint_minus_first'] > 0 for r in rows)
    out['joint_equal'] = sum(r['joint_minus_first'] == 0 for r in rows)
    out['joint_worsened'] = sum(r['joint_minus_first'] < 0 for r in rows)
    out['persistence_match_counts'] = dict(Counter(str(r['persistent_count']) for r in rows))
    out['components'] = {k: sum(r[k] for r in rows) for k in ('first_owned', 'first_unmatched', 'persisted_owned', 'persisted_unmatched')}
    out['context'] = {k: sum(r['second_context'][k] > 0 for r in rows) for k in
                      ('owned_zero_coverage', 'owned_weak_coverage', 'foreign_active', 'foreign_births_in_second')}
    return out


def diagnose(cases, temporal, contexts):
    result = []
    for row in cases:
        rid = row['row_id']; k = row['true_K']
        values = temporal[rid]['triplet_f0' if k == 3 else 'pair_f0']
        expected = np.array([n['frequency_hz'] for n in row['owned_notes']])
        r = {key: row[key] for key in ('row_id', 'fold', 'true_K', 'group')}
        r['arms'] = {}
        matched_first = set()
        for name, value in zip(ARM_NAMES, values):
            count, match = match_frequencies(expected, value)
            if name == 'first':
                matched_first = {m['component'] for m in match}
            r['arms'][name] = {'f0': value.tolist(), 'matches': count, 'complete': count == k}
        count, match = match_frequencies(values[0], values[1])
        persistent = {m['note'] for m in match}
        r.update(persistent_count=count, first_owned=len(matched_first),
                 first_unmatched=k - len(matched_first),
                 persisted_owned=len(persistent & matched_first),
                 persisted_unmatched=len(persistent - matched_first),
                 joint_minus_first=r['arms']['joint']['matches'] - r['arms']['first']['matches'],
                 second_context=contexts[rid])
        result.append(r)
    return result


def run(a):
    require(not a.output.exists(), 'refusing overwrite')
    cases = [json.loads(s) for s in a.cases.read_text().splitlines()]
    require(len(cases) == 488 and all(r['fold'] in FOLDS for r in cases), 'cohort changed')
    config = json.loads(a.config.read_text())
    for r in cases:
        require(config['member_folds'][r['recording_id']] == r['fold'], 'fold mismatch')
    wanted = {r['recording_id'] for r in cases}
    tracks = {t.annotation_member: t for t in index_guitarset(a.dataset) if t.annotation_member in wanted}
    contexts, events = {}, {}
    for member in sorted(wanted):
        require(config['member_folds'][member] in FOLDS, 'excluded annotations')
        notes = load_rich_annotations(tracks[member].annotation_zip, member).notes
        for r in (r for r in cases if r['recording_id'] == member):
            start = r['start_sample']
            nearby = [dict(slot=n.slot, onset_sample=n.onset_sample, offset_sample=n.offset_sample)
                      for n in notes if n.onset_sample < start + 2 * h.WINDOW and n.offset_sample > start + h.WINDOW]
            events[r['row_id']] = nearby
            contexts[r['row_id']] = second_context(r['owned_notes'], nearby, start)
    with np.load(a.temporal / 'temporal.npz', allow_pickle=False) as z:
        temporal = {int(r): {key: z[key][i] for key in ('pair_f0', 'triplet_f0')} for i, r in enumerate(z['row_id'])}
    rows = diagnose(cases, temporal, contexts)
    summary = {}
    for group in (*GROUPS, 'K3', 'K2', 'all'):
        subset = [r for r in rows if (True if group == 'all' else
                  r['true_K'] == int(group[1]) if len(group) == 2 else r['group'] == group)]
        summary[group] = aggregate(subset)
    report = {'status': 'completed', 'cases': len(rows), 'folds': list(FOLDS), 'outer_fold_3_used': False,
              'annotation_use': 'diagnostic only; no inference selection', 'summary': summary,
              'source_sha256': {'cases': sha256_file(a.cases), 'temporal': sha256_file(a.temporal / 'temporal.npz'),
                                'config': sha256_file(a.config), 'script': sha256_file(__file__)},
              'limitations': ['Chosen-frequency persistence does not establish physical source or active amplitude.',
                              'Additional context may contain foreign onsets or owned note offsets.',
                              'Normal audio only; no compressed-path conclusion.']}
    a.output.mkdir(parents=True)
    (a.output / 'cases.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))
    write_json(a.output / 'second-events.json', {str(k): v for k, v in events.items()})
    report['cases_sha256'] = sha256_file(a.output / 'cases.jsonl')
    write_json(a.output / 'report.json', report)
    print(json.dumps({k: summary[k] for k in ('K3', 'K2')}, sort_keys=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cases', 'temporal', 'dataset', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path('analysis/v273-native-paired-config.json'))
    run(p.parse_args())
