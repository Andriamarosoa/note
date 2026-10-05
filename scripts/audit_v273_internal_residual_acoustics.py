"""Annotation and waveform audit of frozen internal residual-guard decisions.

No classifier is trained, no threshold is selected, and no fold-3 recording is
read. Exact-K ownership is independently replayed from ALL original candidates.
Annotation-conditioned measurements are diagnostics, never inference features.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import zipfile

import numpy as np
from scipy.optimize import linear_sum_assignment

from causal_note.guitarset import SAMPLE_RATE, index_guitarset
from causal_note.guitarset_acoustics import load_rich_annotations
from scripts.candidate_timing import full_samples
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts import audit_v273_internal_b_low_harmonic_strata as h
from scripts.v273_residual_audit import FOLDS, sha256_file, require, verify_export, write_json

RADIUS = 882
MATCH_CENTS = 55.0
GROUPS = ("K3_regressed", "K3_preserved", "K2_corrected", "K2_missed")
DATA_MD5 = {"annotation.zip": "b39b78e63d3446f2e54ddb7a54df9b10",
            "audio_mono-pickup_mix.zip": "aecce79f425a44e2055e46f680e10f6a"}


def ownership(groups, onset):
    """Nearest original candidate within 20 ms, first group wins exact ties."""
    onset = np.asarray(onset, dtype=np.int64)
    require(len(groups) > 0 and all(len(g) for g in groups), "empty candidate group")
    distance = np.asarray([np.min(np.abs(np.asarray(g)[:, None] - onset[None, :]), axis=0)
                           for g in groups])
    assigned = np.where(distance.min(axis=0) <= RADIUS, distance.argmin(axis=0), -1)
    return assigned, np.bincount(assigned[assigned >= 0], minlength=len(groups))


def match_frequencies(expected, candidates):
    expected, candidates = np.asarray(expected), np.asarray(candidates)
    if not len(expected) or not len(candidates):
        return 0, []
    cents = np.abs(1200 * np.log2(expected[:, None] / candidates[None, :]))
    # Maximize valid one-to-one matches first; then minimize distance.
    cost = np.where(cents <= MATCH_CENTS, cents / MATCH_CENTS, 1000.0)
    a, b = linear_sum_assignment(cost)
    matches = [{"note": int(i), "component": int(j), "cents": float(cents[i, j])}
               for i, j in zip(a, b) if cents[i, j] <= MATCH_CENTS]
    return len(matches), matches


def timing_metrics(notes, assigned, local, start):
    own = [i for i, owner in enumerate(assigned) if owner == local]
    target = [notes[i] for i in own]
    require(len(target) in (2, 3), "not K2/K3")
    end = start + h.WINDOW
    before = [i for i, n in enumerate(notes) if n.onset_sample < start < n.offset_sample]
    foreign_post = [i for i, n in enumerate(notes)
                    if assigned[i] != local and n.onset_sample < end and n.offset_sample > start]
    foreign_births = [i for i in foreign_post if notes[i].onset_sample >= start]
    rel = np.array([n.onset_sample - start for n in target])
    midis = np.array([n.midi for n in target])
    hz = np.array([n.frequency_hz for n in target])
    cents = np.abs(100 * (midis[:, None] - midis[None, :]))
    min_spacing = float(cents[np.triu_indices(len(midis), 1)].min())
    taper = np.hanning(h.WINDOW) ** 2
    cover = []
    for n in target:
        left, right = max(0, n.onset_sample - start), min(h.WINDOW, n.offset_sample - start)
        cover.append(float(taper[left:right].sum() / taper.sum()) if right > left else 0.0)
    notes_out = [{"slot": n.slot, "midi": n.midi, "frequency_hz": n.frequency_hz,
                  "onset_sample": n.onset_sample, "offset_sample": n.offset_sample,
                  "onset_relative_ms": (n.onset_sample - start) * 1000 / SAMPLE_RATE,
                  "post_taper_energy_coverage": cov}
                 for n, cov in zip(target, cover)]
    overlapping = [i for i, n in enumerate(notes)
                   if n.onset_sample < end and n.offset_sample > start - h.WINDOW]
    events = [{"slot": notes[i].slot, "midi": notes[i].midi,
               "onset_sample": notes[i].onset_sample, "offset_sample": notes[i].offset_sample,
               "owner_track_row": int(assigned[i]), "own": bool(assigned[i] == local)}
              for i in overlapping]
    metrics = {
        "owned_before_start": int(np.sum(rel < 0)),
        "owned_onset_after_post": int(np.sum(rel >= h.WINDOW)),
        "owned_not_overlapping_post": int(sum(n.offset_sample <= start or n.onset_sample >= end for n in target)),
        "owned_weak_post_coverage": int(np.sum(np.asarray(cover) < .1)),
        "min_post_coverage": min(cover),
        "earliest_onset_ms": float(rel.min() * 1000 / SAMPLE_RATE),
        "latest_onset_ms": float(rel.max() * 1000 / SAMPLE_RATE),
        "onset_spread_ms": float(np.ptp(rel) * 1000 / SAMPLE_RATE),
        "min_owned_midi": float(midis.min()), "median_owned_midi": float(np.median(midis)),
        "near_unison_pair": int(min_spacing < h.NMS_CENTS), "min_pitch_spacing_cents": min_spacing,
        "pre_active_notes": len(before), "foreign_post_notes": len(foreign_post),
        "foreign_post_births": len(foreign_births),
        "min_post_cycles": min(max(0, min(n.offset_sample, end) - max(n.onset_sample, start))
                               / SAMPLE_RATE * n.frequency_hz for n in target),
    }
    return metrics, notes_out, events, hz, own


def ordered_pool(freq, x, kind):
    """Replay legacy equal-salience ordering; no labels or probabilities used."""
    scores = np.asarray([h.harmonic_salience(freq, x, float(f)) for f in h.F0_GRID])
    order = (np.lexsort((-np.arange(len(scores)), -scores)) if kind == 'reverse_ties'
             else np.argsort(-scores, kind=kind))
    chosen = []
    for i in order:
        f = float(h.F0_GRID[i])
        if any(h.cents(f, g) < h.NMS_CENTS for g in chosen):
            continue
        chosen.append(f)
        if len(chosen) == h.POOL_SIZE:
            break
    return chosen


def decompose(freq, x, pool):
    D = np.column_stack([h.template(freq, f) for f in pool])
    pair, _, _, x2 = h.fit_best(D, x, 2)
    triplet, _, _, _ = h.fit_best(D, x, 3)
    return pair, triplet, x2


def spectral_evidence(samples, start, target_hz, target_notes, notes, assigned, local,
                      expected_residuals, expected_median):
    taper = np.hanning(h.WINDOW)
    pre = h.pcm_window(samples, start - h.WINDOW, h.WINDOW)
    post = h.pcm_window(samples, start, h.WINDOW)
    all_freq = np.fft.rfftfreq(h.FFT_SIZE, 1 / SAMPLE_RATE)
    keep = (all_freq >= h.MIN_HZ) & (all_freq <= h.MAX_ANALYSIS_HZ)
    freq = all_freq[keep]
    P = (np.abs(np.fft.rfft(pre * taper, n=h.FFT_SIZE)) ** 2)[keep]
    Q = (np.abs(np.fft.rfft(post * taper, n=h.FFT_SIZE)) ** 2)[keep]
    positive = np.maximum(Q - P, 0)
    x = positive / (positive.sum() + 1e-12)
    pool = ordered_pool(freq, x, 'quicksort')
    pair, triplet, x2 = decompose(freq, x, pool)
    legacy_local = [pair[0] / (x2 + 1e-12), triplet[0] / (x2 + 1e-12)]
    local_drift = float(np.max(np.abs(np.asarray(legacy_local) - expected_residuals)))
    order_used = 'quicksort'
    def matches(pair, triplet, pool):
        values = np.array([pair[0], triplet[0]]) / (x2 + 1e-12)
        median = float(np.median(np.asarray(pool)[list(triplet[1])]))
        return np.max(np.abs(values - expected_residuals)) < 1e-8 and abs(median - expected_median) < 1e-8
    if not matches(pair, triplet, pool):
        for kind in ('stable', 'heapsort', 'reverse_ties'):
            alternative = ordered_pool(freq, x, kind)
            ap, at, _ = decompose(freq, x, alternative)
            if matches(ap, at, alternative):
                pool, pair, triplet, order_used = alternative, ap, at, kind
                break
        else:
            raise RuntimeError(f'unexplained residual replay drift: {local_drift}, start {start}')
    J2, c2, a2 = pair
    J3, c3, a3 = triplet
    pair_f0, triplet_f0 = np.asarray(pool)[list(c2)], np.asarray(pool)[list(c3)]
    pool_count, pool_matches = match_frequencies(target_hz, pool)
    trip_count, trip_matches = match_frequencies(target_hz, triplet_f0)
    # Counterfactual representation only: no subtraction, same audio window.
    post_pool = ordered_pool(freq, Q / (Q.sum() + 1e-12), 'stable')
    post_count, _ = match_frequencies(target_hz, post_pool)
    note_support = []
    for f in target_hz:
        mask = np.zeros(len(freq), dtype=bool)
        for harmonic in range(1, h.MAX_HARMONICS + 1):
            mask |= np.abs(freq - harmonic * f) <= h.KERNEL_HZ
        q, p, survived = float(Q[mask].sum()), float(P[mask].sum()), float(positive[mask].sum())
        note_support.append({"band_pre_power": p, "band_post_power": q,
                             "band_positive_retained_fraction": survived / (q + 1e-12)})
    ann_D = np.column_stack([h.template(freq, f) for f in target_hz])
    full, _, _, _ = h.fit_best(ann_D, x, len(target_hz))
    cosine = ann_D.T @ ann_D
    foreign_hz = [n.frequency_hz for i, n in enumerate(notes)
                  if assigned[i] != local and n.onset_sample < start + h.WINDOW and n.offset_sample > start]
    comp_info = []
    for f, amp in zip(triplet_f0, a3):
        direct = min(abs(1200 * np.log2(f / hz)) for hz in target_hz)
        harmonic = min(abs(1200 * np.log2(f / (hz * n))) for hz in target_hz for n in range(2, 7))
        foreign = min((abs(1200 * np.log2(f / hz)) for hz in foreign_hz), default=None)
        comp_info.append({"f0": float(f), "amplitude": float(amp), "nearest_owned_cents": float(direct),
                          "nearest_owned_harmonic_2_to_6_cents": float(harmonic),
                          "nearest_foreign_cents": None if foreign is None else float(foreign)})
    metrics = {
        "pool_owned_matches": pool_count, "triplet_owned_matches": trip_count,
        "pool_all_owned": int(pool_count == len(target_hz)),
        "triplet_all_owned": int(trip_count == len(target_hz)),
        "post_only_pool_owned_matches": post_count,
        "post_only_pool_match_delta": post_count - pool_count,
        "near_zero_triplet_components": int(np.sum(np.asarray(a3) <= 1e-10)),
        "transition_retained_post_power_fraction": float(positive.sum() / (Q.sum() + 1e-12)),
        "pre_to_post_power_ratio": float(P.sum() / (Q.sum() + 1e-12)),
        "minimum_owned_band_retained_fraction": min(s['band_positive_retained_fraction'] for s in note_support),
        "post_rms": float(np.sqrt(np.mean(post ** 2))),
        "max_owned_template_cosine": float(cosine[np.triu_indices(len(target_hz), 1)].max()),
        "annotation_conditioned_residual": float(full[0] / (x2 + 1e-12)),
        "legacy_sort_replay_changed": int(order_used != 'quicksort'),
        "legacy_local_residual_drift": local_drift,
    }
    details = {"pool_f0": pool, "pair_f0": pair_f0.tolist(), "triplet_f0": triplet_f0.tolist(),
               "pair_amplitudes": a2.tolist(), "triplet_amplitudes": a3.tolist(),
               "pool_matches": pool_matches, "triplet_matches": trip_matches,
               "owned_shared_band_support": note_support, "components": comp_info,
               "post_only_pool_f0": post_pool, "replay_sort_order": order_used,
               "local_quicksort_residuals": legacy_local}
    return metrics, details, (J2 / (x2 + 1e-12), J3 / (x2 + 1e-12))


def select_controls(rows):
    pairs = []
    for r in rows:
        if r['group'] != 'K3_regressed':
            continue
        controls = [c for c in rows if c['group'] == 'K3_preserved' and c['recording_id'] == r['recording_id']]
        if not controls:
            pairs.append({"error_row_id": r['row_id'], "control_row_id": None})
            continue
        pitches = sorted(n['midi'] for n in r['owned_notes'])
        def key(c):
            distance = float(np.mean(np.abs(np.asarray(pitches) - sorted(n['midi'] for n in c['owned_notes']))))
            return distance, abs(r['start_sample'] - c['start_sample']), c['row_id']
        c = min(controls, key=key)
        pairs.append({"error_row_id": r['row_id'], "control_row_id": c['row_id'],
                      "recording_id": r['recording_id'], "pitch_distance_semitones": key(c)[0],
                      "time_distance_seconds": key(c)[1] / SAMPLE_RATE})
    return pairs


def summarize_rows(rows):
    out = {}
    for group in GROUPS:
        rr = [r for r in rows if r['group'] == group]
        stats = {}
        for key in rows[0]['measurements']:
            x = np.asarray([r['measurements'][key] for r in rr])
            stats[key] = {"mean": float(x.mean()), "median": float(np.median(x)),
                          "rows_above_zero": int(np.sum(x > 0))} if len(x) else None
        out[group] = {"rows": len(rr), "recordings": len({r['recording_id'] for r in rr}),
                      "measurements": stats}
    return out


def load_cases(root, config):
    cases, contexts, seen = [], {}, set()
    manifests = []
    for fold in FOLDS:
        folder = root / f'fold-{fold}'
        verify_export(folder)
        manifests.append(json.loads((folder / 'manifest.json').read_text()))
        with np.load(folder / 'replay.npz', allow_pickle=False) as z:
            a = dict(z)
        for member in np.unique(a['val_recording']):
            require(config['member_folds'][str(member)] == fold and fold != 3, 'outer fold leak')
            ix = a['val_recording'] == member
            # All groups, not only action cases, are needed to verify ownership.
            contexts[str(member)] = {key: a['val_' + key][ix] for key in ('ids', 'start_sample', 'true_k')}
        for line in (folder / 'rows.jsonl').read_text().splitlines():
            r = json.loads(line)
            if r['split'] != 'val' or not r['valid'] or r['true_K'] not in (2, 3):
                continue
            require(r['fold'] == fold and r['row_id'] not in seen, 'duplicate/misfolded case')
            seen.add(r['row_id'])
            r['group'] = ('K3_regressed' if r['action_applied'] else 'K3_preserved') if r['true_K'] == 3 else ('K2_corrected' if r['action_applied'] else 'K2_missed')
            cases.append(r)
    require(len({m['commit'] for m in manifests}) == 1, 'mixed source runs')
    hashes = manifests[0]['bundle_identity']['source_cache_sha256']
    require(all(m['bundle_identity']['source_cache_sha256'] == hashes for m in manifests), 'mixed cache identities')
    return cases, contexts, hashes, manifests[0]['commit']


def run(a):
    require(not a.output.exists(), 'refusing overwrite')
    config = json.loads(a.config.read_text())
    cases, contexts, hashes, source_commit = load_cases(a.exports, config)
    require(sha256_file(a.config) == json.loads((a.exports / 'fold-0/manifest.json').read_text())['config_sha256'], 'config changed')
    for name, expected in DATA_MD5.items():
        with (a.dataset / name).open('rb') as f:
            got = hashlib.file_digest(f, 'md5').hexdigest()
        require(got == expected, 'dataset checksum changed: ' + name)
    wanted = {r['recording_id'] for r in cases}
    by_hash = {hashes[m]: m for m in wanted}
    indexed = {t.annotation_member: t for t in index_guitarset(a.dataset) if t.annotation_member in wanted}
    require(set(indexed) == wanted, 'audio coverage incomplete')
    a.output.mkdir(parents=True)
    completed, checked_tracks, max_error = [], set(), 0.0
    with (a.output / 'cases.jsonl').open('w') as stream:
        for archive in sorted(a.parts.glob('training-part-*.zip')):
            with zipfile.ZipFile(archive) as zipf:
                inventory = json.loads(zipf.read('file-sha256.json'))
                for path, digest in inventory.items():
                    if not path.endswith('v100-spectral-shard-00.npz') or digest not in by_hash:
                        continue
                    member = by_hash[digest]
                    require(member not in checked_tracks and config['member_folds'][member] in FOLDS, 'duplicate/outer track')
                    raw = zipf.read(path)
                    require(hashlib.sha256(raw).hexdigest() == hashes[member], 'cache checksum changed')
                    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
                        keys = ('cluster_start_samples', 'candidate_samples', 'full_candidate_samples',
                                'full_candidate_offsets', 'mask', 'truncated', 'exact', 'members')
                        cache = {k: np.asarray(z[k]) for k in keys}
                    require(set(cache['members'].astype(str)) == {member}, 'mixed shard')
                    ctx = contexts[member]
                    np.testing.assert_array_equal(cache['cluster_start_samples'], ctx['start_sample'])
                    np.testing.assert_array_equal(np.minimum(cache['exact'], 6), ctx['true_k'])
                    groups = full_samples(cache)
                    track = indexed[member]
                    annotations = load_rich_annotations(track.annotation_zip, member)
                    notes = annotations.notes
                    assigned, exact = ownership(groups, [n.onset_sample for n in notes])
                    np.testing.assert_array_equal(exact, cache['exact'])
                    audio = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
                    samples = np.asarray(audio.samples, np.float64) / 32768.0
                    local_by_id = {int(row): i for i, row in enumerate(ctx['ids'])}
                    for row in sorted((r for r in cases if r['recording_id'] == member), key=lambda r: r['row_id']):
                        local = local_by_id[row['row_id']]
                        start = int(cache['cluster_start_samples'][local])
                        require(start == row['start_sample'] and int(exact[local]) == row['true_K'], 'case target drift')
                        timing, owned, events, hz, own_ids = timing_metrics(notes, assigned, local, start)
                        expected = [row['best_pair_residual_ratio'], row['best_triplet_residual_ratio']]
                        spectral, detail, residuals = spectral_evidence(samples, start, hz, owned, notes, assigned, local,
                                                                      expected, row['median_triplet_f0'])
                        error = float(np.max(np.abs(np.asarray(residuals) - expected)))
                        max_error = max(max_error, error)
                        require(error < 1e-8, f'residual replay drift row {row["row_id"]}: {error}')
                        result = {**row, 'track_row': local, 'measurements': {**timing, **spectral},
                                  'owned_notes': owned, 'nearby_notes': events, 'decomposition': detail,
                                  'candidate_span_ms': (int(groups[local][-1]) - start) * 1000 / SAMPLE_RATE,
                                  'residual_replay_max_abs_error': error}
                        stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + '\n')
                        stream.flush()
                        completed.append(result)
                    checked_tracks.add(member)
                    print(json.dumps({'track': member, 'tracks': len(checked_tracks), 'cases': len(completed),
                                      'residual_replay_max_abs_error': max_error}), flush=True)
    require(checked_tracks == wanted and len(completed) == len(cases), 'incomplete audit')
    controls = select_controls(completed)
    write_json(a.output / 'matched-controls.json', controls)
    by_id = {r['row_id']: r for r in completed}
    deltas = defaultdict(list)
    for pair in controls:
        if pair['control_row_id'] is None:
            continue
        error, control = by_id[pair['error_row_id']], by_id[pair['control_row_id']]
        for key in error['measurements']:
            deltas[key].append(error['measurements'][key] - control['measurements'][key])
    report = {
        'status': 'completed', 'source_run': 37356100423, 'source_commit': source_commit,
        'folds': list(FOLDS), 'outer_fold_3_used': False, 'model_training': False,
        'prediction_changes': False, 'cases': len(completed), 'tracks': len(checked_tracks),
        'annotation_ownership_replayed': True, 'residual_replay_max_abs_error': max_error,
        'window_samples': h.WINDOW, 'window_ms': 1000 * h.WINDOW / SAMPLE_RATE,
        'register_note': 'original register is estimated; min_owned_midi is annotation-derived diagnostic only',
        'groups': summarize_rows(completed),
        'by_fold': {str(f): summarize_rows([r for r in completed if r['fold'] == f]) for f in FOLDS},
        'matched_controls': {'matched': sum(p['control_row_id'] is not None for p in controls),
                             'unmatched': sum(p['control_row_id'] is None for p in controls),
                             'unique_controls': len({p['control_row_id'] for p in controls if p['control_row_id'] is not None}),
                             'rule': 'same recording and true K3; minimum sorted-MIDI distance, then time, then row ID; reuse allowed',
                             'mean_error_minus_control': {k: float(np.mean(v)) for k, v in deltas.items()}},
        'runtime': {'python': platform.python_version(), **{p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'scikit-learn')}},
        'source_md5': DATA_MD5, 'config_sha256': sha256_file(a.config),
        'script_sha256': sha256_file(__file__), 'cases_sha256': sha256_file(a.output / 'cases.jsonl'),
        'limitations': [
            'Case/control selection is descriptive on already inspected internal folds, not an independent validation.',
            'Harmonic bands and inferred components can overlap several notes; they are not isolated sources.',
            'Annotation ownership counts attacks, not the number of sustained pitches in a frame.',
            'Frequency matching uses 55 cents and one-to-one assignment; pitch annotations and audio may differ.',
            'Post-only pool comparison is a representation diagnostic, not a new Exact-K score.',
            'Legacy equal-salience tie order can vary across runtimes; alternate order is used only when it reproduces the two exported residuals and median F0. Those scalars do not uniquely identify all eight original candidates.',
            'Flags overlap and must not be added as disjoint causal explanations.',
        ],
    }
    write_json(a.output / 'report.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('exports', 'dataset', 'parts', 'config', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())


if __name__ == '__main__':
    main()
