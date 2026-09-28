"""Inspect raw audio and group ownership of frozen unweighted low-K errors.

This is a diagnostic on fold 3 only. No model, target, or decoder is changed.
Local eligibility is a protocol counterfactual, not a replacement label.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from causal_note.guitarset import index_guitarset, load_boundary_slots
from scripts import train_v100_spectral_string_slots as spectral
from scripts.audit_v273_low_k import describe
from scripts.candidate_timing import full_samples
from scripts.rebuild_v273_sources import digest, verify_dataset, write_json
from scripts.restore_v273_original_backup import validate_original_inventory
from scripts.spectral_window import COVERED
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import require

RADIUS = 882


def ownership(groups, onset):
    """Independently reproduce nearest-group assignment, including ties."""
    onset = np.asarray(onset, np.int64)
    distance = np.asarray([np.min(np.abs(g[:, None]-onset[None, :]), axis=0) for g in groups])
    winner = distance.argmin(axis=0)
    best = distance.min(axis=0)
    assigned = np.where(best <= RADIUS, winner, -1)
    exact = np.bincount(assigned[assigned >= 0], minlength=len(groups))
    eligible = distance <= RADIUS
    return assigned, exact, eligible, distance


def raw_metrics(segment):
    segment = np.asarray(segment, np.float64)
    def db(x):
        return float(20*np.log10(max(float(np.sqrt(np.mean(x*x))), 1e-10)))
    pre, post = segment[:COVERED.pre_samples], segment[COVERED.pre_samples:]
    late = segment[COVERED.pre_samples+1764:]
    energy = np.abs(np.fft.rfft(segment*np.hanning(len(segment)), n=8192))**2
    freq = np.fft.rfftfreq(8192, 1/44100)
    use = energy[(freq >= 60) & (freq <= 6000)]
    flatness = float(np.exp(np.mean(np.log(use+1e-20)))/(np.mean(use)+1e-20))
    return dict(rms_dbfs=db(segment), pre_rms_dbfs=db(pre), post_rms_dbfs=db(post),
                late_rms_dbfs=db(late), post_minus_pre_db=db(post)-db(pre),
                peak_absolute=float(np.max(np.abs(segment))),
                digitally_silent=bool(not np.any(segment)), spectral_flatness=flatness)


def category_masks(k, pred, eligible, maximum_active, foreign_visible, overlap):
    over = (k < 4) & (pred > k)
    # These sets are deliberately overlapping; never sum them as unique causes.
    return dict(local_eligible_equals_prediction=over & (eligible == pred),
                local_competition_present=over & (eligible > k),
                active_note_count_equals_prediction=over & (maximum_active == pred),
                foreign_birth_visible=over & (foreign_visible > 0),
                no_note_annotation_overlap=over & (overlap == 0))


def run(args):
    require(not args.output.exists(), 'refusing to overwrite results')
    verify_dataset(args.dataset)
    validate_original_inventory(args.outer)
    validate_original_inventory(args.previous)
    cfg = load_config(args.config)
    allowed = {m for m, f in cfg['member_folds'].items() if f == 3}
    with np.load(args.previous/'row-context.npz', allow_pickle=False) as z:
        old = {key: np.asarray(z[key]) for key in z.files}
    with np.load(args.previous/'frames-31-uniform-probes.npz', allow_pickle=False) as z:
        probability = np.asarray(z['baseline'])
    k, predicted = old['k'], probability.argmax(1)
    require(len(k) == 15279 and int(((k < 4) & (predicted > k)).sum()) == 874, 'wrong frozen population')
    require(set(old['member'].astype(str)) == allowed, 'not precisely fold 3')
    reference = json.loads((args.outer/'report.json').read_text())
    hashes = {r['member']: r['covered_sha256'] for r in reference['tracks']}
    indexed = {t.annotation_member: t for t in index_guitarset(args.dataset) if t.annotation_member in allowed}
    fields = {name: np.zeros(len(k), np.float64) for name in (
        'rms_dbfs', 'pre_rms_dbfs', 'post_rms_dbfs', 'late_rms_dbfs', 'post_minus_pre_db',
        'peak_absolute', 'digitally_silent', 'spectral_flatness', 'eligible_births',
        'contested_births', 'max_simultaneous_notes', 'native_positive_peak', 'native_flux_peak')}
    over = (k < 4) & (predicted > k)
    controls = set()
    # One deterministic correct control per error, matched on track and true K.
    for member in sorted(allowed):
        for value in range(4):
            errors = np.flatnonzero((old['member'] == member) & (k == value) & over)
            correct = np.flatnonzero((old['member'] == member) & (k == value) & (predicted == k))
            if len(correct):
                for row in errors:
                    controls.add(int(correct[np.argmin(np.abs(old['cluster_start_samples'][correct]-old['cluster_start_samples'][row]))]))
    verify_rows = set(np.flatnonzero(over).tolist()) | controls
    example_rows = []
    for value in range(4):
        rows = np.flatnonzero((k == value) & over)
        if len(rows):
            example_rows.extend(rows[np.argsort(-probability[rows].max(1), kind='stable')[:2]].tolist())
    examples, traces, clips = [], [], {}
    counts = dict(tracks=0, audio_spectral_replays=0, digitally_silent_overcounts=0,
                  annotation_assignments_replayed=0, annotations=0)
    args.output.mkdir(parents=True)
    seen = set()
    for path in sorted(args.outer.rglob('v100-spectral-shard-*.npz')):
        with np.load(path, allow_pickle=False) as z:
            cache = {key: np.asarray(z[key]) for key in z.files}
        member = str(cache['members'][0])
        require(set(cache['members'].astype(str)) == {member} and member in allowed and member not in seen, 'invalid track shard')
        require(digest(path) == hashes[member], 'outer cache checksum differs')
        seen.add(member)
        ids = np.flatnonzero(old['member'] == member)
        np.testing.assert_array_equal(cache['exact'], k[ids])
        np.testing.assert_array_equal(cache['cluster_start_samples'], old['cluster_start_samples'][ids])
        groups = full_samples(cache)
        track = indexed[member]
        events = np.asarray([(b.onset_sample, b.offset_sample, slot)
                             for slot, notes in enumerate(load_boundary_slots(track.annotation_zip, member)) for b in notes], np.int64)
        assigned, exact, eligible, distances = ownership(groups, events[:, 0])
        np.testing.assert_array_equal(exact, cache['exact'])
        counts['annotation_assignments_replayed'] += int(exact.sum())
        counts['annotations'] += len(events)
        audio = spectral.decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        samples = np.asarray(audio.samples, np.float32)/32768.0
        for local, row in enumerate(ids):
            start = int(old['cluster_start_samples'][row])
            left, right = start-COVERED.pre_samples, start+COVERED.post_samples
            segment = spectral._pcm_window(samples, left, COVERED.segment_samples)
            measurements = raw_metrics(segment)
            for key, value in measurements.items(): fields[key][row] = value
            fields['eligible_births'][row] = int(eligible[local].sum())
            fields['contested_births'][row] = int(np.sum(eligible[local] & (assigned != local)))
            in_window = (events[:, 0] < right) & (events[:, 1] > left)
            probes = np.r_[left, events[in_window, 0], events[in_window, 1]]
            probes = probes[(probes >= left) & (probes < right)]
            active = np.sum((events[:, 0, None] <= probes[None, :]) & (events[:, 1, None] > probes[None, :]), axis=0)
            fields['max_simultaneous_notes'][row] = int(active.max()) if len(active) else 0
            fields['native_positive_peak'][row] = float(cache['spectral'][local, :, :, 1].max())
            fields['native_flux_peak'][row] = float(cache['spectral'][local, :, :, 2].max())
            if int(row) in verify_rows:
                native = spectral._spectral_map_from_segment(segment, window=COVERED).astype(np.float16)
                np.testing.assert_array_equal(native, cache['spectral'][local])
                counts['audio_spectral_replays'] += 1
            if over[row]:
                visible = (events[:, 0] >= left) & (events[:, 0] < right)
                trace = dict(global_index=int(old['global_index'][row]), member=member, track_row=local,
                    start_sample=start, true_k=int(k[row]), predicted_k=int(predicted[row]),
                    probability=probability[row].tolist(), local_eligible_births=int(eligible[local].sum()),
                    contested_births=int(fields['contested_births'][row]),
                    audio=measurements, events=[dict(onset_sample=int(events[i, 0]), offset_sample=int(events[i, 1]),
                        string=int(events[i, 2]), owner_track_row=int(assigned[i]),
                        distance_to_current_candidate=int(distances[local, i]),
                        own=bool(assigned[i] == local)) for i in np.flatnonzero(in_window | visible)])
                traces.append(trace)
                counts['digitally_silent_overcounts'] += int(measurements['digitally_silent'])
                if int(row) in example_rows:
                    examples.append(trace)
                    clips[str(int(old['global_index'][row]))] = spectral._pcm_window(samples, start-11025, 22050)
        counts['tracks'] += 1
        write_json(args.output/'progress.json', counts)
        print(json.dumps(dict(member=member, rows=len(ids), **counts)), flush=True)
    require(seen == allowed and counts['tracks'] == 50 and counts['annotation_assignments_replayed'] == 8812,
            'incomplete source coverage')
    require(counts['audio_spectral_replays'] == len(verify_rows), 'raw audio verification incomplete')
    masks = category_masks(k, predicted, fields['eligible_births'], fields['max_simultaneous_notes'],
                           old['foreign_births_31'], old['overlapping_notes_31'])
    by_k = {}
    for value in range(4):
        same = k == value
        by_k[str(value)] = dict(rows=int(same.sum()), over=int((same & over).sum()),
            categories={key:int((same & mask).sum()) for key,mask in masks.items()},
            measurements={key:dict(over=describe(values[same & over]),
                                   correct=describe(values[same & (predicted == k)])) for key,values in fields.items()})
    label_quiet = over & (old['overlapping_notes_31'] == 0)
    report = dict(status='completed', scope='unweighted native model, 31 frames, fold 3, true K<4',
        source_training_run=36351028493, previous_audit_run=36358846720, total_rows=len(k),
        over_rows=int(over.sum()), raw_checks=counts, independent_label_replay=True,
        matched_correct_controls=len(controls), categories={key:int(value.sum()) for key,value in masks.items()},
        category_counts_overlap=True, by_true_k=by_k,
        annotation_quiet_errors={key:describe(values[label_quiet]) for key,values in fields.items()},
        examples=examples, no_model_training=True, no_output_correction=True, no_relabelled_score=True,
        raw_source_hashes={name:digest(args.dataset/name,'md5') for name in ('annotation.zip','audio_mono-pickup_mix.zip')},
        script_sha256=digest(__file__), config_sha256=digest(args.config),
        limitations=['Only one development fold and seed; no model promotion.',
          'Local eligibility without competing groups changes the assignment rule; it is not a corrected target or a new accuracy score.',
          'Active-note agreement and acoustic measurements are diagnostics, not proof that every extra prediction has that cause.',
          'Nonzero audio does not establish a new musical note; annotation incompleteness, noises and resonance are not distinguished conclusively.',
          'No claim that harmonics are the universal cause; no source-isolated acoustic intervention was performed.'])
    write_json(args.output/'report.json', report)
    write_json(args.output/'overcount-traces.json', traces)
    np.savez_compressed(args.output/'rows.npz', **fields, k=k, predicted=predicted,
                        member=old['member'], global_index=old['global_index'], start=old['cluster_start_samples'])
    np.savez_compressed(args.output/'illustrative-clips.npz', **clips)
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('outer','previous','dataset','config','output'):
        p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
