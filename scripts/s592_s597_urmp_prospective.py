"""S592--S597: prospective, frozen-head URMP ensemble transfer audit.

Only S597 may fetch Notes_ annotations. No fitting, threshold selection,
score-MIDI alignment, S80 promotion or Live change. The prepared audio is
16 kHz, not the original 48 kHz distribution. Source bytes are pinned.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import urllib.parse
import urllib.request
import zipfile

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from scripts.s585_attack_sustain_restrike import (
    need, sha, save, temporal52, predictions, metrics, POLICIES,
)
from scripts.s471_independent_MAESTRO_MIDI_postfrozen_real_V88_K_audit import match_times
from scripts.s591_score_state_note_predictions import match_ledger

REPOSITORY = 'J1mmymm/MIMuT_Data_v2'
REVISION = '3a62b3a1c8bb263a6a6ac37559070339ddb81080'
BASE = f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/'
PREFIX = 'data/urmp/urmp_yourmt3_16k/'
MANIFEST = Path('analysis/v273_s592_urmp_metadata_only.json')
SR = 44100
OFFSET = 5.
SECONDS = 24.
ALL_POLICIES = tuple(POLICIES) + ('BasicPitch_direct',)
STATE_SHA = '555cb092accb6ea0b42394280f646c90ccac0b48746040811ac57da382837d36'


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def write_rows(path, values):
    Path(path).write_text(''.join(json.dumps(v) + '\n' for v in values))


def family(instruments):
    ins = set(instruments)
    for label, members in (
        ('strings', 'vn va vc db'), ('woodwinds', 'fl ob cl sax bn'),
        ('brass', 'tpt tbn hn tba'),
    ):
        if ins <= set(members.split()):
            return label
    return 'mixed'


def entry(item):
    """Hashes from the public repository listing; no content inspection."""
    return dict(path=item['path'], size=item['size'], git_blob_sha1=item['oid'],
                lfs_sha256=item.get('lfs', {}).get('oid'))


def select(metadata):
    need(metadata['revision'] == REVISION, 'S592 pinned metadata revision')
    files = metadata['files']
    pool = []
    for item in files:
        path = item['path']
        if item['type'] != 'file' or '/AuMix_' not in path or not path.endswith('.wav'):
            continue
        folder = path.split('/')[-2]
        parts = folder.split('_')
        annotations = [entry(x) for x in files if x['type'] == 'file'
                       and x['path'].startswith(PREFIX + folder + '/Notes_')
                       and x['path'].endswith('.txt')]
        instruments = parts[2:]
        need(len(annotations) == len(instruments), 'S592 every instrument needs its Notes_ filename')
        pool.append(dict(folder=folder, piece=parts[1], instruments=instruments,
                         family=family(instruments), audio=entry(item),
                         annotations=sorted(annotations, key=lambda z: z['path']),
                         source_id=digest(path.encode()), offset=OFFSET, seconds=SECONDS))
    need(len(pool) == 44 and len({x['folder'][:2] for x in pool}) == 44,
         'S592 all 44 mixtures, without .reapeaks or generated score annotations')
    selected = []
    used = set()
    for category in ('strings', 'woodwinds', 'brass', 'mixed', 'strings'):
        chosen = next(x for x in sorted(pool, key=lambda z: z['source_id'])
                      if x['family'] == category and x['piece'] not in used)
        selected.append(chosen)
        used.add(chosen['piece'])
    return dict(status='S592_URMP_FIVE_ENSEMBLES_METADATA_ONLY', revision=REVISION,
                mirror_repository=REPOSITORY, source_base=BASE,
                prepared_audio='YourMT3 URMP 16kHz mono; NOT original 48kHz',
                selection='SHA256(audio path), category order strings/woodwinds/brass/mixed/strings; distinct piece abbreviation',
                all_44_audio_paths=[x['audio']['path'] for x in sorted(pool, key=lambda z: z['folder'])],
                selected=selected, audio_contents_read=False, annotations_read=False,
                no_training_or_threshold_selection=True, policies=list(ALL_POLICIES),
                source_is_new_to_this_head_experiment=True,
                no_claim_of_unseen_composers_or_globally_unseen_pretraining=True,
                full_S80_NOT_replayed=True, live_NOT_modified=True, no_promotion=True)


def manifest():
    m = read(MANIFEST)
    need(m['status'] == 'S592_URMP_FIVE_ENSEMBLES_METADATA_ONLY'
         and m['revision'] == REVISION and m['source_base'] == BASE
         and not m['audio_contents_read'] and not m['annotations_read']
         and m['policies'] == list(ALL_POLICIES) and len(m['selected']) == 5,
         'S592 public preregistration required')
    for s in m['selected']:
        need(s['source_id'] == digest(s['audio']['path'].encode())
             and s['offset'] == OFFSET and s['seconds'] == SECONDS,
             'S592 identity/excerpt cannot change')
    return m


def fetch(spec, annotation=False):
    path = spec['path']
    expected = '/Notes_' in path and path.endswith('.txt')
    need(path.startswith(PREFIX) and expected == annotation
         and (annotation or ('/AuMix_' in path and path.endswith('.wav'))),
         'Embargo: audio stages may fetch only AuMix_ WAV files')
    req = urllib.request.Request(BASE + urllib.parse.quote(path, safe='/'),
                                 headers={'User-Agent': 'NOTE-S592-prospective-URMP/1.0'})
    with urllib.request.urlopen(req, timeout=120) as response:
        data = response.read(int(spec['size']) + 1)
    need(len(data) == spec['size'], 'Pinned source size changed')
    if spec['lfs_sha256']:
        need(digest(data) == spec['lfs_sha256'], 'Pinned WAV SHA256 changed')
    else:
        gitsha = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        need(gitsha == spec['git_blob_sha1'], 'Pinned annotation git blob SHA changed')
    return data


def decode_prepared(blob):
    rate, signal = wavfile.read(io.BytesIO(blob))
    need(rate == 16000 and signal.ndim == 1 and signal.dtype == np.int16,
         'S593 prepared distribution must be mono int16 at 16kHz')
    signal = resample_poly(signal.astype(np.float32) / 32768., 441, 160)
    need(np.isfinite(signal).all() and len(signal) >= int((OFFSET + SECONDS) * SR),
         'S593 invalid or too-short preselected waveform; no replacement allowed')
    return np.clip(signal, -1., 1.).astype(np.float32)


def prepare(root):
    need(not (root / 'S592').exists(), 'S592 immutable run manifest')
    m = manifest()
    (root / 'S592').mkdir(parents=True)
    # Byte-for-byte preservation of the committed metadata record.
    (root / 'S592/report.json').write_bytes(MANIFEST.read_bytes())
    print('S592 COMMITTED METADATA', sha(MANIFEST), [s['folder'] for s in m['selected']], flush=True)


def v88(root, source_dir, weights):
    from scripts.s277_original_freeze_dataset3_native_prelabel import pcm
    from scripts.s84_external_freeze_native_count import (
        proposal_args, runtime_records, runtime_stack, _load_frozen_stack,
        _predict_score_tracks, _extended_candidate_features, _clusters_window,
        CLUSTER_WINDOW_MS, _cluster_arrays, capture_timing, MAX_CANDIDATES,
        _spectral_maps_for_runtime, COVERED, build_model, predict, SEED,
    )
    out = root / 'S593'
    need(not out.exists(), 'S593 audio inference immutable')
    m = manifest()
    need(sha(root / 'S592/report.json') == sha(MANIFEST), 'S593 metadata provenance')
    out.mkdir(parents=True)
    private = root / 's592_private_audio'
    private.mkdir()
    temp = private / 'audio.zip'
    proof = []
    tracks = []
    names = {}
    with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_STORED) as archive:
        for i, spec in enumerate(m['selected']):
            raw = fetch(spec['audio'])
            signal = decode_prepared(raw)
            clip = signal[int(OFFSET*SR):int((OFFSET+SECONDS)*SR)]
            sid = spec['source_id']
            base = f'urmp_{i:02d}_{sid[:12]}_comp'
            archive.writestr(base + '_mix.wav', pcm(clip))
            tracks.append(SimpleNamespace(annotation_member=base + '.jams',
                          audio_zip=temp, audio_member=base + '_mix.wav'))
            names[base + '.jams'] = sid
            np.save(private / (sid + '.npy'), clip, allow_pickle=False)
            proof.append(dict(source_id=sid, folder=spec['folder'], audio_sha256=digest(raw),
                              bytes=len(raw), native_rate=16000, inference_rate=SR,
                              source_duration=len(signal)/SR, offset=OFFSET, seconds=SECONDS,
                              clip_float32_sha256=sha(private / (sid + '.npy'))))
            print('S593 AUDIO', spec['folder'], digest(raw), flush=True)
    args = proposal_args(source_dir)
    floor, retention, enc86, enc87, model88 = _load_frozen_stack(args)
    scores = _predict_score_tracks(args.base_model, tracks)
    records = runtime_records(tracks, scores, floor)
    need(len(records) > 0, 'S593 no candidate events in entire cohort')
    x88, output88 = runtime_stack(records, enc86, enc87, model88)
    candidates, fused = _extended_candidate_features(x88, output88)
    clusters = _clusters_window(records, CLUSTER_WINDOW_MS)
    sequence, mask, stats, _, _, _ = _cluster_arrays(clusters, {}, records, candidates, output88, fused)
    timing = capture_timing(clusters, records, fused, MAX_CANDIDATES)
    spectra = _spectral_maps_for_runtime(tracks, clusters, records, window=COVERED)
    cache = dict(sequence=np.asarray(sequence, np.float32), mask=np.asarray(mask, np.float32),
                 stats=np.asarray(stats, np.float32), spectral=np.asarray(spectra, np.float32),
                 members=np.array([x['member'] for x in clusters]),
                 cluster_start_samples=np.asarray(timing['cluster_start_samples'], np.int64))
    count_model = build_model('learned_gate', SEED)
    count_model.load_weights(weights)
    prob, kval = predict(count_model, cache, np.arange(len(clusters)))
    start = cache['cluster_start_samples']
    member = np.asarray([names[str(x)] for x in cache['members']])
    ids = np.array([s + ':' + str(int(t)) for s, t in zip(member, start)])
    need(len(set(ids)) == len(ids) and np.isfinite(prob).all()
         and np.all((start >= 0) & (start < SECONDS*SR)) and np.isin(kval, range(7)).all(),
         'S593 invalid source events')
    np.savez_compressed(out / 'events.npz', row_id=ids, member=member, start_samples=start,
                        original_V88_K=np.asarray(kval, np.int8), original_V88_probabilities=prob)
    report = dict(status='S593_ORIGINAL_V88_ON_URMP_AUDIO_PRELABEL', annotations_read=False,
                  manifest_sha256=sha(MANIFEST), events_sha256=sha(out / 'events.npz'),
                  original_count_weights_sha256=sha(weights), audio_proof=proof,
                  event_count=len(ids), K_histogram=np.bincount(kval, minlength=7).tolist(),
                  events_per_source={s['source_id']:int(np.sum(member == s['source_id'])) for s in m['selected']},
                  unused_S466_auxiliary_NOT_executed=True, no_promotion=True)
    save(out / 'report.json', report)
    print('S593 EVENTS SEALED', report['event_count'], report['events_per_source'], flush=True)


def clips(root):
    r = read(root / 'S593/report.json')
    need(not r['annotations_read'] and r['manifest_sha256'] == sha(MANIFEST), 'S593 clip provenance')
    for spec in r['audio_proof']:
        p = root / 's592_private_audio' / (spec['source_id'] + '.npy')
        need(sha(p) == spec['clip_float32_sha256'], 'Frozen clip changed')
        signal = np.load(p, allow_pickle=False)
        need(signal.shape == (int(SECONDS*SR),), 'Frozen clip length changed')
        yield spec['source_id'], signal


def basicpitch(root):
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict, Model
    from scripts.s277_original_freeze_dataset3_native_prelabel import pcm
    out = root / 'S594'
    need(not out.exists(), 'S594 immutable Basic Pitch predictions')
    out.mkdir(parents=True)
    model = Model(ICASSP_2022_MODEL_PATH)
    ledger = []
    counts = {}
    for sid, signal in clips(root):
        p = root / 's592_private_audio' / 'basicpitch.wav'
        p.write_bytes(pcm(signal))
        _, _, notes = predict(str(p), model, onset_threshold=.5, frame_threshold=.3,
                              minimum_note_length=127.7, minimum_frequency=None, maximum_frequency=None)
        count = 0
        for st, en, pitch, amp, bend in notes:
            if 40 <= int(pitch) <= 88 and 0 <= float(st) < SECONDS and en > st:
                ledger.append(dict(WAV_filename_SHA256=sid, predicted_onset_s=float(st),
                                   predicted_offset_s=min(SECONDS, float(en)), pitch_MIDI=int(pitch),
                                   pretrained_note_score=float(amp)))
                count += 1
        counts[sid] = count
        print('S594 BASIC PITCH', sid, count, flush=True)
    write_rows(out / 'notes.jsonl', ledger)
    save(out / 'report.json', dict(status='S594_BASIC_PITCH_URMP_PRELABEL', annotations_read=False,
         S593_sha256=sha(root / 'S593/report.json'), notes_sha256=sha(out / 'notes.jsonl'),
         default_basicpitch_040_unchanged=True, notes_per_source=counts, no_promotion=True))


def features(root):
    from scripts.s589_sixth_pitch_features_WAV import acoustic_inputs
    out = root / 'S595'
    need(not out.exists(), 'S595 immutable acoustic features')
    rv = read(root / 'S593/report.json'); rb = read(root / 'S594/report.json')
    need(rv['events_sha256'] == sha(root / 'S593/events.npz')
         and rb['notes_sha256'] == sha(root / 'S594/notes.jsonl')
         and not rb['annotations_read'] and rb['S593_sha256'] == sha(root / 'S593/report.json'),
         'S595 input hashes')
    with np.load(root / 'S593/events.npz', allow_pickle=False) as z:
        ids, members, starts, kval = (z[k] for k in ('row_id','member','start_samples','original_V88_K'))
    notes = collections.defaultdict(list)
    for r in rows(root / 'S594/notes.jsonl'):
        notes[r['WAV_filename_SHA256']].append(r)
    proof = {x['source_id']: x for x in rv['audio_proof']}
    ledger = []
    for sid, signal in clips(root):
        ix = np.flatnonzero(members == sid)
        if not len(ix):
            continue  # Empty sources remain in ALL score denominators.
        group = [dict(v88_id=str(ids[i]), time_s=float(starts[i]/SR),
                      original_frozen_V88_K=int(kval[i])) for i in ix]
        xx, flags = acoustic_inputs(signal, group, notes[sid])
        need(np.asarray(xx).shape == (len(ix),49,11) and np.isfinite(xx).all(), 'S595 49x11 ABI')
        for g, x, flag in zip(group, xx, flags):
            ledger.append(dict(V88_original_event_id=g['v88_id'], source_recording_sha=sid,
                 original_WAV_sha256=proof[sid]['audio_sha256'],
                 original_V88_candidate_start_s=g['time_s'],
                 original_V88_K_soft_prior_ONLY=g['original_frozen_V88_K'],
                 old_basicpitch_birth_pitches=flag['bp_note_pitches'],
                 old_basicpitch_active_pitches=flag['bp_active_pitches_at_birth_plus30'],
                 per_midi_features_49x11=np.asarray(x, np.float32).round(6).tolist()))
    need(len(ledger) == len(ids), 'S595 every K0--K6 event retained')
    out.mkdir(parents=True)
    write_rows(out / 'features.jsonl', ledger)
    save(out / 'report.json', dict(status='S595_URMP_49X11_AUDIO_FEATURES_PRELABEL', annotations_read=False,
         S593_sha256=sha(root / 'S593/report.json'), S594_sha256=sha(root / 'S594/report.json'),
         feature_sha256=sha(out / 'features.jsonl'), event_count=len(ledger), no_promotion=True))
    print('S595 FEATURES', len(ledger), flush=True)


def freeze(root):
    import joblib
    out = root / 'S596'
    need(not out.exists(), 'S596 immutable preannotation predictions')
    m = manifest(); old = read(root / 'S578/report.json'); new = read(root / 'S585/report.json')
    spec = [x for x in old['weights_before_S577_exposure'] if x['architecture'] == 'general28']
    need(len(spec) == 1 and spec[0]['source_count'] == new['source_count'] == 28,
         'S596 same 28-source fitted checkpoints')
    op = root / 'S578' / spec[0]['filename']; npth = root / 'S585' / new['model_filename']
    need(sha(op) == spec[0]['sha'] and sha(npth) == new['model_sha256'] == STATE_SHA,
         'S596 pretrained checkpoint SHA mismatch')
    for s in (593,594,595):
        need(read(root / f'S{s}/report.json')['annotations_read'] is False, 'S596 annotations embargo')
    rf = read(root / 'S595/report.json')
    need(rf['feature_sha256'] == sha(root / 'S595/features.jsonl')
         and rf['S593_sha256'] == sha(root / 'S593/report.json')
         and rf['S594_sha256'] == sha(root / 'S594/report.json'), 'S596 feature lineage')
    raw = rows(root / 'S595/features.jsonl')
    rec = np.asarray([r['source_recording_sha'] for r in raw])
    X = temporal52(raw)
    control = joblib.load(op); state = joblib.load(npth)
    need(list(state.classes_) == [0,1,2,3], 'S596 four-state ABI')
    gp = control.predict_proba(X.reshape(-1,52))[:,list(control.classes_).index(1)].reshape(len(raw),49)
    sp = state.predict_proba(X.reshape(-1,52)).reshape(len(raw),49,4)
    pred, decisions = predictions(raw, rec, gp, sp)
    pred['BasicPitch_direct'] = {}
    for name in ALL_POLICIES:
        for s in m['selected']:
            pred[name].setdefault(s['source_id'], [])
    for n in rows(root / 'S594/notes.jsonl'):
        pred['BasicPitch_direct'][n['WAV_filename_SHA256']].append(
            dict(time=n['predicted_onset_s'], notes=[n['pitch_MIDI']]))
    out.mkdir(parents=True)
    save(out / 'predictions.json', pred)
    np.savez_compressed(out / 'probabilities.npz', global28=gp, state4=sp, source=rec)
    report = dict(status='S596_URMP_PREDICTIONS_SEALED_BEFORE_NOTES', annotations_read=False,
        policies=list(ALL_POLICIES), old_model_sha256=sha(op), new_model_sha256=sha(npth),
        metadata_sha256=sha(MANIFEST), feature_sha256=sha(root / 'S595/features.jsonl'),
        input_report_hashes={str(s):sha(root / f'S{s}/report.json') for s in (592,593,594,595)},
        prediction_sha256=sha(out / 'predictions.json'), probability_sha256=sha(out / 'probabilities.npz'),
        notes_per_source={p:{s:sum(len(x['notes']) for x in g) for s,g in v.items()} for p,v in pred.items()},
        decoder_counts=decisions, no_fitting_or_target_selection=True, no_promotion=True)
    save(out / 'report.json', report)
    print('S596 SEALED BEFORE NOTES', report['prediction_sha256'], report['notes_per_source'], flush=True)


def parse_notes(blob, instrument, track):
    a = np.loadtxt(io.BytesIO(blob), ndmin=2)
    need(a.ndim == 2 and a.shape[1] == 3 and np.isfinite(a).all()
         and np.all(a[:,0] >= 0) and np.all(a[:,1:] > 0), 'URMP Notes_ schema onset/Hz/duration')
    result = []
    for onset, hz, duration in a:
        midi = int(np.rint(69 + 12*np.log2(hz/440)))
        need(0 <= midi <= 127, 'URMP annotation frequency outside MIDI')
        result.append(dict(time=float(onset-OFFSET), pitch=midi, hz=float(hz),
                           duration=float(duration), instrument=instrument, track=track))
    return result


def truth_groups(notes):
    # Collapse only exact same-time/same-pitch unisons across tracks. No
    # time tolerance, sustain merging, MIDI-score alignment or note filtering.
    events = sorted({(n['time'],n['pitch']) for n in notes
                     if 0 <= n['time'] < SECONDS and 40 <= n['pitch'] <= 88})
    return [dict(time=t, notes=[p]) for t,p in events]


def audit_misses(guess, truth, raw, probs):
    matched, _ = match_ledger(guess, truth)
    result = []
    for pitch in range(40,89):
        gt = sorted(x['time'] for x in truth if pitch in x['notes'])
        for j,t in enumerate(gt):
            if (pitch,j,t) in matched:
                continue
            near = [i for i,r in enumerate(raw) if abs(r['original_V88_candidate_start_s']-t) <= .08]
            if not near:
                result.append(dict(pitch=pitch, truth_time=t, reason='no_V88_site_within80ms'))
                continue
            site = max(near, key=lambda i: float(probs[i,pitch-40,2:].sum()))
            attack = float(probs[site,pitch-40,2:].sum())
            eligible = [i for i in near if not raw[i]['old_basicpitch_birth_pitches']]
            if not eligible:
                reason = 'only_BP_protected_sites'
            elif all(probs[i,pitch-40,2:].sum() < .7 for i in eligible):
                reason = 'attack_probability_below_0.7_at_all_eligible_sites'
            else:
                reason = 'decoder_or_one_to_one_timing_competition'
            result.append(dict(pitch=pitch, truth_time=t, reason=reason,
                strongest_site=raw[site]['original_V88_candidate_start_s'],
                strongest_state_probabilities=probs[site,pitch-40].tolist(),
                strongest_BP_births=raw[site]['old_basicpitch_birth_pitches'],
                attack_probability=attack))
    return result


def score(root):
    out = root / 'S597'
    need(not out.exists(), 'S597 immutable independent score')
    m = manifest(); f = read(root / 'S596/report.json')
    need(f['status'] == 'S596_URMP_PREDICTIONS_SEALED_BEFORE_NOTES' and not f['annotations_read']
         and f['metadata_sha256'] == sha(MANIFEST) and f['policies'] == list(ALL_POLICIES)
         and f['prediction_sha256'] == sha(root / 'S596/predictions.json')
         and f['probability_sha256'] == sha(root / 'S596/probabilities.npz')
         and f['feature_sha256'] == sha(root / 'S595/features.jsonl')
         and all(h == sha(root / f'S{s}/report.json') for s,h in f['input_report_hashes'].items()),
         'S597 embargo: all separate-job prediction proofs must verify first')
    pred = read(root / 'S596/predictions.json')
    selected = {s['source_id']:s for s in m['selected']}
    need(set(pred) == set(ALL_POLICIES) and all(set(v) == set(selected) for v in pred.values()),
         'S597 score every source, even empty predictions')
    for name, sources in pred.items():
        for sid, groups in sources.items():
            need(sum(len(g['notes']) for g in groups) == f['notes_per_source'][name][sid]
                 and all(0 <= g['time'] < SECONDS and len(g['notes']) == len(set(g['notes']))
                         and all(40 <= p <= 88 for p in g['notes']) for g in groups),
                 'S597 prediction count/range drift')
    truth = {}; full = {}; provenance = []; rapid = []
    # FIRST target annotation access, in a separate job after artifact upload.
    for sid, s in selected.items():
        actual = []
        for a in s['annotations']:
            parts = Path(a['path']).name.split('_')
            blob = fetch(a, annotation=True)
            notes = parse_notes(blob, parts[2], int(parts[1]))
            actual.extend(notes)
            bypitch = collections.defaultdict(list)
            for n in notes:
                if 0 <= n['time'] < SECONDS and 40 <= n['pitch'] <= 88:
                    bypitch[n['pitch']].append(n['time'])
            for p, times in bypitch.items():
                for t,u in zip(sorted(times),sorted(times)[1:]):
                    if 0 < u-t <= .100:
                        rapid.append(dict(source_id=sid, track=int(parts[1]), instrument=parts[2], pitch=p, first=t, second=u))
            provenance.append(dict(path=a['path'], source_id=sid, sha256=digest(blob),
                                   git_blob_sha1=a['git_blob_sha1'], total_track_notes=len(notes)))
        truth[sid] = truth_groups(actual)
        full[sid] = actual
    meta = {sid:dict(composer=s['folder']) for sid,s in selected.items()}
    result = metrics(pred, truth, meta)
    changes = {}; ledger = []
    for name in POLICIES[1:]:
        tally = collections.Counter()
        for sid in selected:
            old, fp0 = match_ledger(pred[POLICIES[0]][sid], truth[sid])
            new, fp1 = match_ledger(pred[name][sid], truth[sid])
            need(len(old) == result[POLICIES[0]]['per_song'][sid]['scores']['80']['TP']
                 and len(new) == result[name]['per_song'][sid]['scores']['80']['TP'], 'S597 ledger reconciliation')
            for label, values in (('recovered',new-old), ('lost',old-new)):
                for p,j,t in sorted(values):
                    ledger.append(dict(policy=name, source_id=sid, change=label, pitch=p, truth_time=t, truth_index=j))
                    tally[label] += 1
            aa = {(e['pitch'],e['time']) for e in fp0}; bb = {(e['pitch'],e['time']) for e in fp1}
            for label, values in (('FP_removed',aa-bb), ('FP_added',bb-aa), ('FP_retained',aa&bb)):
                tally[label] += len(values)
                ledger.extend(dict(policy=name,source_id=sid,change=label,pitch=p,time=t) for p,t in sorted(values))
        a = result[POLICIES[0]]['fullstream']['80']; b = result[name]['fullstream']['80']
        need(tally['recovered']-tally['lost'] == b['TP']-a['TP']
             and tally['FP_added']-tally['FP_removed'] == b['FP']-a['FP'], 'S597 exact TP/FP deltas')
        changes[name] = dict(tally, delta_F1_points=100*(b['F1']-a['F1']))
    raw = rows(root / 'S595/features.jsonl')
    with np.load(root / 'S596/probabilities.npz', allow_pickle=False) as z:
        state = z['state4']; source = z['source'].astype(str)
    need(len(raw) == len(source) and all(r['source_recording_sha'] == s for r,s in zip(raw,source)),
         'S597 probability row alignment')
    misses = {}; candidate_coverage = {}; scope = {}
    for sid, s in selected.items():
        ix = np.flatnonzero(source == sid); subset = [raw[i] for i in ix]
        misses[sid] = audit_misses(pred[POLICIES[1]][sid], truth[sid], subset, state[ix])
        need(len(misses[sid]) == result[POLICIES[1]]['per_song'][sid]['scores']['80']['FN'], 'S597 FN audit complete')
        sites = np.array([r['original_V88_candidate_start_s'] for r in subset])
        maximum = 0
        for p in range(40,89):
            real = np.array([g['time'] for g in truth[sid] if p in g['notes']])
            maximum += len(match_times(sites,real,.08))
        candidate_coverage[sid] = dict(real_notes=len(truth[sid]), maximum_TP_on_fixed_V88_sites=maximum,
                                       reasons=dict(collections.Counter(x['reason'] for x in misses[sid])))
        inwindow = [n for n in full[sid] if 0 <= n['time'] < SECONDS]
        inrange = [n for n in inwindow if 40 <= n['pitch'] <= 88]
        scope[sid] = dict(folder=s['folder'], family=s['family'], instruments=s['instruments'],
            all_annotated_notes_in_excerpt=len(inwindow), supported_range_notes_before_unison_collapse=len(inrange),
            notes_outside_MIDI40_88=len(inwindow)-len(inrange), exact_same_pitch_time_unisons_collapsed=len(inrange)-len(truth[sid]),
            active_before_excerpt=sum(n['time'] < 0 < n['time']+n['duration'] for n in full[sid]))
    out.mkdir(parents=True)
    save(out / 'truth_after_freeze.json', truth)
    save(out / 'individual_note_audit.json', ledger)
    save(out / 'missed_note_diagnosis.json', misses)
    save(out / 'annotation_events_after_freeze.json', {s:[n for n in g if 0 <= n['time'] < SECONDS] for s,g in full.items()})
    report = dict(status='S597_URMP_FROZEN_HEAD_PROSPECTIVE_VERDICT', source_count=5, seconds_total=5*SECONDS,
        prepared_audio_16kHz=True, freeze_sha256=sha(root / 'S596/report.json'),
        Notes_annotations_first_read_after_separate_prediction_artifact=True, MIDI_scores_NEVER_used=True,
        true_notes=sum(len(g) for g in truth.values()), source_scope=scope, annotation_provenance=provenance,
        scores=result, comparisons=changes, candidate_coverage=candidate_coverage,
        same_instrument_same_pitch_restrikes_under100ms=rapid,
        audit_hashes={p.name:sha(p) for p in out.glob('*.json')},
        no_instrument_identification_or_offsets_scored=True, no_composer_unseen_claim=True,
        full_S80_ExactK_NOT_replayed=True, Live_NOT_tested_or_modified=True, no_promotion=True)
    save(out / 'report.json', report)
    for name in ALL_POLICIES:
        print('S597 PROSPECTIVE', name, json.dumps(result[name]['fullstream']['80']), flush=True)
    print('S597 AUDIT', json.dumps(changes), flush=True)


def selftest():
    notes = parse_notes(b'5.000 440.0 0.4\n5.090 440.0 0.2\n6.0 27.5 0.3\n', 'vn', 1)
    need([n['pitch'] for n in notes] == [69,69,21] and notes[0]['time'] == 0,
         'S597 Hz conversion or fixed 5s clock offset')
    gt = truth_groups(notes + [dict(notes[0], track=2)])
    need(len(gt) == 2 and gt[1]['time'] > .08, 'S597 exact unison collapse must preserve genuine repeats')
    from scripts.s534_independent_real_repeat_notes_dedup_tolerance_audit import note_metrics
    met = note_metrics([dict(time=0.,notes=[69])], gt, .08)
    need(met['TP'] == 1 and met['FN'] == 1, 'S597 one prediction must not match two notes')
    wrong = note_metrics([dict(time=0.,notes=[68])], gt, .08)
    need(wrong['TP'] == 0 and wrong['FP'] == 1, 'S597 wrong pitch must not match')
    miss = audit_misses([], gt, [], np.empty((0,49,4)))
    need(len(miss) == 2 and all(x['reason'] == 'no_V88_site_within80ms' for x in miss), 'S597 empty-source denominators')
    print('S592--S597 SELFTEST PASS: clocks, Hz, repeats, unisons, wrong pitches, empty predictions')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['register','prepare','v88','bp','features','freeze','score','selftest'])
    parser.add_argument('--root', type=Path, default=Path('model'))
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--source-dir', type=Path)
    parser.add_argument('--weights', type=Path)
    args = parser.parse_args()
    if args.stage == 'register':
        need(not MANIFEST.exists(), 'S592 metadata cannot be overwritten')
        save(MANIFEST, select(read(args.metadata)))
    elif args.stage == 'selftest':
        selftest()
    elif args.stage == 'v88':
        v88(args.root, args.source_dir, args.weights)
    else:
        {'prepare':prepare, 'bp':basicpitch, 'features':features,
         'freeze':freeze, 'score':score}[args.stage](args.root)
