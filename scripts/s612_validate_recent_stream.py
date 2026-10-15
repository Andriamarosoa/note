"""Reproduce frozen recent NOTE decisions, then audit rolling normalization.

No fitting, threshold search or new held-out accuracy claim. S589/S590 and
S595/S596 are existing MAESTRO/URMP recordings, with external frozen V88/BP.
Annotations are read only after both acoustic prediction variants are saved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from causal_note.recent_native import RecentHeads, RecentNoteStream, POLICIES
from scripts.live_recent_notes import read_rows, replay_acoustics, replay_events
from scripts.s569_train_real_23_source_52D_temporal_restrike_note_heads import temporal52
from scripts.s585_attack_sustain_restrike import predictions as original_predictions
from scripts.s515_freeze_real_WAV_49_pitch_frame_onset_harmonic_experts import features
from scripts.s592_s597_urmp_prospective import fetch, decode_prepared, SR, metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def verify_protected_sources():
    import subprocess
    repo = Path(__file__).resolve().parents[1]
    manifest = json.loads((repo/'analysis/evidence/v273-s612-recent-native-stream/protected_recent_sources_SHA256.json').read_text())
    for path, expected in manifest['sha256'].items():
        file = repo/path
        data = file.read_bytes() if file.exists() else subprocess.check_output(
            ['git','show','HEAD:'+path], cwd=repo)
        assert hashlib.sha256(data).hexdigest() == expected, (path,'Protected original changed')
    return manifest['file_count']


def ordered(rows):
    return sorted(rows, key=lambda r:(r['original_V88_candidate_start_s'], r['V88_original_event_id']))


def evaluate(heads, packets):
    stream = RecentNoteStream(heads)
    output, gp, sp, inputs, elapsed = [], [], [], [], []
    for packet in packets:
        started = time.perf_counter()
        result, x, g, s = stream.push(packet['row'], available_sample=packet['available_sample'])
        elapsed.append((time.perf_counter() - started) * 1000)
        output.append(result); gp.append(g); sp.append(s); inputs.append(x)
    return output, np.asarray(inputs), np.asarray(gp), np.asarray(sp), elapsed


def groups(output):
    return {name:[dict(time=o['time'], notes=o['notes'][name])
                  for o in output if o['notes'][name]] for name in POLICIES}


def frozen_ledger(root, heads, features_path, prediction_path, probability_path, report_path):
    report = json.loads((root/report_path).read_text())
    for field, path in (('feature_sha256', features_path),
                        ('notes_sha256' if 'notes_sha256' in report else 'prediction_sha256', prediction_path),
                        ('probabilities_sha256' if 'probabilities_sha256' in report else 'probability_sha256', probability_path)):
        assert sha(root/path) == report[field], (path, 'frozen SHA changed')
    rows = read_rows(root/features_path)
    expected = json.loads((root/prediction_path).read_text())
    expected_prob = np.load(root/probability_path, allow_pickle=False)
    # Different runners can sum tree posteriors in a different order, below 1e-12.
    prob_diff = 0.; count = 0; all_results={p:{} for p in POLICIES}; timings=[]
    for sid in sorted({r['source_recording_sha'] for r in rows}):
        subset=ordered([r for r in rows if r['source_recording_sha']==sid])
        output, x, gp, sp, ms=evaluate(heads, [dict(row=r, available_sample=int(np.ceil((r['original_V88_candidate_start_s']+.25)*SR))) for r in subset])
        np.testing.assert_array_equal(x, temporal52(subset))
        index={r['V88_original_event_id']:i for i,r in enumerate(rows)}
        ix=[index[r['V88_original_event_id']] for r in subset]
        for actual, key in ((gp,'global28'),(sp,'state4')):
            delta=float(np.max(np.abs(actual-expected_prob[key][ix])))
            assert delta < 2e-12, (key,delta)
            prob_diff=max(prob_diff,delta)
        selected=groups(output)
        original,_=original_predictions(subset, np.asarray([sid]*len(subset)), gp, sp)
        for p in POLICIES:
            assert selected[p] == expected[p][sid] == original[p][sid], (p,sid,'decisions changed')
            all_results[p][sid]=selected[p]
        count+=len(subset);timings.extend(ms)
    return dict(events=count, sources=len(all_results[POLICIES[0]]), temporal52_bitwise_equal=True,
                max_probability_difference=prob_diff, changed_note_groups=0,
                predicted_notes={p:sum(len(g['notes']) for song in sources.values() for g in song)
                                 for p,sources in all_results.items()},
                source_feature_sha256=sha(root/features_path)), rows, all_results, timings


def run(root, out, fetch_audio=False):
    if out.exists():
        raise ValueError('Choose a new evidence directory')
    protected_count=verify_protected_sources()
    heads=RecentHeads(root); checks={}; timings=[]
    cases={
        'MAESTRO_two':('sixth/S589/S589_ALL_V88_K0_K6_ORIGINAL_WAV_49_PITCH_FEATURES_PREMIDI.jsonl',
                       'S590/predictions.json','S590/probabilities.npz','S590/report.json'),
        'URMP_five':('urmp/S595/features.jsonl','S596/predictions.json','S596/probabilities.npz','S596/report.json'),
    }
    for name, paths in cases.items():
        check, rows, predicted, ms=frozen_ledger(root,heads,*paths)
        checks[name]=check;timings.extend(ms)
        print('S612 ORIGINAL LEDGER', name, json.dumps(check), flush=True)
    # Here rows/predicted are the URMP cohort, not target MIDI labels.
    manifest=json.loads((root/'urmp/S592/report.json').read_text())
    proof=json.loads((root/'urmp/S593/report.json').read_text())
    audio_sha={p['source_id']:p['clip_float32_sha256'] for p in proof['audio_proof']}
    notes=read_rows(root/'urmp/S594/notes.jsonl')
    bp_report=json.loads((root/'urmp/S594/report.json').read_text())
    # Exact original feature ledger/model and audio are already pinned above;
    # pin the BP ledger to the report rather than trusting a substituted file.
    bp_sha=sha(root/'urmp/S594/notes.jsonl')
    assert bp_sha in bp_report.values(), 'Frozen BP note ledger SHA changed'
    acoustics=[]; variants={m:{p:{} for p in POLICIES} for m in ('reference','rolling3s')}
    out.mkdir(parents=True)
    for spec in manifest['selected']:
        sid=spec['source_id']; audio_file=root/'audio'/f'{sid}.npy'
        if not audio_file.exists() and fetch_audio:
            audio_file.parent.mkdir(parents=True,exist_ok=True)
            sound=decode_prepared(fetch(spec['audio']))
            clip=sound[int(spec['offset']*SR):int((spec['offset']+spec['seconds'])*SR)]
            np.save(audio_file,clip,allow_pickle=False)
        assert sha(audio_file)==audio_sha[sid], 'Original resampled WAV excerpt changed'
        wave=np.load(audio_file,allow_pickle=False)
        subset=ordered([r for r in rows if r['source_recording_sha']==sid])
        bp=[n for n in notes if n['WAV_filename_SHA256']==sid]
        events=replay_events(subset,bp)
        oracle=np.asarray(features(wave,[dict(event_time_seconds=e.time,bp_note_pitches=e.births,
                     bp_active_pitches_at_birth_plus30=e.active) for e in events],bp)).round(6)
        record=dict(source_id=sid,folder=spec['folder'],events=len(subset),audio_sha256=sha(audio_file))
        for mode in variants:
            begin=time.perf_counter()
            packets=list(replay_acoustics(wave,subset,bp,normalization=mode))
            actual=np.asarray([p['row']['per_midi_features_49x11'] for p in packets])
            original=np.asarray([r['per_midi_features_49x11'] for r in subset])
            if mode=='reference':
                np.testing.assert_allclose(actual,oracle,atol=2e-6,rtol=0)
                np.testing.assert_allclose(actual,original,atol=3e-6,rtol=0)
                record.update(max_feature_delta_local=float(np.max(abs(actual-oracle))),
                              max_feature_delta_archived=float(np.max(abs(actual-original))))
            # Verify acoustic output is invariant to arbitrary host block boundaries.
            other=list(replay_acoustics(wave,subset,bp,normalization=mode,blocks=(997,)))
            assert [p['row'] for p in packets]==[p['row'] for p in other], (mode,'block boundaries changed features')
            output, _, gp, sp, ms=evaluate(heads,packets)
            group=groups(output)
            for p in POLICIES:
                variants[mode][p][sid]=group[p]
                if mode=='reference':
                    assert group[p]==predicted[p][sid], (p,sid,'PCM replay changed frozen notes')
            timings.extend(ms)
            delay=[1000*(p['available_sample']/SR-p['row']['original_V88_candidate_start_s']) for p in packets]
            steady=[1000*(p['available_sample']/SR-p['row']['original_V88_candidate_start_s'])
                    for p in packets if not p['end_of_source']
                    and p['row']['original_V88_candidate_start_s'] > .2]
            record[mode]=dict(elapsed_validation_seconds=time.perf_counter()-begin,
                context_availability_ms_min=min(delay), context_availability_ms_max=max(delay),
                steady_context_availability_ms_min=min(steady), steady_context_availability_ms_max=max(steady),
                end_of_source_events=sum(p['end_of_source'] for p in packets),
                note_count={p:sum(len(g['notes']) for g in group[p]) for p in POLICIES})
        acoustics.append(record)
        print('S612 REAL AUDIO',json.dumps(record),flush=True)
    # Seal both outputs before opening known annotations. No thresholds change.
    save(out/'reference_predictions.json',variants['reference'])
    save(out/'rolling3s_predictions.json',variants['rolling3s'])
    truth_path=root/'S597/truth_after_freeze.json'
    truth=json.loads(truth_path.read_text())
    meta={s['source_id']:dict(composer=s['folder']) for s in manifest['selected']}
    scores={mode:metrics(pred,truth,meta) for mode,pred in variants.items()}
    result=dict(status='S612_RECENT_NATIVE_STREAM_PARITY_AND_ROLLING_AUDIT',
        protected_original_source_files_verified=protected_count,
        models_SHA256=heads.hashes, frozen_feature_replays=checks, real_audio_URMP=acoustics,
        policy_scores_on_previously_exposed_URMP=scores,
        true_annotations_SHA256=sha(truth_path),
        frozen_prediction_SHA256={m:sha(out/f'{m}_predictions.json') for m in variants},
        per_event_head_and_selection_ms={k:float(np.percentile(timings,v)) for k,v in [('p50',50),('p95',95),('p99',99),('max',100)]},
        upstream_V88_and_BP_precomputed=True, upstream_latency_unknown=True,
        normalization_reference_reads_future=True,
        normalization_rolling3s_uses_only_frames_available_at_fixed_right_edge=True,
        all_source_audio_and_model_weights_unchanged=True, no_training_or_threshold_search=True,
        no_new_held_out_accuracy_claim=True, physical_microphone_latency_NOT_measured=True,
        full_S80_ExactK_NOT_replayed=True, learned_NoteOff_NOT_integrated=True, no_promotion=True)
    save(out/'report.json',result)
    print('S612 COMPLETE', json.dumps({mode:{p:scores[mode][p]['fullstream']['80'] for p in POLICIES} for mode in variants}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--fetch-urmp',action='store_true')
    a=p.parse_args();run(a.assets,a.out,a.fetch_urmp)
