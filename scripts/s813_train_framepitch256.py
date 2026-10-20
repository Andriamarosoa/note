"""First S813 teacher experiment: two private fresh-DSP256 Frame Pitch heads.

Five outer composition folds; one other composition is reserved for inner
calibration, three for fitting. No refit or tuning on the outer result.
Other teachers and unified MIDI are deliberately not synthesized here.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from causal_note.s813_framepitch_teacher import fit_teacher, FIXED_MODEL_PARAMS
from causal_note.s813_teacher_contract import TeacherEvidence, REQUIRED_METRICS, unification_gate
from scripts.s704_joint_metrics import metric, aggregate, SCORES
from scripts.s813_prepare_native256 import SCHEMA as DATA_SCHEMA, digest_file, save_json

SCHEMA = 'S813_FRAMEPITCH_TWO_PRIVATE_HEADS_FIT80_FRESH_MIDI256_V1'
COUNT_KEYS = ('tp','fp','fn','frames','k_correct','active_frames','active_k_correct',
              'frame_pitch_tp','frame_pitch_fp','frame_pitch_fn','offset_eligible','offset_correct')


def counts(pred, truth, mask=None):
    if mask is None:
        mask = np.ones_like(truth, dtype=bool)
    return dict(tp=int((pred & truth & mask).sum()), fp=int((pred & ~truth & mask).sum()),
                fn=int((~pred & truth & mask).sum()))


def total(rows):
    c = {k:sum(r[k] for r in rows) for k in ('tp','fp','fn')}
    c.update(f1=2*c['tp']/max(1,2*c['tp']+c['fp']+c['fn']),
             precision=c['tp']/max(1,c['tp']+c['fp']), recall=c['tp']/max(1,c['tp']+c['fn']))
    return c


def occupancy(events, ends):
    pred = np.zeros((len(ends),49), bool)
    active = {}
    for event in events:
        pitch = int(event['pitch'])-40
        index = int(np.searchsorted(ends,event['available_sample']))
        if event['type'] == 'note_on':
            if pitch in active:
                raise AssertionError('Invalid source double NoteOn')
            active[pitch] = index
        elif pitch in active:
            pred[active.pop(pitch):index,pitch] = True
    for pitch,index in active.items():
        pred[index:,pitch] = True
    return pred


def run(args):
    if args.out.exists():
        raise FileExistsError('Immutable S813 experiment output already exists')
    data = json.loads((args.cache/'report.json').read_text())
    if data['schema'] != DATA_SCHEMA or not data['complete'] or len(data['tracks']) != 80:
        raise ValueError('Completed authentic fresh256 FIT80 cache required')
    if (any(data[k] != 256 for k in ('feature_hop_samples','decision_hop_samples','metric_hop_samples'))
            or data['learned_frontend'] or data['H0_events_or_posteriors_used_as_features']):
        raise ValueError('Fresh unsupervised 256 DSP required')
    records = []
    baselines = []
    for row in data['tracks']:
        for entry in row['files'].values():
            if digest_file(args.cache/entry['file']) != entry['sha256']:
                raise ValueError('Feature/label cache SHA mismatch')
        source_path = args.traces/f"fit_{row['index']:03d}.json"
        if digest_file(source_path) != row['original_json_sha256']:
            raise ValueError('Source event/annotation SHA mismatch')
        source = json.loads(source_path.read_text())
        files = row['files']
        arrays = {k:np.load(args.cache/v['file'],mmap_mode='r',allow_pickle=False)
                  for k,v in files.items()}
        baseline = metric(source['source_events'],source['fit_truth'],arrays['ends'])
        if any(baseline[k] != source['source_counts'][k] for k in COUNT_KEYS):
            raise AssertionError('Source S708 count parity failed')
        baselines.append(baseline)
        records.append(dict(X=arrays['features'], y=arrays['pitch_target'],
                            ends=arrays['ends'], k=arrays['count_target'],
                            source_events=source['source_events'], **{k:row[k] for k in
                            ('name','composition','texture','index')}))
    compositions = sorted({r['composition'] for r in records})
    if compositions != ['BN1','Funk2','Jazz1','Rock3','SS1']:
        raise ValueError('Unexpected FIT composition protocol')
    args.out.mkdir(parents=True)
    (args.out/'models').mkdir()
    (args.out/'predictions').mkdir()
    baseline = aggregate(baselines)
    config = dict(schema=SCHEMA, source_cache_sha256=digest_file(args.cache/'report.json'),
                  model=FIXED_MODEL_PARAMS, sampling_cap=4000, seed=813,
                  outer_compositions=compositions, inner_calibration='next_sorted_composition_cyclic',
                  refit_after_calibration=False, targets_strictly_above=.85)
    signature = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    report = dict(schema=SCHEMA, complete=False, experiment_sha256=signature, config=config,
        feature_hop_samples=256, decision_hop_samples=256, metric_hop_samples=256,
        sample_rate_hz=44100, resolution_ms=256/44100*1000,
        target_axis='frame_pitch', score_scope='independent_teacher',
        source_cache_sha256=config['source_cache_sha256'],
        source_counts={k:baseline[k] for k in COUNT_KEYS},
        source_six_midi_scores={k:baseline[k] for k in SCORES},
        source_supervised_posteriors_used=False, CAL_opened=False, HELD_opened=False,
        TEST_opened=False, fit_historically_explored=True, generalization_verified=False,
        other_five_teachers_modified=False, unified_midi_evaluated=False,
        physical_latency_measured=False, production_LIVE_modified=False, promotion=False,
        folds={}, tracks=[])
    save_json(args.out/'report.json', report)
    for index, outer in enumerate(compositions):
        cal = compositions[(index+1)%len(compositions)]
        train_rows = [r for r in records if r['composition'] not in (outer,cal)]
        cal_rows = [r for r in records if r['composition'] == cal]
        test_rows = [r for r in records if r['composition'] == outer]
        if (len(train_rows),len(cal_rows),len(test_rows)) != (48,16,16):
            raise AssertionError('Disjoint 48/16/16 protocol changed')
        started = time.perf_counter()
        print('S813_TRAIN',outer,'inner_cal',cal,flush=True)
        teacher = fit_teacher(train_rows, cal_rows, seed=813, max_points=4000)
        training_seconds = time.perf_counter()-started
        checkpoint = args.out/'models'/(outer+'_frame_pitch.joblib')
        joblib.dump(teacher, checkpoint, compress=3)
        proof = teacher.metadata
        if outer in proof['train_compositions']+proof['calibration_compositions']:
            raise AssertionError('Outer composition leaked')
        cpu = []
        example = test_rows[0]['X']
        teacher.predict(example[:1])
        for frame in np.linspace(0,len(example)-1,64,dtype=int):
            start = time.perf_counter()
            teacher.predict(example[frame:frame+1])
            cpu.append((time.perf_counter()-start)*1000)
        report['folds'][outer] = dict(training_seconds=training_seconds,
            checkpoint=checkpoint.relative_to(args.out).as_posix(),
            checkpoint_sha256=digest_file(checkpoint), metadata=proof,
            inference_only_ms_p50_p95_p99=np.percentile(cpu,[50,95,99]).tolist(),
            inference_only_ms_max=max(cpu), cpu_samples=len(cpu),
            extraction_and_device_latency_not_included=True)
        for row in test_rows:
            probabilities = teacher.predict_proba(row['X'])
            predicted = probabilities >= teacher.threshold
            truth = np.asarray(row['y'],dtype=bool)
            source = occupancy(row['source_events'],row['ends'])
            trial = counts(predicted,truth)
            original = counts(source,truth)
            grouped = {}
            for name,selector in [('bass',np.arange(49)<8),
                                  ('mid',(np.arange(49)>=8)&(np.arange(49)<29)),
                                  ('high',np.arange(49)>=29)]:
                grouped[name] = dict(teacher=counts(predicted,truth,selector[None,:]),
                                     source=counts(source,truth,selector[None,:]))
            by_k = {str(k):dict(teacher=counts(predicted,truth,(row['k']==k)[:,None]),
                               source=counts(source,truth,(row['k']==k)[:,None])) for k in range(7)}
            target = args.out/'predictions'/f"fit_{row['index']:03d}.npz"
            np.savez_compressed(target, predicted=predicted, probabilities=probabilities,
                                decision_ends=row['ends'])
            report['tracks'].append(dict(index=row['index'],name=row['name'],
                composition=outer,texture=row['texture'],frames=len(truth),
                teacher=trial,source=original,by_register=grouped,by_k=by_k,
                predictions=target.relative_to(args.out).as_posix(),predictions_sha256=digest_file(target)))
        save_json(args.out/'report.json', report)
        print('S813_OUTER_RESULT',outer,total([r['teacher'] for r in report['tracks']
                                            if r['composition']==outer]),flush=True)
    measured = total([r['teacher'] for r in report['tracks']])
    original = total([r['source'] for r in report['tracks']])
    if abs(original['f1']-baseline['frame_pitch_f1'])>1e-12:
        raise AssertionError('Frame Pitch teacher/source scorer parity failed')
    report.update(complete=True, teacher_frame_pitch=measured, baseline_frame_pitch=original,
                  delta_frame_pitch_pp=100*(measured['f1']-original['f1']),
                  teacher_exceeds_85_fit=measured['f1']>.85)
    for field in ('composition','texture'):
        report['by_'+field] = {name:{kind:total([r[kind] for r in report['tracks'] if r[field]==name])
                                    for kind in ('teacher','source')}
                              for name in sorted({r[field] for r in report['tracks']})}
    for field in ('by_register','by_k'):
        report[field] = {name:{kind:total([r[field][name][kind] for r in report['tracks']])
                              for kind in ('teacher','source')}
                        for name in report['tracks'][0][field]}
    evidence = TeacherEvidence(axis='frame_pitch', score=measured['f1'],
        metric_name=REQUIRED_METRICS['frame_pitch'], source_id=config['source_cache_sha256'],
        evaluation_id=signature, evaluation_role='fit_research', generalization_verified=False,
        inference_uses_labels=False, private_parameters_verified=True,
        private_state_verified=True, causal_observations_verified=True)
    report['teacher_evidence'] = asdict(evidence)
    report['unification_gate'] = unification_gate([evidence])
    save_json(args.out/'report.json', report)
    print('S813_COMPLETE',measured,'source',original,'unification',False,flush=True)


def main():
    p = argparse.ArgumentParser()
    for arg in ('cache','traces','out'):
        p.add_argument('--'+arg,type=Path,required=True)
    run(p.parse_args())


if __name__ == '__main__':
    main()
