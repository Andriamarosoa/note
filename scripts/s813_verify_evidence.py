"""Re-score saved MIDI256 teacher decisions against original signed FIT labels.

This is independent evidence verification, NOT a second training execution.
The trusted S800 manifest must be fetched from run39352880035 by CI.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import joblib
import numpy as np
from scripts.s704_joint_metrics import metric, aggregate, SCORES
from scripts.s813_prepare_native256 import digest_file, save_json, targets_at_ends
from scripts.s813_train_framepitch256 import counts, total, COUNT_KEYS
from causal_note.s813_teacher_contract import TeacherEvidence, unification_gate


def verify(args):
    bundle = json.loads((args.evidence/'bundle.json').read_text())
    report = json.loads((args.evidence/'report.json').read_text())
    extracted = json.loads((args.evidence/'extraction.json').read_text())
    source = json.loads(args.manifest.read_text())
    if not report['complete'] or len(report['tracks']) != 80 or not extracted['complete']:
        raise ValueError('Complete extraction/experiment required')
    if digest_file(args.manifest) != extracted['source_manifest_sha256']:
        raise ValueError('Independent S800 source manifest changed')
    for name,sha in bundle['files'].items():
        if digest_file(args.evidence/name) != sha:
            raise ValueError('Archived evidence checksum changed: '+name)
    for name,sha in bundle['source_code_sha256'].items():
        if digest_file(name) != sha:
            raise ValueError('Scientific code differs from evidence: '+name)
    source_map = {r['index']:r for r in source['fit_files']}
    result_map = {r['index']:r for r in report['tracks']}
    if set(source_map) != set(range(80)) or set(result_map) != set(source_map):
        raise ValueError('80 unique source and result tracks required')
    compositions = ['BN1', 'Funk2', 'Jazz1', 'Rock3', 'SS1']
    if set(report['folds']) != set(compositions):
        raise ValueError('All five outer composition models are required')
    candidate_counts, baselines = [], []
    with np.load(args.evidence/'decisions.npz',allow_pickle=False) as decisions,\
         zipfile.ZipFile(args.evidence/'source_metadata.zip') as metadata:
        if set(decisions.files) != {f'fit_{i:03d}' for i in range(80)}:
            raise ValueError('Missing teacher decisions')
        for index,row in source_map.items():
            if any(result_map[index][key] != row[key]
                   for key in ('name', 'composition', 'frames')):
                raise ValueError('Result track identity differs from signed source')
            stem=f'fit_{index:03d}'
            raw=metadata.read(stem+'.json')
            if hashlib.sha256(raw).hexdigest() != row['json_sha256']:
                raise ValueError('Original source annotation SHA mismatch')
            truth=json.loads(raw)
            ends=np.arange(256,int(truth['samples'])+1,256,dtype=np.int64)
            pred=decisions[stem]
            if pred.dtype != bool or pred.shape != (len(ends),49) or len(ends)!=row['frames']:
                raise ValueError('Teacher MIDI256 dimensions changed')
            target,_=targets_at_ends(truth['fit_truth'],ends)
            c=counts(pred,target)
            if c != result_map[index]['teacher']:
                raise ValueError('Teacher TP/FP/FN differ from persisted predictions')
            candidate_counts.append(c)
            baseline=metric(truth['source_events'],truth['fit_truth'],ends)
            if any(baseline[k]!=truth['source_counts'][k] for k in COUNT_KEYS):
                raise ValueError('Original S708 source counter mismatch')
            baselines.append(baseline)
    measured=total(candidate_counts)
    baseline=aggregate(baselines)
    if measured != report['teacher_frame_pitch']:
        raise ValueError('Aggregated teacher score mismatch')
    if any(baseline[k] != report['source_counts'][k] for k in COUNT_KEYS):
        raise ValueError('Twelve source counters changed')
    for fold,row in report['folds'].items():
        if digest_file(args.evidence/row['checkpoint']) != row['checkpoint_sha256']:
            raise ValueError('Private model checksum changed')
        # These are our generated, SHA-verified project checkpoints only.
        teacher=joblib.load(args.evidence/row['checkpoint'])
        proof=teacher.metadata
        cal = compositions[(compositions.index(fold)+1) % len(compositions)]
        expected_train = sorted(set(compositions)-{fold, cal})
        if (sorted(proof['train_compositions']) != expected_train
                or proof['calibration_compositions'] != [cal]):
            raise ValueError('Exact disjoint 48/16/16 composition protocol changed')
        expected_cal_names = sorted(r['name'] for r in source_map.values()
                                    if r['composition'] == cal)
        if sorted(proof['calibration_names']) != expected_cal_names:
            raise ValueError('Calibration tracks differ from signed source')
        for head in proof['training'].values():
            expected_train_names = sorted(r['name'] for r in source_map.values()
                                          if r['composition'] in expected_train)
            if sorted(r['name'] for r in head['rows']) != expected_train_names:
                raise ValueError('Training tracks differ from signed source')
        if len({id(h.model) for h in teacher.heads.values()}) != 2:
            raise ValueError('Frame Pitch heads share a model')
        if proof['supervised_inputs'] or proof['additional_lookahead_samples']:
            raise ValueError('Supervised backbone or future input')
    ev=TeacherEvidence(**report['teacher_evidence'])
    gate=unification_gate([ev])
    if gate['allow_unification'] or gate['live_ready']:
        raise ValueError('Incomplete research teachers cannot unlock unification/LIVE')
    result=dict(schema='S813_SAVED_DECISIONS_RESCORING_V1', verified=True,
        verification_is_not_a_training_rerun=True, tracks=80,
        frames=sum(r['frames'] for r in report['tracks']),
        teacher_frame_pitch=measured, source_counts={k:baseline[k] for k in COUNT_KEYS},
        source_six_midi_scores={k:baseline[k] for k in SCORES},
        unification_gate=gate,
        audits=dict(data='PASS',acoustics='PASS_structural_local_extraction_not_rerun',
                    architecture='PASS_single_teacher_only',training='PASS_saved_provenance_not_rerun',
                    metrics='PASS',live='UNKNOWN_physical_latency'),
        nvidia_reviews='UNKNOWN_not_executed',promotion=False)
    save_json(args.out,result)
    print('S813_VERIFIED_SAVED_TEACHER_DECISIONS',measured,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for arg in ('evidence','manifest','out'):
        p.add_argument('--'+arg,type=Path,required=True)
    verify(p.parse_args())
