"""Replay saved native count results without training or choosing new thresholds.

Input layout: fold-N/paired and fold-N/anchor, plus the four shared inner
expert archives for outer fold 3 under experts/. Historical fold ZIPs are
used only to verify the composition mapping and the predeclared budgets.
No result from this audit is fed back to the running experiment.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from scripts import train_v273_native_paired as paired
from scripts import audit_v270_class_conditional_fusion as metrics
from scripts import audit_v271_poly_rescue as rescue
from scripts import train_v260_count_weighting as v260
from scripts import train_v272_poly_conditional_count as v272
from scripts.rebuild_v273_sources import digest, write_json
from scripts.train_boundaries import group_stem
from scripts.v273_native_protocol import load_config


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def verify_inventory(root):
    files = read_json(root/'file-sha256.json')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and p.name != 'file-sha256.json'}
    require(actual == set(files), 'archive file inventory mismatch')
    for name, expected in files.items():
        require(digest(root/name) == expected, 'file checksum mismatch: '+name)
    return len(files)


def verify_historical(cfg, directory):
    mapping = {}
    for fold in cfg['folds']:
        n = fold['fold']
        archive = directory/f'fold-{n}.zip'
        require(digest(archive) == fold['source_archive_sha256'], 'historical archive mismatch')
        with zipfile.ZipFile(archive) as z:
            def content(filename, expected):
                names = [name for name in z.namelist() if Path(name).name == filename]
                require(len(names) == 1, 'historical file inventory mismatch')
                value = z.read(names[0])
                require(hashlib.sha256(value).hexdigest() == expected, 'historical file checksum mismatch')
                return value
            report = json.loads(content(f'report-fold-{n}.json', fold['source_report_sha256']))
            prediction = content(f'predictions-fold-{n}.npz', fold['source_prediction_sha256'])
        p = load_npz(io.BytesIO(prediction))
        require(np.all(p['outer_fold'] == n), 'historical fold mismatch')
        for member in set(p['member'].astype(str)):
            require(member not in mapping, 'historical member appears in multiple folds')
            mapping[member] = n
        inner = report['inner']
        expected = {f'v260_{a}': inner['v271_replay'][f'v260_{a}_fixed_replay']['fixed_source_epochs']
                    for a in ('uniform', 'weighted')}
        expected['v272_uniform'] = inner['v272_replay']['uniform_fixed_source_epochs']
        require(expected == fold['epochs'], 'archived epoch budget changed')
    require(mapping == cfg['member_folds'], 'historical member mapping changed')
    return {'archives_verified': 5, 'tracks': len(mapping), 'compositions': len(cfg['composition_folds'])}


def truth_from_shared_experts(root, cfg):
    """Read labels independently of the two count arms; cover all rebuilt rows."""
    k = np.full(cfg['expected_rows'], -1, np.int32)
    members = np.full(cfg['expected_rows'], '', dtype='U96')
    coverage = np.zeros(len(k), np.int8)
    paths = sorted((root/'experts').rglob('v104-nested-outer-3-inner-*.npz'))
    require(len(paths) == 4, 'four outer-3 inner expert packs required')
    audited = 0
    report3 = read_json(root/'fold-3/paired/report-fold-3.json')
    for path in paths:
        audited += verify_inventory(path.parent)
        require(digest(path) == report3['inner_expert_sha256'][path.name], 'shared expert checksum changed')
    paths.append(root/'fold-3/anchor/v104-nested-eval-3.npz')
    for path in paths:
        p = load_npz(path)
        idx = p['global_index'].astype(np.int64)
        require(len(np.unique(idx)) == len(idx) and np.all((idx >= 0) & (idx < len(k))), 'invalid source indices')
        coverage[idx] += 1
        k[idx], members[idx] = p['k'], p['member']
    require(np.all(coverage == 1) and np.all((k >= 0) & (k <= 6)), 'incomplete shared ground truth')
    require(int(np.sum(k >= 2)) == cfg['expected_poly_rows'], 'shared polyphonic count mismatch')
    require(set(members) == set(cfg['member_folds']), 'shared member inventory mismatch')
    return k, members, audited


def score(k, pred):
    poly = k >= 2
    return {'rows': len(k), 'correct': int(np.sum(k == pred)),
            'poly_rows': int(poly.sum()), 'poly_correct': int(np.sum(k[poly] == pred[poly])),
            'exact': float(np.mean(k == pred)), 'poly_exact': float(np.mean(k[poly] == pred[poly]))}


def audit_fold(root, cfg, config_sha, k, members, row_fold, fold):
    stage = root/f'fold-{fold}'
    file_count = verify_inventory(stage)
    base = stage/'paired'
    report = read_json(base/f'report-fold-{fold}.json')
    protocol = read_json(base/'protocol.json')
    p = load_npz(base/f'predictions-fold-{fold}.npz')
    calibration = read_json(base/'frozen-calibration.json')
    require(report['fold'] == fold and protocol['fold'] == fold, 'fold identity mismatch')
    require(report['config_sha256'] == protocol['config_sha256'] == config_sha, 'configuration changed')
    for name, expected in (('identical_pairing_verified', True), ('additional_output_corrector', False),
                           ('outer_labels_used_for_selection', False), ('historical_v273_reproduced', False)):
        require(report[name] is expected, 'unexpected protocol flag: '+name)
    require(calibration == report['calibration'] and digest(base/'frozen-calibration.json') == report['calibration_sha256'],
            'frozen calibration mismatch')
    meta = cfg['folds'][fold]['meta_fold']
    indices = {
        'outer_idx': np.flatnonzero(row_fold == fold),
        'meta_val_idx': np.flatnonzero(row_fold == meta),
        'meta_fit_idx': np.flatnonzero((row_fold != fold) & (row_fold != meta)),
        'final_fit_idx': np.flatnonzero(row_fold != fold),
    }
    for name, idx in indices.items():
        require(v260.array_digest(idx.astype(np.int64)) == protocol['partition_sha256'][name], 'partition hash mismatch')
    outer = indices['outer_idx']
    require(np.array_equal(p['global_index'], outer) and np.all(p['outer_fold'] == fold), 'outer coverage mismatch')
    require(np.array_equal(p['k'], k[outer]) and np.array_equal(p['member'], members[outer]), 'outer truth mismatch')
    anchor_file = stage/f'anchor/v104-nested-eval-{fold}.npz'
    require(digest(anchor_file) == report['shared_outer_anchor_sha256'], 'outer anchor hash mismatch')
    anchor = load_npz(anchor_file)
    require(np.array_equal(anchor['global_index'], outer) and np.array_equal(anchor['pred104_deploy'], p['shared_anchor']),
            'outer anchor alignment mismatch')
    inner_anchor = load_npz(base/'shared-inner-anchor.npz')
    require(np.array_equal(inner_anchor['global_index'], indices['meta_val_idx']), 'inner anchor alignment mismatch')
    stages, records = {}, {}
    for arm in paired.native.ARMS:
        for phase in ('inner', 'final'):
            predict = indices['meta_val_idx' if phase == 'inner' else 'outer_idx']
            fit = indices['meta_fit_idx' if phase == 'inner' else 'final_fit_idx']
            probabilities = {}
            for component in paired.native.COMPONENTS:
                model_dir = base/arm/phase/component
                rec = read_json(model_dir/'training.json')
                require(rec == report['training'][arm][phase+'/'+component], 'training record mismatch')
                poly = component == 'v272_uniform'
                selected_fit = fit[k[fit] >= 2] if poly else fit
                expected_weights = (v272.poly_class_weights(k[selected_fit], 'uniform') if poly else
                                    v260.arm_weights(component.removeprefix('v260_'), k[selected_fit]))
                expected_seed = (v272.SEED + (100 + 10*fold if phase == 'inner' else 1000 + fold) if poly else
                                 v260.SEED + (100 if phase == 'inner' else 1000) + fold)
                require(rec['seed'] == expected_seed and rec['epochs'] == cfg['folds'][fold]['epochs'][component], 'seed or budget changed')
                require(rec['fit_rows'] == len(selected_fit) and np.array_equal(rec['class_weights'], expected_weights), 'fit or class weights mismatch')
                h = hashlib.sha256()
                for epoch in range(rec['epochs']):
                    h.update(paired.epoch_order(selected_fit, rec['seed'], epoch).tobytes())
                require(h.hexdigest() == rec['epoch_row_order_sha256'], 'declared epoch order mismatch')
                require(all(len(v) == rec['epochs'] and np.isfinite(v).all() for v in rec['history'].values()), 'incomplete training history')
                require(digest(model_dir/'model.weights.h5') == rec['weights_sha256'], 'saved model mismatch')
                require(rec['projection_l2'] == 0 if arm == 'control' else rec['projection_l2'] > 0, 'inactive or contaminated evidence branch')
                values = load_npz(model_dir/'probability.npz')
                require(np.array_equal(values['global_index'], predict), 'probability alignment mismatch')
                validator = v272._validate_poly_probability if poly else rescue._validate_probability
                probabilities[component] = validator(values['probability'], component)
                records[(arm, phase, component)] = rec
            if phase == 'inner':
                replay = paired.calibrate(k[predict], np.argmax(inner_anchor['probability'], axis=1), probabilities)
                require(replay == calibration[arm], 'inner threshold selection does not replay')
            else:
                decoded = paired.decode(p['shared_anchor'], probabilities, calibration[arm])
                require(np.array_equal(decoded, p[arm]), 'saved final decisions do not replay')
                recomputed = score(k[outer], decoded)
                require(all(report['arms'][arm]['cardinality'][key] == value for key, value in recomputed.items()), 'score mismatch')
                require(metrics.cardinality_report(k[outer], decoded) == report['arms'][arm]['cardinality'], 'detailed count report mismatch')
                low = metrics.fuse_counts(p['shared_anchor'], np.argmax(probabilities['v260_uniform'], axis=1), 'low_k_fusion')
                mid = rescue.apply_poly_rescue(p['shared_anchor'], low, probabilities['v260_weighted'], calibration[arm]['v271']['threshold'])[0]
                stages[arm] = {name: score(k[outer], values) for name, values in
                               (('low_k_fusion', low), ('poly_rescue', mid), ('final_v273', decoded))}
    for phase in ('inner', 'final'):
        for component in paired.native.COMPONENTS:
            a, b = [records[(arm, phase, component)] for arm in paired.native.ARMS]
            require(all(a[key] == b[key] for key in ('seed', 'epochs', 'fit_rows', 'initial_weights_sha256',
                        'epoch_row_order_sha256', 'class_weights')), 'unpaired training')
    poly = p['k'] >= 2
    improved = (p['decay'] == p['k']) & (p['control'] != p['k'])
    regressed = (p['control'] == p['k']) & (p['decay'] != p['k'])
    return {'fold': fold, 'verified_files': file_count, 'verified_count_models': 12,
            'cardinality': {arm: score(p['k'], p[arm]) for arm in paired.native.ARMS},
            'poly_corrected': int(improved[poly].sum()), 'poly_regressed': int(regressed[poly].sum()),
            'by_k_net_correct': {str(t): int(improved[p['k'] == t].sum() - regressed[p['k'] == t].sum()) for t in range(7)},
            'replayed_stages': stages, 'report_sha256': digest(base/f'report-fold-{fold}.json'),
            'predictions_sha256': digest(base/f'predictions-fold-{fold}.npz')}, p


def audit(args):
    cfg = load_config(args.config)
    sources = read_json(args.sources)
    for archive in sources['artifacts']:
        path = args.input_dir/archive['path']
        require(path.stat().st_size == archive['size'] and digest(path) == archive['sha256'],
                'downloaded artifact checksum mismatch')
    historical = verify_historical(cfg, args.historical_dir)
    k, members, files = truth_from_shared_experts(args.input_dir, cfg)
    row_fold = np.array([cfg['member_folds'][str(m)] for m in members])
    folds = sorted(int(p.parent.parent.name.split('-')[-1]) for p in args.input_dir.glob('fold-*/paired/report-fold-*.json'))
    require(folds and len(folds) == len(set(folds)) and set(folds) <= set(range(5)), 'invalid audited fold inventory')
    results, parts = [], []
    for fold in folds:
        result, p = audit_fold(args.input_dir, cfg, digest(args.config), k, members, row_fold, fold)
        results.append(result)
        parts.append(p)
    merged = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    require(len(np.unique(merged['global_index'])) == len(merged['k']), 'duplicate external rows')
    arms = {arm: score(merged['k'], merged[arm]) for arm in paired.native.ARMS}
    result = {'audited_at_utc': datetime.now(timezone.utc).isoformat(), 'audit_passed': True,
              'sources': sources,
              'completed_folds_audited': folds, 'complete_five_fold_result': folds == list(range(5)),
              'config_sha256': digest(args.config), 'historical_mapping_and_budgets': historical,
              'shared_truth_rows_verified': len(k), 'shared_truth_poly_rows_verified': int(np.sum(k >= 2)),
              'verified_files': files + sum(r['verified_files'] for r in results),
              'verified_count_models': 12*len(folds), 'per_fold': results, 'available_fold_aggregate': arms,
              'paired_poly_delta_pp': 100*(arms['decay']['poly_exact'] - arms['control']['poly_exact']),
              'poly_corrected': sum(r['poly_corrected'] for r in results),
              'poly_regressed': sum(r['poly_regressed'] for r in results),
              'audited_compositions': len({group_stem(str(m)) for m in merged['member']}),
              'audit_scope': {'frozen_thresholds_reselected_from_inner_labels': True,
                              'outer_decisions_replayed_without_labels': True,
                              'count_scores_independently_recomputed': True,
                              'weight_files_checksum_verified': True,
                              'tensorflow_inference_rerun': False,
                              'event_f1_recomputed_from_audio': False,
                              'actual_minibatch_access_trace_recorded': False},
              'automatic_promotion': False}
    write_json(args.output, result)
    print(json.dumps({key: result[key] for key in ('audit_passed', 'completed_folds_audited', 'verified_files',
          'verified_count_models', 'available_fold_aggregate', 'paired_poly_delta_pp', 'poly_corrected', 'poly_regressed')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input-dir', 'config', 'historical-dir', 'sources', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    audit(parser.parse_args())
