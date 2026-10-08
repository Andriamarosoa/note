"""Independent numerical replay, exclusion audit, and complete candidate-memory checks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.evaluate_v273_regression_loops import read, dump
from scripts.v273_regression_loops import FAMILIES, COSTS, design
from scripts.verify_v273_selector_repair_artifacts import require, digest, ids_digest, metrics, named_pairs
from scripts.v273_contextual_risk_contract import retention


def probabilities(z, logits, av, baseline):
    r = 1/(1+np.exp(-z))
    full = np.column_stack([np.where(av, logits, -np.inf), np.zeros(len(z))])
    full -= full.max(1, keepdims=True); q = np.exp(full); q /= q.sum(1, keepdims=True)
    cp = (1-r[:, None])*q[:, :7]; cp[np.arange(len(z)), baseline] = r
    return r, cp, (1-r)*q[:, 7], logits


def replay(data, xr, xq, at, model):
    def transformed(x, scale):
        return np.minimum(6, np.maximum(-6, (x-np.array(scale['mean']))/np.array(scale['scale'])))
    z, logits = data['z'][at].copy(), data['logits'][at].copy()
    if model['r'] is not None:
        coef = np.array(model['r']['parameters'])
        z = coef[0]*z+coef[1]+np.sum(transformed(xr[at], model['r_scale'])*coef[2:], 1)
    if model['q'] is not None:
        coef = np.array(model['q']['parameters'])
        logits = coef[0]*logits+coef[1]+np.sum(transformed(xq[at], model['q_scale'])*coef[2:], 2)
    return probabilities(z, logits, data['available'][at], data['baseline'][at])


def choices(b, av, out):
    r, cp, _, logits = out
    best = np.where(av, logits, -np.inf).argmax(1); p = cp[np.arange(len(b)), best]
    return np.column_stack([np.where(av.any(1) & (p-cost*r > 0), best, b) for cost in COSTS])


def check_close(a, b, message):
    require(np.allclose(a, b, rtol=1e-9, atol=1e-10), message)


def verify(prepared, series1, series2, series3, series4, output):
    require(not output.exists(), 'refusing to overwrite verification')
    data = read(prepared/'inputs.npz'); manifest = json.loads((prepared/'input-manifest.json').read_text())
    require(digest(prepared/'inputs.npz') == manifest['inputs_sha256'], 'prepared identity')
    require(len(data['truth']) == 7493 and len(data['native_truth']) == 59309, 'native scope')
    require(set(data['fold']) == {0, 1, 2, 4}, 'fold exclusions')
    require(all(str(r)[:2] != '05' for r in data['recording_id']), 'player05 exclusion')
    models = json.loads((series1/'models.json').read_text()); splits = json.loads((series1/'splits.json').read_text())
    first = read(series1/'predictions.npz'); probs = read(series1/'probabilities.npz'); inner = read(series1/'inner-evidence.npz')
    n = len(data['truth']); first_eligible = first['predictions'][data['native_position']]
    require(np.array_equal(first_eligible[:, 0], data['original']), 'exact +48 parent replay')
    maximum_difference = 0.; replayed = 0
    for fi, family in enumerate(FAMILIES):
        xr, xq, rn, qn = design(data, family)
        require(rn == splits['designs'][family]['r_names'] and qn == splits['designs'][family]['q_names'], 'declared descriptors')
        for key, model in models.items():
            if model['family'] != family: continue
            train = np.flatnonzero((data['fold'] == model['fold']) & np.isin(data['piece'], model['train_pieces']))
            require(ids_digest(data['global_index'][train]) == model['train_ids_sha256'] and len(train) == model['train_rows'], 'exact fit population')
            for x, scale in [(xr[train], model['r_scale']), (xq[train][data['available'][train]], model['q_scale'])]:
                mean = x.mean(0); sd = x.std(0); sd = np.where(sd > 1e-8, sd, 1.)
                check_close(mean, scale['mean'], 'training-only scaler mean'); check_close(sd, scale['scale'], 'training-only scaler spread')
            for factor in ('r', 'q'):
                if model[factor] is not None:
                    p = model[factor]['parameters']
                    require(model[factor]['success'] and .05 <= p[0] <= 20 and -10 <= p[1] <= 10, 'fitted slope contract')
        for assignment in splits['assignments']:
            if assignment['family'] != family: continue
            block = splits['blocks'][assignment['block']]; model = models[assignment['model']]
            expected_pieces = set(block['learning_pieces'])
            if assignment['role'] == 'inner': expected_pieces.remove(assignment['test_piece'])
            require(set(model['train_pieces']) == expected_pieces and block['piece'] not in expected_pieces, 'nested excluded outer piece')
            require(model['fold'] == block['fold'] and block['fold'] not in model['network_training_folds'], 'network exclusion')
            at = np.flatnonzero(data['piece'] == assignment['test_piece'])
            require(ids_digest(data['global_index'][at]) == assignment['test_ids_sha256'], 'evaluated identity')
            out = replay(data, xr, xq, at, model); decoded = choices(data['baseline'][at], data['available'][at], out)
            if assignment['role'] == 'outer':
                expected = first_eligible[at, fi*7:(fi+1)*7]
                for value, name in zip(out, ('baseline_correct', 'class_probability', 'other_probability', 'logits')):
                    original = probs[name][at, fi]
                    maximum_difference = max(maximum_difference, float(np.max(np.abs(value-original))))
                    check_close(value, original, 'independent probability replay')
            else:
                mask = (inner['rows'][:, 0] == block['index']) & np.isin(inner['rows'][:, 1], at)
                require(np.array_equal(inner['rows'][mask, 1], at), 'inner evidence order')
                expected = inner['predictions'][mask, fi*7:(fi+1)*7]
            require(np.array_equal(decoded, expected), 'independent decisions replay')
            replayed += len(at)
    # Recompute cumulative selection independently from the archived inner decisions.
    for block in splits['blocks']:
        mask = inner['rows'][:, 0] == block['index']; train = inner['rows'][mask, 1]
        test = np.flatnonzero(data['piece'] == block['piece']); y = data['truth'][train]; b = data['baseline'][train]
        require(not set(data['recording_id'][train]) & set(data['recording_id'][test]), 'nested recording overlap')
        ip = inner['predictions'][mask]; original = data['original'][train]
        original_c = int(((b != y) & (original == y)).sum()); original_r = int(((b == y) & (original != b)).sum())
        for stage in range(7):
            scores = []
            for j, p in enumerate(ip[:, :(stage+1)*7].T):
                c = int(((b != y) & (p == y)).sum()); r = int(((b == y) & (p != b)).sum())
                if c >= .9*original_c and r <= original_r and c-r >= original_c-original_r:
                    scores.append((r, r-c, -c, j))
            selected = min(scores)[-1]
            require(selected == block['rounds'][stage]['policy_index'], 'independent inner selection')
            require(np.array_equal(first_eligible[test, 49+stage], first_eligible[test, selected]), 'nested outer application')
    series = [series1, series2, series3, series4]; matrices = []; entries = []
    nonlinear_fit_records = 0
    for root, guard in [(series3, False), (series4, True)]:
        record = json.loads((root/'model-records.json').read_text())
        for model in record['models'].values():
            pieces = model['pieces' if guard else 'train_pieces']
            train = np.flatnonzero((data['fold'] == model['fold']) & np.isin(data['piece'], pieces))
            if guard: train = train[data['original'][train] != data['baseline'][train]]
            require(len(train) == model['train_rows'] and ids_digest(data['global_index'][train]) == model['train_ids_sha256'], 'nonlinear fit identities')
            if not guard:
                wrong = train[data['truth'][train] != data['baseline'][train]]
                require(ids_digest(data['global_index'][wrong]) == model['conditional_train_ids_sha256'], 'conditional fit identities')
            nonlinear_fit_records += 1
        for assignment in record['assignments']:
            block = splits['blocks'][assignment['block']]; model = record['models'][assignment['model']]
            expected = set(block['learning_pieces'])
            if assignment['role'] == 'inner': expected.remove(assignment['test_piece'])
            require(set(model['pieces' if guard else 'train_pieces']) == expected, 'nonlinear nested fit pieces')
            require(block['piece'] not in expected and assignment['test_piece'] not in expected, 'nonlinear receiver exclusion')
            require(model['fold'] == block['fold'], 'nonlinear frozen source fold')
    for index, root in enumerate(series, 1):
        values = read(root/'predictions.npz'); report = json.loads((root/'report.json').read_text())
        for key, source in [('global_index', 'native_global_index'), ('true_K', 'native_truth'),
                            ('frozen_baseline_K', 'native_baseline'), ('fold', 'native_fold')]:
            require(np.array_equal(values[key], data[source]), 'native alignment '+key)
        native = values['predictions']; active = np.zeros(59309, bool); active[data['native_position']] = True
        require(np.array_equal(native[~active], np.broadcast_to(data['native_baseline'][~active, None], native[~active].shape)), 'outside scope untouched')
        outcomes = (native == data['native_truth'][:, None]).astype(np.int8)-(data['native_baseline'] == data['native_truth'])[:, None].astype(np.int8)
        require(np.array_equal(outcomes, values['outcomes_vs_freeze']), 'every outcome retained')
        for j, key in enumerate(values['variant_ids']):
            key = str(key); pred = native[:, j]; s = report['policies'][key]
            require(s['native'] == metrics(data['native_truth'], pred), 'native metrics')
            require(s['versus_freeze'] == named_pairs(data['native_truth'], data['native_baseline'], pred), 'paired metrics')
            require(s['versus_original'] == retention(data['truth'], data['baseline'], data['original'], pred[data['native_position']]), 'retention metrics')
            entries.append(dict(series=index, field=key, variant_id=f'regression_loops_s{index}__{key}',
                retained=True, downstream_ready=False, retention_depends_on_global_gain=False,
                metrics=s['native'], versus_freeze=s['versus_freeze'], versus_original=s['versus_original']))
        matrices.append(native)
    # Independently decode the nonlinear probabilities and complete-policy routing.
    nonlinear = read(series3/'probabilities.npz')
    third = matrices[2][data['native_position']]
    for fi in range(6):
        out = tuple(nonlinear[k][:, fi] for k in ('baseline_correct', 'class_probability', 'other_probability', 'logits'))
        check_close(out[1].sum(1)+out[2], np.ones(n), 'nonlinear simplex')
        require(np.array_equal(choices(data['baseline'], data['available'], out), third[:, fi*7:(fi+1)*7]), 'nonlinear standalone decoding')
    second_archive = read(series2/'predictions.npz'); third_archive = read(series3/'predictions.npz')
    for archive, pool, offset in [(second_archive, first_eligible[:, :49], 0),
                                  (third_archive, np.column_stack([first_eligible[:, :49], third[:, :42]]), 42)]:
        selected = archive['chosen_policy']; target = archive['predictions'][data['native_position']]
        for j in range(selected.shape[1]):
            require(np.array_equal(target[:, offset+j], pool[np.arange(n), selected[:, j]]), 'complete policy routing')
    guard = read(series4/'probabilities.npz'); fourth_archive = read(series4/'predictions.npz')
    fourth = fourth_archive['predictions'][data['native_position']]
    for fi in range(8):
        q = guard['outcomes'][:, fi]; check_close(q.sum(1), np.ones(n), 'guard outcome simplex')
        for ci, cost in enumerate(COSTS):
            prediction = np.where((data['original'] != data['baseline']) & (q[:, 0] > cost*q[:, 1]), data['original'], data['baseline'])
            require(np.array_equal(prediction, fourth[:, fi*7+ci]), 'fixed action guard decoding')
    pool = np.column_stack([first_eligible[:, :7], fourth[:, :56]])
    for j in range(2):
        require(np.array_equal(fourth[:, 56+j], pool[np.arange(n), fourth_archive['chosen_policy'][:, j]]), 'nested guard routing')
    require(len(entries) == 162, 'all 56+3+45+58 policies retained')
    # Router inputs and complete selected policies are checked separately by a full fresh CI replay.
    proof = dict(status='verified', independent_probability_max_difference=maximum_difference,
        replayed_event_assignments=replayed, calibrated_model_records=len(models),
        nonlinear_model_records=nonlinear_fit_records, nonlinear_fit_exclusions_verified=True,
        complete_policy_routing_and_guard_actions_verified=True,
        exact_original_replay=True, outer_test_piece_overlap=0, nested_choice_test_piece_overlap=0,
        calibrated_fit_and_scaler_exclusions_verified=True, native_metrics_and_all_outcomes_verified=True,
        added_candidates=len(entries), total_candidates=208, independent_validation=False,
        entire_outer_fold_label_free=False, entries=entries,
        sources={str(root.name):dict(report_sha256=digest(root/'report.json'), predictions_sha256=digest(root/'predictions.npz')) for root in series})
    dump(output, proof)
    print(json.dumps({k:v for k,v in proof.items() if k not in ('entries','sources')},indent=2),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'series1', 'series2', 'series3', 'series4', 'output'): p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); verify(a.prepared, a.series1, a.series2, a.series3, a.series4, a.output)


if __name__ == '__main__': main()
