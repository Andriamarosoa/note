"""Seven fixed development cycles, with selection nested inside held-piece evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.special import logit

from scripts.v273_regression_loops import (
    FAMILIES, COSTS, available, group_context, design, fit, predict, decode, counts, select_policy,
)
from scripts.evaluate_v273_recording_calibration import load_metadata, fail_family, family_counts
from scripts.verify_v273_selector_repair_artifacts import digest, require, metrics, named_pairs, ids_digest
from scripts.v273_contextual_risk_contract import retention, audit_profiles

FOLDS = (0, 1, 2, 4)


def read(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+'\n')


def prepare(evidence, source, features, output):
    require(not output.exists(), 'refusing to replace prepared evidence')
    proof = json.loads((evidence/'verification.json').read_text())
    require(digest(source) == proof['source_predictions_sha256'], 'frozen contextual model source')
    raw = read(source); swap = read(evidence/'factor-swaps.npz')
    lookup, feature_sources = load_metadata(features)
    ids = swap['eligible_global_index']; native_ids = swap['global_index']
    positions = {int(g): j for j, g in enumerate(native_ids)}
    pos = np.array([positions[int(g)] for g in ids]); y = swap['true_K'][pos]; b = swap['frozen_baseline_K'][pos]
    rows = {}
    for file in sorted(features.rglob('rows.jsonl')):
        for line in file.read_text().splitlines():
            r = json.loads(line); rows[int(r['global_index'])] = r
    names = sorted(rows[int(ids[0])]['features'])
    x = np.array([[rows[int(g)]['features'][name] for name in names] for g in ids])
    gc, gn = group_context(raw['group_proposal'], raw['local_audit_features'], b)
    r = swap['parent_risk_new_correction__baseline_correct']
    require(((r > 0) & (r < 1)).all(), 'finite unmodified risk logits')
    data = dict(global_index=ids, truth=y, baseline=b, fold=swap['fold'][pos],
                proposal=raw['group_proposal'].astype(np.int8),
                available=available(raw['group_proposal'], b),
                z=logit(r), logits=swap['parent_risk_new_correction__pooled_class_logits'],
                features=x, feature_names=np.array(names), group_context=gc, group_names=np.array(gn),
                original=swap['parent_risk_new_correction_K'][pos],
                original_profile=audit_profiles(raw['group_proposal'], raw['local_audit_features'],
                                               swap['parent_risk_new_correction_K'][pos], b),
                native_global_index=native_ids, native_truth=swap['true_K'], native_baseline=swap['frozen_baseline_K'],
                native_fold=swap['fold'], native_position=pos)
    for key in ('piece', 'recording_id', 'start_sample'):
        data[key] = np.array([lookup[int(g)][key] for g in ids])
    require(len(names) == 58 and x.shape == (7493, 58), 'all primary features retained')
    require(set(data['fold']) == set(FOLDS), 'excluded fold3')
    require(all(str(r)[:2] != '05' for r in data['recording_id']), 'excluded player05')
    require(all(lookup[int(g)]['truth'] == y[i] and lookup[int(g)]['baseline'] == b[i]
                and lookup[int(g)]['fold'] == data['fold'][i] for i, g in enumerate(ids)), 'metadata identity')
    output.mkdir(parents=True)
    np.savez_compressed(output/'inputs.npz', **data)
    dump(output/'input-manifest.json', dict(inputs_sha256=digest(output/'inputs.npz'),
        raw_model_predictions_sha256=digest(source), factor_swaps_sha256=digest(evidence/'factor-swaps.npz'),
        feature_sources=feature_sources, previous_registry_sha256=digest(evidence/'candidate-registry.json'),
        feature_names=names, group_descriptor_names=gn, producer_network_excludes_fold=True,
        labels_used_in_features=False, previous_candidates=46))
    print(json.dumps(dict(stage='prepared', rows=len(ids), source_sha256=digest(source),
                         inputs_sha256=digest(output/'inputs.npz'))), flush=True)


def policy_stats(data, prediction, probabilities=None):
    y, b = data['truth'], data['baseline']
    native = data['native_baseline'].copy(); native[data['native_position']] = prediction
    result = dict(eligible=counts(y, b, prediction), native=metrics(data['native_truth'], native),
                  versus_freeze=named_pairs(data['native_truth'], data['native_baseline'], native),
                  versus_original=retention(y, b, data['original'], prediction),
                  failures=family_counts(fail_family(y, b, prediction, data['available'])),
                  folds={str(f): counts(y[data['fold'] == f], b[data['fold'] == f], prediction[data['fold'] == f])
                         for f in FOLDS})
    strong = (b != y) & data['available'][np.arange(len(y)), y] & (
        data['proposal'][:, [2**j-1 for j in range(8)]] != y[:, None]).all(1)
    result['combination_only'] = dict(events=int(strong.sum()), corrected=int((strong & (prediction == y)).sum()))
    if probabilities is not None:
        r, cp, other = probabilities; change = prediction != b
        c = cp[np.arange(len(y)), prediction]
        truth_probability = cp[np.arange(len(y)), y]
        truth_probability = np.where((b != y) & ~data['available'][np.arange(len(y)), y], other, truth_probability)
        expected = float((c-r)[change].sum())
        result['confidence'] = dict(expected_corrections=float(c[change].sum()),
            expected_regressions=float(r[change].sum()), expected_net=expected,
            optimism=expected-result['eligible']['net'], event_nll=float(-np.log(np.maximum(truth_probability, 1e-15)).mean()))
    return result


def diagnostics(data, matrix, keys):
    y, b = data['truth'], data['baseline']; result = {}
    for name, p in [('original', data['original']), ('final_nested', matrix[:, -1])]:
        result[name] = dict(transitions={}, profiles={})
        for source in (2, 3, 4):
            for target in range(2, 7):
                take = (b == source) & (p == target) & (p != b)
                if take.any(): result[name]['transitions'][f'{source}->{target}'] = counts(y[take], b[take], p[take])
        for profile in np.unique(data['original_profile']):
            at = data['original_profile'] == profile
            result[name]['profiles'][str(profile)] = dict(rows=int(at.sum()),
                outcomes=counts(y[at], b[at], p[at]), retention=retention(y[at], b[at], data['original'][at], p[at]))
    # Descriptive, stratified by initial K; no causal or feature-selection claim.
    associations = {}
    original = data['original']; reg = (b == y) & (original != b); cor = (b != y) & (original == y)
    for k in (2, 3, 4):
        a, c = reg & (b == k), cor & (b == k); values = []
        for j, name in enumerate(data['feature_names']):
            x = data['features'][:, j]; sd = np.std(x[a | c])
            values.append(dict(feature=str(name), regressions=int(a.sum()), corrections=int(c.sum()),
                               regression_mean=float(x[a].mean()), correction_mean=float(x[c].mean()),
                               standardized_difference=float((x[a].mean()-x[c].mean())/max(sd, 1e-12))))
        associations[str(k)] = sorted(values, key=lambda r: -abs(r['standardized_difference']))
    result['descriptive_associations_not_causes'] = associations
    pieces = sorted(set(data['piece'])); old = data['original']; final = matrix[:, -1]
    deltas = []
    for piece in pieces:
        t = data['piece'] == piece; before = counts(y[t], b[t], old[t]); after = counts(y[t], b[t], final[t])
        deltas.append([after['corrections']-before['corrections'], before['regressions']-after['regressions'],
                       after['net']-before['net']])
    rng = np.random.default_rng(27402)
    sums = np.asarray(deltas)[rng.integers(0, len(pieces), (10000, len(pieces)))].sum(1)
    result['piece_bootstrap_descriptive_95pct'] = {key: np.quantile(sums[:, j], [.025, .975]).tolist()
        for j, key in enumerate(('corrections_delta', 'regressions_avoided_net', 'net_delta'))}
    result['bootstrap_caveat'] = '19 exposed pieces; overlapping training sets; descriptive intervals, not independent-validation proof.'
    return result


def evaluate(prepared, output):
    require(not output.exists(), 'refusing to replace a previous experiment')
    manifest = json.loads((prepared/'input-manifest.json').read_text())
    require(digest(prepared/'inputs.npz') == manifest['inputs_sha256'], 'frozen prepared inputs')
    data = read(prepared/'inputs.npz'); n = len(data['truth']); output.mkdir(parents=True)
    pieces = sorted(set(data['piece'])); require(len(pieces) == 19, 'piece inventory')
    policy_keys = [f'{family}__cost_{cost:g}' for family in FAMILIES for cost in COSTS]
    predictions = np.empty((n, len(policy_keys)), np.int8)
    rr = np.empty((n, len(FAMILIES))); cp = np.empty((n, len(FAMILIES), 7)); oo = np.empty_like(rr)
    ll = np.empty_like(cp); models = {}; assignments = []; feature_designs = {}
    blocks = []
    for piece in pieces:
        test = np.flatnonzero(data['piece'] == piece); f = int(data['fold'][test[0]])
        learn = np.flatnonzero((data['fold'] == f) & (data['piece'] != piece))
        require(set(data['fold'][test]) == {f}, 'one fold per piece')
        require(not set(data['recording_id'][test]) & set(data['recording_id'][learn]), 'recording overlap')
        blocks.append(dict(piece=str(piece), fold=f, test=test, learn=learn,
                           inner=np.empty((len(learn), len(policy_keys)), np.int8)))
    for family_index, family in enumerate(FAMILIES):
        print(json.dumps(dict(stage='family_started', family=family)), flush=True)
        xr, xq, rn, qn = design(data, family)
        feature_designs[family] = dict(r_names=rn, q_names=qn, r_dimensions=xr.shape[1], q_dimensions=xq.shape[2])
        cache = {}
        def calibrated(train, at):
            train_pieces = tuple(sorted(set(map(str, data['piece'][train]))))
            fold = int(data['fold'][at[0]])
            require(set(data['fold'][train]) == {fold} and set(data['fold'][at]) == {fold}, 'frozen-network fold scope')
            require(not set(data['piece'][at]) & set(train_pieces), 'test-piece calibration leakage')
            require(not set(data['global_index'][train]) & set(data['global_index'][at]), 'test-event calibration leakage')
            key = family+'|'+str(fold)+'|'+','.join(train_pieces)
            if key not in cache:
                model = fit(data, xr, xq, train, family)
                model.update(train_pieces=list(train_pieces), fold=fold,
                             network_training_folds=sorted(set(FOLDS)-{fold}),
                             train_ids_sha256=ids_digest(data['global_index'][train]), train_rows=len(train))
                models[key] = model; cache[key] = model
            else:
                require(cache[key]['train_ids_sha256'] == ids_digest(data['global_index'][train]), 'cache fit identity')
            return predict(data, xr, xq, at, cache[key]), key
        for bi, block in enumerate(blocks):
            train, test = block['learn'], block['test']
            out, key = calibrated(train, test)
            rr[test, family_index], cp[test, family_index], oo[test, family_index], ll[test, family_index] = out
            assignments.append(dict(block=bi, family=family, role='outer', model=key,
                                    test_piece=block['piece'], test_ids_sha256=ids_digest(data['global_index'][test])))
            for ci, cost in enumerate(COSTS):
                predictions[test, family_index*len(COSTS)+ci] = decode(
                    data['baseline'][test], data['available'][test], out[0], out[1], out[3], cost)
            for inner_piece in sorted(set(data['piece'][train])):
                at = train[data['piece'][train] == inner_piece]; fit_at = train[data['piece'][train] != inner_piece]
                inner_out, key = calibrated(fit_at, at)
                require(block['piece'] not in models[key]['train_pieces'], 'outer piece in inner training')
                assignments.append(dict(block=bi, family=family, role='inner', model=key,
                    test_piece=str(inner_piece), test_ids_sha256=ids_digest(data['global_index'][at])))
                local = np.searchsorted(train, at)
                for ci, cost in enumerate(COSTS):
                    block['inner'][local, family_index*len(COSTS)+ci] = decode(
                        data['baseline'][at], data['available'][at], inner_out[0], inner_out[1], inner_out[3], cost)
        print(json.dumps(dict(stage='family_completed', family=family, fits=len(cache),
            cost_one=counts(data['truth'], data['baseline'], predictions[:, family_index*len(COSTS)]))), flush=True)
        dump(output/'models-checkpoint.json', models)
    require(np.array_equal(predictions[:, 0], data['original']), 'original factor replay')
    nested = np.empty((n, len(FAMILIES)), np.int8); selected = np.empty((n, len(FAMILIES)), np.int16)
    block_proofs = []; inner_rows = []; inner_predictions = []
    for bi, block in enumerate(blocks):
        train, test = block['learn'], block['test']; rounds = []
        for stage in range(len(FAMILIES)):
            j, choice = select_policy(data['truth'][train], data['baseline'][train],
                block['inner'][:, :(stage+1)*len(COSTS)], data['original'][train])
            nested[test, stage] = predictions[test, j]; selected[test, stage] = j
            rounds.append(dict(cycle=stage, policy_index=j, policy=policy_keys[j], **choice))
        block_proofs.append(dict(index=bi, piece=block['piece'], fold=block['fold'],
            test_rows=len(test), learning_rows=len(train),
            test_ids_sha256=ids_digest(data['global_index'][test]),
            learning_ids_sha256=ids_digest(data['global_index'][train]),
            learning_pieces=sorted(set(map(str, data['piece'][train]))), rounds=rounds))
        inner_rows.extend((bi, int(i)) for i in train); inner_predictions.append(block['inner'])
    keys = policy_keys + ['nested_cycle_'+str(j) for j in range(len(FAMILIES))]
    matrix = np.column_stack([predictions, nested]); summaries = {}
    for j, key in enumerate(keys):
        if j < len(policy_keys):
            fi = j//len(COSTS); probabilities = rr[:, fi], cp[:, fi], oo[:, fi]
        else:
            fi = selected[:, j-len(policy_keys)]//len(COSTS); row = np.arange(n)
            probabilities = rr[row, fi], cp[row, fi], oo[row, fi]
        summaries[key] = policy_stats(data, matrix[:, j], probabilities)
    native = np.broadcast_to(data['native_baseline'][:, None], (59309, len(keys))).copy()
    native[data['native_position']] = matrix
    np.savez_compressed(output/'predictions.npz', global_index=data['native_global_index'],
        true_K=data['native_truth'], frozen_baseline_K=data['native_baseline'], fold=data['native_fold'],
        eligible_global_index=data['global_index'], variant_ids=np.array(keys), predictions=native.astype(np.int8),
        outcomes_vs_freeze=(native == data['native_truth'][:, None]).astype(np.int8)-
                           (data['native_baseline'] == data['native_truth'])[:, None].astype(np.int8))
    np.savez_compressed(output/'probabilities.npz', baseline_correct=rr, class_probability=cp,
                        other_probability=oo, logits=ll, selected_policy=selected)
    np.savez_compressed(output/'inner-evidence.npz', rows=np.array(inner_rows, np.int32),
                        predictions=np.concatenate(inner_predictions))
    dump(output/'models.json', models); (output/'models-checkpoint.json').unlink()
    dump(output/'splits.json', dict(blocks=block_proofs, assignments=assignments, designs=feature_designs))
    pareto = []
    for key, stat in summaries.items():
        a = stat['eligible']
        dominated = any(v['eligible']['regressions'] <= a['regressions'] and
            v['eligible']['corrections'] >= a['corrections'] and
            (v['eligible']['regressions'] < a['regressions'] or v['eligible']['corrections'] > a['corrections'])
            for v in summaries.values())
        if not dominated: pareto.append(key)
    audit = diagnostics(data, matrix, keys); dump(output/'diagnostics.json', audit)
    columns = ['global_index', 'fold', 'piece', 'recording_id', 'start_sample', 'truth', 'baseline', 'original']
    with (output/'cases.csv').open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n'); writer.writerow(columns+keys)
        for i in range(n): writer.writerow([data[c][i] for c in columns]+list(matrix[i]))
    report = dict(status='completed', protocol='README_V273_PROTOCOLE_BOUCLES_REGRESSIONS.md',
        scope='Development; supervised calibration within network-excluded fold; nested held-piece evaluation.',
        independent_validation=False, entire_outer_fold_label_free=False, promotion=False,
        test_piece_excluded_from_all_fits_and_policy_selection=True,
        inputs_sha256=manifest['inputs_sha256'], previous_candidates=46, added_policies=len(keys),
        retained_candidates=46+len(keys), families=list(FAMILIES), costs=list(COSTS), cycles=7,
        fitted_models=len(models), keep_control=counts(data['truth'], data['baseline'], data['baseline']),
        original=policy_stats(data, data['original']), policies=summaries, pareto_exploratory=pareto,
        blocks=block_proofs, diagnostics=audit,
        software=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        stopping_reason='All seven predefined cycles completed; no outer-label-guided extra search.',
        files={p.name: digest(p) for p in sorted(output.iterdir())})
    dump(output/'report.json', report)
    lines = ['# V27.3 — Résultats des boucles', '',
        '| Cycle | Corrections | Régressions | Net |', '|---|---:|---:|---:|', '| Croisement initial | 566 | 518 | +48 |']
    for j in range(len(FAMILIES)):
        s = summaries['nested_cycle_'+str(j)]['eligible']
        lines.append(f'| {j} : {FAMILIES[j]} | {s["corrections"]} | {s["regressions"]} | {s["net"]:+d} |')
    lines += ['', 'Les familles sont choisies uniquement dans la validation interne. Chaque sortie est conservée.',
              'La contrainte interne de 90 % ne garantit pas 90 % sur les morceaux évalués.',
              'Ces données ont déjà servi au développement ; aucune validation inédite ni promotion.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for name in ('evidence', 'source', 'features', 'output'): prep.add_argument('--'+name, type=Path, required=True)
    run = sub.add_parser('run')
    for name in ('prepared', 'output'): run.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.command == 'prepare': prepare(a.evidence, a.source, a.features, a.output)
    else: evaluate(a.prepared, a.output)


if __name__ == '__main__': main()
