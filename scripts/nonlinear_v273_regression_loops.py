"""Nonlinear calibration of frozen factors; exact nested exclusions and all outputs retained."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from scipy.special import expit, logsumexp
from sklearn.ensemble import HistGradientBoostingClassifier
from scripts.evaluate_v273_regression_loops import read, dump, policy_stats
from scripts.v273_regression_loops import COSTS, design, decode, counts, select_policy
from scripts.route_v273_regression_loops import source_search
from scripts.verify_v273_selector_repair_artifacts import digest, require, ids_digest

REPRESENTATIONS = ('group_audit', 'acoustic58')
BLENDS = (.25, .5, .75)
CONFIG = dict(max_iter=80, learning_rate=.05, max_leaf_nodes=7, max_depth=3,
              min_samples_leaf=40, l2_regularization=10., early_stopping=False,
              random_state=27402)


def nonlinear_features(data, representation):
    xr, _, rn, _ = design(data, representation)
    x = np.column_stack([xr, data['z'], np.where(data['available'], data['logits'], 0.), data['available']])
    return x, rn+['risk_logit']+['conditional_logit_'+str(k) for k in range(7)]+['available_'+str(k) for k in range(7)]


def tree_factors(data, x, at, models):
    rmodel, qmodel = models
    r = rmodel.predict_proba(x[at])[:, list(rmodel.classes_).index(1)]
    q = np.zeros((len(at), 8)); q[:, qmodel.classes_.astype(int)] = qmodel.predict_proba(x[at])
    unsupported = ~data['available'][at]
    q[:, 7] += (q[:, :7]*unsupported).sum(1); q[:, :7][unsupported] = 0
    require(np.allclose(q.sum(1), 1), 'masked nonlinear simplex')
    return r, q


def blended(data, at, rtree, qtree, weight):
    r0 = expit(data['z'][at]); full = np.column_stack([
        np.where(data['available'][at], data['logits'][at], -np.inf), np.zeros(len(at))])
    q0 = np.exp(full-logsumexp(full, axis=1, keepdims=True))
    r = (1-weight)*r0+weight*rtree; q = (1-weight)*q0+weight*qtree
    cp = (1-r[:, None])*q[:, :7]+np.eye(7)[data['baseline'][at]]*r[:, None]
    logits = np.log(np.maximum(q[:, :7], 1e-300))
    return r, cp, (1-r)*q[:, 7], logits


def run(prepared, series, output):
    require(not output.exists(), 'refusing overwrite'); output.mkdir(parents=True)
    data = read(prepared/'inputs.npz'); splits = json.loads((series/'splits.json').read_text())
    first = read(series/'predictions.npz'); inner = read(series/'inner-evidence.npz')
    probabilities1 = read(series/'probabilities.npz'); n = len(data['truth']); original_policies = 49
    keys = [f'tree_{rep}__blend_{alpha:g}__cost_{cost:g}' for rep in REPRESENTATIONS for alpha in BLENDS for cost in COSTS]
    matrix = np.empty((n, 42), np.int8); inner_matrix = np.empty((len(inner['rows']), 42), np.int8)
    rr = np.empty((n, 6)); cp = np.empty((n, 6, 7)); oo = np.empty_like(rr); ll = np.empty_like(cp)
    models = {}; model_records = {}; assignments = []; feature_names = {}
    for rep_index, representation in enumerate(REPRESENTATIONS):
        x, names = nonlinear_features(data, representation); feature_names[representation] = names
        print(json.dumps(dict(stage='nonlinear_started', representation=representation)), flush=True)
        def learn(train, at):
            fold = int(data['fold'][at[0]]); pieces = sorted(set(map(str, data['piece'][train])))
            require(set(data['fold'][train]) == {fold} and set(data['fold'][at]) == {fold}, 'single excluded source model')
            require(not set(data['piece'][at]) & set(pieces), 'piece leakage')
            key = representation+'|'+str(fold)+'|'+','.join(pieces)
            if key not in models:
                rm = HistGradientBoostingClassifier(**CONFIG).fit(x[train], (data['truth'][train] == data['baseline'][train]).astype(int))
                wrong = train[data['truth'][train] != data['baseline'][train]]
                qtarget = np.where(data['available'][wrong, data['truth'][wrong]], data['truth'][wrong], 7)
                qm = HistGradientBoostingClassifier(**CONFIG).fit(x[wrong], qtarget)
                models[key] = rm, qm
                model_records[key] = dict(fold=fold, train_pieces=pieces, train_rows=len(train),
                    train_ids_sha256=ids_digest(data['global_index'][train]),
                    conditional_train_ids_sha256=ids_digest(data['global_index'][wrong]),
                    conditional_train_rows=len(wrong), r_classes=rm.classes_.tolist(), q_classes=qm.classes_.tolist())
            return tree_factors(data, x, at, models[key]), key
        for block in splits['blocks']:
            bi = block['index']; test = np.flatnonzero(data['piece'] == block['piece'])
            m = inner['rows'][:, 0] == bi; locations = np.flatnonzero(m); train = inner['rows'][m, 1]
            (rt, qt), key = learn(train, test)
            assignments.append(dict(block=bi, representation=representation, role='outer', test_piece=block['piece'], model=key))
            for ai, alpha in enumerate(BLENDS):
                fi = rep_index*3+ai; out = blended(data, test, rt, qt, alpha)
                rr[test, fi], cp[test, fi], oo[test, fi], ll[test, fi] = out
                for ci, cost in enumerate(COSTS): matrix[test, fi*7+ci] = decode(data['baseline'][test], data['available'][test], out[0], out[1], out[3], cost)
            for piece in sorted(set(data['piece'][train])):
                local = data['piece'][train] == piece; at = train[local]; learn_at = train[~local]
                (rt, qt), key = learn(learn_at, at)
                assignments.append(dict(block=bi, representation=representation, role='inner', test_piece=str(piece), model=key))
                for ai, alpha in enumerate(BLENDS):
                    fi = rep_index*3+ai; out = blended(data, at, rt, qt, alpha)
                    for ci, cost in enumerate(COSTS): inner_matrix[locations[local], fi*7+ci] = decode(data['baseline'][at], data['available'][at], out[0], out[1], out[3], cost)
        joblib.dump(models, output/'models-checkpoint.joblib', compress=3)
        print(json.dumps(dict(stage='nonlinear_completed', representation=representation, fits=len(models),
            raw_cost_results={str(alpha):counts(data['truth'], data['baseline'], matrix[:, (rep_index*3+ai)*7]) for ai,alpha in enumerate(BLENDS)})), flush=True)
    combined_inner = np.column_stack([inner['predictions'], inner_matrix])
    combined = np.column_stack([first['predictions'][data['native_position'], :49], matrix])
    nested = np.empty((n, 3), np.int8); chosen = np.empty((n, 3), np.int16); proofs = []
    for block in splits['blocks']:
        test = np.flatnonzero(data['piece'] == block['piece']); m = inner['rows'][:, 0] == block['index']; train = inner['rows'][m, 1]
        y, b, original = (data[k][train] for k in ('truth', 'baseline', 'original'))
        j, one = select_policy(y, b, combined_inner[m], original); chosen[test, 0] = j
        source, two = source_search(y, b, original, combined_inner[m], False); chosen[test, 1] = source[data['baseline'][test]-2]
        protected, three = source_search(y, b, original, combined_inner[m], True); chosen[test, 2] = protected[data['baseline'][test]-2]
        for j in range(3): nested[test, j] = combined[test, chosen[test, j]]
        proofs.append(dict(block=block['index'], piece=block['piece'],
            piece_count90=dict(policy=int(chosen[test[0], 0]), **one),
            source_count90=dict(policies=source.tolist(), **two), source_success90=dict(policies=protected.tolist(), **three)))
    nested_keys = ['tree_nested_piece_count90', 'tree_nested_source_count90', 'tree_nested_source_success90']
    all_keys = keys+nested_keys; all_matrix = np.column_stack([matrix, nested]); summaries = {}
    all_r = np.column_stack([probabilities1['baseline_correct'], rr])
    all_cp = np.concatenate([probabilities1['class_probability'], cp], 1)
    all_o = np.column_stack([probabilities1['other_probability'], oo])
    for j, name in enumerate(all_keys):
        fi = j//7+7 if j < 42 else chosen[:, j-42]//7
        ps = all_r[np.arange(n), fi], all_cp[np.arange(n), fi], all_o[np.arange(n), fi]
        summaries[name] = policy_stats(data, all_matrix[:, j], ps)
    native = np.broadcast_to(data['native_baseline'][:, None], (59309, len(all_keys))).copy()
    native[data['native_position']] = all_matrix
    np.savez_compressed(output/'predictions.npz', global_index=data['native_global_index'], true_K=data['native_truth'],
        frozen_baseline_K=data['native_baseline'], fold=data['native_fold'], eligible_global_index=data['global_index'],
        variant_ids=np.array(all_keys), predictions=native.astype(np.int8), chosen_policy=chosen,
        outcomes_vs_freeze=(native == data['native_truth'][:, None]).astype(np.int8)-
                           (data['native_baseline'] == data['native_truth'])[:, None].astype(np.int8))
    np.savez_compressed(output/'probabilities.npz', baseline_correct=rr, class_probability=cp, other_probability=oo, logits=ll)
    np.savez_compressed(output/'inner-evidence.npz', rows=inner['rows'], predictions=inner_matrix)
    (output/'models-checkpoint.joblib').rename(output/'models.joblib')
    dump(output/'model-records.json', dict(models=model_records, assignments=assignments, features=feature_names, configuration=CONFIG))
    dump(output/'router-proofs.json', proofs)
    report = dict(status='completed', policies=summaries, previous_candidates=105, total_candidates=150,
        independent_validation=False, entire_outer_fold_label_free=False, promotion=False,
        source_inputs_sha256=digest(prepared/'inputs.npz'), source_inner_sha256=digest(series/'inner-evidence.npz'),
        files={p.name:digest(p) for p in output.iterdir()})
    dump(output/'report.json', report)
    print(json.dumps({k:dict(**summaries[k]['eligible'], retention=summaries[k]['versus_original']) for k in nested_keys},indent=2),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'series', 'output'): p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); run(a.prepared, a.series, a.output)


if __name__ == '__main__': main()
