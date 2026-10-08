"""Learn M/N/neutral on proposed changes only, with nested piece exclusions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from scripts.evaluate_v273_regression_loops import read, dump, policy_stats
from scripts.v273_regression_loops import COSTS, counts, select_policy
from scripts.route_v273_regression_loops import source_search
from scripts.verify_v273_selector_repair_artifacts import digest, require, ids_digest

WEIGHTS = (.25, .5, .75, 1.)
REPRESENTATIONS = ('group', 'acoustic58')
CONFIG = dict(max_iter=80, learning_rate=.05, max_leaf_nodes=7, max_depth=3,
              min_samples_leaf=20, l2_regularization=10., early_stopping=False, random_state=27402)


def run(prepared, series, output):
    require(not output.exists(), 'refusing overwrite'); output.mkdir(parents=True)
    data = read(prepared/'inputs.npz'); first = read(series/'predictions.npz'); prob = read(series/'probabilities.npz')
    inner = read(series/'inner-evidence.npz'); splits = json.loads((series/'splits.json').read_text())
    n = len(data['truth']); row = np.arange(n); b = data['baseline']; old = data['original']; y = data['truth']
    changed = old != b
    m = prob['class_probability'][:, 0][row, old]; r = prob['baseline_correct'][:, 0]
    prior = np.column_stack([m, r, np.maximum(0., 1-m-r)])
    # Only original changes use these probabilities; KEEP rows receive no inferred outcome.
    prior[~changed] = [0., 0., 1.]
    target = np.where(y == old, 0, np.where(y == b, 1, 2))
    keys = [f'guard_{rep}__blend_{w:g}__cost_{c:g}' for rep in REPRESENTATIONS for w in WEIGHTS for c in COSTS]
    predicted = np.empty((n, 56), np.int8); inner_pred = np.empty((len(inner['rows']), 56), np.int8)
    probabilities = np.zeros((n, 8, 3)); models = {}; records = {}; assignments = []
    for ri, representation in enumerate(REPRESENTATIONS):
        x = np.column_stack([prior, np.eye(7)[b][:, 2:5], np.eye(7)[old][:, 2:7], data['group_context'][row, old]])
        if representation == 'acoustic58': x = np.column_stack([x, data['features']])
        def learn(train, at):
            fold = int(data['fold'][at[0]]); pieces = sorted(set(map(str, data['piece'][train])))
            require(not set(data['piece'][at]) & set(pieces), 'guard piece leakage')
            require(set(data['fold'][train]) == {fold} and set(data['fold'][at]) == {fold}, 'guard source exclusion')
            fit_at = train[changed[train]]; key = representation+'|'+str(fold)+'|'+','.join(pieces)
            if key not in models:
                require(len(fit_at) >= 40, 'insufficient guard training support')
                model = HistGradientBoostingClassifier(**CONFIG).fit(x[fit_at], target[fit_at]); models[key] = model
                records[key] = dict(fold=fold, pieces=pieces, train_rows=len(fit_at),
                                    train_ids_sha256=ids_digest(data['global_index'][fit_at]))
            out = np.zeros((len(at), 3)); out[:, models[key].classes_.astype(int)] = models[key].predict_proba(x[at])
            out[~changed[at]] = [0., 0., 1.]
            return out, key
        for block in splits['blocks']:
            test = np.flatnonzero(data['piece'] == block['piece']); mask = inner['rows'][:, 0] == block['index']
            locations = np.flatnonzero(mask); train = inner['rows'][mask, 1]
            empirical, key = learn(train, test)
            assignments.append(dict(block=block['index'], representation=representation, role='outer', test_piece=block['piece'], model=key))
            for ai, weight in enumerate(WEIGHTS):
                fi = ri*4+ai; q = (1-weight)*prior[test]+weight*empirical; probabilities[test, fi] = q
                for ci, cost in enumerate(COSTS):
                    predicted[test, fi*7+ci] = np.where(changed[test] & (q[:, 0] > cost*q[:, 1]), old[test], b[test])
            for piece in sorted(set(data['piece'][train])):
                local = data['piece'][train] == piece; at = train[local]
                empirical, key = learn(train[~local], at)
                assignments.append(dict(block=block['index'], representation=representation, role='inner', test_piece=str(piece), model=key))
                for ai, weight in enumerate(WEIGHTS):
                    fi = ri*4+ai; q = (1-weight)*prior[at]+weight*empirical
                    for ci, cost in enumerate(COSTS):
                        inner_pred[locations[local], fi*7+ci] = np.where(changed[at] & (q[:, 0] > cost*q[:, 1]), old[at], b[at])
        joblib.dump(models, output/'models-checkpoint.joblib', compress=3)
        print(json.dumps(dict(stage='guard_fitted', representation=representation,
            variants={str(w):counts(y,b,predicted[:,(ri*4+ai)*7]) for ai,w in enumerate(WEIGHTS)})),flush=True)
    pool = np.column_stack([first['predictions'][data['native_position'], :7], predicted])
    inner_pool = np.column_stack([inner['predictions'][:, :7], inner_pred])
    selected = np.empty((n, 2), np.int16); nested = np.empty((n, 2), np.int8); proofs = []
    for block in splits['blocks']:
        test = np.flatnonzero(data['piece'] == block['piece']); mask = inner['rows'][:, 0] == block['index']; train = inner['rows'][mask, 1]
        j, one = select_policy(y[train], b[train], inner_pool[mask], old[train]); selected[test, 0] = j
        source, two = source_search(y[train], b[train], old[train], inner_pool[mask], True)
        selected[test, 1] = source[b[test]-2]
        for j in range(2): nested[test, j] = pool[test, selected[test, j]]
        proofs.append(dict(block=block['index'], piece=block['piece'],
            piece_success90=dict(policy=int(selected[test[0],0]), **one),
            source_success90=dict(policies=source.tolist(), **two)))
    all_keys = keys+['guard_nested_piece_success90', 'guard_nested_source_success90']
    matrix = np.column_stack([predicted, nested]); summaries = {}
    for j, key in enumerate(all_keys):
        summaries[key] = policy_stats(data, matrix[:, j])
        require(summaries[key]['versus_original']['regressions_added'] == 0 and
                summaries[key]['versus_original']['corrections_added'] == 0, 'guard cannot introduce a new action')
        if j < 56: q = probabilities[:, j//7]
        else:
            chosen = selected[:, j-56]; q = prior.copy(); t = chosen >= 7
            q[t] = probabilities[np.flatnonzero(t), (chosen[t]-7)//7]
        take = matrix[:, j] != b
        expected = float((q[:, 0]-q[:, 1])[take].sum())
        summaries[key]['action_confidence'] = dict(expected_corrections=float(q[take, 0].sum()),
            expected_regressions=float(q[take, 1].sum()), expected_net=expected,
            optimism=expected-summaries[key]['eligible']['net'],
            original_changes_outcome_nll=float(-np.log(np.maximum(q[row[changed],target[changed]],1e-15)).mean()))
    native = np.broadcast_to(data['native_baseline'][:,None],(59309,len(all_keys))).copy(); native[data['native_position']] = matrix
    np.savez_compressed(output/'predictions.npz',global_index=data['native_global_index'],true_K=data['native_truth'],
        frozen_baseline_K=data['native_baseline'],fold=data['native_fold'],eligible_global_index=data['global_index'],
        variant_ids=np.array(all_keys),predictions=native.astype(np.int8),chosen_policy=selected,
        outcomes_vs_freeze=(native==data['native_truth'][:,None]).astype(np.int8)-
                           (data['native_baseline']==data['native_truth'])[:,None].astype(np.int8))
    np.savez_compressed(output/'probabilities.npz',outcomes=probabilities,prior=prior)
    np.savez_compressed(output/'inner-evidence.npz',rows=inner['rows'],predictions=inner_pred)
    (output/'models-checkpoint.joblib').rename(output/'models.joblib')
    dump(output/'model-records.json',dict(models=records,assignments=assignments,configuration=CONFIG));dump(output/'router-proofs.json',proofs)
    report=dict(status='completed',policies=summaries,previous_candidates=150,total_candidates=208,
        independent_validation=False,entire_outer_fold_label_free=False,promotion=False,
        source_inputs_sha256=digest(prepared/'inputs.npz'),source_inner_sha256=digest(series/'inner-evidence.npz'),
        files={p.name:digest(p) for p in output.iterdir()})
    dump(output/'report.json',report)
    print(json.dumps({k:dict(**summaries[k]['eligible'],retention=summaries[k]['versus_original']) for k in all_keys[-2:]},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','series','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.prepared,a.series,a.output)


if __name__=='__main__':main()
