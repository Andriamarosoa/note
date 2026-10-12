"""Replay the original risk critics, S1-S4 policies and coherent selector.

No new receiver labels or fitting of predictive models. Historical labels are
restricted to the original fold-excluded audit references. External inference
retains all nineteen held-piece calibration branches; it does not invent an
assignment of a new recording to a historical piece or claim S80 promotion.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import h5py
import joblib
import numpy as np
from scipy.special import erf, expit, logit, softmax
from sklearn.preprocessing import StandardScaler

from scripts.recover_v273_255_producers import (
    EVIDENCE, PREPARED, need, sha, read, write_json, matrices, votes,
)
from scripts.v273_selector_contract import (
    FOLDS, direct_options, audit_receivers, reference_regimes, fallback_metadata,
)
from scripts.prepare_v273_group127_producers import FrozenProducerPool, cache_key
from scripts.prepare_v273_learned_corrector import FrozenCorrector, CataloguePool
from scripts.v273_catalogue_contract import joint_groups, local_audits
from scripts.v273_regression_loops import (
    FAMILIES, COSTS, available, group_context, design, predict, decode,
)
from scripts.nonlinear_v273_regression_loops import nonlinear_features, tree_factors, blended, BLENDS
from scripts.guard_v273_regression_loops import WEIGHTS

REGRESSION = Path('analysis/evidence/v273-regression-loops')
RISK = Path('analysis/evidence/v273-contextual-risk')
CHAMPIONS = ('S1_acoustic', 'S2_source', 'S3_repair', 'S4_repair', 'historical_repair_coherent')


def checked_models(root):
    """Read existing predictive states, preserving every original byte check."""
    manifest = json.loads((root/REGRESSION/'storage-manifest.json').read_text())
    files = {r['stored']: r for r in manifest['files']}
    path = root/REGRESSION/'series1/models.json.gz'
    need(sha(path) == files['series1/models.json.gz']['stored_sha256'], 'S1 archive SHA')
    data = gzip.decompress(path.read_bytes())
    need(hashlib.sha256(data).hexdigest() == files['series1/models.json.gz']['source_sha256'], 'S1 restored SHA')
    states = {1: json.loads(data)}
    for info in manifest['model_archives']:
        chunks = []
        for record in sorted(info['parts'], key=lambda r: r['file']):
            p = root/REGRESSION/record['file']
            need(sha(p) == record['sha256'] and p.stat().st_size == record['bytes'], 'S3/S4 archive part SHA')
            chunks.append(p.read_bytes())
        blob = b''.join(chunks)
        need(hashlib.sha256(blob).hexdigest() == info['compact_sha256'], 'S3/S4 assembled SHA')
        states[info['series']] = joblib.load(io.BytesIO(blob))
    need(set(states) == {1, 3, 4}, 'S1/S3/S4 inventory')
    return states


def dense_weights(blob):
    with h5py.File(io.BytesIO(blob), 'r') as h:
        pairs = {}
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and name.endswith('/vars/0'):
                prefix = name[:-7]
                pairs[prefix] = (obj[:].astype(np.float64), h[prefix+'/vars/1'][:].astype(np.float64))
        h.visititems(visit)
    return pairs


def neural_weights(root):
    manifest = json.loads((root/EVIDENCE/'manifest.json').read_text())['files']['model-replay-bundles.zip']
    path = root/EVIDENCE/'model-replay-bundles.zip'
    need(sha(path) == manifest['sha256'], 'original neural bundle SHA')
    result = {}
    with ZipFile(path) as z:
        for f in FOLDS:
            for name, key in [('catalogue8_local', 'parent'), ('repair_coherent', 'coherent')]:
                member = f'{name}/fold-{f}.weights.h5'
                blob = z.read(member)
                need(hashlib.sha256(blob).hexdigest() == manifest['members'][member]['sha256'], 'original neural member SHA')
                result[key, f] = dense_weights(blob)
    report = json.loads((root/RISK/'report.json').read_text())
    for f in FOLDS:
        path = root/RISK/f'fold-{f}.weights.h5'
        need(sha(path) == report['folds'][str(f)]['weights_sha256'], 'contextual critic SHA')
        result['contextual', f] = dense_weights(path.read_bytes())
    return result


def dense(x, weights, name, gelu=False):
    w, b = weights[name]
    a = np.asarray(x, np.float64) @ w + b
    return a*.5*(1+erf(a/np.sqrt(2))) if gelu else a


def critic(inputs, weights, contextual=False):
    """Original CatalogueCritic equations; stable float64 accumulation."""
    outputs = []
    for start in range(0, len(inputs['baseline']), 96):
        x = {k: v[start:start+96] for k, v in inputs.items()}
        features, base, context, p, v = (x[k] for k in ('group_features', 'baseline', 'context', 'proposal', 'member_votes'))
        n, groups = p.shape
        bgroup = np.broadcast_to(base[:, None], (n, groups, 7))
        cgroup = np.broadcast_to(context[:, None], (n, groups, context.shape[1]))
        state = dense(np.concatenate([features, cgroup, bgroup, np.eye(7)[p]], -1), weights, 'layers/dense', True)
        state = dense(state, weights, 'layers/dense_1', True)
        conditional = dense(state, weights, 'layers/dense_2')[..., 0]
        active = (p != base.argmax(1)[:, None]).astype(np.float64)
        den = np.maximum(active.sum(1, keepdims=True), 1)
        audit_summary = (features[..., 61:]*active[..., None]).sum(1)/den
        row = [context, base, v.reshape(n, -1), audit_summary]
        if contextual:
            row.append((state*active[..., None]).sum(1)/den)
        hidden = dense(np.concatenate(row, 1), weights, 'layers/dense_3', True)
        z = dense(hidden, weights, 'layers/dense_4')[:, 0]
        membership = np.eye(7)[p]
        pooled = (membership*conditional[..., None]).sum(1)/np.maximum(membership.sum(1), 1)
        outputs.append((expit(z).astype(np.float32), pooled.astype(np.float32)))
    return tuple(np.concatenate([o[j] for o in outputs]) for j in range(2))


def coherent_forward(x, weights):
    result = []
    for start in range(0, len(x['baseline']), 96):
        a = {k: v[start:start+96] for k, v in x.items()}
        combo, valid, base, context, metadata = (a[k] for k in ('subset_features', 'subset_mask', 'baseline', 'context', 'keep_heads'))
        n = len(base)
        target = np.broadcast_to(np.eye(5)[None], (n, 5, 5))
        bsub = np.broadcast_to(base[:, None, None], (n, 5, 64, 7))
        tsub = np.broadcast_to(target[:, :, None], (n, 5, 64, 5))
        csub = np.broadcast_to(context[:, None, None], (n, 5, 64, context.shape[1]))
        state = dense(np.concatenate([combo, bsub, tsub, csub], -1), weights, 'subset_embed', True)
        scores = dense(state, weights, 'subset_gate')[..., 0]
        attention = np.where(valid, softmax(np.where(valid, scores, -1e9), axis=2), 0)
        attention /= np.maximum(attention.sum(2, keepdims=True), 1e-9)
        pooled = np.einsum('bks,bksd->bkd', attention, state)
        selected = np.einsum('bks,bksd->bkd', attention, combo)
        ct = np.broadcast_to(context[:, None], (n, 5, context.shape[1]))
        bt = np.broadcast_to(base[:, None], (n, 5, 7))
        hidden = dense(np.concatenate([pooled, selected, ct, bt, target], -1), weights, 'score_hidden', True)
        logits = dense(hidden, weights, 'target_logit')[..., 0]
        legal = valid.any(-1)
        maximum = np.where(legal, logits, -1e9).max(1, keepdims=True)
        mean = np.where(legal, logits, 0).sum(1, keepdims=True)/4
        row = dense(np.concatenate([base, context, metadata, maximum, mean], -1), weights, 'layers/dense', True)
        low = dense(row, weights, 'layers/dense_1')
        poly = np.where(base[:, 2:] > .5, low[:, 2, None], logits)
        result.append(softmax(np.concatenate([low[:, :2], poly], 1), axis=1).astype(np.float32))
    return np.concatenate(result)


def coherent_inputs(reference_p, reference_raw, reference_y, reference_b, p, raw, base, context):
    ref_opt = direct_options(reference_p, reference_b)[0]
    opt, mask, identity = direct_options(p, base)
    ref_group, receiver_group = reference_regimes(reference_raw, raw)
    ga, la = audit_receivers(ref_opt, reference_y, reference_b, base, ref_group, receiver_group)
    features = np.concatenate([opt, ga, identity, la], -1)*mask[..., None]
    need(features.shape == (len(base), 5, 64, 21), 'original coherent input shape')
    return dict(subset_features=features, subset_mask=mask, baseline=np.eye(7, dtype=np.float32)[base],
                context=context, keep_heads=fallback_metadata(p, base))


def branch_data(features, base, proposal, audits, r, logits, feature_names):
    gc, gn = group_context(proposal, audits, base)
    need(np.all((r > 0) & (r < 1)), 'risk probability saturated: cannot reproduce original finite-logit contract')
    d = dict(features=features, baseline=base, available=available(proposal, base),
             group_context=gc, group_names=np.asarray(gn), feature_names=feature_names,
             z=logit(r.astype(np.float64)), logits=logits.astype(np.float64))
    at = np.arange(len(base))
    xr, xq, _, _ = design(d, 'identity')
    original_factors = predict(d, xr, xq, at, dict(r=None, q=None))
    d['original'] = decode(base, d['available'], original_factors[0], original_factors[1], original_factors[3], 1.)
    return d


def family_policies(data, block, states, records):
    """Apply exact original piece-specific models and already frozen routes."""
    at = np.arange(len(data['baseline'])); b = data['baseline']; old = data['original']
    piece = block['piece']; index = block['index']
    first = []
    for family in FAMILIES:
        assignment = [a for a in records[1]['assignments'] if a['family'] == family and a['role'] == 'outer' and a['test_piece'] == piece]
        need(len(assignment) == 1, 'original S1 assignment')
        model = states[1][assignment[0]['model']]
        need(piece not in model['train_pieces'] and model['fold'] == block['fold'], 'S1 calibration lineage')
        xr, xq, rn, qn = design(data, family)
        need(rn == records[1]['designs'][family]['r_names'] and qn == records[1]['designs'][family]['q_names'], 'S1 feature order')
        r, cp, _, lg = predict(data, xr, xq, at, model)
        first.extend(decode(b, data['available'], r, cp, lg, cost) for cost in COSTS)
    first = np.stack(first, 1)
    third = []
    for rep in ('group_audit', 'acoustic58'):
        assignment = [a for a in records[3]['assignments'] if a['representation'] == rep and a['role'] == 'outer' and a['test_piece'] == piece]
        need(len(assignment) == 1, 'original S3 assignment')
        x, names = nonlinear_features(data, rep)
        need(names == records[3]['features'][rep], 'S3 feature order')
        rt, qt = tree_factors(data, x, at, states[3][assignment[0]['model']])
        for alpha in BLENDS:
            r, cp, _, lg = blended(data, at, rt, qt, alpha)
            third.extend(decode(b, data['available'], r, cp, lg, cost) for cost in COSTS)
    third = np.stack(third, 1)
    combined = np.column_stack([first, third])
    route2 = records[2][index]; route3 = records['router3'][index]
    need(route2['piece'] == route3['piece'] == piece, 'frozen router identity')
    p2 = np.asarray(route2['source_count90']['policies'])[b-2]
    p3 = np.asarray(route3['source_count90']['policies'])[b-2]
    xr, xq, _, _ = design(data, 'identity')
    r, cp, _, _ = predict(data, xr, xq, at, dict(r=None, q=None))
    m = cp[at, old]; prior = np.column_stack([m, r, np.maximum(0, 1-m-r)])
    changed = old != b; prior[~changed] = [0, 0, 1]
    fourth = []
    for rep in ('group', 'acoustic58'):
        x = np.column_stack([prior, np.eye(7)[b][:, 2:5], np.eye(7)[old][:, 2:7], data['group_context'][at, old]])
        if rep == 'acoustic58':
            x = np.column_stack([x, data['features']])
        assignment = [a for a in records[4]['assignments'] if a['representation'] == rep and a['role'] == 'outer' and a['test_piece'] == piece]
        need(len(assignment) == 1, 'original S4 assignment')
        model = states[4][assignment[0]['model']]
        empirical = np.zeros((len(b), 3)); empirical[:, model.classes_.astype(int)] = model.predict_proba(x)
        empirical[~changed] = [0, 0, 1]
        for weight in WEIGHTS:
            q = (1-weight)*prior+weight*empirical
            fourth.extend(np.where(changed & (q[:, 0] > cost*q[:, 1]), old, b) for cost in COSTS)
    fourth = np.stack(fourth, 1)
    champions = np.column_stack([first[:, 37], first[at, p2], combined[at, p3], fourth[:, 53]])
    return champions.astype(np.int8), (first, third, fourth)


def model_records(root):
    result = {1: json.loads((root/REGRESSION/'series1/splits.json').read_text())}
    for k in (3, 4):
        result[k] = json.loads((root/REGRESSION/f'series{k}/model-records.json').read_text())
    result[2] = json.loads((root/REGRESSION/'series2/router-proofs.json').read_text())
    result['router3'] = json.loads((root/REGRESSION/'series3/router-proofs.json').read_text())
    return result


def run(root, runtime, output, external=None, s5=None, limit=0, cached=False, historical_proof=None):
    need(not output.exists(), 'refuse to overwrite missing-family evidence')
    proof = json.loads((runtime/'report.json').read_text())
    need(sha(runtime/'specialists.joblib') == proof['reconstructed_models_sha256'], 'verified specialist bytes')
    need(sha(root/PREPARED) == proof['original_inputs_sha256'], 'historical input bytes')
    d = read(root/PREPARED, ('global_index', 'baseline', 'fold', 'features', 'feature_names',
             'native_position', 'piece', 'proposal', 'group_context', 'z', 'logits', 'original'))
    f = read(root/'analysis/evidence/v273-recovered-255/inputs/historical_fundamental58.npz', ('global_index', 'fundamental'))
    need(np.array_equal(f['global_index'], d['global_index']), 'fundamental ordering')
    xhist, rawhist = matrices(d['features'], f['fundamental'], list(map(str, d['feature_names'])))
    original = runtime/'original'
    reference_truth = read(original/'specialists/producers.npz', ('truth',))['truth']
    frozen = FrozenProducerPool({}, reference_truth, d['baseline'], d['fold'], d['global_index'], original/'specialists')
    corrector = FrozenCorrector(reference_truth, d['baseline'], d['fold'], d['global_index'], original/'learned_corrector')
    catalogue = CataloguePool(frozen, corrector)
    corrector_records = {tuple(r['train_folds']): r for r in corrector.manifest['producers']}
    weights = joblib.load(runtime/'specialists.joblib')
    states, records, nets = checked_models(root), model_records(root), neural_weights(root)
    historical = external is None
    if historical:
        targets = {k: d[k] for k in ('features', 'feature_names', 'baseline', 'global_index')}
        target_raw = rawhist
        expected = {}
        for series in (1, 2, 3, 4):
            p = read(root/REGRESSION/f'series{series}/predictions.npz', ('predictions', 'variant_ids'))
            expected[series] = p['predictions'][d['native_position']]
            expected['names'+str(series)] = list(map(str, p['variant_ids']))
        old = read(root/'analysis/evidence/v273-audit-memory/variants/all-variant-decisions.npz', ('predictions', 'variant_ids'))
        expected['coherent'] = old['predictions'][d['native_position'], list(map(str, old['variant_ids'])).index('repair_coherent')]
        archive = read(root/RISK/'probability-evidence.npz', ('parent_baseline_correct_probability', 'candidate_pooled_class_logits'))
    else:
        need(not cached and limit == 0 and s5 is not None, 'external inference requires real producers and complete inputs')
        need(historical_proof is not None, 'complete historical replay proof required before new-audio inference')
        verified = json.loads(historical_proof.read_text())
        need(verified['status'] == 'HISTORICAL_PASS' and verified['complete_historical_cohort'] and
             not verified['cached_receiver_producers'] and verified['script_sha256'] == sha(Path(__file__)),
             'historical replay incomplete, cached-only, or from another runtime')
        with np.load(external, allow_pickle=False) as z:
            need(set(z.files) == {'row_id', 'features', 'fundamental', 'feature_names'}, 'new-audio label-free allowlist')
            full = {k: z[k].copy() for k in z.files}
        source = read(s5, ('row_id', 'freeze_K'))
        need(np.array_equal(source['row_id'], full['row_id']), 'external S5 row alignment')
        need(np.array_equal(d['feature_names'], full['feature_names']), 'acoustic feature ordering')
        eligible = np.flatnonzero(np.isin(source['freeze_K'], (2, 3, 4)))
        targets = dict(features=full['features'][eligible], feature_names=full['feature_names'],
                       baseline=source['freeze_K'][eligible].astype(int), global_index=-np.arange(1, len(eligible)+1))
        xnew, target_raw = matrices(targets['features'], full['fundamental'][eligible], list(map(str, full['feature_names'])))
    output.mkdir(parents=True)
    reports, branch_predictions, branch_names, branch_folds, risk_blocks, conditional_blocks = [], [], [], [], [], []
    historical_predictions = np.full((len(d['baseline']), 5), -1, np.int8)
    for outer in FOLDS:
        train = np.flatnonzero(d['fold'] != outer)
        at = np.flatnonzero(d['fold'] == outer) if historical else np.arange(len(targets['baseline']))
        if limit:
            at = at[:limit]
        base, raw = targets['baseline'][at], target_raw[at]
        key = tuple(f for f in FOLDS if f != outer)
        if cached:
            receiver_votes = catalogue.predict(key, at, {outer})
            p = frozen.predict(key, at, {outer})
        else:
            xx = {name: value[at] for name, value in (xhist if historical else xnew).items()}
            receiver_votes, p, _ = votes(weights[cache_key(key)], xx, raw, base, original/'learned_corrector', corrector_records[key])
        proposal, desc, _ = joint_groups(receiver_votes, base)
        ref_votes, _ = catalogue.crossfit(train, {outer})
        ref_proposal = joint_groups(ref_votes, d['baseline'][train])[0]
        audits = local_audits(rawhist[train], raw, ref_proposal, reference_truth[train], d['baseline'][train],
                              proposal, base, d['global_index'][train], targets['global_index'][at])
        scaler = StandardScaler().fit(rawhist[train])
        context = np.clip(scaler.transform(raw), -6, 6).astype(np.float32)
        inputs = dict(group_features=np.concatenate([desc, audits], -1), member_votes=receiver_votes,
                      baseline=np.eye(7, dtype=np.float32)[base], context=context, proposal=proposal)
        r, _ = critic(inputs, nets['parent', outer])
        _, lg = critic(inputs, nets['contextual', outer], True)
        data = branch_data(targets['features'][at], base, proposal, audits, r, lg, targets['feature_names'])
        ref_p, _ = frozen.crossfit(train, {outer})
        cx = coherent_inputs(ref_p, rawhist[train], reference_truth[train], d['baseline'][train], p, raw, base, context)
        cp = coherent_forward(cx, nets['coherent', outer])
        candidate = cp[:, 2:].argmax(1)+2
        coh = np.where(cp[np.arange(len(base)), candidate] > cp[np.arange(len(base)), base], candidate, base)
        report = dict(outer_fold=outer, receiver_events=len(at), reference_events=len(train),
                      receiver_labels_used=False, original_coherent_tensor_shape=list(cx['subset_features'].shape))
        if historical:
            report.update(original_255_mismatches=int(np.count_nonzero(proposal != d['proposal'][at])),
                original_20D_max_error=float(np.max(np.abs(data['group_context']-d['group_context'][at]))),
                baseline_risk_probability_max_error=float(np.max(np.abs(r-archive['parent_baseline_correct_probability'][at]))),
                conditional_logit_max_error=float(np.max(np.abs(lg-archive['candidate_pooled_class_logits'][at]))),
                original_crossed_factor_decision_mismatches=int(np.count_nonzero(data['original'] != d['original'][at])),
                coherent_decision_mismatches=int(np.count_nonzero(coh != expected['coherent'][at])))
        policy_checks = {1: 0, 3: 0, 4: 0}
        champion_mismatches = np.zeros(4, int)
        champion_checks = 0
        for block in records[1]['blocks']:
            if block['fold'] != outer:
                continue
            local = np.flatnonzero(d['piece'][at] == block['piece']) if historical else np.arange(len(at))
            if not len(local):
                continue
            names = ('feature_names', 'group_names')
            subset = {k: v if k in names else v[local] for k, v in data.items()}
            champions, policies = family_policies(subset, block, states, records)
            if historical:
                source_at = at[local]
                for s, policy in zip((1, 3, 4), policies):
                    policy_checks[s] += int(np.count_nonzero(policy != expected[s][source_at, :policy.shape[1]]))
                reference = np.column_stack([expected[1][source_at, 37], expected[2][source_at, 0],
                    expected[3][source_at, expected['names3'].index('tree_nested_source_count90')], expected[4][source_at, 53]])
                champion_mismatches += np.count_nonzero(champions != reference, axis=0)
                champion_checks += champions.size
                historical_predictions[source_at] = np.column_stack([champions, coh[local]])
            else:
                native = np.broadcast_to(source['freeze_K'][:, None], (len(full['row_id']), 5)).copy()
                native[eligible] = np.column_stack([champions, coh])
                branch_predictions.append(native); branch_names.append(block['piece']); branch_folds.append(outer)
        if historical:
            report.update(raw_policy_mismatches=policy_checks, champion_mismatches=champion_mismatches.tolist(),
                          champion_decisions_checked=champion_checks)
        else:
            risk_blocks.append(r); conditional_blocks.append(lg)
        reports.append(report)
        print(json.dumps(report), flush=True)
        write_json(output/'progress.json', reports)
    if historical:
        checks = sum(r['champion_decisions_checked'] for r in reports)
        complete = bool(checks == 4*7493 and np.all(historical_predictions >= 0))
        failed = any(r['original_255_mismatches'] or r['original_crossed_factor_decision_mismatches'] or
                     r['coherent_decision_mismatches'] or any(r['champion_mismatches']) or
                     any(r['raw_policy_mismatches'].values()) or r['original_20D_max_error'] > 2e-5 or
                     r['baseline_risk_probability_max_error'] > 2e-6 or
                     r['conditional_logit_max_error'] > 2e-5 for r in reports)
        np.savez_compressed(output/'historical_five_families.npz', global_index=d['global_index'],
                            family_ids=np.asarray(CHAMPIONS), predictions=historical_predictions)
        report = dict(status='HISTORICAL_PASS' if not failed else 'HISTORICAL_MISMATCH',
                      complete_historical_cohort=complete, cached_receiver_producers=cached,
                      historical_events=sum(r['receiver_events'] for r in reports),
                      raw_policy_decisions_checked=sum(r['receiver_events'] for r in reports)*(49+42+56),
                      five_champion_decisions_checked=checks+sum(r['receiver_events'] for r in reports),
                      family_ids=list(CHAMPIONS), script_sha256=sha(Path(__file__)),
                      original_prepared_sha256=sha(root/PREPARED),
                      reconstructed_models_sha256=sha(runtime/'specialists.joblib'),
                      output_sha256=sha(output/'historical_five_families.npz'),
                      predictive_models_fitted=False, receiver_labels_used_for_features=False,
                      original_reference_audit_labels_used=True, folds=reports, S80_promoted=False)
        write_json(output/'report.json', report)
        need(not failed, 'original family replay mismatch; see saved report')
    else:
        need(len(branch_predictions) == 19, 'nineteen original held-piece branches required')
        np.savez_compressed(output/'external_five_families.npz', row_id=full['row_id'],
            branch_piece=np.asarray(branch_names), branch_fold=np.asarray(branch_folds), family_ids=np.asarray(CHAMPIONS),
            predictions=np.asarray(branch_predictions), eligible_positions=eligible, fold_ids=np.asarray(FOLDS),
            baseline_correct_probability=np.asarray(risk_blocks), conditional_logits=np.asarray(conditional_blocks))
        report = dict(status='FIVE_ORIGINAL_FAMILIES_EXECUTED_ALL_NINETEEN_BRANCHES', external_events=len(full['row_id']),
            eligible_events=len(eligible), branches=len(branch_predictions), family_ids=list(CHAMPIONS), folds=reports,
            external_input_sha256=sha(external), source_S5_sha256=sha(s5),
            output_sha256=sha(output/'external_five_families.npz'), new_audio_labels_read=False,
            historical_proof_sha256=sha(historical_proof), script_sha256=sha(Path(__file__)),
            predictive_models_fitted=False, outer_reference_audit_labels_used=True,
            cross_branch_aggregation_selected=False, full_S45_16_families_complete=False, S80_promoted=False)
        write_json(output/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--external', type=Path)
    parser.add_argument('--s5', type=Path)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--cached-producers', action='store_true')
    parser.add_argument('--historical-proof', type=Path)
    args = parser.parse_args()
    run(args.root, args.runtime, args.out, args.external, args.s5, args.limit, args.cached_producers, args.historical_proof)
