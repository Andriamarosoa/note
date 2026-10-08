"""Preserve measured policy variants and every event outcome without picking a winner.

This is evidence, not an inference oracle. A downstream learner must regenerate
predictions with all of its forbidden labels excluded from the producers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.verify_v273_selector_repair_artifacts import digest, metrics, named_pairs, require

CORE_SOURCES = [
    ('repair_legacy','selector-repair-results/legacy',None),
    ('repair_contracts','selector-repair-results/contracts','repair_legacy'),
    ('repair_coherent','selector-repair-results/coherent','repair_contracts'),
    ('group127_initial_direct','group127-results/direct','repair_coherent'),
    ('group127_initial_global','group127-results/global','group127_initial_direct'),
    ('group127_initial_local','group127-results/local','group127_initial_global'),
    ('group127_direct','group127-cached-results/direct','repair_coherent'),
    ('group127_global','group127-cached-results/global','group127_direct'),
    ('group127_local','group127-cached-results/local','group127_global'),
    ('consensus_bce','group-consensus-results/pooled_bce','group127_global'),
    ('consensus_ce','group-consensus-results/pooled_ce','consensus_bce'),
    ('catalogue7_global','catalogue-results/control7','consensus_ce'),
    ('catalogue8_global','catalogue-results/corrector8','catalogue7_global'),
    ('catalogue7_calibrated','calibration-results/control7','catalogue7_global'),
    ('catalogue8_calibrated','calibration-results/corrector8','catalogue8_global'),
    ('catalogue7_local','audit-memory-results/control7','catalogue7_global'),
    ('catalogue8_local','audit-memory-results/corrector8','catalogue8_global'),
]


def read(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def preserve_variants(ids, truth, baseline, variants):
    """Never filter a candidate by net gain, copy over its parent, or pick by truth."""
    require(len(set(ids)) == len(ids), 'unique event IDs')
    keys = [v['variant_id'] for v in variants]
    require(len(set(keys)) == len(keys), 'candidate identity collision')
    predictions = np.column_stack([v['prediction'] for v in variants]).astype(np.int8)
    require(predictions.shape == (len(ids), len(variants)) and np.isin(predictions, range(7)).all(), 'aligned native K')
    outcomes = ((predictions == truth[:, None]).astype(np.int8)
                - (baseline == truth).astype(np.int8)[:, None])
    entries = []
    for j, v in enumerate(variants):
        parent = v.get('parent_id')
        before = baseline if parent is None else predictions[:, keys.index(parent)]
        entry = {k: value for k, value in v.items() if k != 'prediction'}
        entry.update(retained=True, retention_depends_on_global_gain=False,
            candidate_kind='measured_policy_candidate', downstream_ready=False,
            required_for_downstream_activation='Aligned nested predictions excluding the downstream receiver from every producer and audit.',
            metrics=metrics(truth, predictions[:, j]),
            versus_freeze=named_pairs(truth, baseline, predictions[:, j]),
            versus_parent=named_pairs(truth, before, predictions[:, j]),
            prediction_values_sha256=hashlib.sha256(np.ascontiguousarray(predictions[:, j]).tobytes()).hexdigest())
        entries.append(entry)
    union = (baseline != truth) & np.any(predictions == truth[:, None], axis=1)
    return entries, predictions, outcomes, union


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts-root', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    require(not args.output.exists(), 'refusing to overwrite existing memory')
    variants = []; native = None; source_files = {}; source_predictions = {}
    for name, directory, parent in CORE_SOURCES:
        path = args.artifacts_root/directory/'predictions.npz'
        require(path.exists(), 'missing source; never silently skip '+str(path))
        d = read(path)
        if native is None:
            native = {k: d[k] for k in ['global_index','true_K','frozen_baseline_K','fold','eligible_global_index']}
        for key in ('global_index','true_K','frozen_baseline_K','fold'):
            require(np.array_equal(d[key], native[key]), 'source alignment '+name+'/'+key)
        require(set(d['eligible_global_index']) == set(native['eligible_global_index']), 'source eligible coverage')
        source_files[name] = dict(path=directory+'/predictions.npz', sha256=digest(path))
        source_predictions[name] = d['predicted_K']
        v = dict(variant_id=name, parent_id=parent, prediction=d['predicted_K'],
                 source=name, source_field='predicted_K',
                 provenance_scope='supervised_calibration_other_pieces_in_excluded_fold' if name.endswith('_calibrated')
                 else 'archived_development_variant_producer_scope_must_be_rechecked_for_each_downstream_use')
        variants.append(v)
        for key in sorted(d):
            if key.endswith('_K') and key not in ('true_K','frozen_baseline_K','predicted_K','original_K') and d[key].shape == native['true_K'].shape:
                variants.append(dict(v, variant_id=name+'__'+key, parent_id=name,
                                     source_field=key, prediction=d[key]))
    ids, y, b, f, eligible = (native[k] for k in ['global_index','true_K','frozen_baseline_K','fold','eligible_global_index'])
    entries, predictions, outcomes, union = preserve_variants(ids, y, b, variants)
    # Duplicate predictions are documented aliases, never evidence of independent experts.
    first_hash = {}
    for entry in entries:
        sha = entry['prediction_values_sha256']
        entry['identical_prediction_alias_of'] = first_hash.get(sha)
        first_hash.setdefault(sha, entry['variant_id'])
    lookup = {}; features_sources = []
    for path in sorted(args.features.rglob('rows.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line); gid = row['global_index']
            require(gid not in lookup, 'duplicate context event')
            lookup[gid] = row
        features_sources.append(dict(file=path.name, fold=int(row['fold']), sha256=digest(path)))
    require(set(eligible) == set(lookup), 'context coverage')
    pos_lookup = {int(v): i for i, v in enumerate(ids)}
    pos = np.array([pos_lookup[int(v)] for v in eligible])
    names = sorted(lookup[int(eligible[0])]['features'])
    context = np.array([[lookup[int(g)]['features'][name] for name in names] for g in eligible], dtype=np.float64)
    require(np.isfinite(context).all(), 'finite original feature values')
    for i, g in enumerate(eligible):
        row = lookup[int(g)]
        require((row['true_k'], row['baseline_k'], row['fold']) == (y[pos[i]], b[pos[i]], f[pos[i]]), 'context labels')
    args.output.mkdir(parents=True)
    files = {}
    # Split into modest durable files; original float64 values are preserved exactly.
    for fold in (0,1,2,4):
        at = np.flatnonzero(f[pos] == fold)
        for part, start in enumerate(range(0, len(at), 512)):
            selected = at[start:start+512]
            file = args.output/f'context-fold-{fold}-part-{part}.npz'
            np.savez_compressed(file, global_index=eligible[selected], feature_names=np.array(names), values=context[selected],
                recording_id=np.array([lookup[int(g)]['recording_id'] for g in eligible[selected]]),
                start_sample=np.array([lookup[int(g)]['start_sample'] for g in eligible[selected]]))
            files[file.name] = dict(sha256=digest(file), rows=len(selected), purpose='join by global_index; observables only')
    file = args.output/'all-variant-decisions.npz'
    np.savez_compressed(file, **native, variant_ids=np.array([v['variant_id'] for v in entries]),
                        predictions=predictions, outcomes_vs_freeze=outcomes)
    files[file.name] = dict(sha256=digest(file), rows=len(ids), columns=len(entries), purpose='audit labels separated from observable contexts')
    casefile = args.output/'complementary-cases.csv'
    fields = ['global_index','fold','recording_id','start_sample','true_K','frozen_K','variants_correct','variants_wrong']
    with casefile.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n'); writer.writeheader()
        for i in np.flatnonzero(union):
            row = lookup[int(ids[i])]
            writer.writerow(dict(global_index=int(ids[i]), fold=int(f[i]), recording_id=row['recording_id'],
                start_sample=row['start_sample'], true_K=int(y[i]), frozen_K=int(b[i]),
                variants_correct='|'.join(str(j) for j in np.flatnonzero(predictions[i] == y[i])),
                variants_wrong='|'.join(str(j) for j in np.flatnonzero(predictions[i] != y[i]))))
    files[casefile.name] = dict(sha256=digest(casefile), rows=int(union.sum()), variant_index_order='same as registry variants')
    comparisons = []
    key_index = {e['variant_id']: i for i, e in enumerate(entries)}
    for left, right in [('catalogue8_global','catalogue8_calibrated'),('catalogue8_global','catalogue8_local'),
                        ('catalogue7_global','catalogue7_local'),('catalogue7_global','catalogue8_global')]:
        a, c = predictions[:, key_index[left]], predictions[:, key_index[right]]
        cora, corc = (b != y) & (a == y), (b != y) & (c == y)
        comparisons.append(dict(left=left,right=right,paired=named_pairs(y,a,c),
            corrections_left_only=int((cora & ~corc).sum()), corrections_right_only=int((corc & ~cora).sum()),
            corrections_both=int((cora & corc).sum()), corrections_union=int((cora | corc).sum()),
            union_is_oracle_diagnostic_not_prediction=True))
    registry = dict(schema='v273-preserved-policy-memory-v1', scope='17 explicitly enumerated available source archives, including all their native diagnostic outputs; not all historical models',
        network_candidates_already_active=8, retained_policy_candidates=len(entries), distinct_prediction_vectors=len(first_hash),
        native_rows=len(ids), eligible_rows=len(eligible), variants=entries, source_files=source_files,
        context_features=names, context_feature_count=len(names), context_sources=features_sources, files=files,
        paired_complements=comparisons, correctable_by_union=int(union.sum()),
        oracle_union_is_not_a_prediction=True, selection_by_true_label_forbidden=True,
        negative_global_net_candidates_preserved=sum(e['versus_freeze']['global']['net'] < 0 for e in entries),
        promotion=False, independent_validation=False)
    (args.output/'registry.json').write_text(json.dumps(registry, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({k:registry[k] for k in ['retained_policy_candidates','distinct_prediction_vectors','correctable_by_union','negative_global_net_candidates_preserved']}))


if __name__ == '__main__':
    main()
