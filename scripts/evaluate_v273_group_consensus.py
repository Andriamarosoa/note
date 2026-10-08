"""Controlled destination pooling and categorical comparison on exposed folds.

Both modes reuse exactly the archived global-audit inputs and producer cache.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from scripts.audit_v273_selector_design import aligned_positions, load_feature_rows
from scripts.v273_selector_contract import ProducerPool, matrices, require, FOLDS
from scripts.v273_group127_contract import (
    CANDIDATES, GROUP_MASKS, GROUP_SIZES, DIRECT_DIM, NEIGHBORS,
    assemble_group_outer, choose_groups, outcome_labels,
)
from scripts.learn_v273_group_consensus import train_consensus, MODES
from scripts.prepare_v273_group127_producers import FrozenProducerPool
from scripts.yourmt3_exactk_common import metrics, paired, digest


def render(report):
    a, b = report['freeze_reference'], report['candidate']
    c = report['archived_coherent']
    p = report['paired']['global']
    rows = [f"# V27.3 — group127 / {report['arm']}", '',
        'Folds 0/1/2/4 exposés ; aucune validation indépendante ni promotion.', '',
        '| Système | Exact-K global | Exact-K poly | Net face à freeze |',
        '|---|---:|---:|---:|',
        f"| freeze_local_combo | {100*a['exact']:.4f}% | {100*a['poly']['exact']:.4f}% | 0 |",
        f"| coherent archivé | {100*c['exact']:.4f}% | {100*c['poly']['exact']:.4f}% | +31 |",
        f"| {report['arm']} | {100*b['exact']:.4f}% | {100*b['poly']['exact']:.4f}% | {p['net']:+d} |", '',
        f"Corrections {p['corrections']} ; régressions {p['regressions']}.", '',
        '| Vrai K | Référence | Variante | Net |', '|---|---:|---:|---:|']
    for k in range(7):
        rows.append(f"| K{k} | {100*a['by_k'][str(k)]['exact']:.4f}% | {100*b['by_k'][str(k)]['exact']:.4f}% | {report['paired']['by_k'][str(k)]['net']:+d} |")
    rows += ['', '| Fold | Net |', '|---|---:|']
    rows += [f"| {f} | {report['folds'][str(f)]['paired']['global']['net']:+d} |" for f in FOLDS]
    rows += ['', 'Les résultats avec seulement les singletons réutilisent le même réseau entraîné sur tous les groupes ; ce ne sont pas des modèles réentraînés.', '',
        f"Net du décodage limité aux singletons : {report['singletons_only']['paired']['global']['net']:+d}.",
        f"Groupes sélectionnés par taille : {report['selected_group_sizes']}.",
        f"Synergies descriptives : {report['synergy']}.",
        f"Gain annoncé : {report['selected_expected_net']:.3f} ; observé : {p['net']:+d}."]
    return '\n'.join(rows)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--archived-global', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--producer-cache', type=Path, required=True)
    args = parser.parse_args()
    args.arm = args.mode
    require(not args.output.exists(), 'refusing overwrite')
    root = Path(__file__).resolve().parents[1]
    comparison = json.loads((root/'analysis/evidence/v273-group127/verification.json').read_text())['arms']['global']
    require(digest(args.archived_global/'predictions.npz') == comparison['predictions_sha256'], 'archived group127 global changed')
    with np.load(args.archived_global/'predictions.npz', allow_pickle=False) as z:
        archived = {k: z[k] for k in z.files}
    old_verification = json.loads((root/'analysis/evidence/v273-selector-repair/verification.json').read_text())
    require(digest(args.reference) == old_verification['arms']['coherent']['predictions_sha256'], 'native archive changed')
    with np.load(args.reference, allow_pickle=False) as z:
        ids, y, base, folds, eligible_ids, old = (z[k] for k in ['global_index','true_K','frozen_baseline_K','fold','eligible_global_index','predicted_K'])
    for key, value in [('global_index',ids),('true_K',y),('frozen_baseline_K',base),('fold',folds),('eligible_global_index',eligible_ids)]:
        require(np.array_equal(archived[key],value), 'archived comparison alignment '+key)
    pos = aligned_positions(ids, eligible_ids)
    require(len(ids) == 59309 and len(pos) == 7493 and set(folds) == set(FOLDS), 'native coverage')
    require(int(np.sum(y == base)) == 48454 and int(np.sum((y == base) & (y >= 2))) == 2530, 'native reference')
    require(metrics(y, old) == old_verification['arms']['coherent']['candidate'], 'archived coherent metrics')
    rows, _ = load_feature_rows(args.features, ids, y, base, folds, pos)
    recording_folds = {}
    for r in rows:
        name = r['recording_id']
        require(name[:2] in ('00','01','02','03','04'), 'player05 present')
        require(name not in recording_folds or recording_folds[name] == r['fold'], 'recording in multiple folds')
        recording_folds[name] = r['fold']
    x, raw, context_names = matrices(rows)
    yc, bc, fc = y[pos], base[pos], folds[pos]
    if args.producer_cache is not None:
        require((args.producer_cache/'source-reference-sha256.txt').read_text().strip() == digest(args.reference), 'cache reference mismatch')
        pool = FrozenProducerPool(x, yc, bc, fc, eligible_ids, args.producer_cache)
    else:
        pool = ProducerPool(x, yc, bc, fc, eligible_ids)
    prediction = base.copy(); singleton_prediction = base.copy()
    selection = np.full(len(pos), -1, np.int32)
    single_selection = np.full_like(selection, -1)
    proposals = np.full((len(pos), 127), -1, np.int32)
    probability = np.full((len(pos), 127, 3), np.nan, np.float32)
    gain = np.full((len(pos), 127), np.nan, np.float32)
    baseline_probability = np.full(len(pos), np.nan, np.float32)
    votes = np.full((len(pos), 7, 5), np.nan, np.float32)
    audits = np.full((len(pos), 127, 9), np.nan, np.float32)
    class_probability = np.full((len(pos),7), np.nan, np.float32)
    other_probability = np.full(len(pos), np.nan, np.float32)
    pooled_logits = np.full((len(pos),7), np.nan, np.float32)
    fold_reports = {}; manifests = []
    args.output.mkdir(parents=True)
    for outer in FOLDS:
        print(json.dumps(dict(arm=args.arm, fold=outer, stage='construct_independent_group_audits')), flush=True)
        train, held, tr, val, manifest = assemble_group_outer(pool, raw, outer)
        manifests.append(manifest)
        proposals[val] = held['proposal']; votes[val] = held['member_votes']
        audits[val] = held['group_features'][..., DIRECT_DIM:]
        for key, value in [('member_votes',held['member_votes']),('group_proposal',held['proposal']),('local_audit_features',audits[val])]:
            require(np.array_equal(value,archived[key][val]), 'inputs differ from archived global: '+key)
        local_columns = [DIRECT_DIM+j for j in (3,4,5,7,8)]
        train['group_features'][..., local_columns] = 0
        held['group_features'][..., local_columns] = 0
        print(json.dumps(dict(arm=args.arm, fold=outer, stage='train_127_groups', epochs=30)), flush=True)
        pred, chosen, out, history, model = train_consensus(train, yc[tr], bc[tr], held, args.mode)
        singleton, singleton_choice = choose_groups(out['expected_gain'], held['proposal'], bc[val], singletons=True)
        prediction[pos[val]] = pred; singleton_prediction[pos[val]] = singleton
        selection[val] = chosen; single_selection[val] = singleton_choice
        probability[val] = out['outcome_probability']; gain[val] = out['expected_gain']
        baseline_probability[val] = out['baseline_correct']
        class_probability[val] = out['class_probability']
        other_probability[val] = out['other_probability']
        pooled_logits[val] = out['pooled_class_logits']
        require(model.count_params()==12994, 'parameter budget changed')
        model.save_weights(str(args.output/f'fold-{outer}.weights.h5'))
        result = dict(eligible_rows=len(val), parameters=int(model.count_params()), training=history,
            paired=paired(yc[val], bc[val], pred),
            selected_expected_net=float(np.where(chosen > 0, out['expected_gain'][np.arange(len(val)), np.maximum(chosen-1, 0)], 0).sum()))
        fold_reports[str(outer)] = result
        (args.output/f'fold-{outer}-result.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
        print(json.dumps(dict(arm=args.arm, fold=outer, stage='evaluated', paired=result['paired']['global'], history=history)), flush=True)
        del train, held, model, out
        tf.keras.backend.clear_session(); gc.collect()
    require(all(np.isfinite(a).all() for a in [probability,gain,baseline_probability,votes,audits,class_probability,other_probability,pooled_logits]), 'missing output')
    require(np.allclose(probability.sum(2), 1, atol=1e-6) and (probability >= -1e-7).all(), 'outcome normalization')
    reproduced, chosen = choose_groups(gain, proposals, bc)
    require(np.array_equal(reproduced, prediction[pos]) and np.array_equal(chosen, selection), 'decoder mismatch')
    active = np.zeros(len(y), bool); active[pos] = True
    require(np.array_equal(prediction[~active], base[~active]), 'coverage expanded')
    labels = outcome_labels(proposals, yc, bc)
    single_cols = np.array([2**i-1 for i in range(7)])
    individually_right = proposals[:, single_cols] == yc[:, None]
    no_right_member = np.einsum('nh,sh->ns', individually_right.astype(np.float32), GROUP_MASKS) == 0
    synergy_groups = (proposals == yc[:, None]) & no_right_member & (GROUP_SIZES[None] >= 2)
    has_synergy = synergy_groups.any(1)
    picked_synergy = synergy_groups[np.arange(len(pos)), np.maximum(selection-1, 0)] & (selection > 0)
    oracle_fix = (bc != yc) & (proposals == yc[:, None]).any(1)
    group_results = []
    for j in range(127):
        selected = selection == j+1
        group_results.append(dict(mask=j+1, size=int(GROUP_SIZES[j]),
            corrections=int(np.sum(labels[:,j] == 0)), regressions=int(np.sum(labels[:,j] == 1)), neutral=int(np.sum(labels[:,j] == 2)),
            selected=int(selected.sum()), selected_corrections=int(np.sum(selected & (labels[:,j] == 0))),
            selected_regressions=int(np.sum(selected & (labels[:,j] == 1))),
            synergy_rows=int(synergy_groups[:,j].sum())))
    unique = [len(set(row.tolist()) | {int(b)}) for row,b in zip(proposals,bc)]
    report = dict(status='completed', arm=args.arm, promotion=False, independent_validation=False,
        freeze_reference=metrics(y,base), archived_coherent=metrics(y,old), candidate=metrics(y,prediction),
        paired=paired(y,base,prediction), paired_vs_coherent=paired(y,old,prediction), folds=fold_reports,
        paired_vs_archived_global=paired(y,archived['predicted_K'],prediction),
        paired_vs_archived_singletons=paired(y,archived['singletons_only_K'],prediction),
        singletons_only=dict(metrics=metrics(y,singleton_prediction), paired=paired(y,base,singleton_prediction),
            comparison_with_all_groups=paired(y,singleton_prediction,prediction), retrained=False),
        selected_group_sizes={str(size):int(np.sum((selection>0) & (GROUP_SIZES[np.maximum(selection-1,0)] == size))) for size in range(1,8)},
        keep_rows=int(np.sum(selection == 0)),
        selected_expected_net=float(np.where(selection>0, gain[np.arange(len(pos)), np.maximum(selection-1,0)],0).sum()),
        groups=group_results,
        synergy=dict(events_with_correct_group_and_all_its_members_wrong=int(has_synergy.sum()),
            among_initially_wrong=int(np.sum(has_synergy & (bc!=yc))), among_initially_correct=int(np.sum(has_synergy & (bc==yc))),
            selected_correct_joint_groups_with_all_members_wrong=int(picked_synergy.sum())),
        oracle=dict(correctable_errors_with_some_group=int(oracle_fix.sum()),
            global_exact_upper_bound=(48454+int(oracle_fix.sum()))/59309,
            poly_exact_upper_bound=(2530+int(oracle_fix.sum()))/7385,
            true_k_counts={str(k):int(np.sum(oracle_fix & (yc==k))) for k in range(7)},
            mean_distinct_verdicts_per_event=float(np.mean(unique)), not_a_prediction=True),
        configuration=dict(mode=args.mode, audit_arm='global', pool='mean conditional logits by proposed K before probability',
            other='fixed zero logit for no available changing proposal correct' if args.mode=='pooled_ce' else 'not modeled in BCE control',
            parameter_budget=12994, archived_group127_run=37777844182,
            candidates=list(CANDIDATES), candidate_count=7, groups=127, extra_KEEP=1,
            optional_frozen_action=True, individual_veto=False,
            fusion='fixed arithmetic mean of candidate vote vectors; tie keeps frozen class',
            seventh_candidate='mean of the same five specialists, not new independent evidence',
            H0='deterministic action encoding; not probability of correctness',
            context_names=context_names, local_neighbors=NEIGHBORS, local_audit_strength=12,
            producer_cache_sha256=getattr(pool,'cache_sha256',None),
            global_audits_enabled=True, local_audits_enabled=False, descriptor_dimensions=64,
            epochs=30, seed=27402, learning_rate=.002, batch_size=192,
            objective='baseline BCE plus '+('original per-event averaged group BCE' if args.mode=='pooled_bce' else 'one conditional categorical CE per baseline-wrong event'),
            shared_baseline_correctness=True, no_posthoc_threshold=True,
            native_rows=59309, eligible_rows=7493, folds=list(FOLDS), player05_used=False, fold3_used=False),
        provenance=manifests, expert_producers=list(pool.manifests.values()),
        limitations=[
            'Development-exposed folds, not independent validation; no promotion.',
            'Seven candidates include one derived mean and may carry redundant evidence; 127 masks are not 127 distinct verdicts.',
            'Fusion is fixed to define honest joint-outcome labels; nonlinear neural learning estimates full-group reliability, not a learned fusion rule.',
            'Both modes use archived global audit features. BCE isolates mean logit pooling; CE additionally changes normalization and conditional likelihood.',
            'The selected mask is only the first representative of the winning K; equal-K groups have identical scores and this mask is not a causal attribution.',
            'No constituent pruning; only the complete group score competes against KEEP.',
            'Proper outcome probabilities and a shared baseline risk do not establish empirical calibration after maximizing across groups.',
            'Full-group oracle and synergy counts use evaluation labels for diagnosis only, never selection.',
            'Only initial K2/K3/K4 can change; sound summaries retain +160ms lookahead.',
        ])
    (args.output/'report.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n')
    (args.output/'report.md').write_text(render(report))
    np.savez_compressed(args.output/'predictions.npz',global_index=ids,true_K=y,frozen_baseline_K=base,
        predicted_K=prediction,fold=folds,eligible_global_index=eligible_ids,
        chosen_group_mask=selection,singletons_only_K=singleton_prediction,singleton_group_mask=single_selection,
        group_proposal=proposals,group_outcome_probability=probability,group_expected_gain=gain,
        baseline_correct_probability=baseline_probability,member_votes=votes,local_audit_features=audits,
        class_probability=class_probability,other_probability=other_probability,pooled_class_logits=pooled_logits)
    print(render(report),flush=True)
    print(json.dumps(dict(mode=args.mode,vs_archived_global=report['paired_vs_archived_global']['global'],vs_archived_singletons=report['paired_vs_archived_singletons']['global'])),flush=True)


if __name__ == '__main__':
    main()
