"""Nested routing of complete measured policies; protect original success identities."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.evaluate_v273_regression_loops import read, dump, policy_stats, diagnostics
from scripts.v273_regression_loops import FAMILIES, COSTS, counts
from scripts.verify_v273_selector_repair_artifacts import digest, require

KINDS = ('source_count90', 'source_success90', 'transition_success90')


def sufficient_statistics(y, b, original, matrix):
    correct = (matrix == y[:, None]) & (b != y)[:, None]
    regression = (matrix != b[:, None]) & (b == y)[:, None]
    old_success = (original == y) & (b != y)
    return np.stack([correct.sum(0), regression.sum(0),
                     (correct & old_success[:, None]).sum(0)], 1).astype(int)


def feasible(stat, reference, protected):
    return ((stat[..., 0] >= .9*reference[0]) & (stat[..., 1] <= reference[1]) &
            (stat[..., 0]-stat[..., 1] >= reference[0]-reference[1]) &
            ((stat[..., 2] >= .9*reference[2]) if protected else True))


def key(stat):
    return int(stat[1]), int(stat[1]-stat[0]), -int(stat[2])


def source_search(y, b, original, matrix, protected):
    table = np.array([sufficient_statistics(y[b == k], b[b == k], original[b == k], matrix[b == k])
                      for k in (2, 3, 4)])
    m = matrix.shape[1]; choices = np.indices((m, m, m)).reshape(3, -1).T
    sums = table[0, choices[:, 0]]+table[1, choices[:, 1]]+table[2, choices[:, 2]]
    reference = sufficient_statistics(y, b, original, original[:, None])[0]
    valid = np.flatnonzero(feasible(sums, reference, protected))
    require(len(valid) > 0, 'original fallback must exist')
    order = np.lexsort((-sums[valid, 2], sums[valid, 1]-sums[valid, 0], sums[valid, 1]))
    at = valid[order[0]]
    return choices[at], dict(considered=len(choices), feasible=len(valid),
                            statistics=sums[at].tolist(), reference=reference.tolist())


def transition_search(y, b, original, matrix, cell, source_policies):
    labels = sorted(set(map(int, cell)))
    table = {c: sufficient_statistics(y[cell == c], b[cell == c], original[cell == c], matrix[cell == c]) for c in labels}
    chosen = {c: int(source_policies[c//7-2]) for c in labels}
    reference = sufficient_statistics(y, b, original, original[:, None])[0]
    movable = []
    for c in labels:
        at = cell == c; old = counts(y[at], b[at], original[at])
        if at.sum() >= 60 and old['corrections'] >= 10 and old['regressions'] >= 10: movable.append(c)
    total = sum(table[c][chosen[c]] for c in labels)
    history = []
    for _ in range(200):
        best = None
        for c in movable:
            alternatives = total-table[c][chosen[c]]+table[c]
            for j in np.flatnonzero(feasible(alternatives, reference, True)):
                if key(alternatives[j]) < key(total):
                    rank = (*key(alternatives[j]), c, int(j))
                    if best is None or rank < best[0]: best = rank, c, int(j), alternatives[j]
        if best is None: break
        _, c, j, new_total = best
        history.append(dict(cell=c, before_policy=chosen[c], after_policy=j,
                            before_statistics=total.tolist(), after_statistics=new_total.tolist()))
        chosen[c] = j; total = new_total
    return chosen, dict(statistics=total.tolist(), reference=reference.tolist(),
        movable_cells=movable, substitutions=history, local_stationary=best is None,
        step_limit_hit=len(history) == 200)


def run(prepared, series, output):
    require(not output.exists(), 'refusing to replace routing experiment'); output.mkdir(parents=True)
    data = read(prepared/'inputs.npz'); inner = read(series/'inner-evidence.npz')
    first = read(series/'predictions.npz'); probability = read(series/'probabilities.npz')
    splits = json.loads((series/'splits.json').read_text()); m = len(FAMILIES)*len(COSTS)
    outer = first['predictions'][data['native_position'], :m]
    old_best = np.where(data['available'], data['logits'], -np.inf).argmax(1)
    cells = data['baseline']*7+old_best
    predictions = np.empty((len(cells), 3), np.int8); policies = np.empty_like(predictions, np.int16)
    proofs = []
    for block in splits['blocks']:
        index = block['index']; test = np.flatnonzero(data['piece'] == block['piece'])
        mask = inner['rows'][:, 0] == index; train = inner['rows'][mask, 1]; matrix = inner['predictions'][mask]
        require(not set(data['piece'][train]) & {block['piece']}, 'outer labels in router selection')
        y, b, original = (data[k][train] for k in ('truth', 'baseline', 'original'))
        count_choice, count_proof = source_search(y, b, original, matrix, False)
        success_choice, success_proof = source_search(y, b, original, matrix, True)
        transition_choice, transition_proof = transition_search(y, b, original, matrix, cells[train], success_choice)
        require(not transition_proof['step_limit_hit'], 'routing failed to reach local stopping condition')
        for j, source in enumerate([count_choice, success_choice]):
            policies[test, j] = source[data['baseline'][test]-2]
        policies[test, 2] = [transition_choice.get(int(c), int(success_choice[c//7-2])) for c in cells[test]]
        for j in range(3): predictions[test, j] = outer[test, policies[test, j]]
        proofs.append(dict(block=index, piece=block['piece'], fold=block['fold'],
            source_count90=dict(policies=count_choice.tolist(), **count_proof),
            source_success90=dict(policies=success_choice.tolist(), **success_proof),
            transition_success90=dict(policies={str(k):v for k,v in transition_choice.items()}, **transition_proof)))
    summaries = {}
    for j, name in enumerate(KINDS):
        fi = policies[:, j]//len(COSTS); row = np.arange(len(fi))
        ps = probability['baseline_correct'][row, fi], probability['class_probability'][row, fi], probability['other_probability'][row, fi]
        summaries[name] = policy_stats(data, predictions[:, j], ps)
    native = np.broadcast_to(data['native_baseline'][:, None], (59309, 3)).copy()
    native[data['native_position']] = predictions
    np.savez_compressed(output/'predictions.npz', global_index=data['native_global_index'],
        true_K=data['native_truth'], frozen_baseline_K=data['native_baseline'], fold=data['native_fold'],
        eligible_global_index=data['global_index'], variant_ids=np.array(KINDS), predictions=native.astype(np.int8),
        outcomes_vs_freeze=(native == data['native_truth'][:, None]).astype(np.int8)-
                           (data['native_baseline'] == data['native_truth'])[:, None].astype(np.int8),
        chosen_policy=policies)
    dump(output/'router-proofs.json', proofs)
    report = dict(status='completed', previous_series_predictions_sha256=digest(series/'predictions.npz'),
        inner_evidence_sha256=digest(series/'inner-evidence.npz'), inputs_sha256=digest(prepared/'inputs.npz'),
        policies=summaries, independent_validation=False, entire_outer_fold_label_free=False,
        promotion=False, test_labels_used_in_selection=False, previous_candidates=102, total_candidates=105,
        diagnostics=diagnostics(data, predictions, KINDS),
        files={p.name:digest(p) for p in output.iterdir()})
    dump(output/'report.json', report)
    print(json.dumps({k:dict(**v['eligible'], retention=v['versus_original']) for k,v in summaries.items()},indent=2),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'series', 'output'): p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); run(a.prepared, a.series, a.output)


if __name__ == '__main__': main()
