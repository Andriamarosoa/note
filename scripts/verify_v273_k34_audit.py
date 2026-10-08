"""Verify archived K3/K4 diagnostic predictions and exported event accounting."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.audit_v273_selector_design import aligned_positions
from scripts.verify_v273_selector_repair_artifacts import named_pairs, require


def decode(probability, baseline):
    row = np.arange(len(baseline))
    destination = probability[:, 2:].argmax(1)+2
    return np.where(probability[row, destination] > probability[row, baseline], destination, baseline)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--coherent', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    archive_sha = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    require(archive_sha == '5979dc1b465f467a8b463035fbfb1c1c41d91fb9362549e73bd58c69f7eb3043', 'audit archive SHA-256')
    with np.load(args.coherent/'predictions.npz', allow_pickle=False) as z:
        original = {k: z[k] for k in z.files}
    ids, y, b, current = (original[k] for k in ['global_index','true_K','frozen_baseline_K','predicted_K'])
    pos = aligned_positions(ids, original['eligible_global_index'])
    yc, bc, pc = y[pos], b[pos], current[pos]
    old_good = yc == bc; now_good = yc == pc
    corrections = ~old_good & now_good
    regressions = old_good & ~now_good & np.isin(yc, [3,4])
    report = json.loads((args.audit/'report.json').read_text())
    require(report['regression_case_count'] == int(regressions.sum()) == 594, 'case count')
    require(report['original_paired'] == named_pairs(y,b,current), 'original metric')
    events = list(csv.DictReader((args.audit/'event-audit.csv').open()))
    require(np.array_equal([int(e['global_index']) for e in events], ids[pos]), 'event ID alignment')
    q0 = original['class_probability']
    row = np.arange(len(pos))
    for i, e in enumerate(events):
        require((int(e['true_k']),int(e['baseline_k']),int(e['predicted_k'])) == (yc[i],bc[i],pc[i]), 'case classes')
        require(abs(float(e['expected_advantage'])-float(q0[i,pc[i]]-q0[i,bc[i]])) < 1e-6, 'case margin')
    cases = list(csv.DictReader((args.audit/'k3k4-regressions.csv').open()))
    require(np.array_equal([int(e['global_index']) for e in cases], ids[pos][regressions]), '594 regression IDs')
    require(len({e['global_index'] for e in cases}) == 594, 'duplicate regression case')
    evidence = dict(status='verified', run_id=37769970667, artifact_id=11546734202,
        source_commit='4329829cf9953363b95f6e13dab68c42eaa2367c', archive_sha256=archive_sha,
        independent_validation=False, promotion=False, regression_cases=594,
        original_paired=report['original_paired'], all_changes=report['all_changes'],
        transitions=report['transitions'], margin_bins=report['margin_bins'],
        replay_checks=report['replay_checks'], regression_overlap=report['regression_overlap'], interventions={})
    with np.load(args.audit/'diagnostic-replays.npz', allow_pickle=False) as z:
        require(np.array_equal(z['eligible_global_index'], ids[pos]), 'replay IDs')
        expert = z['expert_probability']
        require(expert.shape == (7493,6,5) and np.isfinite(expert).all(), 'expert array')
        require(np.allclose(expert.sum(2),1,atol=1e-6), 'expert normalization')
        for i,e in enumerate(events):
            delta = expert[i,1:,pc[i]-2]-expert[i,1:,bc[i]-2]
            require(abs(float(e['expert_mean_advantage'])-float(delta.mean())) < 1e-6, 'expert mean')
            require(int(e['experts_supporting_decision']) == int(np.sum(delta>0)), 'expert support')
        for name, recorded in report['interventions'].items():
            q = z[name.replace(':','__')]
            require(q.shape == (7493,7) and np.isfinite(q).all() and (q>=0).all(), 'intervention probabilities')
            require(np.allclose(q.sum(1),1,atol=1e-6), 'intervention normalization')
            pred = decode(q,bc)
            full = b.copy();full[pos] = pred
            paired_freeze = named_pairs(y,b,full);paired_current = named_pairs(y,current,full)
            require(paired_freeze == recorded['paired_vs_freeze'], 'intervention native metric')
            require(paired_current == recorded['paired_vs_coherent'], 'intervention paired metric')
            require(int(np.sum(corrections & (pred!=yc))) == recorded['original_corrections_lost'], 'lost corrections')
            require(int(np.sum(regressions & (pred==yc))) == recorded['original_k34_regressions_restored'], 'restored regressions')
            for f in (0,1,2,4):
                take = original['fold'][pos] == f
                require(named_pairs(yc[take],pc[take],pred[take])['global'] == recorded['by_fold'][str(f)], 'intervention fold metric')
            if name.startswith('identity_control:'):
                require(np.array_equal(pred,pc), 'negative control prediction')
            evidence['interventions'][name] = recorded
    require(len(evidence['interventions']) == 28, 'intervention coverage')
    evidence['files_sha256'] = {name:hashlib.sha256((args.audit/name).read_bytes()).hexdigest()
        for name in ['diagnostic-replays.npz','report.json','event-audit.csv','k3k4-regressions.csv','feature-contrasts.csv']}
    evidence['count_shift_summary'] = {name:dict(selected_rows=sum(v['selected_rows'] for v in report['count_shifts'] if v['channel']==name),
        selected_above_train_max=sum(v['held_selected_above_train_max'] for v in report['count_shifts'] if v['channel']==name))
        for name in ['global_count','context_count']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(evidence,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(dict(status='verified',event_rows=len(events),regression_rows=len(cases),interventions=28)))


if __name__ == '__main__':
    main()
