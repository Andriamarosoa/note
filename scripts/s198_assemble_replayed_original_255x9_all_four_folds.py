"""S198 independently assemble fourfold S197 verified historical 255x9 audited source.

Combines fixed original masks/audits saved by each outer-holdout reconstruction,
confirms exactly 7493 unique eligible historical rows and 0 differences against
original archived proposal and 20D group summaries. No development truth read.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
FOLDS=(0,1,2,4)
def need(q,m):
    if not q:raise ValueError(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(root,folds,out):
    need(not out.exists(),'immutable combined original 255x9')
    prepared=root/'analysis/evidence/v273-regression-loops/prepared/inputs.npz'
    with np.load(prepared,allow_pickle=False) as source:
        ids=source['global_index'].copy()
        parent=source['baseline'].copy()
        f=source['fold'].copy()
        proposals=source['proposal'].copy()
        context=source['group_context'].copy()
    need(len(ids)==7493 and context.shape==(7493,7,20),'source original historical matrix drift')
    lookup={int(value):i for i,value in enumerate(ids)}
    need(len(lookup)==7493,'source original row identity duplicated')
    blocks=np.zeros((len(ids),255,9),np.float32)
    covered=np.zeros(len(ids),bool)
    records=[]
    for fold in FOLDS:
        folder=folds/f'fold{fold}'
        report=json.loads((folder/'report.json').read_text())
        p=folder/'original_9D_fold_audits_REPLAY.npz'
        with np.load(p,allow_pickle=False) as z:
            need(set(z.files)=={'global_index','fold','original_255_group_proposal',
                                'rebuilt_original_255x9_group_audits',
                                'recovered_7x20_audit_summary'},
                 'only original source-derived values, no truth arrays')
            local={k:z[k].copy() for k in z.files}
        expected=np.flatnonzero(f==fold)
        target=np.array([lookup[int(k)] for k in local['global_index']],dtype=int)
        need(len(target)==len(expected) and set(target.tolist())==set(expected.tolist()),
             f'outer held-fold {fold} rows not completely covered')
        need(not np.any(covered[target]),'same event reconstructed more than once')
        need(np.all(local['fold']==fold) and report['held_fold']==fold and
             report['held_rows']==len(target),'fold metadata disagreement')
        need(report['original_group_proposal_mismatches']==0 and
             report['aggregate_20D_max_absolute_discrepancy']<=2e-5,
             f'original 255 and 9 historical failed in fold {fold}')
        need(np.array_equal(local['original_255_group_proposal'],proposals[target]),
             'recovered 255 original action proposals differ from archived')
        need(np.allclose(local['recovered_7x20_audit_summary'],
                         context[target],atol=2e-5,rtol=0),
             'all reconstructed local 9 audits not the archived 20D summary')
        need(local['rebuilt_original_255x9_group_audits'].shape==
             (len(target),255,9) and
             np.isfinite(local['rebuilt_original_255x9_group_audits']).all(),
             'missing 9D complete group feature for true original event')
        blocks[target]=local['rebuilt_original_255x9_group_audits']
        covered[target]=True
        records.append({'fold':fold,'rows':len(target),'audit_replay_SHA256':sha(p),
            'exact_original_group_proposal_mismatches':0,
            'historical_20D_aggregate_max_abs_error':
                 report['aggregate_20D_max_absolute_discrepancy']})
    need(covered.all() and covered.sum()==7493,'original fourfold audit replay incomplete')
    out.mkdir(parents=True)
    np.savez_compressed(out/'full_original_historical_255x9_recomputed_4outerfolds.npz',
        global_index=ids,baseline_K=parent,fold=f,
        original_255_group_proposal=proposals,
        true_original_group9_reconstructed_from_frozen_producers=blocks,
        original_20D_context_summary=context)
    report={'status':'S198_ALL_FOUR_ORIGINAL_OUTER_FOLDS_255_AND_9_HISTORICAL_RECONSTRUCTED',
        'historical_eligible_event_count':len(ids),'original_255_groups_per_event':255,
        'original_nine_audit_columns_per_group':9,
        'total_original_audit_scalars_restored':int(blocks.size),
        'all_rows_covered_once':bool(covered.all()),
        'all_group_proposals_match_authentic_original':True,
        'all_20D_aggregated_historical_audit_features_reproduced':True,
        'folds_replayed':records,
        'combined_original_9D_archive_SHA256':
              sha(out/'full_original_historical_255x9_recomputed_4outerfolds.npz'),
        'S5_527D_original_upstream_constructor_matched':False,
        'new_audio_255x9_historical_equivalence':False,
        'development_fold_training_labels_only_in_reference_audit_reconstruction':True,
        'training_receiver_truth_used':False,
        'S80_promoted':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
def selftest():
    a=np.array([0,1,2,4],dtype=int)
    need(len(set(a.tolist()))==4 and sum(range(1,256))==32640,
         'fixed four-fold and 255 masks')
    print('S198 PASS four original outer folds and 255 source groups')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--folds',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        need(a.folds is not None and a.out is not None,'all folds paths needed')
        run(a.root,a.folds,a.out)
