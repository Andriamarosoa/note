"""Recompute failure partitions and learning gaps from frozen audit arrays."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.rebuild_v273_sources import digest, write_json
from scripts.restore_v273_original_backup import validate_original_inventory


def summarize(audits, models, output):
    result = dict(status='completed_and_cross_checked', outer_fold=3, frames=31,
        model_scope='native count component; not the full V27.3 pipeline',
        variants={}, training_performed=False, output_corrector=False, model_promoted=False,
        oracle_interpretation='Uses true polyphony; diagnostic failure partition, not deployable accuracy.')
    predictions = {}
    for arm in ('uniform', 'weighted'):
        source = models / ('frames-31-' + arm)
        root = audits / arm
        validate_original_inventory(source)
        validate_original_inventory(root)
        report = json.loads((root / 'report.json').read_text())
        assert report['status'] == 'completed' and report['outer_fold'] == 3
        protocol = json.loads((source / 'protocol.json').read_text())
        assert report['protocol_sha256'] == digest(source / 'protocol.json')
        variant = dict(report_sha256=digest(root / 'report.json'), phases={})
        for phase in ('inner', 'final'):
            record = json.loads((source / phase / 'training.json').read_text())
            assert report['phases'][phase]['frozen_training'] == record
            weights = np.asarray(record['class_weights'], np.float64)
            splits = {}
            for split in ('fit', 'heldout'):
                with np.load(root / f'{phase}-{split}.npz', allow_pickle=False) as z:
                    data = {name:np.asarray(z[name]) for name in z.files}
                ids,k,pred,p = (data[x] for x in ('global_index','k','predicted','probability'))
                assert len(np.unique(ids)) == len(ids)
                assert np.isfinite(p).all() and (p >= 0).all() and p.shape == (len(k),7)
                np.testing.assert_allclose(p.sum(1),1,atol=1e-5)
                np.testing.assert_array_equal(pred,p.argmax(1))
                if split == 'heldout':
                    with np.load(source / phase / 'predictions.npz', allow_pickle=False) as saved:
                        for key in ('global_index','k','member','predicted'):
                            np.testing.assert_array_equal(data[key],saved[key])
                        np.testing.assert_allclose(p,saved['probability'],rtol=2e-4,atol=2e-5)
                name = (('inner_fit' if phase == 'inner' else 'final_fit') if split == 'fit'
                        else ('inner_val' if phase == 'inner' else 'outer'))
                np.testing.assert_array_equal(np.bincount(k,minlength=7),protocol['partitions'][name]['classes'])
                cm=np.zeros((7,7),np.int64)
                np.add.at(cm,(k,pred),1)
                observed=report['phases'][phase]['splits'][split]
                assert cm.tolist() == observed['confusion_true_by_predicted']
                assert int(np.trace(cm[2:,2:])) == observed['poly_correct']
                poly=k>=2
                within=p[:,2:].argmax(1)+2
                failure=dict(
                    lost_to_0_or_1_true_class_best_within_poly=int(np.sum(poly&(pred<2)&(within==k))),
                    lost_to_0_or_1_also_wrong_within_poly=int(np.sum(poly&(pred<2)&(within!=k))),
                    predicted_poly_wrong_count=int(np.sum(poly&(pred>=2)&(pred!=k))))
                assert sum(failure.values()) == int(np.sum(poly&(pred!=k)))
                assert observed['poly_correct_if_true_poly_known'] == int(np.sum(poly&(within==k)))
                loss=-np.log(np.maximum(p[np.arange(len(k)),k].astype(np.float64),1e-12))
                np.testing.assert_allclose(np.mean(loss*weights[k]),observed['weighted_nll'],atol=1e-12,rtol=1e-12)
                by_k={str(v):dict(rows=int(cm[v].sum()),correct=int(cm[v,v]),
                    under=int(cm[v,:v].sum()),over=int(cm[v,v+1:].sum()),
                    exact=float(cm[v,v]/cm[v].sum()) if cm[v].sum() else None)
                    for v in range(7)}
                splits[split]=dict(rows=len(k),correct=int(np.trace(cm)),exact=float(np.trace(cm)/len(k)),
                    poly_rows=int(poly.sum()),poly_correct=int(np.trace(cm[2:,2:])),
                    poly_exact=float(np.trace(cm[2:,2:])/poly.sum()),by_true_k=by_k,
                    failures_disjoint=failure,
                    common_k2_k3_errors=int(np.sum((k>=2)&(k<=3)&(pred!=k))),
                    rare_k5_k6_errors=int(np.sum((k>=5)&(pred!=k))),
                    nll=float(np.mean(loss)),weighted_nll=float(np.mean(loss*weights[k])),
                    npz_sha256=digest(root / f'{phase}-{split}.npz'))
                if phase == 'final' and split == 'heldout':
                    predictions[arm] = data
            gap=dict(poly_exact_fit_minus_heldout=splits['fit']['poly_exact']-splits['heldout']['poly_exact'],
                     exact_fit_minus_heldout=splits['fit']['exact']-splits['heldout']['exact'],
                     by_true_k={str(v):splits['fit']['by_true_k'][str(v)]['exact']-
                                splits['heldout']['by_true_k'][str(v)]['exact'] for v in range(7)})
            variant['phases'][phase]=dict(splits=splits,generalization_gap=gap,
                class_weights=weights.tolist(),training_history=record['history'],
                original_fit_classes=protocol['partitions']['inner_fit' if phase=='inner' else 'final_fit']['classes'])
        result['variants'][arm]=variant
    a,b=predictions['uniform'],predictions['weighted']
    for key in ('global_index','k','member'):
        np.testing.assert_array_equal(a[key],b[key])
    k=a['k'];pa,pb=a['predicted'],b['predicted'];poly=k>=2
    result['weighting_intervention']=dict(
        poly_correct_before=int(np.sum(poly&(pa==k))),poly_correct_after=int(np.sum(poly&(pb==k))),
        poly_corrected=int(np.sum(poly&(pa!=k)&(pb==k))),
        poly_regressed=int(np.sum(poly&(pa==k)&(pb!=k))),
        by_true_k={str(v):dict(rows=int(np.sum(k==v)),
            correct_delta=int(np.sum((k==v)&(pb==k))-np.sum((k==v)&(pa==k))),
            corrected=int(np.sum((k==v)&(pa!=k)&(pb==k))),
            regressed=int(np.sum((k==v)&(pa==k)&(pb!=k)))) for v in range(7)},
        paired_training_proof='analysis/v273-window-pair-audit.json; same inputs, initial weights, seeds, orders and budget')
    result['verification']=dict(all_archives_file_inventories_verified=True,
        all_confusions_recomputed=True,all_heldout_rows_and_decisions_reproduced=True,
        all_class_counts_and_failure_partitions_recomputed=True,
        summarizer_sha256=digest(__file__))
    write_json(output,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('audits','models','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    r=summarize(a.audits,a.models,a.output)
    print(json.dumps(dict(status=r['status'],weighting_intervention=r['weighting_intervention'])))
