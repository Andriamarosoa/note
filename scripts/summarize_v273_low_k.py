"""Independently check saved low-K probes and extract illustrative cases."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.audit_v273_low_k import VARIANTS
from scripts.audit_v273_window_pair import read_predictions
from scripts.rebuild_v273_sources import digest, write_json
from scripts.restore_v273_original_backup import validate_original_inventory


def summarize(root, models, output):
    validate_original_inventory(root)
    report=json.loads((root/'report.json').read_text())
    assert report['status']=='completed' and report['outer_fold']==3
    with np.load(root/'row-context.npz',allow_pickle=False) as z:
        c={key:np.asarray(z[key]) for key in z.files}
    k=c['k']; low=k<4
    saved={v:read_predictions(models/v,'final')[2] for v in VARIANTS}
    result={'report_sha256':digest(root/'report.json'),'row_context_sha256':digest(root/'row-context.npz'),
            'all_saved_probe_counts_recomputed':True,'variants':{},'examples':[],
            'example_selection':'Highest confidence per named stratum; illustrative, not a representative sample.'}
    for variant,p in saved.items():
        for key in ('global_index','member','cluster_start_samples','k'):
            np.testing.assert_array_equal(c[key],p[key])
        with np.load(root/(variant+'-probes.npz'),allow_pickle=False) as z:
            probe={key:np.asarray(z[key]) for key in z.files}
        np.testing.assert_allclose(probe['baseline'],p['probability'],rtol=1e-4,atol=1e-5)
        a=p['predicted']; over=low&(a>k)
        stats={}
        for name,prob in probe.items():
            assert prob.shape==(len(k),7) and np.isfinite(prob).all()
            assert np.allclose(prob.sum(1),1,atol=1e-5)
            b=prob.argmax(1)
            if name=='baseline':
                np.testing.assert_array_equal(b,a);continue
            observed=dict(before_over=int(over.sum()),after_over=int((low&(b>k)).sum()),
                over_to_correct=int((over&(b==k)).sum()),over_to_under=int((over&(b<k)).sum()),
                correct_to_over=int((low&(a==k)&(b>k)).sum()),under_to_over=int((low&(a<k)&(b>k)).sum()))
            expected=report['replay_and_sensitivity'][variant]['interventions'][name]['low_k']
            assert all(expected[key]==value for key,value in observed.items())
            stats[name]=observed
        flags={key:c[key]>0 for key in ('foreign_births_31','foreign_births_added_tail','carried_notes_at_start')}
        flags['no_annotation_overlap']=c['overlapping_notes_31']==0
        rows={'over_low':int(over.sum()),'probes':stats,'context_of_overcounts':{key:int((over&v).sum()) for key,v in flags.items()},
              'by_true_k':{str(i):{'rows':int((k==i).sum()),'over':int(((k==i)&(a>k)).sum()),
                 'context_of_overcounts':{key:int(((k==i)&(a>k)&v).sum()) for key,v in flags.items()}} for i in range(4)}}
        if 'tail_zero' in probe:
            zero,hold,head=[probe[n].argmax(1) for n in ('tail_zero','tail_hold','head_zero')]
            previous=saved[variant.replace('31','23')]['predicted']
            consensus=over&(zero==k)&(hold==k)&(head>k)
            rows['tail_both_correct_head_control_still_over']=int(consensus.sum())
            rows['tail_both_correct_with_foreign_added_tail']=int((consensus&flags['foreign_births_added_tail']).sum())
            rows['tail_both_correct_on_new_overcounts']=int((consensus&(previous<=k)).sum())
            rows['tail_both_correct_by_true_k']={str(i):int((consensus&(k==i)).sum()) for i in range(4)}
            if variant=='frames-31-weighted':
                strata={'new_over_tail_specific_foreign':consensus&flags['foreign_births_added_tail']&(previous<=k),
                        'k0_with_foreign_birth':over&(k==0)&flags['foreign_births_31'],
                        'k0_carried_without_foreign_birth':over&(k==0)&flags['carried_notes_at_start']&~flags['foreign_births_31'],
                        'k0_no_annotated_note_overlap':over&(k==0)&flags['no_annotation_overlap'],
                        'k3_to4_created_by_weighting':(k==3)&(a==4)&(saved['frames-31-uniform']['predicted']==3)}
                for name,take in strata.items():
                    ids=np.flatnonzero(take)
                    if not len(ids):continue
                    row=int(ids[np.argmax(p['probability'][ids].max(1))])
                    result['examples'].append(dict(stratum=name,eligible_rows=len(ids),global_index=int(c['global_index'][row]),
                        member=str(c['member'][row]),track_row=int(c['track_row'][row]),
                        start_seconds=float(c['cluster_start_samples'][row]/44100),true_k=int(k[row]),
                        predictions={v:int(q['predicted'][row]) for v,q in saved.items()},
                        probability=p['probability'][row].tolist(),probe_predictions={n:int(q[row].argmax()) for n,q in probe.items()},
                        foreign_births_31=int(c['foreign_births_31'][row]),foreign_births_added_tail=int(c['foreign_births_added_tail'][row]),
                        carried_notes_at_start=int(c['carried_notes_at_start'][row]),overlapping_notes_31=int(c['overlapping_notes_31'][row])))
        result['variants'][variant]=rows
    write_json(output,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','models','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();summarize(a.root,a.models,a.output)
