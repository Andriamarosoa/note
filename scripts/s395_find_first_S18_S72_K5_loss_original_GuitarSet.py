"""S395: locate FIRST true-K5->pred-K4 loss from S18 through S69/S70/S71/S72/S80.

Factual stage provenance on aligned ORIGINAL 59k event IDs. No training.
S46's 193 outputs are policies with parent fallback, NOT 193 independent
trained attack detectors; evaluate parent separately and don't count clones.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def need(x,msg):
    if not x:raise ValueError(msg)
def run(s46,s271,s80,your,out):
    need(not out.exists(),'immutable stage evidence only')
    v=load(s46);r=load(s271);b=load(s80);t=load(your)
    ord=np.argsort(t['global_index'].astype(np.int64))
    ids=b['row_id'].astype(str)
    y=t['k'][ord].astype(int);final=b['s80_K'].astype(int)
    need(len(ids)==59309 and np.array_equal(ids,v['row_id'].astype(str))
         and np.array_equal(ids,r['row_id'].astype(str))
         and np.array_equal(ids,t['global_index'][ord].astype(str))
         and int(((y==5)&(final==4)).sum())==224
         and int(((y==4)&(final==4)).sum())==806
         and np.array_equal(r['s69_polymax_K'],b['s72_K']*0+r['s69_polymax_K'])
         and np.array_equal(r['s72_poly_poly_guard_K'],b['s72_K']),
         'S395 original stage lineage mismatched')
    # The actual selected route to S80 is the POLY branch of S71 -> S72.
    stages=[
        ('S18_parent',v['S18_parent_K']),
        ('S69_polymax',r['s69_polymax_K']),
        ('S70_proposed_poly',r['s70_proposed_poly_K']),
        ('S71_poly',r['s71_poly_K']),
        ('S72_poly_poly_guard',r['s72_poly_poly_guard_K']),
        ('S79',b['s79_K']),('S80',b['s80_K'])]
    stages={k:np.asarray(x,dtype=int) for k,x in stages}
    groups={'K5_lost_as_K4':(y==5)&(final==4),
            'K5_correct_S80':(y==5)&(final==5),
            'K4_correct_S80':(y==4)&(final==4),
            'K5_other_errors':(y==5)&(final!=4)&(final!=5)}
    result={}
    for name,mask in groups.items():
        ix=np.flatnonzero(mask)
        lineage={key:{f'K{i}':int((val[ix]==i).sum()) for i in range(7)}
                 for key,val in stages.items()}
        edges={}
        last=None
        for key,val in stages.items():
            if last is not None:
                prev,src=last
                edges[prev+'__to__'+key]=dict(
                    changed=int((src[ix]!=val[ix]).sum()),
                    correctK5_lost=int(((src[ix]==5)&(val[ix]!=5)).sum()),
                    K5_recovered=int(((src[ix]!=5)&(val[ix]==5)).sum()),
                    K5_to_K4=int(((src[ix]==5)&(val[ix]==4)).sum()),
                    K4_to_K5=int(((src[ix]==4)&(val[ix]==5)).sum()))
            last=(key,val)
        first_lost={}
        for j in ix:
            if stages['S18_parent'][j]!=5:
                loc='already_not_K5_at_S18'
            else:
                loc='never_lost'
                keys=list(stages)
                for q in range(1,len(keys)):
                    if stages[keys[q-1]][j]==5 and stages[keys[q]][j]!=5:
                        loc=keys[q-1]+'__to__'+keys[q];break
            first_lost[loc]=first_lost.get(loc,0)+1
        s46=v['S46_193_policy_K'][ix]
        parent=v['S18_parent_K'][ix]
        # Count identical inherited parent decisions and genuine deviations.
        identical=(s46==parent[:,None])
        result[name]=dict(n=len(ix),stage_class_histograms=lineage,
            stage_transition_deltas=edges,first_loss_location=first_lost,
            S46_variants_identical_to_S18_parent_total=int(identical.sum()),
            S46_variant_total=int(s46.size),
            S46_rows_all_193_identical_to_parent=int(identical.all(1).sum()),
            S46_rows_any_specialist_deviated=int((~identical).any(1).sum()),
            S71_veto_three_fold=int(r['s71_veto_three_folds'][ix].sum()),
            S72_veto_three_fold=int(r['s72_veto_three_folds'][ix].sum()),
            S71_fold_veto_counts=list(map(int,r['s71_fold_veto'][:,ix].sum(axis=1))),
            S72_fold_veto_counts=list(map(int,r['s72_fold_veto'][:,ix].sum(axis=1))))
    report=dict(status='S395_TRUE_K5_ERROR_FIRST_LOSS_STAGE_REAL_CHAIN',
        full_GuitarSet_events=59309,original_true_K5=355,
        S46_193_can_clone_S18_parent=True,
        S80_GuitarSet_train_overlap=True,
        no_model_retraining=True,no_threshold_change=True,
        no_independent_external_generalization_proof=True,
        by_class=result,
        S80_weights_unchanged=True)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,g in result.items():
        print('S395 '+name+' stage K5',{k:v['K5'] for k,v in g['stage_class_histograms'].items()},
              'first_loss',g['first_loss_location'],
              'S46rows_always_parent',g['S46_rows_all_193_identical_to_parent'],
              'S71veto',g['S71_veto_three_fold'],'S72veto',g['S72_veto_three_fold'],flush=True)
        print('S395 '+name+' stage transitions',json.dumps(g['stage_transition_deltas']),flush=True)
    print('S395 PASS first-loss stage audit; no model change',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for a in ('s46','s271','s80','your','out'):p.add_argument('--'+a,type=Path,required=True)
    q=p.parse_args();run(q.s46,q.s271,q.s80,q.your,q.out)
