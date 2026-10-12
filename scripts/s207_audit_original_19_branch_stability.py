"""S207: label-free independent consistency audit of the 19 S206 branches.

No model training, labels, invented final fold aggregation or promotion.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

FOLDS=(0,1,2,4)
BAD={'true_K','truth','label','labels','target','midi','correct'}
def need(v,msg):
    if not v: raise ValueError(msg)
def read(path):
    with np.load(path,allow_pickle=False) as z:
        need(not BAD.intersection(z.files),'leaked annotation: '+str(path))
        return {key:z[key].copy() for key in z.files}

def audit(root,out):
    need(not out.exists(),'do not overwrite S207 audit')
    folders=sorted(p for p in root.glob('branch*_fold*') if p.is_dir())
    need(len(folders)==19,'nineteen original held-piece branches required')
    bank={}
    original_ids=None
    original_names=None
    original_families=None
    for folder in folders:
        fold=int(folder.name.rsplit('_fold',1)[1])
        need(fold in FOLDS,'invalid original fold')
        p=read(folder/'s46'/'predictions.npz')
        r=read(folder/'risk'/'s45_contract.npz')
        need(set(p)=={'row_id','variant_ids','predictions'},'193 policies contract')
        need(set(r)=={'row_id','parent_K','family_ids','proposal_K','risks','independent_votes'},
             '16 families contract')
        ids=p['row_id']; names=p['variant_ids']; predictions=p['predictions']
        families=r['family_ids']
        need(predictions.shape==(57,193) and r['proposal_K'].shape==(57,16) and
             r['risks'].shape==(57,16,2),'wrong source geometry')
        need(np.array_equal(r['row_id'],ids) and np.array_equal(predictions[:,0],r['parent_K']),
             'wrong parent ID or S46 first policy')
        need(len(families)==len(set(map(str,families)))==16,'duplicate family')
        need(np.isin(predictions,range(7)).all(),'invalid classes')
        if original_ids is None:
            original_ids=ids;original_names=names;original_families=families
        else:
            need(np.array_equal(ids,original_ids) and np.array_equal(names,original_names)
                 and np.array_equal(families,original_families),
                 'branches not aligned to same events and variant identities')
        bank.setdefault(fold,[]).append((folder.name,predictions,r))
    need([len(bank[f]) for f in FOLDS]==[5,4,5,5],'missing original branch in fold')
    details=[]
    representatives=[]
    for fold in FOLDS:
        rows=bank[fold];base=rows[0][1]
        ref=rows[0][2]
        diff_cells=0; diff_rows=set(); diff_family=0
        max_risk_delta=0.
        for _,pred,r in rows[1:]:
            delta=(pred!=base)
            diff_cells+=int(delta.sum())
            diff_rows.update(np.flatnonzero(delta.any(1)).tolist())
            diff_family+=int(np.count_nonzero(r['proposal_K']!=ref['proposal_K']))
            q=r['risks'];p=ref['risks'];finite=np.isfinite(q)&np.isfinite(p)
            need(np.array_equal(np.isfinite(q),np.isfinite(p)), 'risk null mask inconsistency')
            if np.any(finite):
                max_risk_delta=max(max_risk_delta,float(np.abs(q[finite]-p[finite]).max()))
        details.append({'fold':fold,'branches':len(rows),
                        'S46_different_cells_vs_first_in_same_fold':diff_cells,
                        'S46_events_with_any_different_policy':len(diff_rows),
                        'S45_source_family_different_cells':diff_family,
                        'max_absolute_Ridge_risk_difference':max_risk_delta})
        representatives.append(base)
    stack=np.stack(representatives)
    stable=np.all(stack==stack[0:1],axis=0)
    across_fold_diff=int((~stable).sum())
    changes_from_own_parent=[
        int(np.count_nonzero(rep[:,1:]!=rep[:,0,None])) for rep in representatives]
    report={'status':'S207_ALL_19_BRANCHES_NO_LABEL_BRANCH_STABILITY_AUDIT',
            'branch_count':len(folders),'fold_branch_counts':[len(bank[f]) for f in FOLDS],
            'events':57,'family_count':16,'S46_columns':193,
            'within_fold':details,'across_four_folds_unstable_event_policy_cells':across_fold_diff,
            'across_four_folds_stable_event_policy_cells':int(stable.sum()),
            'total_event_policy_cells':int(stable.size),
            'S46_alternative_vs_own_S18_parent_cells_by_fold':changes_from_own_parent,
            'truth_labels_read':False,'new_final_aggregation_rule_selected':False,
            'independent_ExactK_accuracy_measured':False,
            'historical_S80_equivalence_proven':False,'promoted':False}
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--s206',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    audit(a.s206,a.out)
