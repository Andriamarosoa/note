"""S107 identify S5 feature coordinates actually used by original four trees.
No input reconstruction or validation claims. Read only trusted pinned archives.
"""
from pathlib import Path
import argparse,hashlib,json
from collections import Counter
import joblib,numpy as np
FOLDS=(0,1,2,4)
def audit(root,out):
    byfold={};all_counts=Counter();total=0
    for fold in FOLDS:
        p=root/f'fold{fold}';report=json.loads((p/'report.json').read_text())
        ck=p/'model.joblib'
        if hashlib.sha256(ck.read_bytes()).hexdigest()!=report['model_sha256']:
            raise ValueError('original model SHA drift')
        stored=joblib.load(ck);model=stored['model']
        if model.n_features_in_!=527:raise ValueError('unexpected dimensions')
        c=Counter()
        for step in model._predictors:
            for tree in step:
                nodes=tree.nodes
                splits=nodes[~nodes['is_leaf']]
                c.update(int(x) for x in splits['feature_idx'])
        if not c:raise ValueError('no trained splits detected')
        if not all(0<=i<527 for i in c):raise ValueError('invalid split index')
        all_counts.update(c);total+=sum(c.values())
        byfold[str(fold)]={'total_splits':sum(c.values()),
                          'active_feature_count':len(c),
                          'active_feature_indices':sorted(c),
                          'top20':[{'feature_index':i,'split_count':n} for i,n in c.most_common(20)]}
    report={'status':'S107_ORIGINAL_S5_527_TREE_FEATURE_AUDIT',
            'total_splits':total,'active_unique_features':len(all_counts),
            'active_feature_indices':sorted(all_counts),
            'unused_feature_count':527-len(all_counts),
            'top30':[{'feature_index':i,'split_count':n} for i,n in all_counts.most_common(30)],
            'folds':byfold,
            'semantics_per_feature_available':False,
            'authentic_527D_generator_verified':False,
            'new_Goat_S5_probabilities':0,
            'S80_promoted':False}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ['status','active_unique_features','unused_feature_count','top30']},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();audit(a.root,a.out)
