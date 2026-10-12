"""S216 — independent decomposition of FULL-SOURCE S215 S46 fold divergence.

Read ONLY S215 label-free sealed full sixteen-source S45/S46 predictions. No MIDI/true K, no tuning.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
FOLDS=(0,1,2,4)
def need(v,msg):
    if not v: raise ValueError(msg)
def load(path):
    with np.load(path,allow_pickle=False) as z:
        need(set(z.files)=={'row_id','variant_ids','predictions'},'strict S46 output contract')
        return {k:z[k].copy() for k in z.files}
def run(root,out):
    need(not out.exists(),'immutable output')
    all_rows=[];within_fold_diff=[]
    for f in FOLDS:
        paths=sorted(root.glob(f'branch*_fold{f}/s46/predictions.npz'))
        need(len(paths)=={0:5,1:4,2:5,4:5}[f],'missing original held-piece outputs')
        ref=load(paths[0])
        diff=0
        for p in paths[1:]:
            trial=load(p)
            need(np.array_equal(trial['row_id'],ref['row_id']) and
                 np.array_equal(trial['variant_ids'],ref['variant_ids']) and
                 trial['predictions'].shape==ref['predictions'].shape,
                 'within-fold policy and event identity mismatch')
            diff+=int(np.count_nonzero(trial['predictions']!=ref['predictions']))
        within_fold_diff.append(diff)
        all_rows.append(ref)
    ids=all_rows[0]['row_id']
    for row in all_rows:
        need(np.array_equal(ids,row['row_id']) and
             np.array_equal(all_rows[0]['variant_ids'],row['variant_ids']),
             'row or policy identity shifted')
    pred=np.stack([z['predictions'] for z in all_rows])
    need(pred.shape==(4,57,193),'S215 4fold matrices geometry')
    parent=pred[:,:,0]
    parent_unstable=(~np.all(parent==parent[0:1],axis=0))
    changes=(pred[:,:,1:]!=parent[:,:,None])
    all_same=np.all(pred==pred[0:1],axis=0)
    inconsistent=~all_same
    parent_only=np.all(pred==parent[:,:,None],axis=0)
    # Representative branch per fold for cross-fold comparison; intra-fold differences separately reported.
    pure_parent=inconsistent & parent_only
    gate_related=inconsistent & ~parent_only
    need(int((pure_parent|gate_related).sum())==int(inconsistent.sum()),'decomposition mismatch')
    # Across-fold decisions that differ while parents agree are a direct gate-only divergence.
    parent_agree=np.all(parent==parent[0:1],axis=0)
    gate_only=inconsistent & parent_agree[:,None]
    # With disagreeing parents, some gates might restore consensus (not necessarily accuracy).
    restored=all_same & parent_unstable[:,None]
    output=dict(status='S216_ALL_16_SOURCE_FOLD_STABILITY_AND_GATE_ATTRIBUTION',
        audio_events=57,policies=193,folds=list(FOLDS),
        within_fold_changed_cells_vs_first=within_fold_diff,
        parent_S18_unstable_events=int(parent_unstable.sum()),
        parent_S18_stable_events=int(parent_agree.sum()),
        unstable_event_policy_cells=int(inconsistent.sum()),
        explained_by_four_parent_outputs_without_S46_changes=int(pure_parent.sum()),
        unstable_cells_with_some_S46_gate_change=int(gate_related.sum()),
        unstable_cells_when_parent_already_unanimous=int(gate_only.sum()),
        stable_cells_despite_different_S18_parents=int(restored.sum()),
        unstable_events_with_any_policy_difference=int(inconsistent.any(axis=1).sum()),
        each_fold_gate_action_cells=[int(v.sum()) for v in changes],
        no_label_inputs=True,full_historical_S80_equivalent=False,
        new_deployment_rule_chosen=False,independent_accuracy_measured=False,
        promoted=False)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2),flush=True)
    return output
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--s215',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); run(a.s215,a.out)
