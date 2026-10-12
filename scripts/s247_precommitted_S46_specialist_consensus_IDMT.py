"""S247: precommitted S219 sparse conditional specialist fold consensus on
1034 untouched IDMT V88 native event IDs from genuine original S246
16-family/304-Ridge/193-S46 all-19-branch output bank.

Fold silent is abstention not veto; require strict majority within the
4 or 5 same-fold historical held-piece networks, exact S18 parent
compatibility, unanimity among active folds, parent fallback on conflict.
No held-piece oracle, no annotations, original S219 rule unchanged.
"""
from __future__ import annotations
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np
from scripts.s219_sparse_s46_specialist_branch_fusion import solve
FOLDS=(0,1,2,4)
def need(v,msg):
    if not v:raise ValueError(msg)
def read(p,fields):
    with np.load(p,allow_pickle=False) as z:
        need(set(z.files)==set(fields),'S247 input schema must remain strictly label-free')
        return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(root,s239,out):
    need(not out.exists(),'S247 frozen IDMT S46 output cannot overwrite')
    parent=read(s239,{'row_id','freeze_K','original_fold_ids','original_S18_by_fold_K',
       'S18_piece_neutral_consensus_K','S18_quorum_of_3_supported',
       'S18_vote_strength','S18_abstention_to_freeze'})
    ids=parent['row_id'];n=len(ids)
    need(n==1034 and len(set(map(str,ids)))==n and
         np.array_equal(parent['original_fold_ids'],FOLDS),
         'S239 1034 IDMT true canonical events required')
    files=sorted(root.glob('branch*_fold*/s46/predictions.npz'))
    need(len(files)==19,'missing one or more original S246 piece branches')
    ks=[];folds=[];branches=[];variant_names=None
    for file in files:
        m=re.fullmatch(r'branch(\d+)_fold(\d+)',file.parent.parent.name)
        need(m is not None,'S246 original fold source naming drift')
        branch=int(m.group(1));fold=int(m.group(2))
        need(fold in FOLDS,'not a historical source fold')
        z=read(file,{'row_id','variant_ids','predictions'})
        need(np.array_equal(ids,z['row_id']) and z['predictions'].shape==(n,193),
             'S246 S46 source model event IDs or exact 193 policy geometry mismatch')
        if variant_names is None:variant_names=z['variant_ids']
        need(np.array_equal(variant_names,z['variant_ids']),
             'same original 193 S46 models not selected in every branch')
        original=read(file.parent.parent/'proposals.npz',
             {'row_id','parent_K','family_ids','proposal_K'})
        need(np.array_equal(ids,original['row_id']) and
             np.array_equal(z['predictions'][:,0],original['parent_K']),
             'original S46 parent disagrees with actual S45 source')
        ks.append(z['predictions']);folds.append(fold);branches.append(branch)
    need(sorted(branches)==list(range(19)) and
         [folds.count(f) for f in FOLDS]==[5,4,5,5],
         'entire 19 original heldpiece branch bank required')
    raw=np.stack(ks).astype(np.int8);folds=np.array(folds,np.int8)
    s18=parent['S18_piece_neutral_consensus_K'].astype(np.int8)
    output=solve(raw,folds,s18)
    need(np.array_equal(output['fold_parent_K'],parent['original_S18_by_fold_K']),
         'original historic S18 fold lineage incompatible with candidate S246')
    for permutation in (np.arange(18,-1,-1),np.roll(np.arange(19),7)):
        same=solve(raw[permutation],folds[permutation],s18)
        need(all(np.array_equal(output[k],same[k]) for k in output),
             'S219 original deployed source rule depends on arbitrary piece order')
    pred=output['predictions']
    need(pred.shape==(1034,193) and np.array_equal(pred[:,0],s18) and
         np.isin(pred,range(7)).all(),
         'full precommitted S46 193 policy labels invalid')
    change=(pred[:,1:]!=s18[:,None])
    out.mkdir(parents=True)
    path=out/'S247_original_193_S46_foldneutral_1034_IDMT.npz'
    np.savez_compressed(path,row_id=ids,variant_ids=variant_names,
       original_freeze_K=parent['freeze_K'],S18_parent_K=s18,
       original_19_branch_folds=folds,
       S46_193_policy_K=pred,per_fold_specialist_K=output['fold_proposals'],
       active_fold_count=output['active_fold_support'],
       parent_incompatible_specialists=output['wrong_parent_proposals_ignored'],
       contradictory_specialists=output['contradictory_fold_proposal'])
    report=dict(status='S247_ALL_GENUINE_S46_SPECIALIST_VOTES_PRECOMMITTED_1034_NATIVE_IDMT',
       original_S246_source_branch_count=19,
       original_S246_family_count=16,
       original_S246_ridge_model_count=304,
       original_S246_policy_count=193,
       original_native_reference_event_count=n,
       source_fold_counts=[5,4,5,5],
       original_precommitted_S219_new_piece_voting_identical_to_GOAT=True,
       original_S18_parents_identical_to_S238_and_S239=True,
       original_S46_source_branch_intervention_cells=int(np.count_nonzero(
          raw[:,:,1:]!=raw[:,:,0,None])),
       S46_active_S219_policy_event_cells=int(change.sum()),
       S46_events_with_any_active_specialist=int(np.count_nonzero(change.any(1))),
       S46_parent_incompatible_proposals_ignored=int(output['wrong_parent_proposals_ignored'].sum()),
       S46_conflicting_active_specialist_cells=int(output['contradictory_fold_proposal'].sum()),
       source_branch_reordering_invariance_verified=True,
       models_refitted=False,IDMT_XML_MIDI_content_read=False,
       independent_accuracy_measured=False,S80_promoted=False,
       output_SHA256=sha(path))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for x in ('s246','s239','out'):p.add_argument('--'+x,type=Path,required=True)
    a=p.parse_args();run(a.s246,a.s239,a.out)
