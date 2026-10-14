"""S282: precommitted label-blind S217 3/4 fold consensus on unseen IDMT S18.

Exactly the frozen decision policy established from GOAT *before* IDMT
unseen song labels were accessed. Four original heldfold heads have
equal weight, 3/4 votes required; otherwise return the immutable
original S233 official freeze_local_combo K. Independently crosscheck
every vote by Counter loops. S18/S45/S80 are NOT scored and S80 is
NOT yet evaluated on the 506 native events.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from scripts.s217_label_blind_hierarchical_branch_consensus import count_and_vote
FOLDS=(0,1,2,4)
BAD={'true_K','truth','label','labels','target','midi','correct','y','accuracy','regression'}
def req(x,msg):
    if not x:raise ValueError(msg)
def read(path,fields):
    with np.load(path,allow_pickle=False) as z:
        req(set(z.files)==set(fields) and not BAD.intersection(z.files),
            'original complete label-blind S281 source required '+str(z.files))
        return {f:z[f].copy() for f in z.files}
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(src,out):
    req(not out.exists(),'S282 never overwrites frozen outcomes')
    z=read(src,{'row_id','freeze_K','fold_ids','global_K','poly_K',
       'S8_K','S9_K','S13_K','S14_K','S15_K','S16_K','S17_K','S18_K'})
    ids=z['row_id'];b=z['freeze_K'];fold=z['S18_K'];n=len(ids)
    req(n==506 and len(set(ids.tolist()))==n and
        np.array_equal(z['fold_ids'],FOLDS) and
        fold.shape==(4,n) and b.shape==(n,) and
        np.isin(fold,range(7)).all() and np.isin(b,range(7)).all(),
        'unmodified four original 506-fold predictions and source reference required')
    selected,support,votes=count_and_vote(fold[:,:,None],
                                            b[:,None],3)
    selected=selected[:,0];support=support[:,0];votes=votes[:,0]
    verify=[];argcounts=[]
    for k in range(n):
        counts=Counter(map(int,fold[:,k]))
        strength=max(counts.values())
        winners=[a for a,q in counts.items() if q==strength]
        chosen=int(winners[0]) if strength>=3 else int(b[k])
        req(int(selected[k])==chosen and
            bool(support[k])==(strength>=3) and
            int(votes[k])==strength,
            'S282 independent original S217 fold-vote implementation diverged')
        argcounts.append(len(winners))
    disagree=np.any(fold!=fold[:1],axis=0)
    abstain=~support
    out.mkdir(parents=True)
    data=out/'S282_original_S18_foldneutral_506_IDMT.npz'
    np.savez_compressed(data,row_id=ids,freeze_K=b,
       original_fold_ids=np.asarray(FOLDS),
       original_S18_by_fold_K=fold,
       S18_piece_neutral_consensus_K=selected,
       S18_quorum_of_3_supported=support,
       S18_vote_strength=votes,
       S18_abstention_to_freeze=abstain)
    report=dict(status='S282_IDMT_S217_PRECOMMITTED_4FOLD_S18_CONSENSUS',
      baseline_original_reference_run=38319155819,
      genuine_original_S5_to_S18_run=38319911630,
      native_event_count=n,original_fold_count=4,
      original_S18_disagreeing_events=int(disagree.sum()),
      original_S18_parent_abstentions_to_baseline=int(abstain.sum()),
      original_S18_changes_vs_official_freeze=int((selected!=b).sum()),
      original_S18_changed_event_ids=[str(ids[i]) for i in np.flatnonzero(selected!=b)],
      original_S18_K_histogram=np.bincount(selected,minlength=7).tolist(),
      original_S217_precommitted_rule='four equal folds, majority3of4, fallback unchanged official reference',
      all_506_votes_independently_reproduced_with_Collections_Counter=True,
      all_original_V88_native_reference_IDs_unchanged=True,
      possible_tied_fold_strength_events=int(sum(v==2 for v in votes)),
      labelled_song_annotations_read=False,
      original_trained_model_weights_untouched=True,
      S45_S46_S69_S80_not_replayed_on_independent_506_events=True,
      independent_exactK_accuracy_measured=False,S80_promoted=False,
      output_SHA256=digest(data))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items()
          if k!='original_S18_changed_event_ids'},indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--s281',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    z=p.parse_args();run(z.s281,z.out)
