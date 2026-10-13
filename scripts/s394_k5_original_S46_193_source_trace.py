"""S394: trace all 193 S46 source policy candidates to final S80 K5.
Read-only benchmark, same GuitarSet event IDs and labels; no training/changes.
Disentangles: absent K5 source vs K5 offered but lost by downstream S69-S72.
The 193 trained source outputs share ORIGINAL dev/training overlap, so nothing
here demonstrates generalization or acoustically identifies the fifth pitch.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
def need(x,m):
    if not x:raise ValueError(m)
def read(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def checksum(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(s46,s80,your,out):
    need(not out.exists(),'S394 output must be immutable')
    a=read(s46);b=read(s80);mt=read(your)
    ord=np.argsort(mt['global_index'].astype(np.int64))
    ids=b['row_id'].astype(str)
    yy=mt['k'][ord].astype(int);final=b['s80_K'].astype(int)
    allp=a['S46_193_policy_K']
    need(len(ids)==59309 and len(np.unique(ids))==59309
         and np.array_equal(ids,a['row_id'].astype(str))
         and np.array_equal(ids,mt['global_index'][ord].astype(str))
         and allp.shape==(59309,193)
         and np.isin(allp,np.arange(7)).all()
         and np.array_equal(a['original_freeze_K'],b['incumbent_freeze_K'])
         and int((yy==5).sum())==355
         and int(((yy==5)&(final==4)).sum())==224
         and int(((yy==4)&(final==4)).sum())==806,
         'source 193 S46 and 59k benchmark schema/alignment drift')
    v5=np.count_nonzero(allp==5,axis=1)
    v4=np.count_nonzero(allp==4,axis=1)
    groups={'K5_missed_as_K4':(yy==5)&(final==4),
            'K4_correct_as_K4':(yy==4)&(final==4),
            'K5_correct_as_K5':(yy==5)&(final==5),
            'K5_other_missed':(yy==5)&(final!=5)&(final!=4)}
    stats={}
    for name,mask in groups.items():
        p=v5[mask];r=np.flatnonzero(mask)
        stats[name]=dict(n=int(mask.sum()),
            no_S46_candidate_K5=int(np.count_nonzero(p==0)),
            any_S46_candidate_K5=int(np.count_nonzero(p>=1)),
            S46_2plus_K5=int(np.count_nonzero(p>=2)),
            S46_5plus_K5=int(np.count_nonzero(p>=5)),
            S46_K5_votes_median=float(np.median(p)),
            S46_K4_votes_median=float(np.median(v4[r])),
            PR16_policy22_K5=int(np.count_nonzero(b['original_PR16_policy22_K'][r]==5)),
            S72_selected_K5=int(np.count_nonzero(b['s72_K'][r]==5)),
            S79_selected_K5=int(np.count_nonzero(b['s79_K'][r]==5)),
            YourMT3_correct=int(np.count_nonzero(mt['predicted'][ord][r]==yy[r])))
    lost=groups['K5_missed_as_K4']
    s46_candidate=v5>0
    p22=b['original_PR16_policy22_K']==5
    s80_parent=b['s72_K']==5
    mtcorrect=mt['predicted'][ord]==5
    result=dict(
        status='S394_COMPLETE_193_S46_ORIGINAL_K5_SOURCE_COVERAGE_TRACE',
        inputs_sha256={'S46_193':checksum(s46),'S80':checksum(s80),'YourMT3':checksum(your)},
        native_events=59309,true_K5=355,
        by_observed_true_K_groups=stats,
        lost_K5_count=int(lost.sum()),
        lost_with_any_S46_K5=int((lost&s46_candidate).sum()),
        lost_without_any_S46_K5=int((lost&~s46_candidate).sum()),
        lost_with_S46_or_PR16_policy22_K5=int((lost&(s46_candidate|p22)).sum()),
        lost_without_S46_or_PR16_policy22_K5=int((lost&~(s46_candidate|p22)).sum()),
        lost_with_S46_K5_but_final_S72_K4=int((lost&s46_candidate&~s80_parent).sum()),
        among_lost_recognized_by_YourMT3=int((lost&mtcorrect).sum()),
        among_lost_YourMT3_correct_S46_also_offered_K5=int((lost&mtcorrect&s46_candidate).sum()),
        true_K5_any_S46_K5_count=int(((yy==5)&s46_candidate).sum()),
        all_S46_193_policy_candidates_are_ORIGINAL_source_folds=True,
        original_S80_dev_train_overlap=True,
        labels_can_only_prove_proposal_presence_not_correct_pitch=True,
        promotion=False,model_weights_changed=False)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    print('S394 PASS complete 193 S46 stage trace, NO S80 change',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for x in ('s46','s80','your','out'):p.add_argument('--'+x,required=True,type=Path)
    v=p.parse_args();run(v.s46,v.s80,v.your,v.out)
