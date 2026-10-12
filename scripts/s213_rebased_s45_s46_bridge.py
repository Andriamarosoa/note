"""S213: replace five S173-adapted S45 sources with genuine S212 replays.

Original S1/S2/S3/S4/coherent recovered, original S11/S12/S38/S40/S41
S212 on true S5/S9/S18 parents, and original S29-fed S35 S210.
Five recurrent S19/S20/S24/S25/S27 still use documented S173 adapters.
Recompute 304 original S45 Ridge risk scores and 193 S46 decisions in
all 19 branches, compare with S211, no labels/training or S80 claims.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scripts.s206_all_19_s45_s46_branches import FOLDS,FIVE_KEYS,ELEVEN_KEYS,need,load,take_s18
from scripts.loop_v273_native_risk import designs
from scripts.s96_external_s45_risk_runtime import predict as risk_predict
from scripts.s88_unlabeled_s46_runtime import run as s46_run

S210_KEYS={'row_id','piece','fold','S29_K','S35_K','S35_actions'}
S212_KEYS={'row_id','branch_piece','branch_fold','family_ids','proposals_K'}
AUDIO_KEYS={'row_id','summary58','flow245','geometry3','sequence42x49',
            'morphology490','physics294','member','start_samples','end_samples'}
REBASED=('S11_votes_time_audio','S12_audio_logistic',
         'S38_morphology','S40_temporal_CNN','S41_neural_fusion')
STILL_ADAPTED=('S19_recurrent_ABA','S20_repeated_ABA',
               'S24_Bfirst','S25_Bfirst','S27_early_stop')

def run(five_path,eleven_path,s210_path,s212_path,s18_path,s97_path,
        models,catalog,previous211,out):
    need(not out.exists(),'refuse overwrite S213 evidence')
    five=load(five_path,FIVE_KEYS)
    eleven=load(eleven_path,ELEVEN_KEYS)
    real35=load(s210_path,S210_KEYS)
    five_new=load(s212_path,S212_KEYS)
    chain=take_s18(load(s18_path))
    audio=load(s97_path,AUDIO_KEYS)
    ids=chain['row_id'];n=len(ids)
    need(n==57 and all(np.array_equal(ids,a['row_id']) for a in
         (five,eleven,real35,five_new,audio)),'unaltered GOAT source row order')
    names5=list(map(str,five['family_ids']))
    names11=list(map(str,eleven['original_family_ids']))
    names_new=list(map(str,five_new['family_ids']))
    expected=[r['family'] for r in json.loads(catalog.read_text())['original_families']]
    need(len(names5)==5 and len(names11)==11 and len(set(names5+names11))==16
         and set(names5+names11)==set(expected),'original 16 champion families')
    need(set(names_new)==set(REBASED) and
         set(STILL_ADAPTED).issubset(names11) and 'S35_multiroute' in names11,
         'S212 exact five re-based families + 5 remaining adapted')
    need(five['predictions'].shape==(19,n,5) and
         five_new['proposals_K'].shape==(19,n,5) and
         real35['S35_K'].shape==(19,n),'all 19 original held-piece outputs')
    need(chain['S18_K'].shape==(4,n) and
         [int(np.sum(five['branch_fold']==f)) for f in FOLDS]==[5,4,5,5],
         'four genuine S18 fold configurations')
    j35=names11.index('S35_multiroute')
    stats=[];out.mkdir(parents=True)
    for fi,fold in enumerate(FOLDS):
        parent=chain['S18_K'][fi].astype(np.int8)
        vote=designs({'native_position':np.arange(n),'global_index':np.arange(n),
                      'features':audio['summary58']},np.arange(n),
                     chain['freeze_K'],chain['global_K'][fi],
                     chain['poly_K'][fi],audio['geometry3'])['votes_time']
        x=np.column_stack((vote,audio['summary58'],audio['flow245'])).astype(np.float32)
        need(x.shape==(n,335) and np.isfinite(x).all(),'true reconstructed S5 to S45 335D contract')
        for branch in np.flatnonzero(five['branch_fold']==fold):
            piece=str(five['branch_piece'][branch])
            hit35=np.flatnonzero((real35['fold']==fold)&(real35['piece']==piece))
            hit212=np.flatnonzero((five_new['branch_fold']==fold)&(five_new['branch_piece']==piece))
            need(len(hit35)==len(hit212)==1,'original held-piece join S210 S212: '+piece)
            source=eleven['source_K'].copy()
            source[:,j35]=real35['S35_K'][hit35[0]]
            for fam in REBASED:
                source[:,names11.index(fam)]=five_new['proposals_K'][hit212[0],names_new.index(fam)]
            keys=names5+names11
            raw=np.column_stack((five['predictions'][branch],source)).astype(np.int8)
            proposals=np.column_stack([raw[:,keys.index(k)] for k in expected])
            folder=out/f'branch{int(branch):02d}_fold{fold}'
            prev=previous211/folder.name
            need((prev/'s46/predictions.npz').is_file() and
                 (prev/'proposals.npz').is_file(),'immutable S211 comparison absent')
            oldp=load(prev/'proposals.npz')
            need(set(oldp)=={'row_id','parent_K','family_ids','proposal_K'} and
                 np.array_equal(oldp['row_id'],ids) and
                 np.array_equal(oldp['parent_K'],parent) and
                 np.array_equal(oldp['family_ids'],np.asarray(expected)),
                 'S211 baseline S45 join')
            delta=oldp['proposal_K']!=proposals
            allowed=np.array([k in REBASED for k in expected])
            need(not np.any(delta[:,~allowed]),'S213 changed source outside original five re-based families')
            changes={fam:int(delta[:,expected.index(fam)].sum()) for fam in REBASED}
            folder.mkdir()
            np.savez_compressed(folder/'proposals.npz',row_id=ids,parent_K=parent,
                                family_ids=np.asarray(expected),proposal_K=proposals)
            np.savez_compressed(folder/'features.npz',row_id=ids,acoustic_features=x)
            risk_predict(folder/'proposals.npz',folder/'features.npz',models,folder/'risk')
            s46_run(folder/'risk'/'s45_contract.npz',folder/'s46')
            old=load(prev/'s46/predictions.npz')
            new=load(folder/'s46/predictions.npz')
            need(set(old)==set(new)=={'row_id','variant_ids','predictions'} and
                 np.array_equal(old['row_id'],new['row_id']) and
                 np.array_equal(old['variant_ids'],new['variant_ids']) and
                 new['predictions'].shape==(n,193),
                 'S46 exactly same policies and inputs across S211 and S213')
            changed=old['predictions']!=new['predictions']
            stats.append(dict(branch=int(branch),fold=fold,piece=piece,
                 five_new_source_changes=changes,
                 S46_changed_decision_cells=int(changed.sum()),
                 S46_changed_events=int(changed.any(axis=1).sum())))
    need(len(stats)==19 and
         [sum(r['fold']==f for r in stats) for f in FOLDS]==[5,4,5,5],
         'all original 19 S45/S46 research branches')
    res=dict(status='S213_SIX_ORIGINAL_S45_SOURCE_REBASE_AUDITED',
       events=57,original_branches=19,families=16,
       S45_original_ridge_checkpoints=304,S46_original_policy_count=193,
       newly_rebased_families=list(REBASED),other_recovered_families=names5,
       original_S29_S35_connected=True,
       still_adapted_families=list(STILL_ADAPTED),
       S212_source_differences_vs_S211={
          fam:sum(s['five_new_source_changes'][fam] for s in stats) for fam in REBASED},
       S46_decision_differences_vs_S211=sum(s['S46_changed_decision_cells'] for s in stats),
       changed_S46_branches=sum(s['S46_changed_decision_cells']>0 for s in stats),
       branches=stats,labels_read=False,predictive_models_retrained=False,
       complete_historical_S45_equivalence=False,independent_accuracy_measured=False,
       S80_promoted=False)
    (out/'report.json').write_text(json.dumps(res,indent=2)+'\n')
    print(json.dumps({k:v for k,v in res.items() if k!='branches'},indent=2),flush=True)
    return res

if __name__=='__main__':
    a=argparse.ArgumentParser()
    for k in ('five','eleven','s210','s212','s18','s97','models','catalog','previous211','out'):
        a.add_argument('--'+k,type=Path,required=True)
    p=a.parse_args()
    run(p.five,p.eleven,p.s210,p.s212,p.s18,p.s97,p.models,p.catalog,p.previous211,p.out)
