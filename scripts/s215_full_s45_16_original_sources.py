"""S215: all sixteen S45 families reconstructed from ORIGINAL frozen models.

Five recovered S1..S4/coherent historical branches, five genuine S212
families S11/S12/S38/S40/S41, five genuine S214 recurrent sources
S19/S20/S24/S25/S27, and S210 authentic S29-fed S35. Re-score all
304 original S45 Ridge risk models and 193 S46 policies on each of
19 held-piece research branches. No training, annotations or new-piece
aggregation policy; NOT original historical identity or S80 promotion.
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
FIVE_SOURCE_KEYS={'row_id','branch_piece','branch_fold','family_ids','proposals_K'}
AUDIO_KEYS={'row_id','summary58','flow245','geometry3','sequence42x49',
            'morphology490','physics294','member','start_samples','end_samples'}
RECURRENT=('S19_recurrent_ABA','S20_repeated_ABA',
           'S24_Bfirst','S25_Bfirst','S27_early_stop')
REBASED=('S11_votes_time_audio','S12_audio_logistic',
         'S38_morphology','S40_temporal_CNN','S41_neural_fusion')

def run(five_path,eleven_path,s210_path,s212_path,s214_path,
        s18_path,s97_path,models,catalog,previous213,out):
    need(not out.exists(),'immutable S215 source proof only')
    five=load(five_path,FIVE_KEYS)
    eleven=load(eleven_path,ELEVEN_KEYS)
    true35=load(s210_path,S210_KEYS)
    rebased=load(s212_path,FIVE_SOURCE_KEYS)
    recurrent=load(s214_path,FIVE_SOURCE_KEYS)
    chain=take_s18(load(s18_path))
    audio=load(s97_path,AUDIO_KEYS)
    ids=chain['row_id'];n=len(ids)
    need(n==57 and all(np.array_equal(ids,x['row_id']) for x in
          (five,eleven,true35,rebased,recurrent,audio)),
         'label-free original native GOAT alignment')
    names5=list(map(str,five['family_ids']))
    names11=list(map(str,eleven['original_family_ids']))
    names212=list(map(str,rebased['family_ids']))
    names214=list(map(str,recurrent['family_ids']))
    expected=[v['family'] for v in json.loads(catalog.read_text())['original_families']]
    need(len(expected)==16 and len(set(expected))==16 and
         set(names5+names11)==set(expected) and
         len(names5)==5 and len(names11)==11 and
         set(names212)==set(REBASED) and set(names214)==set(RECURRENT)
         and 'S35_multiroute' in names11,
         '16 ORIGINAL families, 5+5+5+1 no H9 or substitutes')
    need(all(z['proposals_K'].shape==(19,n,5) for z in (rebased,recurrent))
         and five['predictions'].shape==(19,n,5)
         and true35['S35_K'].shape==(19,n) and
         eleven['source_K'].shape==(n,11),
         'original 19 branch source widths')
    need(chain['S18_K'].shape==(4,n),'four original S18 folds')
    stats=[];out.mkdir(parents=True)
    for fi,fold in enumerate(FOLDS):
        parent=chain['S18_K'][fi].astype(np.int8)
        votes=designs({'native_position':np.arange(n),'global_index':np.arange(n),
                       'features':audio['summary58']},
                      np.arange(n),chain['freeze_K'],chain['global_K'][fi],
                      chain['poly_K'][fi],audio['geometry3'])['votes_time']
        x=np.column_stack((votes,audio['summary58'],audio['flow245'])).astype(np.float32)
        need(x.shape==(n,335) and np.isfinite(x).all(),'real frozen S45 input geometry')
        for branch in np.flatnonzero(five['branch_fold']==fold):
            piece=str(five['branch_piece'][branch])
            def unique_index(bank,piece_key,fold_key):
                indexes=np.flatnonzero((bank[piece_key]==piece)&(bank[fold_key]==fold))
                need(len(indexes)==1,'historical held-piece branch join failed '+piece)
                return int(indexes[0])
            j35=unique_index(true35,'piece','fold')
            j212=unique_index(rebased,'branch_piece','branch_fold')
            j214=unique_index(recurrent,'branch_piece','branch_fold')
            src=eleven['source_K'].copy()
            src[:,names11.index('S35_multiroute')]=true35['S35_K'][j35]
            for fam in REBASED:
                src[:,names11.index(fam)]=rebased['proposals_K'][j212,:,names212.index(fam)]
            for fam in RECURRENT:
                src[:,names11.index(fam)]=recurrent['proposals_K'][j214,:,names214.index(fam)]
            nameorder=names5+names11
            raw=np.column_stack((five['predictions'][branch],src)).astype(np.int8)
            complete=np.column_stack([raw[:,nameorder.index(f)] for f in expected])
            need(complete.shape==(n,16) and np.isin(complete,range(7)).all(),
                 'all real sixteen source proposal columns available')
            folder=out/f'branch{int(branch):02d}_fold{fold}'
            prior=previous213/folder.name
            need((prior/'proposals.npz').is_file() and
                 (prior/'s46/predictions.npz').is_file(),
                 'prior original S213 frozen proof missing')
            oldprops=load(prior/'proposals.npz')
            need(set(oldprops)=={'row_id','parent_K','family_ids','proposal_K'} and
                 np.array_equal(oldprops['row_id'],ids) and
                 np.array_equal(oldprops['parent_K'],parent) and
                 np.array_equal(oldprops['family_ids'],np.asarray(expected)),
                 'S213 historical branch source mismatch')
            delta=oldprops['proposal_K']!=complete
            legal=np.array([fam in RECURRENT for fam in expected])
            need(not np.any(delta[:,~legal]),'S215 modified outside five recurrent families')
            folder.mkdir()
            np.savez_compressed(folder/'proposals.npz',row_id=ids,parent_K=parent,
                                family_ids=np.asarray(expected),proposal_K=complete)
            np.savez_compressed(folder/'features.npz',row_id=ids,acoustic_features=x)
            risk_predict(folder/'proposals.npz',folder/'features.npz',models,folder/'risk')
            s46_run(folder/'risk'/'s45_contract.npz',folder/'s46')
            before=load(prior/'s46/predictions.npz')
            after=load(folder/'s46/predictions.npz')
            need(set(before)==set(after)=={'row_id','variant_ids','predictions'} and
                 np.array_equal(before['row_id'],after['row_id']) and
                 np.array_equal(before['variant_ids'],after['variant_ids'])
                 and after['predictions'].shape==(n,193),
                 'strict original 193 S46 policies across S213/S215')
            changes=before['predictions']!=after['predictions']
            stats.append(dict(fold=fold,branch=int(branch),piece=piece,
                source_diff_by_recurrent_family={
                    fam:int(delta[:,expected.index(fam)].sum()) for fam in RECURRENT},
                S46_changed_policy_event_cells=int(changes.sum()),
                S46_changed_events=int(changes.any(axis=1).sum())))
    need(len(stats)==19 and [sum(z['fold']==f for z in stats) for f in FOLDS]==[5,4,5,5],
         '19 historical branches preserved intact')
    report=dict(status='S215_SIXTEEN_ORIGINAL_CHECKPOINT_S45_S46_NEW_AUDIO_FORWARD',
       audio_events=57,branch_count=19,original_S45_sources=16,
       historical_S45_ridge_checkpoints=304,
       S46_policy_columns=193,recovered_s1_s4_coherent=5,
       reexecuted_genuine_S11_S12_S38_S40_S41=5,
       reexecuted_genuine_recurrent_S19_S20_S24_S25_S27=5,
       reexecuted_original_S29_S35=1,
       remaining_adapted_S45_original_sources=[],
       recurrent_proposal_changes_vs_S213={
           fam:sum(s['source_diff_by_recurrent_family'][fam] for s in stats)
           for fam in RECURRENT},
       S46_decision_cells_changed_vs_S213=sum(s['S46_changed_policy_event_cells'] for s in stats),
       changed_S46_branches=sum(s['S46_changed_policy_event_cells']>0 for s in stats),
       branches=stats,truth_labels_opened=False,models_retrained=False,
       original_S5_527D_matrix_byte_equivalence_unproven=True,
       historic_piece_specific_policy_equivalence_on_unseen_music_unproven=True,
       prespecified_new_piece_aggregation_unresolved=True,
       S69_to_S80_new_audio_unexecuted=True,
       independent_ExactK_performance_measured=False,
       original_S80_promoted=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='branches'},indent=2),flush=True)
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('five','eleven','s210','s212','s214','s18','s97',
                'models','catalog','previous213','out'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    run(a.five,a.eleven,a.s210,a.s212,a.s214,a.s18,a.s97,
        a.models,a.catalog,a.previous213,a.out)
