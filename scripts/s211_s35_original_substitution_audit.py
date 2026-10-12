"""S211: measure original S29->S35 substitution inside all 16 S45/193 S46.

Five true recovered champion families and ten unchanged *adapted* S173
sources remain. Only S35 is replaced with its authentic frozen original
S29-fed 19-branch S210 forward. This is a source-attribution test, NOT a
fully equivalent historical S45 nor an independent Exact-K score.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.s206_all_19_s45_s46_branches import (
    FOLDS,FIVE_KEYS,ELEVEN_KEYS,need,load,take_s18)
from scripts.loop_v273_native_risk import designs
from scripts.s96_external_s45_risk_runtime import predict as risk_predict
from scripts.s88_unlabeled_s46_runtime import run as s46_run

S210_KEYS={'row_id','piece','fold','S29_K','S35_K','S35_actions'}
AUDIO_KEYS={'row_id','summary58','flow245','geometry3',
            'sequence42x49','morphology490','physics294',
            'member','start_samples','end_samples'}

def run(five_path, eleven_path, s210_path, s18_path, s97_path, models,
        catalog, original206, out):
    need(not out.exists(),'immutable S211 output')
    five=load(five_path,FIVE_KEYS)
    eleven=load(eleven_path,ELEVEN_KEYS)
    s210=load(s210_path,S210_KEYS)
    s18=take_s18(load(s18_path))
    audio=load(s97_path,AUDIO_KEYS)
    ids=s18['row_id'];n=len(ids)
    need(n==57 and all(np.array_equal(ids,z['row_id']) for z in
                         (five,eleven,s210,audio)),'57 native GOAT row IDs')
    five_names=list(map(str,five['family_ids']))
    eleven_names=list(map(str,eleven['original_family_ids']))
    expected=[z['family'] for z in json.loads(catalog.read_text())['original_families']]
    need(len(five_names)==5 and len(eleven_names)==11 and
         set(five_names+eleven_names)==set(expected) and
         'S35_multiroute' in eleven_names,
         'original exact 16 S45 family names')
    need(len(set(five_names+eleven_names))==16,'duplicate S45 family')
    j35=eleven_names.index('S35_multiroute')
    need(s210['S35_K'].shape==(19,n) and five['predictions'].shape==(19,n,5)
         and eleven['source_K'].shape==(n,11),'original 19-branch source geometry')
    need(s18['S18_K'].shape==(4,n) and
         np.isin(s210['S35_K'],range(7)).all(),
         'authentic recovered 4-fold S18 and S35 source K')
    seen=set();reports=[];out.mkdir(parents=True)
    for fi,fold in enumerate(FOLDS):
        parent=s18['S18_K'][fi].astype(np.int8)
        v=designs({'native_position':np.arange(n),'global_index':np.arange(n),
                   'features':audio['summary58']},
                  np.arange(n),s18['freeze_K'],s18['global_K'][fi],
                  s18['poly_K'][fi],audio['geometry3'])['votes_time']
        x=np.column_stack((v,audio['summary58'],audio['flow245'])).astype(np.float32)
        need(x.shape==(n,335) and np.isfinite(x).all(),'real original S45 audio 335D')
        for branch in np.flatnonzero(five['branch_fold']==fold):
            piece=str(five['branch_piece'][branch])
            hits=np.flatnonzero((s210['fold']==fold)&(s210['piece']==piece))
            need(len(hits)==1,'S29/S35 exact held-piece branch join: '+piece)
            src=eleven['source_K'].copy()
            src[:,j35]=s210['S35_K'][hits[0]]
            family_names=five_names+eleven_names
            raw=np.column_stack((five['predictions'][branch],src)).astype(np.int8)
            proposed=np.column_stack([raw[:,family_names.index(f)] for f in expected])
            folder=out/f'branch{int(branch):02d}_fold{fold}'
            previous=original206/folder.name/'s46'/'predictions.npz'
            need(previous.is_file(),'sealed original S206 comparison unavailable')
            folder.mkdir()
            np.savez_compressed(folder/'proposals.npz',row_id=ids,parent_K=parent,
                                family_ids=np.asarray(expected),proposal_K=proposed)
            np.savez_compressed(folder/'features.npz',row_id=ids,acoustic_features=x)
            risk_predict(folder/'proposals.npz',folder/'features.npz',models,folder/'risk')
            s46_run(folder/'risk'/'s45_contract.npz',folder/'s46')
            old=load(previous)
            current=load(folder/'s46'/'predictions.npz')
            need(set(old)==set(current)=={'row_id','variant_ids','predictions'} and
                 np.array_equal(old['row_id'],current['row_id']) and
                 np.array_equal(old['variant_ids'],current['variant_ids']) and
                 old['predictions'].shape==current['predictions'].shape==(n,193),
                 'old/new S46 precise policy alignment')
            changed=(old['predictions']!=current['predictions'])
            changed_s35=src[:,j35]!=eleven['source_K'][:,j35]
            seen.add((fold,piece))
            reports.append(dict(fold=fold,branch=int(branch),piece=piece,
                S35_new_source_differs_from_S173_events=int(changed_s35.sum()),
                S35_new_source_changes_vs_true_S18=int(np.count_nonzero(src[:,j35]!=parent)),
                S46_changed_event_policy_cells=int(changed.sum()),
                S46_changed_events=int(changed.any(axis=1).sum()),
                S46_changed_policies=int(changed.any(axis=0).sum())))
    need(len(reports)==len(seen)==19 and
         [sum(r['fold']==f for r in reports) for f in FOLDS]==[5,4,5,5],
         'missing or repeated held-piece branch')
    r=dict(status='S211_REAL_S29_S35_SOURCE_REPLACED_IN_MIXED_S45_S46',
       ics='actual original 19 S29 -> per-fold S35, 5 reconstructed and 10 adapted S45 families',
       events=n,branches=19,families=16,
       S45_original_ridge_checkpoints=304,S46_policy_columns=193,
       S35_source_changed_events_across_branches=sum(v['S35_new_source_differs_from_S173_events'] for v in reports),
       S46_changed_event_policy_cells_vs_S206=sum(v['S46_changed_event_policy_cells'] for v in reports),
       S46_changed_branches_vs_S206=sum(v['S46_changed_event_policy_cells']>0 for v in reports),
       branches_evidence=reports,truth_labels_read=False,models_trained=False,
       ten_other_S45_families_still_adapted=True,
       historical_16_family_S45_equivalence=False,
       full_S80_validated=False,independent_ExactK_measured=False,
       S80_promoted=False)
    (out/'report.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k!='branches_evidence'},indent=2),flush=True)
    print(json.dumps(reports),flush=True)
    return r
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('five','eleven','s210','s18','s97','models','catalog','original206','out'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    run(a.five,a.eleven,a.s210,a.s18,a.s97,a.models,a.catalog,a.original206,a.out)
