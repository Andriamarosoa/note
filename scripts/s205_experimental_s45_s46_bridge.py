"""S205: explicitly EXPERIMENTAL S45-16/S46-193 bridge on exposed GOAT.

Never treat this mixed-parent bridge as original S80. Five champions have a
recovered original source, eleven are S173 adapted checkpoints. Preserve
S18 fold branches and reject absent/duplicated family columns or truth.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.loop_v273_native_risk import designs
from scripts.s96_external_s45_risk_runtime import predict as risk_predict
from scripts.s88_unlabeled_s46_runtime import run as s46_run

FOLDS = (0, 1, 2, 4)
BAD = {'true_K', 'truth', 'label', 'labels', 'target', 'midi', 'correct'}
FIVE_KEYS = {'row_id','branch_piece','branch_fold','family_ids','predictions',
             'eligible_positions','fold_ids','baseline_correct_probability','conditional_logits'}
ELEVEN_KEYS = {'row_id','S18_parent_K','original_family_ids','source_K',
               'S35_new_K','S35_selected_actions','agreement',
               'trained_Ridge_risks','partial_eleven_selected_head',
               'partial_eleven_candidate_K'}

def need(ok,why):
    if not ok:
        raise ValueError(why)

def load(path,expected=None):
    with np.load(path,allow_pickle=False) as z:
        need(not BAD.intersection(z.files),'annotation leak: '+str(path))
        if expected is not None:
            need(set(z.files)==expected,'input schema mismatch: '+str(path))
        return {k:z[k].copy() for k in z.files}

def take_s18(d):
    required={'row_id','freeze_K','fold_ids','global_K','poly_K',
              'S8_K','S9_K','S13_K','S14_K','S15_K','S16_K','S17_K','S18_K'}
    need(set(d)==required,'recovered S18 source schema')
    need(tuple(map(int,d['fold_ids']))==FOLDS,'source S18 folds')
    return d

def run(five_path, eleven_path, s18_path, s97_path, models, catalog, out):
    need(not out.exists(),'immutable S205 output required')
    five=load(five_path,FIVE_KEYS)
    eleven=load(eleven_path,ELEVEN_KEYS)
    chain=take_s18(load(s18_path))
    audio=load(s97_path,{'row_id','summary58','flow245','geometry3',
                          'sequence42x49','morphology490','physics294',
                          'member','start_samples','end_samples'})
    ids=chain['row_id']
    n=len(ids)
    need(n==57 and all(np.array_equal(ids,x['row_id']) for x in (five,eleven,audio)),
         'same real GOAT event IDs/order across all sources')
    five_names=list(map(str,five['family_ids']))
    eleven_names=list(map(str,eleven['original_family_ids']))
    expected=[x['family'] for x in json.loads(catalog.read_text())['original_families']]
    need(len(five_names)==5 and len(eleven_names)==11 and
         len(set(five_names+eleven_names))==16 and
         set(five_names+eleven_names)==set(expected),
         'sixteen exact S44 champion names, no substitutes')
    need(five['predictions'].shape==(19,n,5) and
         eleven['source_K'].shape==(n,11),'historical 19 branch and 11 source shapes')
    need(chain['S18_K'].shape==(4,n) and chain['global_K'].shape==(4,n) and
         chain['poly_K'].shape==(4,n),'four source fold branches missing')
    need(five['branch_fold'].shape==(19,) and
         set(map(int,five['branch_fold']))==set(FOLDS),'19 branch fold provenance')
    need(all(np.isin(q,range(7)).all() for q in
             (five['predictions'],eleven['source_K'],chain['S18_K'],chain['freeze_K'])),
         'invalid K classes')
    need(audio['summary58'].shape==(n,58) and audio['flow245'].shape==(n,245)
         and audio['geometry3'].shape==(n,3),'original acoustic feature dimensions')
    out.mkdir(parents=True)
    summary=[]
    for fi,fold in enumerate(FOLDS):
        candidates=np.flatnonzero(five['branch_fold']==fold)
        need(len(candidates)>0,'no original held-piece branch for fold')
        branch=int(candidates[0]) # one diagnostic branch, NOT a deployment aggregator
        parent=chain['S18_K'][fi].astype(np.int8)
        names=five_names+eleven_names
        raw=np.column_stack((five['predictions'][branch],eleven['source_K'])).astype(np.int8)
        proposed=np.column_stack([raw[:,names.index(f)] for f in expected])
        # Original 335D S45 risk inputs, recomputed from genuine frozen S5 parents.
        features=designs({'native_position':np.arange(n),'global_index':np.arange(n),
                          'features':audio['summary58']},
                         np.arange(n),chain['freeze_K'],chain['global_K'][fi],
                         chain['poly_K'][fi],audio['geometry3'])['votes_time']
        x=np.column_stack((features,audio['summary58'],audio['flow245'])).astype(np.float32)
        need(x.shape==(n,335) and np.isfinite(x).all(),'original 335D audio contract')
        folder=out/f'fold{fold}'
        folder.mkdir()
        np.savez_compressed(folder/'proposals.npz',row_id=ids,parent_K=parent,
                            family_ids=np.asarray(expected),proposal_K=proposed)
        np.savez_compressed(folder/'features.npz',row_id=ids,acoustic_features=x)
        risk_predict(folder/'proposals.npz',folder/'features.npz',
                     models,folder/'risk')
        s46_run(folder/'risk'/'s45_contract.npz',folder/'s46')
        z=load(folder/'s46'/'predictions.npz')
        need(z['predictions'].shape==(n,193),'all S46 193 outputs')
        need(not any('h9' in str(f).lower() for f in expected),'H9 forbidden')
        mismatch=parent!=eleven['S18_parent_K']
        source_disagree=np.any(eleven['source_K']!=parent[:,None],axis=1)
        summary.append({
            'fold':fold,'selected_historical_branch_index':branch,
            'other_historical_branches_not_evaluated':int(len(candidates)-1),
            'S18_true_vs_S173_adapter_parent_disagreements':int(mismatch.sum()),
            'S18_true_vs_S173_adapter_parent_disagreements_poly':int(
                np.count_nonzero(mismatch & np.isin(parent,(2,3,4,5,6)))),
            'eleven_adapted_source_alternatives':int(source_disagree.sum()),
            'five_recovered_source_alternatives':int(
                np.any(five['predictions'][branch]!=parent[:,None],axis=1).sum()),
            'S46_policy_count':int(z['predictions'].shape[1]),
            'S46_changes_across_193_policies':int(np.count_nonzero(
                z['predictions'][:,1:]!=parent[:,None]))})
    report={'status':'S205_EXPERIMENTAL_MIXED_PARENT_16_FAMILY_S46_DIAGNOSTIC',
            'fold_results':summary,'audio_events':n,
            'family_count':16,'risk_checkpoint_count':304,
            'S46_variant_count_including_parent':193,
            'original_five_champions_recovered':True,
            'eleven_sources_adapted_to_earlier_S5_S18':True,
            'uses_true_S5_S18_fold_for_risk_parent':True,
            'S29_original_producer_recovered':False,
            'full_19_branch_aggregation_deployed':False,
            'truth_labels_read':False,'independent_accuracy_measured':False,
            'original_S80_equivalence_proven':False,'S80_promoted':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('five','eleven','s18','s97','models','catalog','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    run(a.five,a.eleven,a.s18,a.s97,a.models,a.catalog,a.out)
