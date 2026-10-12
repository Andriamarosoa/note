"""S267: full genuine 16 source-family S45 original Ridge + S46 193 policy
on 59309 untouched GUITARSET rank33-48 reference V88 native events.

Requires 5 historical rare S1–S4/coherent (S245) + 5 original S11/S12/
S38/S40/S41 (S242) + 5 original S19/S20/S24/S25/S27 (S243) + original
S29-fed S35 (S241). All four original S18 parents from S238 and actual
S240 native acoustic views. Executes frozen 304 original trained S45
risk checkpoints and all 193 S46 policies through unchanged S96/S88.
NO use of S173 proxies, GOAT results, MIDI, XML, or new model fitting.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scripts.loop_v273_native_risk import designs
from scripts.s96_external_s45_risk_runtime import predict as risk_predict
from scripts.s88_unlabeled_s46_runtime import run as s46_run
from scripts.s206_all_19_s45_s46_branches import FOLDS,FIVE_KEYS,need,load,take_s18
from scripts.s215_full_s45_16_original_sources import FIVE_SOURCE_KEYS,S210_KEYS,RECURRENT,REBASED,AUDIO_KEYS
FORBIDDEN={'true_K','truth','label','labels','target','midi','correct','regression'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(five_path,s35_path,s242_path,s243_path,s18_path,s97_path,models,catalog,out):
    need(not out.exists(),'never overwrite a frozen S46 candidate')
    five=load(five_path,FIVE_KEYS)
    s35=load(s35_path,S210_KEYS)
    six=load(s242_path,FIVE_SOURCE_KEYS)
    recurrent=load(s243_path,FIVE_SOURCE_KEYS)
    chain=take_s18(load(s18_path))
    audio=load(s97_path,AUDIO_KEYS)
    ids=chain['row_id'];n=len(ids)
    need(n==59309 and len(set(map(str,ids)))==n and
         all(np.array_equal(ids,v['row_id']) for v in (five,s35,six,recurrent,audio)),
         'all original unseen native 59309 row IDs/ordering must coincide')
    five_names=list(map(str,five['family_ids']))
    six_names=list(map(str,six['family_ids']))
    recurrent_names=list(map(str,recurrent['family_ids']))
    expected=[v['family'] for v in json.loads(catalog.read_text())['original_families']]
    need(len(expected)==16 and len(set(expected))==16 and
         len(five_names)==5 and
         set(six_names)==set(REBASED) and
         set(recurrent_names)==set(RECURRENT) and
         set(five_names+six_names+recurrent_names+['S35_multiroute'])==set(expected),
         '16 exact original S44 champion trained families missing or duplicated')
    need(five['predictions'].shape==(19,n,5) and
         six['proposals_K'].shape==(19,n,5) and
         recurrent['proposals_K'].shape==(19,n,5) and
         s35['S35_K'].shape==(19,n) and
         chain['S18_K'].shape==(4,n),
         '19 saved genuine branches of all 16 original source families required')
    for origin in (five,six,recurrent):
        need(len(set(map(str,origin['branch_piece'])))==19 and
             [int(np.count_nonzero(origin['branch_fold']==f)) for f in FOLDS]==[5,4,5,5],
             'historical 19 held-piece branch fold provenance drift')
    need([int(np.count_nonzero(s35['fold']==f)) for f in FOLDS]==[5,4,5,5],
         'S35 true training pieces differ from original folds')
    models=Path(models)
    need(len(list(models.glob('*.joblib')))==304,
         'exact original 304 S45 trained risk checkpoints required')
    out.mkdir(parents=True)
    records=[];policy_names=None
    branch_count=0
    for fi,fold in enumerate(FOLDS):
        parent=chain['S18_K'][fi].astype(np.int8)
        original335=designs({'native_position':np.arange(n),'global_index':np.arange(n),
            'features':audio['summary58']},np.arange(n),chain['freeze_K'],
            chain['global_K'][fi],chain['poly_K'][fi],audio['geometry3'])['votes_time']
        X=np.column_stack((original335,audio['summary58'],audio['flow245'])).astype(np.float32)
        need(X.shape==(n,335) and np.isfinite(X).all(),
             'original historic trained Ridge risk input contract')
        for j in np.flatnonzero(five['branch_fold']==fold):
            piece=str(five['branch_piece'][j])
            def locate(d,pk,fk):
                p=np.flatnonzero((d[pk].astype(str)==piece)&(d[fk]==fold))
                need(len(p)==1,'historical same-piece same-fold original branch absent '+piece)
                return int(p[0])
            j35=locate(s35,'piece','fold')
            j242=locate(six,'branch_piece','branch_fold')
            j243=locate(recurrent,'branch_piece','branch_fold')
            family_proposals={name:five['predictions'][j,:,k]
                              for k,name in enumerate(five_names)}
            family_proposals.update({name:six['proposals_K'][j242,:,k]
                                     for k,name in enumerate(six_names)})
            family_proposals.update({name:recurrent['proposals_K'][j243,:,k]
                                     for k,name in enumerate(recurrent_names)})
            family_proposals['S35_multiroute']=s35['S35_K'][j35]
            need(set(family_proposals)==set(expected),'original 16 model source composition incomplete')
            proposals=np.column_stack([family_proposals[name] for name in expected]).astype(np.int8)
            need(proposals.shape==(n,16) and np.isin(proposals,range(7)).all(),
                 'source model K0–K6 invariant')
            folder=out/f'branch{branch_count:02d}_fold{fold}'
            folder.mkdir()
            np.savez_compressed(folder/'proposals.npz',row_id=ids,
                parent_K=parent,family_ids=np.asarray(expected),proposal_K=proposals)
            np.savez_compressed(folder/'features.npz',row_id=ids,acoustic_features=X)
            risk_predict(folder/'proposals.npz',folder/'features.npz',models,folder/'risk')
            s46_run(folder/'risk'/'s45_contract.npz',folder/'s46')
            pred=load(folder/'s46/predictions.npz')
            need(set(pred)=={'row_id','variant_ids','predictions'} and
                 np.array_equal(pred['row_id'],ids) and
                 pred['predictions'].shape==(n,193) and
                 np.array_equal(pred['predictions'][:,0],parent),
                 'genuine original 193 S46 outputs misaligned to original fold parent')
            if policy_names is None:policy_names=pred['variant_ids']
            need(np.array_equal(pred['variant_ids'],policy_names),
                 'original 193 policies differ between historical branches')
            records.append(dict(branch_id=branch_count,fold=int(fold),piece=piece,
                changed_source_proposals={
                    k:int(np.count_nonzero(family_proposals[k]!=parent)) for k in expected},
                policy_event_cells_changed_vs_S18=int(np.count_nonzero(
                    pred['predictions'][:,1:]!=parent[:,None]))))
            branch_count+=1
        print(json.dumps(dict(fold=int(fold),completed_branches=branch_count,
                              S46_policy_variants=193)),flush=True)
    need(branch_count==19 and [sum(r['fold']==f for r in records) for f in FOLDS]==[5,4,5,5],
         'original 19 branch confidence source census')
    manifest=dict(status='S267_FULL_16_GENUINE_ORIGINAL_S45_S46_59309_GUITARSET_NATIVE',
        original_native_reference_event_count=n,
        original_16_source_champion_family_names=expected,
        original_s45_ridge_checkpoint_count=304,
        original_s46_policy_count=193,
        original_fold_held_piece_branches=19,
        original_4fold_branch_sizes=[5,4,5,5],
        original_rare_S1_S2_S3_S4_coherent=5,
        original_S11_S12_S38_S40_S41_sources=5,
        original_recurrent_S19_S20_S24_S25_S27_sources=5,
        original_true_S29_S35_sources=1,
        S173_adapted_proxies_used=False,
        H9_models_used=False,
        per_branch_results=records,
        new_song_historic_held_piece_route_equivalence_proven=False,
        GuitarSet_fourfold_training_overlap_for_three_folds=True,
        original_GuitarSet_OOF_comparability_not_claimed=True,
        original_trained_checkpoints_unchanged=True,
        original_source_models_refitted=False,
        S46_label_free=True,
        GUITARSET_XML_MIDI_content_read=False,
        S69_S80_tail_not_yet_executed=True,
        independent_ExactK_accuracy_measured=False,
        S80_promoted=False)
    (out/'report.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='per_branch_results'},indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('five','s35','s242','s243','s18','s97','models','catalog','out'):
        p.add_argument('--'+k,type=Path,required=True)
    q=p.parse_args()
    run(q.five,q.s35,q.s242,q.s243,q.s18,q.s97,q.models,q.catalog,q.out)
