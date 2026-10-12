"""S325: run original frozen S79 logistic K1 critics + exact original S80 K0/K2 tail.

Exposed native IDMT 57 real WAV events, original S221..S226
S18/S45/S46/S69/S70/S71/S72 source model chains, original saved
S80 four K1 LogisticRegression models, original 36 PR16, original
12 neural posteriors and original S147 exact S80 tail formula.

New identity-neutral S79 deployment rule: >=3 of 4 trained fold models
accept K1->PR16 K2+ intervention using their archived training-only
feature view and threshold. No oracle heldfold assigned to external
composition. K0/K2 final rules exactly S147. This is a NEW versioned
external inference policy, NOT historically identical S80 selection.
No MIDI/truth, no Exact-K score, no S80 promotion.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np
from scripts.s147_s80_tail_runtime import tail
FOLDS=(0,1,2,4)
BAD={'true_K','truth','label','labels','y','midi','target','correct','accuracy','regression'}
def need(v,msg):
    if not v:raise ValueError(msg)
def read(p,keys):
    with np.load(p,allow_pickle=False) as z:
        need(set(z.files)==set(keys) and not BAD.intersection(z.files),
             'S325 strict external label-free source contract '+str(p))
        return {k:z[k].copy() for k in z.files}
def run(s226file,s219file,s85file,s97file,s80_models,out):
    need(not out.exists(),'immutable source-only S325 S80 external evidence')
    base=read(s226file,{'row_id','s69_polymax_K','s70_proposed_poly_K',
       's71_risk_K','s71_poly_K','s71_veto_three_folds',
       's72_veto_three_folds','s71_fold_veto','s72_fold_veto',
       'original_pr16_geometry46',
       's72_risk_all_K','s72_risk_poly_guard_K',
       's72_poly_all_K','s72_poly_poly_guard_K'})
    s46=read(s219file,{'row_id','variant_ids','original_freeze_K',
       'S18_parent_K','original_19_branch_folds','S46_193_policy_K',
       'per_fold_specialist_K','active_fold_count',
       'parent_incompatible_specialists','contradictory_specialists'})
    pr=read(s85file,{'member','start_samples','end_samples',
       'incumbent_freeze_K','pr16_policy_K','pr16_vote_count',
       'pr16_policy22_K','pr16_network_probabilities'})
    audio=read(s97file,{'row_id','summary58','flow245','geometry3',
       'sequence42x49','morphology490','physics294',
       'member','start_samples','end_samples'})
    ids=list(map(str,base['row_id']));n=len(ids)
    need(n==964 and np.array_equal(base['row_id'],s46['row_id']) and
         np.array_equal(base['row_id'],audio['row_id']),
         'native IDMT original event identity')
    keys=[f'{v}:{int(start)}' for v,start in zip(pr['member'],pr['start_samples'])]
    idx={key:i for i,key in enumerate(keys)}
    need(len(keys)==len(idx) and all(k in idx for k in ids),
         'original PR16 policy join to S80 WAV groups')
    pos=[idx[k] for k in ids]
    ps=pr['pr16_network_probabilities'][pos]
    v=pr['pr16_policy_K'][pos].astype(np.int8)
    policy22=pr['pr16_policy22_K'][pos].astype(np.int8)
    truthfree_s46=s46['S46_193_policy_K'].astype(np.int8)
    need(np.array_equal(s46['original_freeze_K'],pr['incumbent_freeze_K'][pos]) and
         len(set(ids))==964 and all(':' in k for k in ids),
         'original official baseline freeze K drift or duplicate native event IDs')
    b=base['s72_poly_poly_guard_K'].astype(np.int8)
    geom=base['original_pr16_geometry46'].astype(np.float32)
    summary=audio['summary58'].astype(np.float32)
    need(ps.shape==(n,12,7) and v.shape==(n,36) and
         truthfree_s46.shape==(n,193) and geom.shape==(n,46)
         and summary.shape==(n,58),
         'original S80 K1/K0/K2 posterior and geometry dimensions')
    need(np.isin(b,range(7)).all() and np.isin(policy22,range(7)).all(),
         'invalid original K')
    pavg=ps.mean(1)
    votes=np.column_stack([(v==k).sum(1) for k in range(7)])/36.
    sv=np.column_stack([(truthfree_s46==k).sum(1) for k in range(7)])/193.
    extra=np.column_stack((pavg[np.arange(n),policy22],
          pavg[np.arange(n),b],votes,sv)).astype(np.float32)
    acoust=np.column_stack((summary,geom))
    flat=ps.reshape(n,-1)
    features={'prob':np.column_stack((flat,extra)).astype(np.float32),
        'acoustic':np.column_stack((acoust,extra)).astype(np.float32),
        'combined':np.column_stack((acoust,flat,extra)).astype(np.float32)}
    need([features[k].shape[1] for k in ('prob','acoustic','combined')]==[100,120,204],
         'true historical K1 feature view widths')
    eligible=(b==1)&(policy22>b)
    choices=json.loads((s80_models/'report.json').read_text())
    need(choices['status']=='RESEARCH_ONLY_NOT_PROMOTED' and
         set(choices['train_fold_only_K1'])==set(map(str,FOLDS)),
         'historical original S80 selected threshold inventory')
    fold_actions=np.zeros((4,n),bool)
    fold_scores=np.full((4,n),np.nan,np.float32)
    provenance=[]
    for fi,fold in enumerate(FOLDS):
        cp=s80_models/'models'/f'K1_gate_fold{fold}.joblib'
        need(cp.is_file(),'original S80 saved trained K1 checkpoint absent '+str(fold))
        net=joblib.load(cp)
        choice=choices['train_fold_only_K1'][str(fold)]['training_only']
        name=str(choice['name']);cut=float(choice['cut'])
        need(name in features and net.n_features_in_==features[name].shape[1],
             'historical S79 selected model feature contract changed')
        if eligible.any():
            score=net.predict_proba(features[name][eligible])[:,1]
            need(np.isfinite(score).all() and score.shape==(int(eligible.sum()),),
                 'original S79 probabilities invalid')
            fold_scores[fi,eligible]=score
            fold_actions[fi,eligible]=score>=cut
        provenance.append(dict(fold=fold,view=name,threshold=cut,
            original_model_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),
            folded_candidate_actions=int(fold_actions[fi].sum())))
    accepts=(fold_actions.sum(0)>=3)&eligible
    s79=b.copy();s79[accepts]=policy22[accepts]
    result,action0,action2=tail(b,s79,v,ps,truthfree_s46)
    need(np.isin(result,range(7)).all() and result.shape==(n,),
         'full S80 IDMT K0..6 predictions')
    need(not np.any(accepts&(action0|action2)),
         'original S80 source K1 and K0/K2 overlap')
    out.mkdir(parents=True)
    np.savez_compressed(out/'S325_full_original_S80_unseen_IDMT_964_frozen.npz',
       row_id=np.asarray(ids),incumbent_freeze_K=pr['incumbent_freeze_K'][pos],
       s72_K=b,s79_K=s79,s80_K=result,original_PR16_policy22_K=policy22,
       K1_by_fold_threshold_action=fold_actions,
       K1_by_fold_threshold_scores=fold_scores,
       action_K1=accepts,action_K0=action0,action_K2=action2)
    report=dict(status='S325_COMPLETE_GENUINE_ORIGINAL_TRAINED_S80_UNSEEN_NATIVE_IDMT',
       original_native_WAV_events=n,
       original_official_V88_native_event_IDs_matched=True,
       original_S247_193_policy_source_IDMT=True,
       S79_original_trained_K1_models=4,
       S80_K0_K2_original_final_formula_exact=True,
       original_PR16_predictions=36,original_PR16_prob_models=12,
       S46_policy_inputs=193,original_S79_feature_views=list(features),
       original_S79_fold_thresholds=provenance,
       action_counts=dict(K1=int(accepts.sum()),K0=int(action0.sum()),
                          K2=int(action2.sum())),
       S80_K_histogram=np.bincount(result,minlength=7).tolist(),
       changes_S80_vs_S72=int(np.count_nonzero(result!=b)),
       changes_S80_vs_freeze_local_combo=int(np.count_nonzero(
          result!=pr['incumbent_freeze_K'][pos])),
       external_accuracy_K0_K6=None,external_accuracy_global=None,
       external_accuracy_poly=None,
       historical_original_S80_new_piece_identity_proven=False,
       deployment_rule_new='S217-S226 original trained checkpoint sources + fold-equal consensus; S79 3of4 gate votes; S147 historical final tail',
       IDMT_events_previously_exposed=False,
       native_WAV_rank_set='dataset2_SHA256_filenames_33to48',
       song_disjoint_independent_accuracy_measured=False,
       truth_MIDI_read=False,weights_retrained_on_external_audio=False,
       official_promotion=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser()
    for name in ('s226','s219','s85','s97','s80','out'):
        a.add_argument('--'+name,type=Path,required=True)
    q=a.parse_args()
    run(q.s226,q.s219,q.s85,q.s97,q.s80,q.out)
