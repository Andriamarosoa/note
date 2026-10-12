"""S241: reconnect original S29 outputs to original pretrained S35 per-fold model.

This is strictly label-free evidence on UNLABELLED IDMT ranking 33–48. No choice of new
recording's historical held-piece branch. Historical S35 training used a
piece-specific S29 OOF source, so all 19 S29 pieces remain separate.
No full S80, external Exact-K, promotion, or fold aggregation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import joblib
from scripts.s173_original_s35_weights_new_goat import FOLDS, TRAINABLE, ACTIONS, POLY_HEADS, need
from scripts.train_v273_extended_dynamic_router import dynamic_route
from scripts.v273_extended_head_bank import bank_feature_views
from scripts.loop_v273_native_risk import designs

BAD={'true_K','truth','label','labels','target','midi','correct'}

def read(path,keys):
    with np.load(path,allow_pickle=False) as z:
        need(set(z.files)==set(keys) and not set(z.files).intersection(BAD),
             'label-free S210 input schema: '+str(path))
        return {k:z[k].copy() for k in z.files}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fold_models(root,fold,views):
    """Only genuine source outer-fold experts and Q gates from the same fold."""
    n=len(next(iter(views.values())))
    bank={};gates={};checks=[]
    for head in TRAINABLE:
        file=root/'models'/f'fold{fold}__outer__{head}.joblib'
        need(file.is_file(),'absent S35 outer acoustic checkpoint '+str(file))
        state=joblib.load(file)
        need(state['name']==head and int(state['outer_fold'])==fold and
             state['inner_fold'] is None,'original S35 acoustic provenance')
        scaled=np.clip(state['scaler'].transform(views[head]),-5,5)
        q=np.asarray(state['model'].predict_proba(scaled),np.float32)
        classes=list(map(int,state['model'].classes_))
        expected=([1,2] if head=='E12_second_note' else
                  [2,3,4,5,6] if head in POLY_HEADS else list(range(7)))
        need(classes==expected and q.shape==(n,len(expected)) and
             np.allclose(q.sum(axis=1),1,atol=1e-4),
             'S35 expert classes/probabilities mismatch')
        p=np.zeros((n,7),np.float32)
        p[:,classes]=q
        need(np.isfinite(p).all(),'expert nonfinite')
        bank[head]=p
        checks.append(dict(fold=fold,type='expert',head=head,sha256=sha(file)))
    for head in ACTIONS:
        file=root/'models'/f'outer{fold}__Q__{head}.joblib'
        if not file.is_file(): continue
        state=joblib.load(file)
        need(state['head']==head and int(state['outer'])==fold and
             state['model'].n_features_in_==115 and
             state['scaler'].n_features_in_==115,
             'original S35 dynamic risk gate provenance')
        gates[head]={'model':state['model'],'scaler':state['scaler']}
        checks.append(dict(fold=fold,type='quality',head=head,sha256=sha(file)))
    need(len(bank)==9 and len(gates)>0,'frozen S35 per-fold bank incomplete')
    return bank,gates,checks

def run(s209,s18,s97,s35,out):
    need(not out.exists(),'immutable S210 evidence only')
    src=read(s209,{'row_id','piece','fold','predicted_K','actions','calls'})
    chain=read(s18,{'row_id','freeze_K','fold_ids','global_K','poly_K',
                    'S8_K','S9_K','S13_K','S14_K','S15_K','S16_K','S17_K','S18_K'})
    audio=read(s97,{'row_id','summary58','flow245','geometry3','sequence42x49',
                    'morphology490','physics294','member','start_samples','end_samples'})
    ids=src['row_id'];n=len(ids)
    need(n==1034 and all(np.array_equal(ids,z['row_id']) for z in (chain,audio)),
         'genuine S29/S35 source row alignment')
    need(src['predicted_K'].shape==(19,n) and
         len(set(map(str,src['piece'])))==19 and
         chain['S18_K'].shape==(4,n),'all S29 routes and 4 S18 upstream parents')
    need(tuple(map(int,chain['fold_ids']))==FOLDS and
         [int(np.sum(src['fold']==f)) for f in FOLDS]==[5,4,5,5],
         'original S29 piece-held lineage')
    out_preds=np.full((19,n),-1,np.int8)
    out_routes=np.full((19,n,3),-2,np.int8)
    checks=[];fold_stats=[]
    for fi,fold in enumerate(FOLDS):
        parent=chain['S18_K'][fi].astype(np.int8)
        votes=designs({'native_position':np.arange(n),'global_index':np.arange(n),
                       'features':audio['summary58']},np.arange(n),
                      chain['freeze_K'],chain['global_K'][fi],
                      chain['poly_K'][fi],audio['geometry3'])['votes_time']
        x=np.column_stack((votes,audio['summary58'],audio['flow245'])).astype(np.float32)
        need(x.shape==(n,335) and np.isfinite(x).all(),'genuine S35 335D feature contract')
        views=bank_feature_views(x,audio['morphology490'],audio['physics294'])
        bank,gates,proof=fold_models(s35,fold,views)
        checks.extend(proof)
        per=[]
        for i in np.flatnonzero(src['fold']==fold):
            ab=src['predicted_K'][i]
            pred,route=dynamic_route(x,parent,bank,ab,gates,1.,0.)
            need(pred.shape==(n,) and route.shape==(n,3) and
                 np.isin(pred,range(7)).all() and
                 np.all((route>=-1)&(route<len(ACTIONS))),
                 'S35 authentic 3-pass routing output')
            out_preds[i]=pred
            out_routes[i]=route
            per.append(dict(piece=str(src['piece'][i]),source_S29_to_S18_changes=
                            int(np.count_nonzero(ab!=parent)),
                            S35_to_true_S18_changes=int(np.count_nonzero(pred!=parent)),
                            S35_action_count=int(np.count_nonzero(route>=0))))
        fold_stats.append(dict(fold=fold,pieces=len(per),
                original_experts=len(bank),quality_gates=len(gates),
                piece_results=per))
    need(np.all(out_preds>=0) and np.all(out_routes>=-1),
         'incomplete S35 original source replay')
    out.mkdir(parents=True)
    np.savez_compressed(out/'original_s29_to_s35_IDMT.npz',
         row_id=ids,piece=src['piece'],fold=src['fold'],
         S29_K=src['predicted_K'],S35_K=out_preds,S35_actions=out_routes)
    summary=dict(status='S241_FROZEN_S29_19_BRANCH_TO_S35_PER_FOLD_ROUTING',
         source_S29_run=37865173636,source_S35_run=37868356632,
         events=1034,branches=19,original_S29_Q_models=38,
         original_S35_expert_checkpoints=36,
         original_S35_quality_gate_checkpoints=len([r for r in checks if r['type']=='quality']),
         fixed_historical_S35_policy='series35__lambda1__threshold0',
         same_fold_source_inputs=True,source_models_unchanged=True,
         fold_results=fold_stats,
         checkpoints=checks,labels_read=False,models_refitted=False,
         source_S20_for_S29_proxy_eliminated=True,
         historical_piece_heldout_assignment_possible_on_new_audio=False,
         new_piece_branch_aggregation_fixed=False,
         source_S45_16_families_recomputed=False,
         full_S80_reconstructed=False,independent_ExactK_scored=False,
         S80_promoted=False)
    (out/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('checkpoints','fold_results')},indent=2),flush=True)
    print(json.dumps(fold_stats),flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('s209','s18','s97','s35','out'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    run(a.s209,a.s18,a.s97,a.s35,a.out)
