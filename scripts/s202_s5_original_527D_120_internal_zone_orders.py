"""S202 systematic fixed 5! within-zone S5 527D order falsification.

Each historical 104D zone in the S118 *unverified* suffix contains an
activation20 and four lag21 blocks (1/2/4/8-frame). Test all 5! consistent
block orderings across zones, including previous exact identity S123.
Only historical original SAVED S5 posteriors are used as fidelity references.
No MIDI, true K, tuning, training, new-WAV accuracy or promotion.
"""
from __future__ import annotations
import argparse,itertools,json
from pathlib import Path
import numpy as np,joblib
from scripts.s201_s5_original_527D_four_zone_permutation_fidelity import (
    inputs,subset,need,digest)
from scripts.s118_exploratory_s4_owned527 import extra416

LENGTHS=(20,21,21,21,21)
def run(features,original,model_file,model_report,cached,out,num=512):
    need(not out.exists(),'immutable zone internal permutation study')
    record=json.loads(model_report.read_text())
    need(digest(model_file)==record['model_sha256'],'pinned model SHA drift')
    need(digest(original)==record['prediction_sha256'],'pinned original S5 posteriors SHA drift')
    model=joblib.load(model_file)['model']
    need(model.n_features_in_==527 and np.array_equal(model.classes_,np.arange(7)),
         'trained S5 original model mismatch')
    a=inputs(features,original,cached)
    at=subset(a['global_index'],num)
    suffix=np.stack([extra416(a['sequence'][i],a['geometry'][i,:42]) for i in at])
    prefix=np.column_stack([a['summary'][at],a['geometry'][at],
                 np.eye(7,dtype=np.float32)[a['baseline'][at].astype(int)]])
    need(prefix.shape==(num,111) and suffix.shape==(num,416),'original 111+416 candidate contract')
    original_prob=a['saved_original_prob'][at].astype(np.float64)
    saved_s123=a['cached_replay_prob'][at].astype(np.float64)
    zones=suffix.reshape(num,4,104)
    starts=np.cumsum((0,)+LENGTHS)
    parts=[zones[:,:,starts[j]:starts[j+1]] for j in range(5)]
    results=[]
    for perm in itertools.permutations(range(5)):
        new=np.concatenate([parts[k] for k in perm],axis=2).reshape(num,416)
        X=np.column_stack([prefix,new]).astype(np.float32)
        prob=model.predict_proba(X).astype(np.float64)
        need(np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=1e-5),'real S5 bad posterior')
        error=np.abs(prob-original_prob)
        verdict={'within_each_zone_5block_order':list(perm),
                 'S5_saved_argmax_disagreements':int(np.count_nonzero(prob.argmax(1)!=original_prob.argmax(1))),
                 'saved_posterior_MAE':float(error.mean()),
                 'saved_posterior_max_abs_diff':float(error.max()),
                 'full_probability_vectors_1e_minus_6':int(np.sum(np.all(error<1e-6,axis=1)))}
        if perm==(0,1,2,3,4):
            identity_error=float(np.max(np.abs(prob-saved_s123)))
            need(identity_error<5e-6,'identity must preserve original S123 candidate posterior')
            verdict['S123_posterior_max_abs_identity_diff']=identity_error
        results.append(verdict)
    best=sorted(results,key=lambda x:(x['S5_saved_argmax_disagreements'],
                  x['saved_posterior_MAE'],x['within_each_zone_5block_order']))
    detail={'status':'S202_ORIGINAL_S5_527D_FIVE_WITHIN_ZONE_BLOCKS_120_FIXED_ORDER_FIDELITY',
        'historical_source_fold':0,'source_original_audio_events_tested':num,
        'source_original_104D_hypothesis':'activation20 and lag1+lag2+lag4+lag8 each21D, four temporal ownership zones',
        'source_candidate_416D_semantics':'S118 exploratory, not proven exact original',
        'current_known_original_four_104D_zone_order_unmodified':True,
        'permutations_predeclared':120,
        'original_S5_saved_prediction_SHA256':digest(original),
        'original_trained_S5_checkpoint_SHA256':digest(model_file),
        'all_120_fixed_hypothesis_results':results,
        'best_historical_fidelity_by_Kdisagreements':best[0],
        'historical_identity_S123_metrics':results[0],
        'permutations_matching_all_original_probability_vectors_1e_minus_6':[
            x['within_each_zone_5block_order'] for x in results
            if x['full_probability_vectors_1e_minus_6']==num],
        'historical_truth_or_MIDI_used':False,
        'source_checkpoint_refit_or_retrained':False,
        'S5_527D_builder_exact_original_verified':False,
        'new_audio_S80_ExactK_measured':False,
        'S80_promoted':False}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(detail,indent=2)+'\n')
    print('S202_SUMMARY '+json.dumps({k:detail[k] for k in (
      'permutations_predeclared','best_historical_fidelity_by_Kdisagreements',
      'historical_identity_S123_metrics',
      'permutations_matching_all_original_probability_vectors_1e_minus_6',
      'S5_527D_builder_exact_original_verified','S80_promoted')},sort_keys=True),flush=True)
def selftest():
    need(sum(LENGTHS)==104 and len(tuple(itertools.permutations(range(5))))==120,
         'complete five blocks/24 lags 120 intra-zone orders')
    t=np.arange(104).reshape((1,1,104))
    start=np.cumsum((0,)+LENGTHS)
    parts=[t[:,:,start[j]:start[j+1]] for j in range(5)]
    need(np.array_equal(np.concatenate(parts,axis=2),t),'identity internal order')
    print('S202 PASS 104D 5! complete internal ordering identity')
if __name__=='__main__':
    a=argparse.ArgumentParser()
    for v in ('features','original','model','checkpoint-report','cached','out'):
        a.add_argument('--'+v,type=Path)
    a.add_argument('--count',type=int,default=512)
    a.add_argument('--self-test',action='store_true')
    p=a.parse_args()
    if p.self_test:selftest()
    else:
        need(all((p.features,p.original,p.model,p.checkpoint_report,p.cached,p.out)),
             'authentic original S5 raw probability and acoustic checkpoint files required')
        run(p.features,p.original,p.model,p.checkpoint_report,p.cached,p.out,p.count)
