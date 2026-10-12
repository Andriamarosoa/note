"""S150: forward ORIGINAL S18 K4->K3 and K2->K1 frozen risk models on GOAT.

Experimental NEW chain S11-adapted -> S18 original veto weights.
Original S16/S17 K12/K23/K32/K34 vetoes are not yet used;
thus genuine S18 source parent NOT reproduced. No labels at inference.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np
from scripts.s141_external_original_s40_cnn import load_input
from scripts.loop_v273_k43_k21 import (DIRECTIONS,ARMS,SCOPES,
                                        THRESHOLDS,make_policies)
def need(c,m):
    if not c:raise ValueError(m)
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def run(s97,s85,s135,s11,s18,out):
    need(not out.exists(),'immutable results')
    ids,X,_,parents=load_input(s97,s85,s135)
    b=parents['official_freeze'].astype(np.int8)
    with np.load(s11,allow_pickle=False) as z:
        need(set(z.files)=={'row_id','variant_ids','predicted_K',
                            'probability_variant_ids',
                            'original_model_mean_probabilities','freeze_K'},
             'strict S149 input schema')
        need(np.array_equal(z['row_id'].astype(str),ids) and
             np.array_equal(z['freeze_K'],b),'S11/S18 native events align')
        names=list(map(str,z['variant_ids']))
        need('series11__votes_time_audio__hgb__all__cost0' in names,
             'pre-fixed S11 deployed adapter parent not available')
        # Preregistered policy before ever accessing GOAT labels:
        parent=z['predicted_K'][:,names.index(
            'series11__votes_time_audio__hgb__all__cost0')].astype(np.int8)
    need(X.shape==(57,335),'genuine S18 source input: 32+58+245')
    view={'summary':X[:,:90],
          'flow':X}
    bank={};records=[];model_counts={}
    for (source,dest),prefix in zip(DIRECTIONS,('k43','k21')):
        for rep in ('summary','flow'):
            for alg in ('logistic','hgb'):
                files=sorted((s18/'models').glob(prefix+'__*__'+rep+'_'+alg+'.joblib'))
                # A specific historical model may have no eligible test piece.
                # Record absence rather than manufacturing a probability.
                key=f'{prefix}__{rep}_{alg}'
                arr=[]
                for file in files:
                    state=joblib.load(file)
                    need(state['direction']==prefix and
                        state['piece'] in file.name and
                        state['fold'] in (0,1,2,4),'original S18 checkpoint identity')
                    estimator=state['model'];sc=state['scaler']
                    need(estimator.n_features_in_==sc.n_features_in_==view[rep].shape[1],
                         'S18 fit feature width changed')
                    score=estimator.predict_proba(
                        np.clip(sc.transform(view[rep]),-6,6))
                    classes=list(map(int,estimator.classes_))
                    need(1 in classes,'original S18 true-baseline class absent')
                    p=np.asarray(score[:,classes.index(1)],np.float64)
                    need(p.shape==(57,) and np.isfinite(p).all() and
                         np.all((p>=0)&(p<=1)),'S18 source binary risk probabilities')
                    arr.append(p)
                    records.append({'direction':prefix,'representation':rep,
                                    'algo':alg,'piece':state['piece'],
                                    'fold':int(state['fold']),'sha256':sha(file)})
                # In case an original direction had no saved classifier,
                # report zero genuine coverage and fail, not fill fake 0.5.
                need(len(arr)>0,'no original S18 model for '+key)
                bank[key]=np.stack(arr).mean(0)
                model_counts[key]=len(arr)
        bank[f'{prefix}__flow_both']=np.minimum(
            bank[f'{prefix}__flow_logistic'],
            bank[f'{prefix}__flow_hgb'])
        bank[f'{prefix}__flow_mean']=(
            bank[f'{prefix}__flow_logistic']+
            bank[f'{prefix}__flow_hgb'])/2
    policies,eligible=make_policies(b,parent,bank)
    need(len(policies)==110 and len(records)>0,'108 original S18 veto policies')
    out.mkdir(parents=True)
    np.savez_compressed(out/'s18_original_weights_GOAT_from_adapted_S11.npz',
        row_id=ids,parent_s11_K=parent,freeze_K=b,
        risk_variant_ids=np.array(list(bank)),
        risk_probs=np.column_stack(list(bank.values())),
        policy_variant_ids=np.array(list(policies)),
        policy_predicted_K=np.column_stack(list(policies.values())))
    result={'status':'S150_S18_ORIGINAL_SAVED_WEIGHTS_NEW_GOAT_ADAPTED_S11_PARENT',
       'source_original_run':37856499212,
       'original_saved_S18_models_executed':len(records),
       'model_count_by_direction_rep_algorithm':model_counts,
       'checkpoint_evidence':records,
       'GOAT_real_events':len(ids),
       'eligible_K4_to_K3':int(eligible['k43'].sum()),
       'eligible_K2_to_K1':int(eligible['k21'].sum()),
       '108_original_veto_policy_formulas_executed':True,
       'changed_K_by_policy':{k:int((v!=parent).sum()) for k,v in policies.items()},
       'S11_adapted_parent_name':'series11__votes_time_audio__hgb__all__cost0',
       'S16_S17_original_parent_chain_connected':False,
       'original_S18_historical_parent_equivalence':False,
       'GOAT_MIDI_labels_read':False,'external_exactK_measured':False,
       'full_S80_completed':False,'promotion':False}
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in (
      'status','original_saved_S18_models_executed','GOAT_real_events',
      'model_count_by_direction_rep_algorithm',
      'eligible_K4_to_K3','eligible_K2_to_K1',
      'S16_S17_original_parent_chain_connected','promotion')},indent=2))
def selftest():
    b=np.array([4,2,2,1],np.int8)
    parent=np.array([3,1,2,1],np.int8)
    probs={f'{name}__{arm}':np.full(4,.99,np.float64)
       for name in ('k43','k21') for arm in ARMS}
    policies,eligible=make_policies(b,parent,probs)
    need(len(policies)==110 and eligible['k43'].sum()==1 and
         eligible['k21'].sum()==1,'original S18 veto masks changed')
    print('S150 PASS 108 exact original S18 K43/K21 gate rules')
if __name__=='__main__':
    a=argparse.ArgumentParser()
    for k in ('s97','s85','s135','s11','s18','out'):a.add_argument('--'+k,type=Path)
    a.add_argument('--self-test',action='store_true')
    x=a.parse_args()
    if x.self_test:selftest()
    else:
        need(all(getattr(x,k) is not None for k in
             ('s97','s85','s135','s11','s18','out')),'six provenance sources')
        run(x.s97,x.s85,x.s135,x.s11,x.s18,x.out)
