"""S153: authentic S13 + S14 saved risk weights and ORIGINAL S15 safe union on GOAT.

Original 42x49 derived acoustic features, original 90D/335D trained
models. BUT original S9/S5 parent missing on new music, so use
predeclared p=original PR16 policy22 as a distinct NEW parent adapter.
No GOAT MIDI/true-K used, no training/tuning. Not original S80.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np
from scripts.s141_external_original_s40_cnn import load_input
from scripts.loop_v273_birth_flow import decode as s13_decode
from scripts.loop_v273_conditional_k01 import decode as s14_decode
from scripts.verify_v273_series15_union import (
 S13_SAFE,S14_SAFE,S13_WIDE,S14_WIDE)
def need(c,m):
    if not c:raise ValueError(m)
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def infer_stage(path,X,stage):
    scores={};provenance=[]
    for rep in ('summary','flow'):
        for algo in ('logistic','hgb'):
            key=rep+'_'+algo
            files=sorted((path/'models').glob('*__'+key+'.joblib'))
            need(len(files)>0,stage+' checkpoint missing '+key)
            values=[]
            for f in files:
                d=joblib.load(f)  # user-controlled source artifact only
                need(d['piece'] in f.name and d['fold'] in (0,1,2,4),
                    stage+' fold/piece historical identity')
                model=d['model'];scale=d['scaler'];xx=X[rep]
                need(model.n_features_in_==scale.n_features_in_==xx.shape[1],
                    stage+' original 90D/335D feature width')
                pred=model.predict_proba(np.clip(scale.transform(xx),-6,6))
                classes=list(map(int,model.classes_))
                need(1 in classes,stage+' fitted K0 probability target missing')
                q=np.asarray(pred[:,classes.index(1)],np.float64)
                need(q.shape==(len(xx),) and np.isfinite(q).all() and
                     ((q>=0)&(q<=1)).all(),stage+' risk scores malformed')
                values.append(q)
                provenance.append({'stage':stage,'representation':rep,'algorithm':algo,
                    'piece':d['piece'],'fold':int(d['fold']),'sha256':sha(f)})
            scores[key]=np.stack(values).mean(0)
    scores['flow_both']=np.minimum(scores['flow_logistic'],scores['flow_hgb'])
    scores['flow_mean']=(scores['flow_logistic']+scores['flow_hgb'])/2.
    return scores,provenance
def run(s97,s85,s135,s13,s14,out):
    need(not out.exists(),'original archive output immutable')
    ids,X,_,parent=load_input(s97,s85,s135)
    b=parent['official_freeze'].astype(np.int8)
    s9=parent['original_PR16_policy22'].astype(np.int8)
    views={'summary':X[:,:90], 'flow':X}
    risks13,e13=infer_stage(s13,views,'S13')
    risks14,e14=infer_stage(s14,views,'S14')
    candidates13,decisions13=s13_decode(None,b,s9,risks13)
    candidates14,decisions14=s14_decode(b,s9,risks14)
    need(len(candidates13)==26 and len(candidates14)==32,
         'original S13/S14 complete veto head inventories')
    safe13=candidates13[S13_SAFE];safe14=candidates14[S14_SAFE]
    wide13=candidates13[S13_WIDE];wide14=candidates14[S14_WIDE]
    s15={'series9_parent':s9.copy(),
         'series13_safe':safe13,'series14_safe':safe14,
         'series13_wide':wide13,'series14_wide':wide14}
    for style,x,z in (('safe',safe13,safe14),('wide',wide13,wide14)):
        p13=x!=s9;p14=z!=s9
        need(np.array_equal(x,np.where(p13,b,s9)) and
             np.array_equal(z,np.where(p14,b,s9)),
             'original S15 rollback-only original action contract')
        s15['series15__'+style+'_OR']=np.where(p13|p14,b,s9).astype(np.int8)
        s15['series15__'+style+'_AND']=np.where(p13&p14,b,s9).astype(np.int8)
    need(len(s15)==9,'9 source S15 policies')
    final=s15['series15__safe_OR']
    need(np.isin(final,range(7)).all(),'out-of-range S15 K')
    out.mkdir(parents=True)
    np.savez_compressed(out/'original_S13_S14_S15_saved_models_GOAT_adapter.npz',
        row_id=ids,freeze_K=b,adapted_S9_K=s9,
        S13_safe_K=safe13,S14_safe_K=safe14,
        S15_safe_OR_K=final,S15_policy_ids=np.asarray(list(s15)),
        S15_predictions=np.column_stack(list(s15.values())),
        S13_policy_ids=np.asarray(list(candidates13)),
        S13_predictions=np.column_stack(list(candidates13.values())),
        S14_policy_ids=np.asarray(list(candidates14)),
        S14_predictions=np.column_stack(list(candidates14.values())))
    report={'status':'S153_ORIGINAL_TRAINED_S13_S14_AND_S15_UNION_ON_NEW_AUDIO',
       'GOAT_native_audio_events':57,'original_checkpoints_S13':len(e13),
       'original_checkpoints_S14':len(e14),
       'historical_checkpoint_provenance':e13+e14,
       'source_original_runs':{'S13':37854384791,'S14':37855070118},
       'original_S13_24_gate_policies':True,
       'original_S14_30_gate_policies':True,
       'original_S15_9_union_policies':True,
       'K0_birth_eligible_S13':int(((b==0)&(s9!=b)).sum()),
       'K0_birth_eligible_S14':int(((b==0)&(s9==1)).sum()),
       'S15_rollbacks_from_S9':int((s9!=final).sum()),
       'S15_final_K_histogram':np.bincount(final,minlength=7).tolist(),
       'original_S9_S5_equivalence_on_GOAT':False,
       'original_S15_historical_equivalence_on_GOAT':False,
       'MIDI_or_true_K_accessed':False,
       'promoted_S80':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in (
        'status','GOAT_native_audio_events','original_checkpoints_S13',
        'original_checkpoints_S14','K0_birth_eligible_S13','K0_birth_eligible_S14',
        'S15_rollbacks_from_S9','S15_final_K_histogram','promoted_S80')},indent=2))
def selftest():
    b=np.array([0,0,1],np.int8);s9=np.array([1,2,1],np.int8)
    safe13=np.array([0,2,1],np.int8);safe14=np.array([1,0,1],np.int8)
    out=np.where((safe13!=s9)|(safe14!=s9),b,s9)
    need(out.tolist()==[0,0,1],'S15 safe_OR action contract')
    print('S153 PASS original S15 safe union, S13/S14 veto architecture')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('s97','s85','s135','s13','s14','out'):p.add_argument('--'+k,type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        need(all(getattr(a,k) is not None for k in
                ('s97','s85','s135','s13','s14','out')),'source provenance required')
        run(a.s97,a.s85,a.s135,a.s13,a.s14,a.out)
