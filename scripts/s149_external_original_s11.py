"""S149: real ORIGINAL S11 risk checkpoints on 57 new GOAT audio groups.

ORIGINAL weights + scaler + design(32D/91D), but new-piece parents
are not original S5 global/poly and original S9. b=freeze, g=S135,
p=PR16 policy22, and S9 adapter = p. Models aggregated across original
held-out piece classifiers by equal averaging. Zero training, zero MIDI.
S11 historical/original S18 equivalence not claimed.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np
from scripts.s141_external_original_s40_cnn import load_input
from scripts.loop_v273_native_risk import designs,ARMS,ALGORITHMS,SCOPES,COSTS
def require(test,msg):
    if not test:raise ValueError(msg)
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def run(s97,s85,s135,s11,out):
    require(not out.exists(),'immutable original S11 GOAT report')
    ids,_,_,parents=load_input(s97,s85,s135)
    with np.load(s97,allow_pickle=False) as z:
        summary=z['summary58'].astype(np.float32)
        geo=z['geometry3'].astype(np.float32)
    b=parents['official_freeze'].astype(np.int8)
    g=parents['student_S135'].astype(np.int8)
    p=parents['original_PR16_policy22'].astype(np.int8)
    x=designs({'native_position':np.arange(len(ids)),
               'global_index':np.arange(len(ids)),
               'features':summary},np.arange(len(ids)),b,g,p,geo)
    require(x['votes_time'].shape==(57,32) and
            x['votes_time_audio'].shape==(57,91),'original S11 32D 91D source')
    # S9 historical original policy depended on missing S8 source vote.
    # Explicit deterministic NEW parent adapter, not exact original S9.
    source_s9=p.copy()
    changed=(source_s9!=b)
    allpol={'new_S9_parent_adapter':source_s9.copy(),
            'freeze_local_combo_parent':b.copy()}
    model_proof=[];decisions={};probs={}
    for arm in ARMS:
        for alg in ALGORITHMS:
            key=arm+'__'+alg
            files=sorted((s11/'models').glob('*__'+key+'.joblib'))
            require(len(files)>=10,'missing original S11 trained models '+key)
            predictions=[]
            for file in files:
                entry=joblib.load(file) # only original pinned user GitHub archive
                require(entry['arm']==arm and entry['kind']==alg and
                        entry['piece'] in file.name,'original S11 saved identity')
                model=entry['model'];sc=entry['scaler']
                size=x[arm].shape[1]
                require(model.n_features_in_==sc.n_features_in_==size,
                        'historical S11 input feature dimensionality')
                sample=np.clip(sc.transform(x[arm]),-6,6)
                score=model.predict_proba(sample)
                full=np.zeros((len(ids),7),np.float64)
                full[:,np.asarray(model.classes_,int)]=score
                require(np.isfinite(full).all() and
                        np.allclose(full.sum(1),1,atol=1e-5),
                        'original S11 model output normalization')
                predictions.append(full)
                model_proof.append(dict(source_piece=entry['piece'],
                                  fold=int(entry['fold']),arm=arm,kind=alg,
                                  checkpoint_sha256=sha(file)))
            average=np.mean(predictions,axis=0)
            probs[key]=average.astype(np.float32)
            for scope in SCOPES:
                for cost in COSTS:
                    mask=((b==0) if scope=='k0' else np.ones(len(ids),bool))
                    advantage=average[np.arange(len(ids)),b]-average[np.arange(len(ids)),source_s9]
                    veto=changed&mask&(advantage>cost)
                    result=source_s9.copy();result[veto]=b[veto]
                    label=f'series11__{key}__{scope}__cost{cost:g}'
                    allpol[label]=result
                    decisions[label]={'veto_count':int(veto.sum()),
                                      'changed_from_adapter_s9':int((result!=source_s9).sum())}
    require(len(allpol)==26 and len(probs)==4,'original S11 24 gate rule coverage')
    out.mkdir(parents=True)
    np.savez_compressed(out/'S11_real_original_weights_new_GOAT_adapted.npz',
       row_id=ids,variant_ids=np.array(list(allpol)),
       predicted_K=np.column_stack(list(allpol.values())),
       probability_variant_ids=np.array(list(probs)),
       original_model_mean_probabilities=np.stack(list(probs.values()),axis=1),
       freeze_K=b)
    report={'status':'S149_ORIGINAL_S11_WEIGHTS_EXECUTED_GOAT_WITH_NEW_PARENT_ADAPTER',
            'source_run':37853663514,'original_models_executed':len(model_proof),
            'GOAT_real_audio_events':len(ids),
            'S11_24_original_policy_formulas_executed':True,
            'S11_original_weights_and_scalers_used':True,
            'model_evidence':model_proof,
            'S11_26_candidate_K_distributions':{k:np.bincount(a,minlength=7).tolist()
                                                for k,a in allpol.items()},
            'gate_decisions':decisions,
            'source_parent_mapping_NEW':'b=freeze,g=S135,p=PR16#22; S9=p',
            'original_series9_parent_verified_on_GOAT':False,
            'original_S11_historical_equivalence_on_GOAT':False,
            'original_S18_external_inference_complete':False,
            'truth_MIDI_read':False,'full_S80_validated':False,'promotion':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in
          ('status','source_run','original_models_executed','GOAT_real_audio_events',
           'S11_24_original_policy_formulas_executed',
           'original_series9_parent_verified_on_GOAT','promotion')},indent=2))
def selftest():
    b=np.array([0,1,2],np.int8);p=np.array([1,1,3],np.int8)
    scores=np.zeros((3,7));scores[np.arange(3),b]=.8
    scores[0,1]=.1;scores[2,3]=.1
    advantage=scores[np.arange(3),b]-scores[np.arange(3),p]
    v=(p!=b)&(advantage>.25)
    assert v.tolist()==[True,False,True]
    print('S149 PASS original S11 veto math and 32/91D contract')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('s97','s85','s135','s11','out'):p.add_argument('--'+n,type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        require(all(getattr(a,n) is not None for n in
                    ('s97','s85','s135','s11','out')),'require all model/audio paths')
        run(a.s97,a.s85,a.s135,a.s11,a.out)
