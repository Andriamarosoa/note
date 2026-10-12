"""S201 exhaustive fixed 4! temporal-zone permutation *falsification* on genuine S5.

Uses original PINNED S5 checkpoint and saved S5 raw probabilities as a
historical OUTPUT FIDELITY REFERENCE. Selects 512 original fold0 events by
SHA256(global_index), without viewing truth, recording ID or song. Each of
24 permutations reorders only 4 genuine 104D hypothetical *candidate* blocks
from S118 exploratory log-activation suffix; no retraining or tuning.
Zero posterior equality is necessary (not sufficient) for original 527D.
"""
from __future__ import annotations
import argparse,hashlib,itertools,json
from pathlib import Path
import numpy as np,joblib
from scripts.s118_exploratory_s4_owned527 import extra416

def need(x,m):
    if not x:raise ValueError(m)
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def inputs(features,raw_archive,cached):
    with np.load(features,allow_pickle=False) as z:
        a={k:z[k].copy() for k in ('global_index','summary','geometry','sequence','baseline')}
        # explicitly no 'k', labels, MIDI or truth
    with np.load(raw_archive,allow_pickle=False) as z:
        a['saved_original_id']=z['global_index'].copy()
        a['saved_original_prob']=z['raw'].copy()
        a['saved_original_fold']=z['fold'].copy()
    with np.load(cached,allow_pickle=False) as z:
        a['cached_id']=z['global_index'].copy()
        a['cached_replay_prob']=z['probabilities'].copy()
    n=len(a['global_index'])
    need(n==15952 and np.array_equal(a['global_index'],a['saved_original_id'])
         and np.array_equal(a['global_index'],a['cached_id']),
         'original fold0 alignment')
    need(np.all(a['saved_original_fold']==0),'unexpected original fold')
    need(a['summary'].shape==(n,58) and a['geometry'].shape==(n,46)
         and a['sequence'].shape==(n,42,49)
         and a['saved_original_prob'].shape==(n,7),
         'source original acoustic and posterior array contract')
    return a
def subset(ids,count):
    hashes=np.asarray([hashlib.sha256(str(int(x)).encode()).hexdigest()
                       for x in ids])
    return np.argsort(hashes,kind='stable')[:count]
def run(features,original,model_file,model_report,cached,out,num=512):
    need(not out.exists(),'immutable 24 zone reorder hypothesis')
    report=json.loads(model_report.read_text())
    need(digest(model_file)==report['model_sha256'],'pinned original S5 model SHA changed')
    need(digest(original)==report['prediction_sha256'],'pinned original S5 probability evidence changed')
    model=joblib.load(model_file)['model']
    need(model.n_features_in_==527 and np.array_equal(model.classes_,np.arange(7)),
         'original S5 trained 527D class contract changed')
    a=inputs(features,original,cached)
    at=subset(a['global_index'],num)
    need(len(at)==num and len(set(at))==num,'SHA256 candidate subset changed')
    suffix=np.stack([extra416(a['sequence'][i],a['geometry'][i,:42]) for i in at])
    need(suffix.shape==(num,416),'original hypothesis suffix must be four 104D')
    prefix=np.column_stack([a['summary'][at],a['geometry'][at],
         np.eye(7,dtype=np.float32)[a['baseline'][at].astype(int)]])
    target=a['saved_original_prob'][at]
    baseline=a['cached_replay_prob'][at]
    need(prefix.shape==(num,111),'source 111D acoustic+geometry+freeze')
    batches=suffix.reshape((num,4,104))
    results=[]
    for perm in itertools.permutations(range(4)):
        candidate=np.column_stack([prefix,batches[:,perm].reshape(num,416)]).astype(np.float32)
        actual=model.predict_proba(candidate).astype(np.float64)
        need(actual.shape==target.shape and np.isfinite(actual).all(),'S5 checkpoint outputs invalid')
        errors=np.abs(actual-target)
        v={'permutation':list(perm),'argmax_K_mismatches':int(np.sum(actual.argmax(1)!=target.argmax(1))),
           'posterior_mean_abs_error':float(errors.mean()),
           'posterior_max_abs_error':float(errors.max()),
           'posterior_vectors_all7_within_1e_minus_6':int(np.sum(np.all(errors<1e-6,axis=1))),
           'KL_target_candidate':float(np.mean(np.sum(target*np.log(np.maximum(target,1e-10)/np.maximum(actual,1e-10)),axis=1)))}
        if perm==(0,1,2,3):
            cached_error=np.abs(actual-baseline)
            v['historical_S123_previous_replay_posterior_max_diff']=float(cached_error.max())
            need(np.allclose(actual,baseline,atol=5e-6,rtol=5e-6),
                 'S201 identity did not reproduce S123 fixed log candidate')
        results.append(v)
        print(json.dumps(v,sort_keys=True),flush=True)
    sorted_k=sorted(results,key=lambda x:(x['argmax_K_mismatches'],
        x['posterior_mean_abs_error'],x['permutation']))
    report={'status':'S201_ORIGINAL_S5_527D_FOUR_104D_ZONE_ORDER_24_FIXED_FIDELITY_TEST',
        'original_historical_fold0_sample_size':num,
        'full_original_historical_fold0_rows':len(a['global_index']),
        'sample_selected_before_reading_saved_S5_probabilities':True,
        'source_sample_selection':'SHA256(int(global_index) ASCII) stable sort first N',
        'archived_S5_training_labels_read':False,
        'source_wave_true_pitch_K_or_MIDI_used':False,
        'original_trained_model_unchanged':True,
        'fixed_hypothesis_suffix':'S118 unverified log_activation_first 4 x 104D zones',
        'only_zone_order_changed':True,
        'permutations_tested':len(results),
        'fixed_original_raw_posterior_evidence_SHA256':digest(original),
        'fixed_checkpoint_SHA256':digest(model_file),
        'hypothesis_results_in_order':results,
        'smallest_original_S5_predicted_K_mismatches':sorted_k[0]['argmax_K_mismatches'],
        'best_diagnostic_permutation_on_exposed_historical_output':sorted_k[0]['permutation'],
        'permutation_proving_full_original_per_event_posterior_fidelity':
            [x['permutation'] for x in results if x['posterior_vectors_all7_within_1e_minus_6']==num],
        'S5_original_527D_builder_recovered':False,
        'original_S5_new_WAV_accuracy_measured':False,
        'S80_promoted':False}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print('S201_SUMMARY '+json.dumps({k:report[k] for k in (
      'permutations_tested','smallest_original_S5_predicted_K_mismatches',
      'best_diagnostic_permutation_on_exposed_historical_output',
      'permutation_proving_full_original_per_event_posterior_fidelity',
      'S5_original_527D_builder_recovered','S80_promoted')},sort_keys=True),flush=True)
def selftest():
    need(len(list(itertools.permutations(range(4))))==24,'all 24 permutations')
    tmp=np.arange(2*416).reshape(2,4,104)
    need(np.array_equal(tmp[:,(0,1,2,3)].reshape(2,416),tmp.reshape(2,416)),
         'identity 104D block order')
    print('S201 PASS 4 factorial and exact 104D blocks; zero label reads')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--features',type=Path)
    p.add_argument('--original',type=Path)
    p.add_argument('--model',type=Path)
    p.add_argument('--checkpoint-report',type=Path)
    p.add_argument('--cached',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--count',type=int,default=512)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        need(all((a.features,a.original,a.model,a.checkpoint_report,a.cached,a.out)),
             'genuine S5 originals required')
        run(a.features,a.original,a.model,a.checkpoint_report,a.cached,a.out,a.count)
