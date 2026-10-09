"""S38 independent numeric replay of all 16 genuine cross-fold polyphonic experts."""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from scripts.train_v273_aba_recurrent import prepare
from scripts.train_v273_extended_dynamic_router import original_temporal_data,read
from scripts.train_v273_s38_poly_harmonic_expert import score_predictions,VIEWS,ALGOS,DOMAINS,GATES
from scripts.yourmt3_exactk_common import FOLDS,metrics,require

def sha(x):
    return hashlib.sha256(np.asarray(x,dtype='<i8').tobytes()).hexdigest()

def run(a):
    require(not a.output.exists(),'cannot overwrite independent verification')
    d=prepare(a)
    pred=read(a.input/'predictions.npz')
    scores=read(a.input/'new_expert_probs.npz')
    report=json.loads((a.input/'report.json').read_text())
    require(np.array_equal(pred['global_index'],d['ids']) and
            np.array_equal(scores['global_index'],d['ids']),
            'source native identity mismatch')
    require(np.array_equal(pred['true_K'],d['y']) and
            np.array_equal(pred['fold'],d['fold']),
            'true fold source drift')
    require(report['status']=='completed' and
            report['new_polyphonically_trained_models']==16,
            'one or more acoustic models missing')
    original=d['parent']
    names=list(map(str,pred['variant_ids']))
    require(len(names)==25 and names[0]=='series18_parent',
            '24 policies missing')
    require(np.array_equal(pred['predictions'][:,0],original),
            'baseline changed')
    require(metrics(d['y'],original)['correct']==49178 and
            metrics(d['y'],original)['poly']['correct']==2998,
            'S18 no longer matches certified reference')
    morph,_,_=original_temporal_data(a.features,d)
    viewdata={'raw335':d['X'].astype(np.float32),
              'morph825':np.column_stack((d['X'],morph)).astype(np.float32)}
    verified={}
    all_held_probs=np.zeros((len(d['ids']),4,7),np.float32)
    for model_idx,(view,algo) in enumerate(
            [(v,m) for v in VIEWS for m in ALGOS]):
        original_key=f'{view}__{algo}'
        for fold in FOLDS:
            spec=next((e for e in report['model_provenance']
                 if e['fold']==fold and e['view']==view and e['model']==algo),None)
            require(spec is not None,'missing per-fold classifier evidence')
            path=a.input/'models'/spec['file']
            require(path.is_file(),'missing fitted classifier weights '+str(path))
            ckpt=joblib.load(path)
            tr=(d['fold']!=fold)&(d['y']>=1)
            held=(d['fold']==fold)
            require(np.array_equal(ckpt['train_native_ids'],d['ids'][tr]) and
                    np.array_equal(ckpt['held_native_ids'],d['ids'][held]),
                    'fitted acoustic expert saw test-fold labels')
            require(spec['fit_ids_sha256']==sha(d['ids'][tr]) and
                    spec['held_ids_sha256']==sha(d['ids'][held]),
                    'crossfold provenance changed')
            with threadpool_limits(limits=2):
                pp=ckpt['estimator'].predict_proba(
                    np.clip(ckpt['scaler'].transform(
                    viewdata[view][held]),-5,5).astype(np.float32))
            require(np.array_equal(ckpt['estimator'].classes_,np.arange(1,7)),
                    'K1..K6 output alignment missing')
            all_held_probs[held,model_idx,1:7]=pp
            verified[spec['file']]=hashlib.sha256(path.read_bytes()).hexdigest()
        expert=all_held_probs[:,model_idx]
        original_probs=scores['poly_probs'][:,model_idx]
        require(np.allclose(expert,original_probs,atol=1e-5),
                'expert numeric prediction not reproduced '+original_key)
        proposed=expert.argmax(1).astype(np.int8)
        for domain in DOMAINS:
            for gate in GATES:
                name=f'series38__{original_key}__{domain}__{gate}'
                require(name in names,'missing reported intervention '+name)
                saved=pred['predictions'][:,names.index(name)]
                computed=score_predictions(original,proposed,expert,domain,gate)
                require(np.array_equal(saved,computed),
                        'policy output differs from independently replayed model')
    require(len(verified)==16,'all 16 model artifacts required')
    a.output.mkdir(parents=True)
    result=dict(status='all_models_numerically_verified',
        source_S38_run=report.get('source_run','S38'),
        held_outer_fold_never_in_fit=True,
        all_16_acoustic_models_loaded=True,
        all_24_policies_matched_exactly=True,
        H9_excluded=True,
        no_new_independent_composition_validation=True,
        no_production_promotion=True,
        model_sha256=verified)
    (a.output/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('PASS: all 16 models reloaded; every held-fold posterior and every one of 24 K-class decisions matches exactly')
    print('EXPLICIT LIMIT: producer evaluation uses historically inspected development cohort, not new compositions')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--s18',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
