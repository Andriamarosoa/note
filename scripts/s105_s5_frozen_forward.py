"""S105 read-only inference adapter for four frozen original S5 527D models.

Never use true_K or MIDI at inference. A diagnostic forward on constant input
tests checkpoint executability ONLY, not music predictions, equivalence, or
promotion. Normalization/feature order must match original constructor.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np
REQUIRED={'row_id','design527'}
BANNED={'truth','true_K','target','labels','y','annotation'}
FOLDS=(0,1,2,4)

def load(root):
    models=[]
    for fold in FOLDS:
        p=root/f'fold{fold}'
        manifest=json.loads((p/'report.json').read_text())
        weights=p/'model.joblib'
        digest=hashlib.sha256(weights.read_bytes()).hexdigest()
        if digest!=manifest['model_sha256']:raise ValueError('S5 model checksum drift')
        saved=joblib.load(weights) # only trusted original pinned checkout
        model=saved['model']
        if model.n_features_in_!=527 or not np.array_equal(model.classes_,np.arange(7)):
            raise ValueError('original 527/7 K schema mismatch')
        if int(saved['held_fold'])!=fold: raise ValueError('checkpoint fold mismatch')
        if str(saved['designs_sha256'])!=str(manifest['designs_sha256']):
            raise ValueError('training 527D design digest drift')
        models.append(model)
    return models

def forward(models,x):
    if x.ndim!=2 or x.shape[1]!=527 or not np.isfinite(x).all():
        raise ValueError('real 527D S5 design required')
    p=np.stack([m.predict_proba(x).astype(np.float64) for m in models],axis=1)
    if p.shape!=(len(x),4,7) or not np.isfinite(p).all():
        raise ValueError('checkpoint probability shape drift')
    if not np.allclose(p.sum(2),1,atol=1e-5):raise ValueError('invalid probabilities')
    return p

def run(args):
    models=load(args.models)
    if args.checkpoints_only:
        # This is deliberately not an acoustic sample; never archive it as a prediction.
        p=forward(models,np.zeros((1,527),np.float32))
        print(json.dumps({'status':'S105_FROZEN_S5_WEIGHTS_FORWARD_PASS',
                          'original_models_loaded':4,
                          'all_seven_probabilities_valid':bool(np.isfinite(p).all()),
                          'input_was_diagnostic_not_audio':True,
                          'GOAT_predictions':0,'S80_independent':False,'promotion':False}))
        return
    if args.features is None or args.output is None:
        raise ValueError('requires authentic externally built design527 and output')
    if args.output.exists():raise ValueError('immutable output already exists')
    with np.load(args.features,allow_pickle=False) as z:
        if set(z.files)!=REQUIRED or BANNED.intersection(z.files):
            raise ValueError('label-blind 527D data contract mismatch')
        x=z['design527'].copy()
        ids=z['row_id'].copy()
    if len(ids)!=len(set(map(str,ids))) or len(ids)!=len(x):
        raise ValueError('event keys inconsistent')
    prob=forward(models,x)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'s5_posteriors.npz',row_id=ids,
             fold_ids=np.array(FOLDS),probabilities=prob)
    report={'status':'S105_NEW_AUDIO_S5_PREDICTION_COMPONENT_ONLY',
      'events':len(x),'original_checkpoint_count':4,
      'original_527_feature_generator_verified':False,
      'new_audio_feature_provenance_verified':False,
      'ensemble_policy':'four frozen fold outputs; NO historical S5 policy equivalence asserted',
      'S80_external_score':None,'promotion':False}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--features',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--checkpoints-only',action='store_true')
    run(p.parse_args())
