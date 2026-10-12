"""S117 inference-only deployment smoke of frozen original S4 control models.

8 original HGB weights, 57 genuine unseen GOAT events, 111D historical
protocol input. Crucially no accuracy or full S80 claim: the 111D column
order remains unproven by old-fold output replay and each model was
trained on historical GuitarSet, not new GOAT labels.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np
FOLDS=(0,1,2,4)
def require(value,msg):
    if not value:raise ValueError(msg)
def run(root,inp,out):
    require(not out.exists(),'never overwrite experimental S4 external result')
    with np.load(inp,allow_pickle=False) as z:
        require(set(z.files)=={'row_id','control111','freeze_K','geometry46'},'external input schema drift')
        keys=z['row_id'];X=z['control111'];base=z['freeze_K']
    require(X.shape==(57,111) and np.isfinite(X).all(),'unseen S4 control input invalid')
    require(np.isin(base,range(7)).all(),'invalid frozen parent')
    prob=np.full((57,4,2,7),np.nan,np.float32)
    records=[]
    for a,fold in enumerate(FOLDS):
        for b,poly in enumerate((1,2)):
            path=root/f'fold{fold}__control__poly{poly}.joblib'
            require(path.is_file(),'original S4 checkpoint missing: '+str(path))
            state=joblib.load(path)
            model=state['model']
            require(state['family']==f'control__poly{poly}' and state['held_fold']==fold,
                    'frozen S4 provenance mismatch')
            require(model.n_features_in_==111 and np.array_equal(model.classes_,np.arange(7)),
                    'original S4 feature/K schema mismatch')
            scores=np.asarray(model.predict_proba(X),np.float64)
            require(scores.shape==(57,7) and np.isfinite(scores).all() and
                    np.allclose(scores.sum(1),1,atol=1e-5),'bad original model probabilities')
            prob[:,a,b]=scores.astype(np.float32)
            records.append({'fold':fold,'poly_weight':poly,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                            'predicted_K_histogram':np.bincount(scores.argmax(1),minlength=7).tolist()})
    require(np.isfinite(prob).all(),'original model forward incomplete')
    out.mkdir(parents=True)
    ensemble=prob.mean(axis=1)
    predictions=ensemble.argmax(axis=-1).astype(np.int8)
    np.savez_compressed(out/'s4_control_external_probabilities.npz',
       row_id=keys,freeze_K=base,fold_ids=np.array(FOLDS),
       original_fold_probs=prob,ensemble_probs=ensemble,
       ensemble_K=predictions)
    result={'status':'S117_EIGHT_ORIGINAL_S4_CONTROL_MODELS_REAL_GOAT_FORWARD_COMPLETE',
       'external_events':57,'original_checkpoints_executed':8,
       'K_histograms_by_polyweight':[np.bincount(predictions[:,i],minlength=7).tolist() for i in range(2)],
       'changed_vs_frozen_count':[int((predictions[:,i]!=base).sum()) for i in range(2)],
       'checkpoints':records,'authentic_wav_features':True,'GOAT_truth_read':False,
       'S4_control111_column_replay_verified':False,
       'S4_owned527_available':False,
       'S80_external_accuracy':None,'S80_promoted':False}
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','external_events','original_checkpoints_executed',
           'K_histograms_by_polyweight','changed_vs_frozen_count','GOAT_truth_read','S80_promoted')},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoints',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.checkpoints,a.features,a.out)
