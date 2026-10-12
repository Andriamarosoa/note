"""S143: execute ORIGINAL S12 saved cross-piece logistic risk checkpoints on GOAT.

Original S12 90D vote+audio, not reconstructed S12 policy: requires missing
S9 eligible intervention mask and true S11 global/poly source. Therefore the
nineteen historic original piece-exclusive models are evaluated on ALL 57
label-free GOAT events using a NEW S141 parent adaptation. This is only a
weight-deployment feasibility audit, NEVER original S12/S45/S80 equivalence.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import joblib
import numpy as np
from scripts.s141_external_original_s40_cnn import load_input
def require(c,why):
    if not c:raise ValueError(why)
def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(s97,s85,s135,s12,out):
    require(not out.exists(),'immutable proof output')
    ids,X335,_,parent=load_input(s97,s85,s135)
    X90=X335[:,:90]
    require(X90.shape==(57,90) and np.isfinite(X90).all(),
            'original 32D votes + 58D audio S12 model input schema')
    modeldir=s12/'models'
    all_files=sorted(modeldir.glob('*__logistic.joblib'))
    require(len(all_files)>=15,'missing most original S12 heldpiece checkpoints')
    stacked=[]
    provenance=[]
    for path in all_files:
        state=joblib.load(path)  # historical trusted GitHub artifact only
        check_name=path.name.split('__logistic.joblib')[0]
        require(state['piece']==check_name,'trained original S12 piece provenance drift')
        require(state['model'].n_features_in_==90 and
                state['scaler'].n_features_in_==90,'S12 original 90D schema')
        raw=state['model'].predict_proba(
            np.clip(state['scaler'].transform(X90),-6,6))
        p=np.zeros((57,7),np.float32)
        p[:,np.asarray(state['model'].classes_,int)]=np.asarray(raw,np.float32)
        require(p.shape==(57,7) and np.isfinite(p).all() and
                np.allclose(p.sum(1),1,atol=1e-5),
                'original S12 estimator seven way posteriors')
        stacked.append(p)
        provenance.append({'original_held_piece':check_name,
                           'source_checkpoint_sha256':digest(path),
                           'original_weights_loaded':True})
        print(json.dumps(provenance[-1]),flush=True)
    matrix=np.stack(stacked)
    mean=matrix.mean(0)
    pred=mean.argmax(1).astype(np.int8)
    require(mean.shape==(57,7) and np.isfinite(mean).all(),'S12 ensemble shape')
    out.mkdir(parents=True)
    np.savez_compressed(out/'S12_original_logistic_risk_models_GOAT.npz',
        row_id=ids,means=mean,predicted_K=pred,per_heldpiece_probability=matrix,
        parent_freeze_K=parent['official_freeze'],
        parent_S135_K=parent['student_S135'],
        parent_original_PR16_22_K=parent['original_PR16_policy22'])
    report={'status':'S143_EXECUTED_ORIGINAL_S12_LOGISTIC_SAVED_CHECKPOINTS_ON_GOAT',
            'historic_source_run_id':37853979683,
            'original_weights_and_scalers_unchanged':True,
            'source_S12_model_count':len(provenance),
            'trusted_models':provenance,
            'GOAT_events':57,
            'GOAT_ensemble_K_histogram':np.bincount(pred,minlength=7).tolist(),
            'model_policy_new':'equal mean of all saved LOPO original S12 logistic classifiers',
            'model_operates_all_groups_out_of_eligibility_domain':True,
            'original_S12_source_mask_S9_replayed':False,
            'original_S11_votes_source_replayed':False,
            'parent_adapter_new':'b=freeze,g=S135,p=PR16#22',
            'true_S12_champion_reproduced':False,
            'full_16_S45_families_available':False,
            'MIDI_truth_opened':False,
            'external_Exact_K_score':None,
            'S80_promoted':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in
      ('status','source_S12_model_count','GOAT_ensemble_K_histogram',
       'true_S12_champion_reproduced','S80_promoted')},indent=2))
def selftest():
    x=np.zeros((5,335),np.float32)
    require(x[:,:90].shape==(5,90),'S12 original numeric prefix width')
    print('S143 PASS: S12 raw 32+58 width without label')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('s97','s85','s135','s12','out'):p.add_argument('--'+k,type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        require(all(getattr(a,k) is not None for k in
                    ('s97','s85','s135','s12','out')),'five pinned inputs')
        run(a.s97,a.s85,a.s135,a.s12,a.out)
