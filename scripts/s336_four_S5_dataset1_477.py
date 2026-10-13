"""S336 original S5 saved models on 477 Dataset1 native V88 rows, label free.

Weights are downloaded by workflow from immutable commit
7165e1ef54510aa9effc9f597b2783c832ca907a, SHA256-verified against
original historical S5 recovery source. The recovered S4 527D builder
was tested on all 59309 original saved probabilities with zero drift.
All 4 S5 original heldfold K0..6 estimators each score EVERY 477
unlabelled original official native event. Keep four distinct fold
hypotheses and corrected class priors, do not select held-piece oracle.
No labels, no fitted models, no full S18/S80 or promotion claim.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np
from scripts.verify_v273_s5_owned527_recovery import (
    FOLDS,MODEL_DIGEST,ORIGINAL_DESIGN_DIGEST)
BAD={'true_K','truth','label','labels','midi','correct','accuracy','target','y','regression'}
def need(v,msg):
    if not v:raise ValueError(msg)
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(s335file,model_dir,out):
    need(not out.exists(),'original S5 folded inference immutable')
    with np.load(s335file,allow_pickle=False) as z:
        keys={'row_id','member','start_samples','end_samples',
            'incumbent_freeze_K','original_PR16_geometry46',
            'original_S5_input_owned527'}
        need(set(z.files)==keys and not BAD.intersection(z.files),
             'S236 true original 527D input only')
        ids=z['row_id'].astype(str)
        x=z['original_S5_input_owned527'].copy()
        parent=z['incumbent_freeze_K'].copy()
    need(x.shape==(477,527) and
         len(set(ids))==477 and np.isfinite(x).all(),
         'original official native 527D S5 source')
    all_raw=[];all_prior=[];proof=[]
    for fold in FOLDS:
        cp=model_dir/f'fold{fold}'/'model.joblib'
        need(cp.is_file() and sha(cp)==MODEL_DIGEST[fold],
             'original immutable S5 pretrained weights SHA256 mismatch: '+str(fold))
        state=joblib.load(cp)
        need(int(state['held_fold'])==fold and
             state['designs_sha256']==ORIGINAL_DESIGN_DIGEST,
             'historical S5 trained metadata drift')
        model=state['model']
        need(model.n_features_in_==527,
             'original S5 model expects non527 acoustic input')
        raw=model.predict_proba(x)
        weights=np.asarray(state['class_weights'],float)
        need(raw.shape==(len(x),7) and weights.shape==(7,) and
             np.isfinite(raw).all() and np.all(weights>0) and
             np.allclose(raw.sum(1),1.,atol=1e-6),
             'trained S5 K0–K6 logits invalid')
        corrected=raw/weights[None,:]
        corrected/=corrected.sum(1,keepdims=True)
        need(np.isfinite(corrected).all() and
             np.allclose(corrected.sum(1),1.,atol=1e-6),
             'original class prior correction invalid')
        all_raw.append(raw)
        all_prior.append(corrected)
        proof.append(dict(fold=fold,checkpoint_SHA256=sha(cp),
            historical_training_design_SHA256=ORIGINAL_DESIGN_DIGEST,
            original_S5_K_histogram=np.bincount(corrected.argmax(1),
                                                minlength=7).tolist()))
    raw=np.stack(all_raw,axis=0)
    probs=np.stack(all_prior,axis=0)
    pred=probs.argmax(2).astype(np.int8)
    unanimous=np.all(pred==pred[:1],axis=0)
    out.mkdir(parents=True)
    dest=out/'S336_original_S5_fourfold_477_dataset1_native_unlabelled.npz'
    np.savez_compressed(dest,row_id=ids,incumbent_freeze_K=parent,
        fold_ids=np.asarray(FOLDS),
        original_S5_raw_probability_by_fold=raw,
        original_S5_prior_probability_by_fold=probs,
        original_S5_K_by_fold=pred,all_four_folds_agree=unanimous)
    report=dict(status='S336_ALL_FOUR_ORIGINAL_TRAINED_S5_FOLDS_ON_IDMT_DATASET1',
       original_native_reference_event_count=len(x),S5_model_inputs=527,
       original_trained_S5_fold_count=4,
       original_held_folds=list(FOLDS),
       original_S5_fold_model_proofs=proof,
       S5_all_four_folds_agree_event_count=int(unanimous.sum()),
       S5_different_fold_predictions_event_count=int((~unanimous).sum()),
       official_reference_v88_native_K_original= True,
       original_fold_models_never_retrained=True,
       new_composition_original_fold_oracle_not_selected=True,
       original_S5_historical_training_predictions_bit_exact_in_prior_run=True,
       output_sha256=sha(dest),
       S18_through_S80_not_yet_executed_on_477_native_events=True,
       unseen_IDMT_XML_MIDI_contents_read=False,
       independent_accuracy_measured=False,official_S80_promotion=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('s335','models','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();run(a.s335,a.models,a.out)
