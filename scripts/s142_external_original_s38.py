"""S142: restore real ORIGINAL S38 16-fold view/algorithm trained estimator inference.

Use immutable original Series38 trained ExtraTrees *morph825* champions
on new GOAT native audio, plus source score_predictions K1+ gating.
Input 335D vote parent b/g/p uses the documented NEW adapter from S141:
b=freeze original, g=S135 student, p=original PR16 policy22.
Original S18/S45 upstream equivalence is NOT asserted.
No MIDI or truth access; cannot promote original S80.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib
import numpy as np
from scripts.s141_external_original_s40_cnn import load_input
from scripts.train_v273_s38_poly_harmonic_expert import score_predictions
FOLDS=(0,1,2,4)
def check(x,why):
    if not x:raise ValueError(why)
def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(s97,s85,s135,s38,out):
    check(not out.exists(),'immutable S142 output')
    ids,x,seq,parent=load_input(s97,s85,s135)
    with np.load(s97,allow_pickle=False) as z:
        check(np.array_equal(ids,z['row_id'].astype(str)),'S38 morph source row alignment')
        morph=z['morphology490'].astype(np.float32)
    X=np.column_stack((x,morph)).astype(np.float32)
    check(X.shape==(57,825) and np.isfinite(X).all(),'original 825D genuine GOAT morph')
    b=parent['official_freeze']
    foldp=[];evidence=[]
    for fold in FOLDS:
        name=f'fold{fold}__morph825__extra_trees.joblib'
        path=s38/'models'/name
        check(path.is_file(),f'original S38 checkpoint missing {name}')
        # Safe only because the user-authored original GitHub Actions archive
        # is trusted; joblib/pickle must not be loaded from unknown sources.
        bundle=joblib.load(path)
        check(bundle['fold']==fold and bundle['view']=='morph825' and
              bundle['algo']=='extra_trees','S38 original trainer checkpoint identity')
        clf=bundle['estimator'];sc=bundle['scaler']
        check(clf.n_features_in_==825 and sc.n_features_in_==825,'825D model source')
        check(np.array_equal(clf.classes_,np.arange(1,7)),
              'original S38 K1..K6 output classes')
        z=np.clip(sc.transform(X),-5,5).astype(np.float32)
        r=np.asarray(clf.predict_proba(z),np.float32)
        check(r.shape==(57,6) and np.allclose(r.sum(1),1,atol=1e-5),
              'original S38 probability conservation')
        p=np.column_stack((np.zeros(57,np.float32),r))
        foldp.append(p)
        proposed=p.argmax(1).astype(np.int8)
        gated=score_predictions(b,proposed,p,'source_K1to6','argmax')
        evidence.append({'fold':fold,'checkpoint_sha256':digest(path),
                         'raw_K_distribution':np.bincount(proposed,minlength=7).tolist(),
                         'gated_K_distribution':np.bincount(gated,minlength=7).tolist(),
                         'original_S38_weights_executed':True})
        print(json.dumps(evidence[-1]),flush=True)
    stack=np.stack(foldp)
    mean=stack.mean(axis=0)
    chosen=mean.argmax(1).astype(np.int8)
    gated=score_predictions(b,chosen,mean,'source_K1to6','argmax')
    check(mean.shape==(57,7) and np.allclose(mean.sum(1),1,atol=1e-5),
          'mean probabilities')
    check(np.all(gated[b==0]==0),'original source_K1to6 cannot change K0')
    out.mkdir(parents=True)
    np.savez_compressed(out/'s38_original_morph825_real_GOAT_unlabeled.npz',
          row_id=ids,fold_ids=np.array(FOLDS),fold_probabilities=stack,
          mean_probabilities=mean,original_S38_proposed_K=chosen,
          adapted_K1plus_K=gated,source_parent_freeze_K=b,
          parent_s135_K=parent['student_S135'],
          parent_policy22_K=parent['original_PR16_policy22'])
    result={'status':'S142_ORIGINAL_S38_MORPH825_EXTRA_TREES_ON_GOAT',
            'source_run_id':37880386031,'original_S38_weights_unchanged':True,
            'S38_archived_original_family':'series38__morph825__extra_trees__source_K1to6__argmax',
            'GOAT_events':57,'source_checkpoints':evidence,
            'GOAT_adapted_K_histogram':np.bincount(gated,minlength=7).tolist(),
            'new_deploy_rule':'4 fold equal-mean posterior, gate iff official freeze K>=1',
            'new_parent_mapping_different_from_S18':'b=freeze,g=S135,p=PR16#22',
            'original_S18_S45_source_equivalent':False,
            'full_original_S80_available':False,
            'MIDI_truth_read':False,'independent_accuracy_measured':False,
            'promotion':False}
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
def selftest():
    old=np.array([0,1,2,3],np.int8)
    new=np.array([2,3,2,4],np.int8)
    p=np.eye(7,dtype=np.float32)[new]
    assert score_predictions(old,new,p,'source_K1to6','argmax').tolist()==[0,3,2,4]
    print('S142 PASS S38 genuine K1-K6 source gating')
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('s97','s85','s135','s38','out'):ap.add_argument('--'+name,type=Path)
    ap.add_argument('--self-test',action='store_true')
    a=ap.parse_args()
    if a.self_test:selftest()
    else:
        check(all(getattr(a,n) is not None for n in ('s97','s85','s135','s38','out')),
              'requires fixed models and original audio')
        run(a.s97,a.s85,a.s135,a.s38,a.out)
