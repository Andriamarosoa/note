"""S392: read-only original K5-vs-K4 score separability of S390/S391 OOF experts.

No model training, no policy threshold selection, no reference update.
Measures ROC AUC/AP only on unchanged S80=S272 K4 proposals and original
same-row GuitarSet truth, with material TRAIN OVERLAP in S80 explicitly noted.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
F=['audio104_crossfit_prediction_NO_TRUTH.npz',
   'audio1133_crossfit_prediction_NO_TRUTH.npz']
G=['K4_K5_training_heldfold_probabilities_NO_TRUTH.npz',
   'K3_K4_K5_training_heldfold_probabilities_NO_TRUTH.npz']
def need(ok,msg):
    if not ok:raise ValueError(msg)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def run(s80,your,s390,s391,out):
    need(not out.exists(),'source audit must not overwrite past report')
    b=load(s80);a=load(your)
    pos=np.argsort(a['global_index'].astype(np.int64))
    ids=b['row_id'].astype(str);yy=a['k'][pos];kk=b['s80_K']
    need(len(ids)==59309 and np.array_equal(ids,a['global_index'][pos].astype(str))
         and int((yy==5).sum())==355 and int((kk==4).sum())==1115
         and int(((yy==5)&(kk==4)).sum())==224 and int((kk==yy).sum())==53327,
         'original S80 matched benchmark drift')
    scope=kk==4;target=yy==5
    results={}
    for prefix,folder,files in [('S390',s390,F),('S391',s391,G)]:
        for file in files:
            src=load(folder/file)
            need(np.array_equal(ids,src['row_id'].astype(str)),
                'OOF source event IDs mismatch '+file)
            field='posterior_K5_oof' if prefix=='S390' else 'p_K5_crossfit'
            proba=src[field]
            need(proba.shape==(59309,) and np.isfinite(proba).all() and
                 np.all((0<=proba)&(proba<=1)),
                 'complete fourfold expert probs required')
            a_auc=float(roc_auc_score(target[scope],proba[scope]))
            a_ap=float(average_precision_score(target[scope],proba[scope]))
            results[prefix+'/'+file]=dict(
                n=int(scope.sum()),true_K5=int(target[scope].sum()),
                true_K5_fraction=float(target[scope].mean()),
                rank_ROC_AUC=a_auc,average_precision=a_ap,
                median_score_true_K5=float(np.median(proba[scope&target])),
                median_score_true_K4=float(np.median(proba[scope&(yy==4)])),
                max_score=float(proba[scope].max()),
                finite_values_pass=True)
    need(len(results)==4 and all(.5<x['rank_ROC_AUC']<.8 for x in results.values()),
         'K5 original ranking audit changed')
    report=dict(status='S392_K5_SOURCE_STATIC_SCORE_SEPARABILITY_DIAGNOSTIC',
        original_S80_K4_count=int(scope.sum()),
        true_K5_among_S80_K4=int((scope&target).sum()),
        true_K4_among_S80_K4=int((scope&(yy==4)).sum()),
        other_true_K_among_S80_K4=int((scope&~np.isin(yy,[4,5])).sum()),
        sha256={'S272':digest(s80),'YourMT3':digest(your)},
        scores=results,
        no_crossfold_model_training_in_this_audit=True,
        new_head_training_S390_S391_foldheld=True,
        S80_S272_training_overlap_unresolved=True,
        external_validation_not_claimed=True,
        learned_heads_not_promoted=True,
        S80_reference_unchanged=True,
        implication='Audio feature K5 scores carry some signal but strongly overlap correct K4; tested static architectures cannot produce a favorable reliable K5->K4 correction gate.')
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    print('S392 PASS: four true K5 score overlap measures, NO PROMOTION',flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('s80','your','s390','s391','out'):
        ap.add_argument('--'+name,required=True,type=Path)
    x=ap.parse_args();run(x.s80,x.your,x.s390,x.s391,x.out)
