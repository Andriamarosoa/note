"""S391: new upstream acoustic K5 producer with true K4/K5 learning targets.

Unlike S390, DO NOT train conditionally on historical S80 K4 predictions.
Train on TRUE class 4/5, or 3/4/5; use only acoustic features and TRAINING
fold labels. Whole held-fold is always unseen by its corresponding model.
Evaluation against S272 S80 is descriptive (S272 itself has train overlap).
No promotion, threshold selection, or change to any existing S80 checkpoint.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import ExtraTreesClassifier
FOLDS=(0,1,2,4);N=59309; CUTS=(.20,.30,.40,.50,.60,.70)
AUDIO={'row_id','member','start_samples','end_samples','incumbent_freeze_K',
'summary58','sequence42x49','flow245','morphology490',
'physics294','original_PR16_geometry46','geometry3'}
TRUTH={'row_id','true_K','original_fold','original_freeze_K'}
S272={'row_id','incumbent_freeze_K','s72_K','s79_K','s80_K',
'original_PR16_policy22_K','K1_by_fold_threshold_action',
'K1_by_fold_threshold_scores','action_K1','action_K0','action_K2'}
def need(v,m):
    if not v:raise ValueError(m)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p,keys):
    with np.load(p,allow_pickle=False) as z:
        need(set(z.files)==keys,'wrong source schema '+str(p))
        return {k:z[k].copy() for k in z.files}
def score(y,b,c,m):
    k=np.flatnonzero(m)
    bb=b[k]==y[k];cc=c[k]==y[k]
    return dict(n=int(len(k)),correct_before=int(bb.sum()),
       correct_after=int(cc.sum()),corrections=int((~bb&cc).sum()),
       regressions=int((bb&~cc).sum()),net=int(cc.sum()-bb.sum()))
def run(audio,truth,historical,s272,out):
    need(not out.exists(),'immutable S391 new original evidence only')
    a=read(audio,AUDIO);t=read(truth,TRUTH);s=read(s272,S272)
    with np.load(historical,allow_pickle=False) as z:
        need({'global_index','true_K','s80_K','fold'}.issubset(z.files),
             'original heldfold benchmark absent')
        ix=np.argsort(z['global_index'].astype(np.int64))
        histid=z['global_index'][ix].astype(str)
        historic=z['s80_K'][ix].astype(np.int8)
        hy=z['true_K'][ix].astype(np.int8)
        hf=z['fold'][ix].astype(int)
    ids=a['row_id'].astype(str)
    y=t['true_K'].astype(np.int8);f=t['original_fold'].astype(int)
    b=s['s80_K'].astype(np.int8);m=a['member'].astype(str)
    need(len(ids)==N and len(np.unique(ids))==N
        and np.array_equal(ids,t['row_id'].astype(str))
        and np.array_equal(ids,s['row_id'].astype(str))
        and np.array_equal(ids,histid)
        and np.array_equal(y,hy) and np.array_equal(f,hf)
        and np.array_equal(a['incumbent_freeze_K'],t['original_freeze_K'])
        and np.array_equal(s['incumbent_freeze_K'],t['original_freeze_K'])
        and set(np.unique(f))==set(FOLDS)
        and int((y==5).sum())==355
        and int((b==y).sum())==53327
        and int((historic==y).sum())==49281,
        'original held-fold source alignment mismatch')
    X=np.column_stack((a['summary58'],a['original_PR16_geometry46'],
       a['flow245'],a['morphology490'],a['physics294'])).astype(np.float32)
    need(X.shape==(N,1133) and np.isfinite(X).all(),
         'original upstream acoustic 1133D source failure')
    out.mkdir(parents=True)
    results={}
    for view,allowed in [('K4_K5_training',(4,5)),
                         ('K3_K4_K5_training',(3,4,5))]:
        p=np.full(N,np.nan,np.float32);check=[]
        for hold in FOLDS:
            val=f==hold
            train=(f!=hold)&np.isin(y,allowed)
            held=set(m[val])
            train&=~np.isin(m,list(held))
            pos=int((y[train]==5).sum());neg=int((y[train]!=5).sum())
            need(pos>50 and neg>100 and val.sum()>1000
                 and len(set(m[train]).intersection(held))==0,
                 'genuine group-OOF class balance failed fold '+str(hold))
            clf=ExtraTreesClassifier(n_estimators=192,max_depth=16,
                 min_samples_leaf=3,max_features='sqrt',class_weight=None,
                 random_state=391+hold+(1000 if view=='K3_K4_K5_training' else 0),
                 n_jobs=2)
            clf.fit(X[train],(y[train]==5).astype(np.int8))
            p[val]=clf.predict_proba(X[val])[:,list(clf.classes_).index(1)]
            save=out/(view+'_fold'+str(hold)+'.joblib')
            joblib.dump(clf,save,compress=3)
            check.append(dict(fold=hold,train_events=int(train.sum()),
               train_K5=pos,train_other=neg,val_events=int(val.sum()),
               val_K5=int((y[val]==5).sum()),member_overlap=0,
               model_sha256=digest(save)))
        need(np.isfinite(p).all() and (p>=0).all() and (p<=1).all(),
             'new K5 source OOF probability missing')
        trials=[]
        for cut in CUTS:
            take=(b==4)&(p>=cut)
            changed=b.copy();changed[take]=5
            oldtake=(historic==4)&(p>=cut)
            oldnew=historic.copy();oldnew[oldtake]=5
            row=dict(threshold=cut,proposals=int(take.sum()),
              S80_all_DESCRIPTIVE=score(y,b,changed,np.ones(N,bool)),
              S80_poly_DESCRIPTIVE=score(y,b,changed,y>=2),
              S80_K5_DESCRIPTIVE=score(y,b,changed,y==5),
              S80_K4_DESCRIPTIVE=score(y,b,changed,y==4),
              S80_byK_DESCRIPTIVE={f'K{k}':score(y,b,changed,y==k) for k in range(7)},
              historical_OOF_all=score(y,historic,oldnew,np.ones(N,bool)),
              historical_OOF_K5=score(y,historic,oldnew,y==5),
              historical_OOF_K4=score(y,historic,oldnew,y==4),
              fold_results={str(h):score(y,historic,oldnew,f==h) for h in FOLDS})
            trials.append(row)
        np.savez_compressed(out/(view+'_heldfold_probabilities_NO_TRUTH.npz'),
             row_id=ids,fold=f,p_K5_crossfit=p)
        results[view]=dict(training_trueK_categories=list(allowed),
            fold_proofs=check,predeclared_threshold_ledger=trials)
    report=dict(status='S391_ACOUSTIC_K5_PRODUCER_CROSS_FOLD_TARGET_REDESIGN',
       model_inputs='original S258 audio only, 1133D pre-S72 source features',
       original_S80_reference_S272_unchanged=True,
       historic_S80_genuine_heldpiece_comparison=True,
       real_S80_S272_not_pure_OOF=True,
       original_GuitarSet_train_source_already_exposed=True,
       thresholds_set_in_source_before_score=list(CUTS),
       true_group_fold_leakage=0,
       original_native_events=N,
       original_true_K5_count=355,
       sources={'audio':digest(audio),'truth':digest(truth),
                'historic_S80_OOF':digest(historical),'S272':digest(s272)},
       results=results,no_new_IDMT_labels_read=True,
       newly_trained_heads=8,
       candidate_policy_promoted=False,S80_reference_changed=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for view,item in results.items():
        print('S391 TRAIN',view,json.dumps(item['fold_proofs']))
        for q in item['predeclared_threshold_ledger']:
            print('S391 EVAL',view,q['threshold'],'actions',q['proposals'],
             'K5 fixes',q['S80_K5_DESCRIPTIVE']['corrections'],
             'K4 regressions',q['S80_K4_DESCRIPTIVE']['regressions'],
             'S80 net',q['S80_all_DESCRIPTIVE']['net'],
             'true historic OOF net',q['historical_OOF_all']['net'],flush=True)
    print('S391 PASS independent fold experts only: S80 unchanged',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('audio','truth','historical','s272','out'):
        p.add_argument('--'+k,type=Path,required=True)
    z=p.parse_args();run(z.audio,z.truth,z.historical,z.s272,z.out)
