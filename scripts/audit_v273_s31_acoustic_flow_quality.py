"""S31: can acoustic/harmonic-flow features validate or reject S29 changes?

The previous cross-expert corroboration (S30) did not distinguish the 4
regressions. Here only 245 flow/harmonic or 303 audio+flow or all 335
original variables can inform reliability, with observable parent/candidate.
All train labels from OTHER pieces, never held-event labels.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from scripts.train_v273_aba_recurrent import prepare
from scripts.yourmt3_exactk_common import FOLDS,metrics,paired,require

BASE='series29__lambda2__threshold0.02'
VIEWS={'flow245':slice(90,335),'audio_plus_flow303':slice(32,335),
       'all335':slice(0,335)}
MODELS=('logistic','hgb')
CUTS=(.25,.40,.55,.70)


def load(path):
    with np.load(path,allow_pickle=False) as d:return {k:d[k] for k in d.files}

def option(z,name):
    names=list(map(str,z['variant_ids']))
    require(name in names,'no variant '+name)
    return z['predictions'][:,names.index(name)].astype(np.int8)

def unique_proposals(z,parent):
    ids=[];ks=[]
    for name in list(map(str,z['variant_ids'])):
        if not name.startswith('series29__'):continue
        candidate=option(z,name)
        hit=np.flatnonzero(candidate!=parent)
        ids.append(hit);ks.append(candidate[hit])
    vals=np.unique(np.column_stack((np.concatenate(ids),np.concatenate(ks))),axis=0)
    require(len(vals)>70,'too few changed training candidates')
    return vals[:,0].astype(np.int64),vals[:,1].astype(np.int64)

def features(ix,k,view,X,parent,A20,A25):
    ix=np.asarray(ix,np.int64);k=np.asarray(k,np.int64)
    require(len(ix)==len(k),'bad proposed class input')
    old=parent[ix].astype(np.int64)
    ii=np.arange(len(ix))
    af=A20[ix,3,:];bf=A25[ix,3,:]
    cls=np.eye(7,dtype=np.float32)
    margin=np.column_stack((
         af[ii,k]-af[ii,old],bf[ii,k]-bf[ii,old])).astype(np.float32)
    raw=X[ix,VIEWS[view]]
    res=np.column_stack((raw,cls[old],cls[k],margin)).astype(np.float32)
    require(res.shape==(len(ix),(VIEWS[view].stop-VIEWS[view].start)+16),
            'feature dimensions are incorrect')
    require(np.isfinite(res).all(),'nonfinite raw audio')
    return res

def selftest():
    x=np.ones((5,335),np.float32)
    parent=np.asarray([0,1,2,3,4],np.int8)
    cand=np.asarray([2,3,4,5,6],np.int8)
    a=np.zeros((5,5,7),np.float32)
    for view,s in VIEWS.items():
        q=features(np.arange(5),cand,view,x,parent,a,a)
        assert q.shape==(5,s.stop-s.start+16)
    print('PASS: three independent acoustic/flow groups, two model heads and four fixed thresholds')


def audit(args):
    require(not args.output.exists(),'cannot overwrite prior test output')
    d=prepare(args)
    z=load(args.s29);a20=load(args.s20);a25=load(args.s25)
    for b in (z,a20,a25):
        require(np.array_equal(b['global_index'],d['ids']),
                'source event identity changed')
    require(np.array_equal(z['true_K'],d['y']) and
            np.array_equal(z['fold'],d['fold']),'truth/fold misaligned')
    parent=option(z,'series18_parent')
    require(np.array_equal(parent,d['parent']),'S18 prior changed')
    original=option(z,BASE)
    diff=(original!=parent)
    require(int(diff.sum())==26 and
            int(((parent!=d['y'])&(original==d['y'])).sum())==17 and
            int(((parent==d['y'])&(original!=d['y'])).sum())==4,
            'all 26 S29 changed events including 5 neutral required')
    original_ix=np.flatnonzero(diff)
    P20=np.asarray(a20['A_probs'],np.float32)
    P25=np.asarray(a25['A_probs'],np.float32)
    require(P20.shape==P25.shape==(59309,5,7),'A-first/B-first source mismatch')
    train_idx,train_candidate=unique_proposals(z,parent)
    pred_scores={f'{v}_{m}':np.full(len(parent),np.nan,np.float32)
                 for v in VIEWS for m in MODELS}
    args.output.mkdir(parents=True)
    (args.output/'models').mkdir()
    records=[]
    started=time.monotonic()
    pieces=np.asarray(d['pieces'])
    for piece in sorted(set(pieces.tolist())):
        test=(pieces==piece)&diff
        if not test.any():continue
        fold=int(np.unique(d['fold'][pieces==piece])[0])
        fit=(pieces[train_idx]!=piece)&(d['fold'][train_idx]==fold)
        target=(train_candidate[fit]==d['y'][train_idx[fit]]).astype(int)
        if int(fit.sum())<50 or len(np.unique(target))<2:
            fit=(pieces[train_idx]!=piece)
            target=(train_candidate[fit]==d['y'][train_idx[fit]]).astype(int)
        require(len(np.unique(target))==2,'training split only one label')
        choose=np.flatnonzero(test)
        for view in VIEWS:
            xx=features(train_idx[fit],train_candidate[fit],view,
                d['X'],parent,P20,P25)
            yy=features(choose,original[choose],view,
                d['X'],parent,P20,P25)
            scale=StandardScaler().fit(xx)
            Xfit=np.clip(scale.transform(xx),-6,6)
            Xheld=np.clip(scale.transform(yy),-6,6)
            for algo in MODELS:
                estimator=LogisticRegression(C=.1,
                    class_weight='balanced',solver='liblinear',max_iter=300,
                    random_state=27331) if algo=='logistic' else (
                    HistGradientBoostingClassifier(max_iter=80,max_depth=3,
                       max_leaf_nodes=10,min_samples_leaf=20,
                       l2_regularization=10,learning_rate=.05,random_state=27331))
                with threadpool_limits(limits=2):
                    estimator.fit(Xfit,target)
                    pp=estimator.predict_proba(Xheld)[:,1]
                pred_scores[f'{view}_{algo}'][choose]=pp.astype(np.float32)
                path=f'{piece}__{view}__{algo}.joblib'
                joblib.dump(dict(model=estimator,scaler=scale,view=view,
                    algorithm=algo,piece=piece,fold=fold,
                    train_ids=d['ids'][train_idx[fit]],
                    train_candidates=train_candidate[fit],
                    held_ids=d['ids'][choose]),args.output/'models'/path,compress=3)
                records.append(dict(filename=path,piece=piece,view=view,algo=algo,
                    train_count=len(target),train_correct=int(target.sum()),
                    train_incorrect=int((target==0).sum()),
                    held_count=len(choose),
                    same_fold_fit=bool((d['fold'][train_idx[fit]]==fold).all()),
                    fit_sha256=hashlib.sha256(d['ids'][train_idx[fit]].astype('<i8').tobytes()).hexdigest()))
        print(json.dumps(dict(piece=piece,models=len(records),
            time_seconds=round(time.monotonic()-started,1))),flush=True)
    outcomes={'series18_parent':parent.copy(),BASE:original.copy()}
    for name,score in pred_scores.items():
        require(np.isfinite(score[original_ix]).all(),
                'missing meta-prediction for '+name)
        for cut in CUTS:
            label=f'series31__{name}__p_gt{cut:g}'
            outcomes[label]=np.where(score>cut,original,parent).astype(np.int8)
    require(len(outcomes)==26,'24 flow quality policies plus reference and parent')
    diagnostics={}
    for name,pred in outcomes.items():
        mask=pred!=parent
        diagnostics[name]=dict(metrics=metrics(d['y'],pred),
            vs_parent=paired(d['y'],parent,pred),
            vs_freeze=paired(d['y'],d['freeze'],pred),
            corrected=int((mask&(pred==d['y'])&(parent!=d['y'])).sum()),
            regressed=int((mask&(pred!=d['y'])&(parent==d['y'])).sum()),
            neutral=int((mask&(pred!=d['y'])&(parent!=d['y'])).sum()),
            by_true_K={str(k):dict(
                corrected=int(((d['y']==k)&mask&(pred==d['y'])&(parent!=d['y'])).sum()),
                regressed=int(((d['y']==k)&mask&(pred!=d['y'])&(parent==d['y'])).sum()))
                for k in range(7)},
            per_fold={str(f):dict(metrics=metrics(d['y'][d['fold']==f],
                pred[d['fold']==f]),versus_S18=paired(d['y'][d['fold']==f],
                parent[d['fold']==f],pred[d['fold']==f])) for f in FOLDS})
    keep=[key for key in outcomes if key.startswith('series31__')
          and diagnostics[key]['corrected']>0
          and diagnostics[key]['regressed']==0
          and diagnostics[key]['metrics']['poly']['correct']>=2998]
    ranks=sorted(diagnostics,key=lambda s:(diagnostics[s]['metrics']['correct'],
           diagnostics[s]['metrics']['poly']['correct']),reverse=True)
    with (args.output/'26_changed_audio_flow_audit.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['native_id','fold','piece','true_K','S18_K','S29_K',
            'outcome','flow_logistic','flow_hgb','audio_flow_logistic',
            'audio_flow_hgb','all335_logistic','all335_hgb',
            'accepted_policy_names'])
        for i in original_ix:
            agreed=[k for k,v in outcomes.items() if k.startswith('series31__')
                    and v[i]==original[i]]
            writer.writerow([int(d['ids'][i]),int(d['fold'][i]),
                str(pieces[i]),int(d['y'][i]),int(parent[i]),int(original[i]),
                'corrected' if original[i]==d['y'][i] else
                'regressed' if parent[i]==d['y'][i] else 'neutral',
                *[round(float(pred_scores[k][i]),5) for k in pred_scores],
                ';'.join(agreed)])
    report=dict(status='completed',source_run_S29=37865173636,
        original_changes=26,corrected=17,regressed=4,neutral=5,
        independent_new_music_validation=False,
        meta_train_excludes_held_piece=True,
        original_cohort_previously_exposed=True,
        acoustic_features_measured_not_navier_stokes_law=True,
        models_saved=records,models_count=len(records),
        zero_loss_candidates=keep,
        best_global=ranks[0],audit=diagnostics,
        elapsed_seconds=round(time.monotonic()-started,2))
    (args.output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    np.savez_compressed(args.output/'predictions.npz',
        global_index=d['ids'],true_K=d['y'],fold=d['fold'],
        variant_ids=np.asarray(list(outcomes)),
        predictions=np.column_stack(list(outcomes.values())))
    np.savez_compressed(args.output/'acoustic_scores.npz',
        global_index=d['ids'],variant_ids=np.asarray(list(pred_scores)),
        probabilities=np.column_stack(list(pred_scores.values())))
    lines=['# Série31 — flux harmonique vs audio vs toutes caractéristiques','',
        '26 S29 changed events (17 corrected, 4 regressed, 5 neutral); no held label input to any gate.',
        '**Exploratory previously exposed development set; no production promotion.**',
        '', '| Policy | Exact global | Exact poly | Fixes vs S18 | Breaks vs S18 | Original four blocked |',
        '|---|---:|---:|---:|---:|---:|']
    for k in ranks:
        v=diagnostics[k];m=v['metrics']
        lines.append(f"| {k} | {100*m['exact']:.4f}% | "
          f"{100*m['poly']['exact']:.4f}% | {v['corrected']} | "
          f"{v['regressed']} | {4-v['regressed']} |")
    lines+=['',f'Positive zero-loss research candidates: {len(keep)}',
       f'Cross-piece saved fit models: {len(records)}',
       'All scores, 26 cases, K0–K6 counts and folds are preserved.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--features',type=Path)
    p.add_argument('--s18',type=Path)
    p.add_argument('--s29',type=Path)
    p.add_argument('--s20',type=Path)
    p.add_argument('--s25',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:selftest()
    else:
        require(args.features and args.s18 and args.s29 and
             args.s20 and args.s25 and args.output,'source artifacts absent')
        audit(args)
