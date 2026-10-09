"""S30: test whether corroboration blocks S29 errors without destroying corrections.

All decisions are label-free, pre-registered in analysis/README_V273_SERIE30...
Learned trust models train on changed proposals from OTHER pieces only.
The four observed broken events NEVER become handwritten exceptions.
"""
from __future__ import annotations

import argparse,csv,hashlib,json,time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.train_v273_aba_recurrent import prepare
from scripts.yourmt3_exactk_common import FOLDS,metrics,paired,require

REFERENCE='series29__lambda2__threshold0.02'
DSETS=('AF2','AF4','BF2','BF4')
MARGINS=(0.,.05,.10,.20)
CUTS=(.3,.5,.7)


def read(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def get(source,name):
    variants=list(map(str,source['variant_ids']))
    require(name in variants,'missing source variant '+name)
    return source['predictions'][:,variants.index(name)].astype(np.int8)


def confirmations(parent,proposed,A20,A25,B25):
    n=len(parent)
    ii=np.arange(n)
    P={'AF2':A20[:,1,:], 'AF4':A20[:,3,:],
       'BF2':A25[:,1,:], 'BF4':A25[:,3,:]}
    support={name:matrix.argmax(1)==proposed for name,matrix in P.items()}
    diffs={name:(matrix[ii,proposed]-matrix[ii,parent]) for name,matrix in P.items()}
    old_b=B25[:,0,:]
    bsupport=old_b[ii,proposed]>old_b[ii,parent]
    gates={
        'pass_AF2':support['AF2'],
        'pass_AF4':support['AF4'],
        'pass_BF2':support['BF2'],
        'pass_BF4':support['BF4'],
        'pass_both_final':support['AF4']&support['BF4'],
        'pass_either_final':support['AF4']|support['BF4'],
        'pass_both_second':support['AF2']&support['BF2'],
        'pass_either_second':support['AF2']|support['BF2'],
        'B0_supports_candidate':bsupport
    }
    for name in ('minimum','maximum'):
        joint=np.minimum(diffs['AF4'],diffs['BF4']) if name=='minimum' else np.maximum(diffs['AF4'],diffs['BF4'])
        for cut in MARGINS:
            gates[f'{name}_final_margin_gt{cut:g}']=(joint>cut)
    return gates,P,diffs,bsupport


def feature_rows(index,proposed,X,parent,A20,A25,B25):
    """Only acoustic evidence and observable proposed selection; never truth."""
    index=np.asarray(index,np.int64)
    proposed=np.asarray(proposed,np.int64)
    old=parent[index]
    ii=np.arange(len(index))
    one=np.eye(7,dtype=np.float32)
    s20=A20[index]
    s25=A25[index]
    b0=B25[index,0]
    components=[]
    for name,arr in (('AF2',s20[:,1]),('AF4',s20[:,3]),
                     ('BF2',s25[:,1]),('BF4',s25[:,3])):
        components.extend([arr[ii,proposed],arr[ii,old],
              arr[ii,proposed]-arr[ii,old],
              (arr.argmax(1)==proposed).astype(np.float32)])
    components.extend([
        b0[ii,proposed]-b0[ii,old],
        b0[ii,proposed],
        b0[ii,old],
        abs(proposed-old).astype(np.float32),
        (proposed>=2).astype(np.float32),
        (old>=2).astype(np.float32)])
    stats=np.column_stack(components).astype(np.float32)
    ret=np.column_stack((X[index,:90],one[old],one[proposed],
         s20[:,1],s20[:,3],s25[:,1],s25[:,3],b0,stats)).astype(np.float32)
    require(ret.shape[1]==90+14+28+7+len(components),'feature width drift')
    require(np.isfinite(ret).all(),'non-finite live evidence')
    return ret


def all_train_proposals(source,parent):
    ids=[];ks=[]
    for key in source['variant_ids']:
        name=str(key)
        if not name.startswith('series29__'):continue
        v=get(source,name)
        changed=np.flatnonzero(v!=parent)
        ids.append(changed)
        ks.append(v[changed])
    pairs=np.unique(np.column_stack((np.concatenate(ids),np.concatenate(ks))),axis=0)
    require(len(pairs)>70,'too few training S29 proposals for reliability')
    return pairs[:,0].astype(np.int64),pairs[:,1].astype(np.int64)


def selftest():
    p=np.array([2,1,3,2],np.int8);c=np.array([3,2,4,2],np.int8)
    A=np.tile(np.eye(7,dtype=np.float32)[c][:,None,:],(1,5,1))
    B=np.ones((4,4,7),np.float32)*.2
    B[np.arange(4),0,c]=.7
    gates,prob,diff,bs=confirmations(p,c,A,A,B)
    assert gates['pass_both_final'].all()
    assert gates['B0_supports_candidate'][:3].all()
    X=np.zeros((4,335),np.float32)
    f=feature_rows(np.arange(4),c,X,p,A,A,B)
    assert f.shape==(4,90+14+28+7+22)
    for i in (0,1,2):
        assert diff['AF4'][i]>0
    print('PASS: fixed 26 corroboration rules, acoustic quality features and no ground truth selection')


def main(args):
    require(not args.output.exists(),'no overwrite of research files')
    started=time.monotonic()
    d=prepare(args)
    data=read(args.s29)
    paths=read(args.routes)
    p20=read(args.s20)
    p25=read(args.s25)
    for p in (data,paths,p20,p25):
        require(np.array_equal(p['global_index'],d['ids']),'source event index mismatch')
    require(np.array_equal(data['true_K'],d['y']) and
            np.array_equal(data['fold'],d['fold']),'S29 scoring identity drift')
    parent=get(data,'series18_parent')
    require(np.array_equal(parent,d['parent']),'S18 parent not conserved')
    original=get(data,REFERENCE)
    require(int(((original==d['y'])&(parent!=d['y'])).sum())==17 and
            int(((original!=d['y'])&(parent==d['y'])).sum())==4,
            '17 corrections and 4 regressions reference drift')
    A20=np.asarray(p20['A_probs'],np.float32)
    A25=np.asarray(p25['A_probs'],np.float32)
    B25=np.asarray(p25['B_compatibility'],np.float32)
    require(A20.shape==A25.shape==(59309,5,7) and
            B25.shape==(59309,4,7),'cross-order sources incomplete')
    gates,P,diffs,bs=confirmations(parent,original,A20,A25,B25)
    require(len(gates)==9+2*len(MARGINS),'wrong preregistered tests')
    result={'series18_parent':parent.copy(),'freeze_parent':d['freeze'].copy(),
            REFERENCE:original.copy()}
    for key,gate in gates.items():
        result[f'series30__{key}']=np.where(gate,original,parent).astype(np.int8)

    # Meta-quality learned on every distinct S29 changed proposal from
    # other pieces of the SAME fold whenever both label classes are present.
    # These are NOT fitted on the 4 held event regressions.
    srcid,srck=all_train_proposals(data,parent)
    scores={algo:np.full(len(parent),np.nan,np.float32)
            for algo in ('logistic','hgb')}
    models_ledger=[]
    args.output.mkdir(parents=True)
    (args.output/'models').mkdir()
    chosen=np.flatnonzero(original!=parent)
    for piece in sorted(set(d['pieces'].tolist())):
        held=d['pieces']==piece
        if not np.any(held& (original!=parent)):
            continue
        fold=int(np.unique(d['fold'][held])[0])
        fit=(d['pieces'][srcid]!=piece)&(d['fold'][srcid]==fold)
        if len(np.unique((srck[fit]==d['y'][srcid[fit]]).astype(int)))<2 or fit.sum()<50:
            fit=(d['pieces'][srcid]!=piece)
        fit_i=srcid[fit];fit_k=srck[fit]
        trainlabel=(fit_k==d['y'][fit_i]).astype(np.int32)
        require(len(np.unique(trainlabel))==2,
            'single-class training cannot learn reliability without test truth')
        xi=np.flatnonzero(held&(original!=parent))
        xin=feature_rows(fit_i,fit_k,d['X'],parent,A20,A25,B25)
        xout=feature_rows(xi,original[xi],d['X'],parent,A20,A25,B25)
        scaler=StandardScaler().fit(xin)
        train=np.clip(scaler.transform(xin),-6,6)
        test=np.clip(scaler.transform(xout),-6,6)
        for algo in scores:
            model=LogisticRegression(C=.1,class_weight='balanced',max_iter=350,
                     solver='liblinear',random_state=27330) if algo=='logistic' else (
                HistGradientBoostingClassifier(max_iter=80,max_depth=3,
                    max_leaf_nodes=10,min_samples_leaf=20,l2_regularization=10,
                    learning_rate=.05,random_state=27330))
            with threadpool_limits(limits=2):
                model.fit(train,trainlabel)
                pp=model.predict_proba(test)[:,1]
            scores[algo][xi]=pp.astype(np.float32)
            filename=f'{piece}__{algo}.joblib'
            joblib.dump(dict(model=model,scaler=scaler,
                held_piece=piece,fold=fold,algo=algo,
                train_event_ids=d['ids'][fit_i],
                held_event_ids=d['ids'][xi],train_proposed_K=fit_k),
                args.output/'models'/filename,compress=3)
            models_ledger.append(dict(file=filename,piece=piece,fold=fold,
                train_count=len(trainlabel),correct_train=int(trainlabel.sum()),
                incorrect_train=int((trainlabel==0).sum()),
                held_count=len(xi),
                trained_on_same_fold=bool((d['fold'][fit_i]==fold).all()),
                fitted_hash=hashlib.sha256(d['ids'][fit_i].astype('<i8').tobytes()).hexdigest()))
    for algo,v in scores.items():
        require(np.isfinite(v[chosen]).all(),'unscored changed event '+algo)
        for cut in CUTS:
            key=f'series30__learned_{algo}__p_gt{cut:g}'
            result[key]=np.where(v>cut,original,parent).astype(np.int8)
    require(len(result)==3+len(gates)+len(scores)*len(CUTS),
            'fixed policy inventory unexpectedly changed')

    audits={}
    for name,vec in result.items():
        changed=vec!=parent
        audits[name]=dict(metrics=metrics(d['y'],vec),
            vs_S18=paired(d['y'],parent,vec),
            vs_freeze=paired(d['y'],d['freeze'],vec),
            corrected=int((changed&(vec==d['y'])&(parent!=d['y'])).sum()),
            regressed=int((changed&(vec!=d['y'])&(parent==d['y'])).sum()),
            neutral=int((changed&(vec!=d['y'])&(parent!=d['y'])).sum()),
            by_true_K={str(k):dict(total=int((d['y']==k).sum()),
                correct=int(((d['y']==k)&(vec==d['y'])).sum()),
                corrections=int(((d['y']==k)&changed&
                       (vec==d['y'])&(parent!=d['y'])).sum()),
                regressions=int(((d['y']==k)&changed&
                       (vec!=d['y'])&(parent==d['y'])).sum()))
                for k in range(7)},
            folds={str(f):dict(metrics=metrics(d['y'][d['fold']==f],
                 vec[d['fold']==f]),vs_S18=paired(d['y'][d['fold']==f],
                 parent[d['fold']==f],vec[d['fold']==f])) for f in FOLDS})
    # All original 21 changed events, even when correctly blocked, are retained.
    original_changed=np.flatnonzero(original!=parent)
    require(len(original_changed)==26 and int(((original!=parent)&(parent!=d['y'])&(original!=d['y'])).sum())==5, 'changed cases include 5 neutral events')
    with (args.output/'all_26_S29_changed_cases.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['native_id','fold','piece','true_K','old_K','S29_K',
            'original_outcome','AF2_agree','AF4_agree','BF2_agree',
            'BF4_agree','B0_support','AF4_delta','BF4_delta',
            'learned_logistic','learned_hgb','retained_by_policy_names'])
        for i in original_changed:
            accepted=[k for k,v in result.items()
                      if k.startswith('series30__') and v[i]==original[i]]
            writer.writerow([int(d['ids'][i]),int(d['fold'][i]),str(d['pieces'][i]),
                int(d['y'][i]),int(parent[i]),int(original[i]),
                'corrected' if original[i]==d['y'][i] else
                'regressed' if parent[i]==d['y'][i] else 'neutral',
                *[int(gates[f'pass_{key}'][i]) for key in DSETS],
                int(bs[i]),round(float(diffs['AF4'][i]),5),
                round(float(diffs['BF4'][i]),5),
                round(float(scores['logistic'][i]),5),
                round(float(scores['hgb'][i]),5),';'.join(accepted)])
    ranking=sorted(audits,key=lambda name:(
        audits[name]['metrics']['correct'],
        audits[name]['metrics']['poly']['correct']),reverse=True)
    positive_zero_loss=[k for k in audits if k.startswith('series30__')
            and audits[k]['corrected']>0 and audits[k]['regressed']==0
            and audits[k]['metrics']['poly']['correct']>=2998]
    report=dict(status='completed',source_run_S29=37865173636,
        original_26_changes=26,original_corrected=17,original_regressed=4,
        parent_S18_unmodified=True,independent_unseen_validation=False,
        historically_exposed_development_dataset=True,
        source_experts_per_piece_out_of_fit=True,
        trained_reliability_piece_exclusion=True,
        meta_producers_could_have_second_order_label_leakage=True,
        prespecified_policy_count=len(result),
        learned_quality_models=models_ledger,
        original_changed_native_ids=d['ids'][original_changed].tolist(),
        audits=audits,best_global=ranking[0],
        positive_zero_loss_research_candidates=positive_zero_loss,
        experimental_only_no_auto_promotion=True,
        elapsed_seconds=round(time.monotonic()-started,2))
    (args.output/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    np.savez_compressed(args.output/'predictions.npz',global_index=d['ids'],
        true_K=d['y'],fold=d['fold'],variant_ids=np.asarray(list(result)),
        predictions=np.column_stack(list(result.values())))
    np.savez_compressed(args.output/'gate_scores.npz',global_index=d['ids'],
        quality_logistic=scores['logistic'],quality_hgb=scores['hgb'])
    lines=['# Série30 — corroboration indépendante A-first/B-first des 4 régressions S29','',
        '**Étude de développement déjà exposée, pas validation sur musique inédite.**',
        'Pour chaque proposition de la S29, toutes les règles sont dérivées des sorties A/B et des entrées de signal, pas du vrai K.',
        f"Changed S29 events: {len(original_changed)} = 17 fixed + 4 broken + 5 neutral.",
        '', '| Policy | Exact global | Exact poly | Fix vs S18 | Break vs S18 | Original 4 blocked | Original 17 retained |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for k in ranking:
        v=audits[k];m=v['metrics']
        lines.append(f"| {k} | {100*m['exact']:.4f}% | "
             f"{100*m['poly']['exact']:.4f}% | {v['corrected']} | "
             f"{v['regressed']} | {4-v['regressed']} | {v['corrected']} |")
    lines+=['',f"Loss-free research candidates (still NOT promoted): {len(positive_zero_loss)}",
        f"Learned models: {len(models_ledger)}",
        'Detailed 26 changed-case CSV (17 fixes, 4 regressions, 5 neutral); every rejected selection and fold/per-K metric preserved.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--features',type=Path)
    p.add_argument('--s18',type=Path)
    p.add_argument('--s29',type=Path)
    p.add_argument('--routes',type=Path)
    p.add_argument('--s20',type=Path)
    p.add_argument('--s25',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        require(a.features and a.s18 and a.s29 and a.routes and
             a.s20 and a.s25 and a.output,'complete S29/S20/S25 evidence required')
        main(a)
