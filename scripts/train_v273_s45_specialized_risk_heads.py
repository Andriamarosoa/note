"""S45: fit a separate correction-vs-regression risk head for EACH S43/S44 family.

The original prediction vectors remain immutable. For each held piece,
train the family-specific risk model on OTHER pieces in that piece's fold.
This is exploratory: sources were chosen post-hoc and old historical
producer OOF provenance is not universally verifiable. No H9.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from scripts.train_v273_aba_recurrent import prepare
from scripts.yourmt3_exactk_common import FOLDS,metrics,paired,require

LAMS=(1.,2.,4.)
CUTS=(0.,.02,.05,.10)
N=59309
BAN=('h9_','yourmt3','your_mt3','oracle')

def sha(a):
    return hashlib.sha256(np.asarray(a,dtype='<i8').tobytes()).hexdigest()

def source_data(path):
    with np.load(path,allow_pickle=False) as q:
        return {k:q[k] for k in q.files}

def make_inputs(X,old,cand,votes):
    """Audio/votes/flux and candidate-vote agreement, no ground truth."""
    n=len(old)
    cls=np.eye(7,dtype=np.float32)
    x=np.column_stack([
        X[:,:139],cls[old],cls[cand],
        np.abs(cand.astype(np.float32)-old.astype(np.float32))[:,None]/6.,
        np.sign(cand.astype(np.float32)-old.astype(np.float32))[:,None],
        np.asarray(votes,dtype=np.float32)[:,None]/16.,
    ]).astype(np.float32)
    require(x.shape==(n,156),'risk evidence schema drift')
    require(np.isfinite(x).all(),'source signal non-finite')
    return x

def choose_best(baseline,proposal,risk,lam,cut):
    """Either one best-scoring proposal or untouched parent K."""
    score=risk[:,:,0]-float(lam)*risk[:,:,1]
    score=np.where(np.isfinite(score),score,-np.inf)
    j=np.argmax(score,axis=1)
    top=score[np.arange(len(baseline)),j]
    good=np.isfinite(top)&(top>cut)
    return np.where(good,proposal[np.arange(len(baseline)),j],baseline).astype(np.int8),np.where(good,j,-1).astype(np.int8)

def selftest():
    x=np.zeros((4,335),np.float32)
    old=np.array([0,2,3,4],np.int8)
    proposal=np.array([1,1,4,3],np.int8)
    feat=make_inputs(x,old,proposal,np.array([1,0,2,5]))
    require(feat.shape==(4,156),'bad risk features')
    proposals=np.array([[1,2],[3,4],[1,3],[5,4]],np.int8)
    r=np.full((4,2,2),np.nan,np.float32)
    r[:,0]=[.8,.1];r[:,1]=[.15,.25]
    p,j=choose_best(old,proposals,r,2.,.1)
    assert j.tolist()==[0,0,0,0]
    assert p.tolist()==[1,3,1,5]
    r[:]=[.01,.9]
    p,j=choose_best(old,proposals,r,2.,.1)
    assert j.tolist()==[-1]*4 and np.array_equal(p,old)
    print('PASS: family-specific learned correction-risk feature/action rules, no true K/H9')

def run(a):
    require(not a.output.exists(),'never overwrite experiment')
    started=time.monotonic()
    d=prepare(a)
    old=d['parent'].astype(np.int8)
    require(len(old)==N and metrics(d['y'],old)['correct']==49178
           and metrics(d['y'],old)['poly']['correct']==2998,
           'S18 native reference drift')
    native=source_data(a.bank)
    require(np.array_equal(native['global_index'],d['ids']) and
            np.array_equal(native['true_K'],d['y']) and
            np.array_equal(native['fold'],d['fold']) and
            np.array_equal(native['S18_reference_K'],old),
            'cross-source archive mismatch; no valid alignment')
    manifest=json.loads(a.manifest.read_text())
    require(manifest['status']=='completed' and
            len(manifest['family_champions'])>=16,
            'incomplete S44 family candidate inventory')
    digest_index={s:i for i,s in enumerate(map(str,native['distinct_vector_sha256']))}
    champions={}
    for q in manifest['family_champions']:
        if q['kind']!='gross':continue
        fam=q['family']
        if any(tag in (q['variant']+' '+fam).lower() for tag in BAN):
            raise ValueError('H9 or oracle source in available experts')
        if fam in champions:raise ValueError('duplicate source family')
        if q['sha256'] not in digest_index:raise ValueError('missing native S43 SHA')
        champions[fam]=q
    require(len(champions)>=16,'not enough original correction families')
    families=sorted(champions)
    candidate=np.column_stack([
        native['predictions'][:,digest_index[champions[f]['sha256']]]
        for f in families]).astype(np.int8)
    require(candidate.shape==(N,len(families)),'family prediction bank incomplete')
    # Agreement is only one vote per DIFFERENT family (not correlated
    # alternate candidates from the same source group).
    agreement=np.zeros((N,len(families)),np.int16)
    for j in range(len(families)):
        agreement[:,j]=np.sum(
            (candidate==candidate[:,j,None])&
            (candidate!=old[:,None]),axis=1).astype(np.int16)-
            (candidate[:,j]!=old).astype(np.int16)
    risk=np.full((N,len(families),2),np.nan,np.float32)
    models=[]
    a.output.mkdir(parents=True)
    (a.output/'models').mkdir()
    pieces=np.asarray(d['pieces'])
    for piece in sorted(set(pieces.tolist())):
        held=np.flatnonzero(pieces==piece)
        fold=int(np.unique(d['fold'][held])[0])
        assert (d['fold'][held]==fold).all()
        other=(d['fold']==fold)&(pieces!=piece)
        for j,fam in enumerate(families):
            cand=candidate[:,j]
            fit=np.flatnonzero(other&(cand!=old))
            test=held[cand[held]!=old[held]]
            require(not np.intersect1d(fit,held).size,
                    'held-piece training leakage')
            require(len(fit)>15,
                    'insufficient training proposals for '+str((piece,fam)))
            # Two targets use ONLY true K of training pieces to label whether
            # proposal fixes an old error or destroys a correct prediction.
            fix=((old[fit]!=d['y'][fit])&(cand[fit]==d['y'][fit])).astype(np.float32)
            lose=((old[fit]==d['y'][fit])&(cand[fit]!=d['y'][fit])).astype(np.float32)
            y=np.column_stack([fix,lose])
            x=make_inputs(d['X'][fit],old[fit],cand[fit],agreement[fit,j])
            scaler=StandardScaler().fit(x)
            model=Ridge(alpha=80.,random_state=None)
            with threadpool_limits(limits=2):
                model.fit(np.clip(scaler.transform(x),-6,6),y)
                if len(test):
                    xt=make_inputs(d['X'][test],old[test],cand[test],agreement[test,j])
                    value=model.predict(np.clip(scaler.transform(xt),-6,6))
                    risk[test,j]=np.clip(value,0,1).astype(np.float32)
            path=a.output/'models'/f'{piece}__{fam}.joblib'
            joblib.dump(dict(model=model,scaler=scaler,
                family=fam,piece=piece,fold=fold,
                train_native_ids=d['ids'][fit],
                held_native_ids=d['ids'][test],
                source_original_sha256=champions[fam]['sha256'],
                train_corrected=int(fix.sum()),train_regressed=int(lose.sum())),
                path,compress=3)
            models.append(dict(file=str(path.relative_to(a.output)),
                family=fam,piece=piece,fold=fold,
                trained=int(len(fit)),evaluated=int(len(test)),
                source_sha256=champions[fam]['sha256'],
                train_ids_SHA256=sha(d['ids'][fit]),
                held_ids_SHA256=sha(d['ids'][test]),
                corrected_train=int(fix.sum()),regressed_train=int(lose.sum())))
        print(json.dumps(dict(piece=piece,fold=fold,
            models_completed=len(models),seconds=round(time.monotonic()-started,1))),flush=True)
    require(np.isfinite(risk[candidate!=old]).all(),
            'unscored proposed correction risk')
    # There is no action at unchanged candidate K; it is never selected.
    per_family={}
    outputs={'series18_parent':old.copy()}
    selected={}
    for j,fam in enumerate(families):
        for lam in LAMS:
            val=risk[:,j,0]-lam*risk[:,j,1]
            for cut in CUTS:
                key=f'series45__{fam}__lambda{lam:g}__cut{cut:g}'
                accepted=(candidate[:,j]!=old)&np.isfinite(val)&(val>cut)
                outputs[key]=np.where(accepted,candidate[:,j],old).astype(np.int8)
    for lam in LAMS:
        for cut in CUTS:
            key=f'series45__ALLfamilies__lambda{lam:g}__cut{cut:g}'
            vec,j=choose_best(old,candidate,risk,lam,cut)
            outputs[key]=vec;selected[key]=j
    measures={}
    for name,pred in outputs.items():
        diff=old!=pred
        correction=diff&(old!=d['y'])&(pred==d['y'])
        regression=diff&(old==d['y'])&(pred!=d['y'])
        neutral=diff&(old!=d['y'])&(pred!=d['y'])
        poly=(d['y']>=2)
        measures[name]=dict(corrected=int(correction.sum()),
            regressed=int(regression.sum()),
            neutral=int(neutral.sum()),
            net=int(correction.sum()-regression.sum()),
            poly_corrected=int((correction&poly).sum()),
            poly_regressed=int((regression&poly).sum()),
            poly_net=int((correction&poly).sum()-(regression&poly).sum()),
            metrics=metrics(d['y'],pred),
            versus_S18=paired(d['y'],old,pred),
            per_true_K={str(k):dict(
                corrected=int((correction&(d['y']==k)).sum()),
                regressed=int((regression&(d['y']==k)).sum())) for k in range(7)},
            per_source_K={str(k):dict(
                corrected=int((correction&(old==k)).sum()),
                regressed=int((regression&(old==k)).sum())) for k in range(7)},
            per_fold={str(f):dict(
                corrected=int((correction&(d['fold']==f)).sum()),
                regressed=int((regression&(d['fold']==f)).sum())) for f in FOLDS},
            selection_by_family={families[i]:int((selected[name]==i).sum())
                for i in range(len(families))} if name in selected else {})
    rank=sorted(measures,key=lambda k:(measures[k]['metrics']['correct'],
             measures[k]['metrics']['poly']['correct']),reverse=True)
    bestfam={}
    for fam in families:
        vals=[k for k in outputs if k.startswith(f'series45__{fam}__')]
        bestfam[fam]=max(vals,key=lambda k:(measures[k]['net'],
                                measures[k]['poly_net']))
    safe=[k for k in outputs if k!='series18_parent' and
          measures[k]['corrected']>0 and measures[k]['regressed']==0 and
          measures[k]['metrics']['poly']['correct']>=2998]
    metadata=dict(status='completed',source_S43_complete_bank=True,
        source_S44_per_family_champions=True,source_trained_provenance_historical_incomplete=True,
        models_saved=len(models),model_provenance=models,
        family_sources=champions,families=families,
        observed_cross_family_agreement_in_inputs=True,
        H9_excluded=True,S18_unchanged=True,
        held_piece_labels_excluded_from_meta_fit=True,
        historical_source_selection_informed_by_test_cohort=True,
        not_unseen_composition_validation=True,no_production_promotion=True,
        evaluations=len(outputs),best_global=rank[0],top_global=rank[:20],
        best_per_family=bestfam,
        candidate_zero_regressions=safe,
        audits=measures,elapsed_seconds=round(time.monotonic()-started,1))
    (a.output/'report.json').write_text(json.dumps(metadata,ensure_ascii=False,
             sort_keys=True,indent=2)+'\n')
    np.savez_compressed(a.output/'predictions.npz',global_index=d['ids'],
        fold=d['fold'],true_K=d['y'],
        variant_ids=np.asarray(list(outputs)),
        predictions=np.column_stack(list(outputs.values())))
    np.savez_compressed(a.output/'risk_scores.npz',global_index=d['ids'],
        family_ids=np.asarray(families),
        proposal_K=candidate,risks=risk,independent_votes=agreement)
    np.savez_compressed(a.output/'selected_family.npz',global_index=d['ids'],
        variant_ids=np.asarray(list(selected)),
        selected_family=np.column_stack(list(selected.values())),
        family_ids=np.asarray(families))
    lines=['# S45 — nouveaux sélecteurs apprenants par boucle historique',
        '', f"{len(families)} familles, {len(models)} modèles piece-held Ridge "
        f"et {len(outputs)-1} politiques candidates.",
        '**Risques non calibrés ; cohortes et anciennes propositions choisies après connaissance des erreurs.**',
        '**Aucune promotion / H9 exclue / S18 intact.**','',
        '| Famille | Meilleur critère lambda/seuil | Corrigés | Régressions | Poly corrigés | Poly régressés |',
        '|---|---|---:|---:|---:|---:|']
    for fam in families:
        k=bestfam[fam];v=measures[k]
        lines.append(f"| {fam} | {k.split(fam)[-1]} | {v['corrected']} | "
            f"{v['regressed']} | {v['poly_corrected']} | {v['poly_regressed']} |")
    lines+=['','## Meilleurs scores globaux (K0–K6)','',
        '| Selection | Exact-K global | Exact-K poly | Corrections | Régressions |',
        '|---|---:|---:|---:|---:|']
    for k in rank[:25]:
        v=measures[k];m=v['metrics']
        lines.append(f"| {k} | {m['exact']*100:.4f}% | "
            f"{m['poly']['exact']*100:.4f}% | "
            f"{v['corrected']} | {v['regressed']} |")
    lines+=['',f"Critère strict sans régression + correction positive + maintien du poly : {len(safe)} variantes.",
       'Les poids individuels, refus et métriques K0–K6/fold sont entièrement sauvegardés.']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('.'))
    ap.add_argument('--features',type=Path)
    ap.add_argument('--s18',type=Path)
    ap.add_argument('--bank',type=Path)
    ap.add_argument('--manifest',type=Path)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:selftest()
    else:
        require(args.features and args.s18 and args.bank and
                args.manifest and args.output,'native S43/S44 records and features required')
        run(args)
