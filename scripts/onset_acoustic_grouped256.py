"""Second direct-acoustic experiment: four-composition inner calibration.

Models trained on three compositions can be reused for the unordered pair of
excluded OUTER and CAL compositions. Ten pairs cover all twenty inner folds.
The first experiment supplies five of these pairs; the other five are trained
here. The selected model is then refitted on four inner compositions, saved,
and only then used on OUTER. No CPU/TP-loss/FP-budget acceptance constraint.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import argparse
import itertools
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from onset256.acoustic_attack import AttackPolicy,SCHEMA,DIM
from onset256.acoustic_calibration import decode_cache,count_cache,truth_arrays
from scripts.onset_acoustic_direct256 import (
    COMPS,PREREGISTRATION,SEED,baseline_mask,probabilities,mask_events,
    digest,dump,metric,counts,sum_scores)

GROUPED_SEED=2026102305
SPEC=PREREGISTRATION['model']
CUTS=(.75,.80,.85,.90,.95,.975,.99)


def pair_name(pair):return '__'.join(sorted(pair))


def fit(inner,directory,seed):
    groups=[np.load(directory/'samples'/f'{c}_w12.npz') for c in inner]
    x=np.concatenate([g['X'] for g in groups]);y=np.concatenate([g['y'] for g in groups])
    for g in groups:g.close()
    model=HistGradientBoostingClassifier(**SPEC,early_stopping=False,random_state=seed)
    model.fit(x,y)
    return model


def prepare(args):
    if args.out.exists():raise FileExistsError('immutable grouped experiment')
    args.out.mkdir()
    for name in ('features','samples'):
        (args.out/name).symlink_to((args.previous/name).resolve(),target_is_directory=True)
    shutil.copyfile(args.previous/'collection_report.json',args.out/'collection_report.json')
    (args.out/'pairs').mkdir();(args.out/'trained').mkdir()
    pairs=list(itertools.combinations(COMPS,2))
    source_pairs={pair_name((outer,COMPS[(i+1)%5])):outer for i,outer in enumerate(COMPS)}
    dump(args.out/'preregistration.json',dict(schema=SCHEMA,experiment='grouped_inner_four',
        seed=GROUPED_SEED,width_hops=12,model=SPEC,cuts=CUTS,reset_ratios=[.25,.5],
        gap_hops=[7,14],modes=PREREGISTRATION['modes'],
        width_reason='12-hop supervision won every available first-loop CAL comparison',
        baseline_fallback_allowed=True,objective='pooled four-composition inner OOF event F1',
        inner_training_compositions=3,final_training_compositions=4,
        existing_pair_sources=source_pairs,pairs=pairs,trained_hop_samples=256,
        future_audio_samples=0,added_hold_samples=0,
        corpus_already_explored=True,historical_upstream_overlap=True,
        external_generalization_proven=False,no_outer_threshold_tuning=True))


def pair_fit(task):
    directory,previous,pair=Path(task[0]),Path(task[1]),tuple(task[2])
    name=pair_name(pair);out=directory/'pairs'/name;out.mkdir()
    inner=[c for c in COMPS if c not in pair]
    source_pairs=json.loads((directory/'preregistration.json').read_text())['existing_pair_sources']
    source=source_pairs.get(name)
    cp=out/'model.joblib'
    with threadpool_limits(limits=1):
        if source:
            src=previous/'trained'/source/'acoustic_w12.joblib'
            # The last first-loop fold can still be finishing; each read waits
            # for its completed immutable result rather than a partial model.
            deadline=time.monotonic()+5400
            while not (src.parent/'result.json').exists():
                if time.monotonic()>deadline:raise TimeoutError('source fold missing')
                time.sleep(5)
            old=json.loads((src.parent/'frozen_model_and_policy.json').read_text())
            if sorted(old['train_compositions'])!=sorted(inner):
                raise AssertionError('pair reuse would leak an excluded composition')
            shutil.copyfile(src,cp);model=joblib.load(cp)
        else:
            print('GROUPED_PAIR_TRAIN_BEGIN',name,flush=True)
            model=fit(inner,directory,GROUPED_SEED+list(itertools.combinations(COMPS,2)).index(pair))
            joblib.dump(model,cp,compress=3)
            print('GROUPED_PAIR_TRAIN_DONE',name,flush=True)
        dump(out/'metadata.json',dict(excluded=list(pair),train_compositions=inner,
            source=source,checkpoint_sha256=digest(cp),trained_hop_samples=256,
            frozen_before_predictions=True))
        rows=json.loads((directory/'collection_report.json').read_text())['per_track']
        for row in rows:
            if row['composition'] not in pair:continue
            target=out/f"probability_{row['index']:03d}.npy"
            reusable=(previous/'trained'/source/f"probability_{row['index']:03d}.npy") if source else None
            old=json.loads((previous/'trained'/source/'frozen_model_and_policy.json').read_text()) if source else None
            if (reusable is not None and reusable.exists() and
                    old['selected']['name']=='direct_acoustic' and old['selected']['width']==12):
                shutil.copyfile(reusable,target)
            else:np.save(target,probabilities(model,row,directory))
        dump(out/'complete.json',dict(complete=True,excluded=list(pair),
            probability_sha256={p.name:digest(p) for p in out.glob('probability_*.npy')}))
        print('GROUPED_PAIR_READY',name,flush=True)


def candidate_policies():
    for cut,ratio,gap,mode in itertools.product(CUTS,(.25,.5),(7,14),PREREGISTRATION['modes']):
        yield AttackPolicy(cut=cut,reset_ratio=ratio,gap_hops=gap,
            use_baseline=mode!='direct_only',
            baseline_cut=.05 if mode=='veto_baseline_005' else .15 if mode=='veto_baseline_015' else 0.)


def final_fold(task):
    directory,outer=Path(task[0]),task[1]
    rows=json.loads((directory/'collection_report.json').read_text())['per_track']
    reference=json.loads(Path('series/ONSET_SCORE_RECOVERY_HGB256/evidence/sealed_predictions.json').read_text())
    inner=[r for r in rows if r['composition']!=outer]
    train=[c for c in COMPS if c!=outer]
    out=directory/'trained'/outer;out.mkdir()
    caches=[]
    for row in inner:
        p=directory/'pairs'/pair_name((outer,row['composition']))
        meta=json.loads((p/'metadata.json').read_text())
        if outer in meta['train_compositions'] or row['composition'] in meta['train_compositions']:
            raise AssertionError('invalid inner holdout')
        q=np.load(p/f"probability_{row['index']:03d}.npy")
        caches.append((row,q,baseline_mask(row,reference),truth_arrays(row['truth'])))
    baseline=sum_scores([metric(*count_cache(b,*t)) for _,_,b,t in caches])
    best=dict(name='reference',score=baseline,width=None,policy=asdict(AttackPolicy(enabled=False)))
    candidates=[]
    with threadpool_limits(limits=1):
        for policy in candidate_policies():
            score=sum_scores([metric(*count_cache(decode_cache(q,b,policy),*t))
                              for _,q,b,t in caches])
            candidate=dict(name='direct_acoustic',width=12,policy=asdict(policy),score=score)
            candidates.append(candidate)
            if score['f1']>best['score']['f1']+1e-12:best=candidate
        policy=AttackPolicy(**best['policy'])
        for row,q,b,t in caches:
            mask=decode_cache(q,b,policy)
            a=counts(mask_events(mask),row['truth']);bscore=metric(*count_cache(mask,*t))
            if any(a[k]!=bscore[k] for k in ('tp','fp','fn')):raise AssertionError('matching mismatch')
        print('GROUPED_CAL_SELECTED',outer,json.dumps(best),flush=True)
        model=None;cp=out/'acoustic_w12.joblib'
        if best['name']!='reference':
            model=fit(train,directory,GROUPED_SEED+COMPS.index(outer)+100)
            joblib.dump(model,cp,compress=3)
        metadata=dict(schema=SCHEMA,trained_hop_samples=256,dim=DIM,
            outer_composition=outer,cal_composition='four_inner_compositions',
            cal_compositions=train,train_compositions=train,selected=best,
            cal_reference=baseline,history=candidates,calibration='four_inner_grouped_OOF',
            frozen_before_outer=True,historical_upstream_overlap=True,
            external_generalization_proven=False,added_hold_samples=0,future_audio_samples=0)
        if model is not None:metadata['checkpoint_sha256']=digest(cp)
        dump(out/'frozen_model_and_policy.json',metadata)
        predicted={};hashes={};outer_rows=[r for r in rows if r['composition']==outer]
        for row in outer_rows:
            base=baseline_mask(row,reference)
            q=probabilities(model,row,directory) if model is not None else np.zeros(base.shape,np.float32)
            p=out/f"probability_{row['index']:03d}.npy";np.save(p,q);hashes[p.name]=digest(p)
            predicted[row['track']]=decode_cache(q,base,policy)
        dump(out/'outer_prediction_freeze.json',dict(probabilities=hashes,
            model_and_policy_sha256=digest(out/'frozen_model_and_policy.json')))
        records=[];sealed={}
        for row in outer_rows:
            events=mask_events(predicted[row['track']])
            records.append(dict(track=row['track'],composition=outer,score=counts(events,row['truth'])))
            sealed[row['track']]=sorted((e['sample'],e['pitch']) for e in events)
        result=dict(outer=outer,selected=best,outer_score=sum_scores([r['score'] for r in records]))
        dump(out/'predictions.json',sealed);dump(out/'result.json',result)
        print('GROUPED_OUTER',outer,json.dumps(result['outer_score']),flush=True)
        return result,records,sealed


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=5)
    args=p.parse_args();prepare(args)
    pairs=list(itertools.combinations(COMPS,2))
    # New pairs first; the one still-running source fold is checked later.
    source=json.loads((args.out/'preregistration.json').read_text())['existing_pair_sources']
    pairs.sort(key=lambda pair:pair_name(pair) in source)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(pair_fit,[(str(args.out),str(args.previous),pair) for pair in pairs]))
        results=list(pool.map(final_fold,[(str(args.out),c) for c in COMPS]))
    folds=[];rows=[];sealed={}
    for fold,records,predicted in results:
        folds.append(fold);rows.extend(records);sealed.update(predicted)
    report=dict(complete=True,folds=folds,per_track=rows,
        totals=sum_scores([r['score'] for r in rows]),new_pair_models=5,reused_pair_models=5,
        final_models_trained=sum(f['selected']['name']!='reference' for f in folds),
        hop_samples=256,added_hold_samples=0,future_audio_samples=0,
        fresh_audio_replay_required=True,historical_upstream_overlap=True,
        external_generalization_proven=False,promoted=False)
    dump(args.out/'trained/report.json',report)
    dump(args.out/'trained/sealed_predictions.json',sealed)
    print('GROUPED_TOTAL',json.dumps(report['totals']),flush=True)


if __name__=='__main__':main()
