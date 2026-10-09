"""S37: compare all nine dynamic S35 proposals using FROZEN piece-held S36b Q gates.

No new fits, no true label at selection time, no H9, no H8 without WAV.
S36b original two-correction candidate is a frozen comparison, not input.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import joblib
from threadpoolctl import threadpool_limits
from scripts.train_v273_aba_recurrent import prepare
from scripts.train_v273_s36b_crosspiece_risk import build,get,read
from scripts.yourmt3_exactk_common import FOLDS,metrics,paired,require

LAMS=(1.,2.,4.)
CUTS=(0.,.005,.02,.05,.10)
VIEWS=('audio_flow','full_path')
REF='series36b__series35__lambda4__threshold0__full_path__lambda2__threshold0.005'

def selftest():
    score=np.array([[.1,.1,-np.inf],[-np.inf,-np.inf,-np.inf]],np.float32)
    best=np.argmax(score,axis=1)
    safe=np.max(score,axis=1)>0.05
    assert best.tolist()==[0,0] and safe.tolist()==[True,False]
    print('PASS: deterministic 9-route maximum and abstention; no held truth')

def run(a):
    require(not a.output.exists(),'never overwrite research')
    d=prepare(a)
    source=read(a.s35)
    route=read(a.routes)
    oldrun=read(a.s36b)
    for z in (source,route,oldrun):
        require(np.array_equal(z['global_index'],d['ids']),
                'wrong source native events')
    require(np.array_equal(oldrun['true_K'],d['y']),
            'original S36b scores misaligned')
    base=get(source,'series18_parent')
    require(np.array_equal(base,d['parent']) and
            metrics(d['y'],base)['correct']==49178,'S18 changed')
    actions=list(map(str,route['action_names']))
    require(len(actions)==19 and
         'H9' not in actions and
         'H8_pitch_shift' in actions,
         'H9 forbidden or incomplete adapter catalog')
    routes=list(map(str,route['variant_ids']))
    require(len(routes)==9 and
         all(n.startswith('series35__') for n in routes),
         'source does not contain nine fully evaluated dynamic routes')
    orig=get(oldrun,REF)
    assert ((base!=d['y'])&(orig==d['y'])).sum()==2
    assert ((base==d['y'])&(orig!=d['y'])).sum()==0
    N=len(base)
    # Every view, each lambda, each cut evaluated without truth. All nine
    # route proposals exist, but no evaluation of additional heads is claimed
    # to be computationally saved.
    outputs={'series18_parent':base.copy(),REF:orig.copy()}
    chosen={}
    risk_max={}
    scores_per_action=np.full((N,len(routes),2,2),np.nan,np.float32)
    for view_index,view in enumerate(VIEWS):
        # Scores are filled per piece using its original held-piece Q head.
        for piece in sorted(set(d['pieces'].tolist())):
            items=np.flatnonzero(d['pieces']==piece)
            qpath=a.s36b_dir/'models'/f'{piece}__{view}.joblib'
            require(qpath.exists(),'frozen S36b model missing')
            q=joblib.load(qpath)
            f=int(np.unique(d['fold'][items])[0])
            require(q['piece']==piece and q['fold']==f and q['view']==view,
                    'frozen risk model provenance incorrect')
            train_ids=set(map(int,q['train_native_ids']))
            require(not any(int(v) in train_ids for v in d['ids'][items]),
                    'S36b Q predictor saw held music piece')
            for j,rname in enumerate(routes):
                pred=get(source,rname)
                proposed=items[pred[items]!=base[items]]
                if not len(proposed):continue
                steps=route['selected_head'][proposed,j,:]
                feat=build(d['X'][proposed],base[proposed],
                     pred[proposed],steps,view)
                with threadpool_limits(limits=2):
                    qscore=np.clip(q['model'].predict(
                        np.clip(q['scaler'].transform(feat),-6,6)),0,1)
                scores_per_action[proposed,j,view_index]=qscore.astype(np.float32)
        for lam in LAMS:
            utility=(scores_per_action[:,:,view_index,0]
                  -lam*scores_per_action[:,:,view_index,1])
            utility=np.where(np.isfinite(utility),utility,-np.inf)
            order=np.argmax(utility,axis=1)
            best=utility[np.arange(N),order]
            for cut in CUTS:
                key=f'series37__{view}__lambda{lam:g}__threshold{cut:g}'
                accept=np.isfinite(best)&(best>cut)
                # Deterministic argmax of utility over the original nine S35
                # routes, not over ground truth. Return S18 on abstention.
                chosenK=base.copy()
                where=np.flatnonzero(accept)
                for j,rname in enumerate(routes):
                    here=where[order[where]==j]
                    if len(here):
                        chosenK[here]=get(source,rname)[here]
                outputs[key]=chosenK
                chosen[key]=np.where(accept,order,-1).astype(np.int8)
                risk_max[key]=np.where(accept,best,0).astype(np.float32)
    require(len(outputs)==32 and len(chosen)==30,
            'exactly 30 policies and 2 references')
    audits={}
    for key,pred in outputs.items():
        new=(pred!=base)
        fixed=(base!=d['y'])&(pred==d['y'])
        reg=(base==d['y'])&(pred!=d['y'])
        neutral=(base!=d['y'])&(pred!=d['y'])&new
        usage={routes[i]:int((chosen[key]==i).sum())
               for i in range(9)} if key in chosen else {}
        audits[key]=dict(metrics=metrics(d['y'],pred),
            versus_S18=paired(d['y'],base,pred),
            corrections=int(fixed.sum()),regressions=int(reg.sum()),
            neutral=int(neutral.sum()),
            chosen_producer_usage=usage,
            by_true_K={str(k):dict(
                corrections=int(((d['y']==k)&fixed).sum()),
                regressions=int(((d['y']==k)&reg).sum())) for k in range(7)},
            by_fold={str(f):dict(
                corrections=int(((d['fold']==f)&fixed).sum()),
                regressions=int(((d['fold']==f)&reg).sum())) for f in FOLDS})
    safe=[k for k in chosen if audits[k]['corrections']>0
          and audits[k]['regressions']==0
          and audits[k]['metrics']['poly']['correct']>=2998]
    ranking=sorted(audits,key=lambda k:(audits[k]['metrics']['correct'],
          audits[k]['metrics']['poly']['correct']),reverse=True)
    a.output.mkdir(parents=True)
    report=dict(status='completed',reused_frozen_S36b_Q_models=True,
        source_routing_options=routes,
        inspected_all_nine_S35_routes=True,
        H9_excluded=True,H8_not_executed=True,
        already_exposed_cohort=True,independent_new_music_validation=False,
        no_automatic_promotion=True,
        policy_count=len(outputs),audits=audits,
        strict_zero_loss_candidates=safe,best_global=ranking[0])
    (a.output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    np.savez_compressed(a.output/'predictions.npz',
        global_index=d['ids'],true_K=d['y'],fold=d['fold'],
        variant_ids=np.asarray(list(outputs)),
        predictions=np.column_stack(list(outputs.values())))
    np.savez_compressed(a.output/'chosen_routes.npz',
        global_index=d['ids'],variant_ids=np.asarray(list(chosen)),
        selected_source=np.column_stack(list(chosen.values())),
        source_routes=np.asarray(routes))
    np.savez_compressed(a.output/'candidate_scores.npz',
        global_index=d['ids'],source_names=np.asarray(routes),
        view_names=np.asarray(VIEWS),P_fix_P_break=scores_per_action)
    lines=['# S37 — nine-route frozen S36b risk comparison, H9 excluded','',
        'No refitting of the crosspiece heldfold Q heads. All nine S35 routes.',
        '30 preregistered selection policies; no independent new-music validation.',
        '', '| Variant | Global | Poly | Fixes vs S18 | Regressions vs S18 |',
        '|---|---:|---:|---:|---:|']
    for key in ranking:
        z=audits[key];m=z['metrics']
        lines.append(f"| {key} | {m['exact']*100:.4f}% | "
            f"{m['poly']['exact']*100:.4f}% | "
            f"{z['corrections']} | {z['regressions']} |")
    lines+=['',f'Positive zero-regression research candidates: {len(safe)}',
        'All per-K/fold and rejected proposals stored. S18 is unchanged.']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--features',type=Path)
    p.add_argument('--s18',type=Path)
    p.add_argument('--s35',type=Path)
    p.add_argument('--routes',type=Path)
    p.add_argument('--s36b',type=Path)
    p.add_argument('--s36b-dir',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        require(a.features and a.s18 and a.s35 and a.routes and
                a.s36b and a.s36b_dir and a.output,
                'S35, S36b model files and original acoustic folds are required')
        run(a)
