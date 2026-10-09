"""S34: learned acoustic 1-vs-2-note expert for S29 K1/K2 contradictions.

Training is on all K1/K2 original acoustic events of OTHER pieces, not
only the four known regressions. No test true K in the runtime gating rule.
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

VIEWS={'audio58':slice(32,90),'flow245':slice(90,335),'all335':slice(0,335)}
ALGOS=('logistic','hgb')
LIMITS=(.35,.50,.65,.80)
S29='series29__lambda2__threshold0.02'
S31='series31__all335_logistic__p_gt0.25'


def load(p):
    with np.load(p,allow_pickle=False) as f:
        return {k:f[k] for k in f.files}


def pred(z,name):
    names=list(map(str,z['variant_ids']))
    require(name in names,'missing source '+name)
    return z['predictions'][:,names.index(name)].astype(np.int8)


def accept(parent,proposed,prob1,threshold):
    swap_2_to_1=(parent==2)&(proposed==1)
    swap_1_to_2=(parent==1)&(proposed==2)
    # All non K1/K2 transitions leave unchanged.
    permitted=(~(swap_2_to_1|swap_1_to_2))|(
        swap_2_to_1&(prob1>threshold))|(
        swap_1_to_2&(prob1<(1-threshold)))
    return np.where(permitted,proposed,parent).astype(np.int8)


def selftest():
    before=np.array([2,1,3,2],np.int8)
    after=np.array([1,2,4,1],np.int8)
    p=np.array([.9,.1,np.nan,.2],np.float32)
    final=accept(before,after,p,.65)
    assert final.tolist()==[1,2,4,2]
    assert len(VIEWS)==3 and len(ALGOS)==2 and len(LIMITS)==4
    print('PASS: 24 K1↔K2 acoustic gates; non-K1/K2 untouched; no held truth')


def execute(a):
    require(not a.output.exists(),'refuse to overwrite research files')
    time0=time.monotonic()
    d=prepare(a)
    z=load(a.s29);s31=load(a.s31)
    require(np.array_equal(z['global_index'],d['ids']) and
         np.array_equal(s31['global_index'],d['ids']),
         'source producer ID mismatch')
    require(np.array_equal(z['fold'],d['fold']) and
         np.array_equal(z['true_K'],d['y']),'K/fold source changed')
    parent=pred(z,'series18_parent')
    original=pred(z,S29)
    latest=pred(s31,S31)
    require(np.array_equal(parent,d['parent']),'source S18 parent mismatch')
    changed=(original!=parent)
    require(int(changed.sum())==26,'all 26 baseline changes required')
    conflict=((parent==2)&(original==1))|((parent==1)&(original==2))
    require(conflict.sum()>14,'no actual K1/K2 decisions')
    est={f'{v}_{m}':np.full(len(d['y']),np.nan,np.float32)
         for v in VIEWS for m in ALGOS}
    a.output.mkdir(parents=True)
    (a.output/'models').mkdir()
    trained=[]
    for piece in sorted(set(d['pieces'][conflict].tolist())):
        held=(d['pieces']==piece)&conflict
        f=int(np.unique(d['fold'][d['pieces']==piece])[0])
        fit=(d['pieces']!=piece)&(d['fold']==f)&np.isin(d['y'],[1,2])
        if len(np.unique(d['y'][fit]))<2 or fit.sum()<100:
            fit=(d['pieces']!=piece)&np.isin(d['y'],[1,2])
        require(fit.sum()>100 and not (held&fit).any(),
                'no valid specialist training split')
        yfit=(d['y'][fit]==1).astype(np.int8)
        held_idx=np.flatnonzero(held)
        for view,part in VIEWS.items():
            trainx=d['X'][fit,part]
            testx=d['X'][held,part]
            scale=StandardScaler().fit(trainx)
            xx=np.clip(scale.transform(trainx),-6,6)
            yy=np.clip(scale.transform(testx),-6,6)
            for algo in ALGOS:
                m=LogisticRegression(C=.1,
                     solver='liblinear',class_weight='balanced',
                     max_iter=500,random_state=27334) if algo=='logistic' else (
                     HistGradientBoostingClassifier(max_iter=150,
                      max_depth=4,max_leaf_nodes=25,min_samples_leaf=80,
                      l2_regularization=10,learning_rate=.05,
                      random_state=27334))
                with threadpool_limits(limits=2):
                    m.fit(xx,yfit)
                    p1=m.predict_proba(yy)[:,list(m.classes_).index(1)]
                est[f'{view}_{algo}'][held_idx]=p1.astype(np.float32)
                file=f'{piece}__{view}__{algo}.joblib'
                joblib.dump(dict(model=m,scaler=scale,piece=piece,
                    fold=f,view=view,algo=algo,training_ids=d['ids'][fit],
                    held_ids=d['ids'][held],truth_used_only_in_fit=True),
                    a.output/'models'/file,compress=3)
                trained.append(dict(file=file,piece=piece,fold=f,
                    train_rows=int(fit.sum()),train_K1=int((yfit==1).sum()),
                    train_K2=int((yfit==0).sum()),held_rows=int(held.sum()),
                    crosspiece_fit=True,
                    training_hash=hashlib.sha256(
                         d['ids'][fit].astype('<i8').tobytes()).hexdigest()))
        print(json.dumps(dict(piece=piece,models=len(trained),
             elapsed=round(time.monotonic()-time0,1))),flush=True)
    require(all(np.isfinite(sc[conflict]).all() for sc in est.values()),
        'missing binary class confidence for observed transitions')
    out={'series18_parent':parent.copy(),S29:original.copy(),S31:latest.copy()}
    for method,probs in est.items():
        for limit in LIMITS:
            name=f'series34__{method}__class_certainty_gt{limit:g}'
            out[name]=accept(parent,original,probs,limit)
    require(len(out)==27,'expected 24 learnable specialist gates plus 3 sources')
    audits={}
    for name,y in out.items():
        mask=y!=parent
        audits[name]=dict(metrics=metrics(d['y'],y),
            vs_S18=paired(d['y'],parent,y),
            vs_freeze=paired(d['y'],d['freeze'],y),
            corrected=int((mask&(y==d['y'])&(parent!=d['y'])).sum()),
            regressed=int((mask&(y!=d['y'])&(parent==d['y'])).sum()),
            neutral=int((mask&(y!=d['y'])&(parent!=d['y'])).sum()),
            per_true_K={str(k):dict(
              correct=int(((d['y']==k)&(y==d['y'])).sum()),
              fixed=int(((d['y']==k)&mask&(y==d['y'])&(parent!=d['y'])).sum()),
              broken=int(((d['y']==k)&mask&(y!=d['y'])&(parent==d['y'])).sum()))
               for k in range(7)},
            per_fold={str(f):dict(metrics=metrics(d['y'][d['fold']==f],
                y[d['fold']==f]),vs_S18=paired(d['y'][d['fold']==f],
                parent[d['fold']==f],y[d['fold']==f]))
                for f in FOLDS})
    winners=[s for s in out if s.startswith('series34__') and
        audits[s]['corrected']>0 and audits[s]['regressed']==0
        and audits[s]['metrics']['poly']['correct']>=2998]
    ranking=sorted(audits,key=lambda s:(audits[s]['metrics']['correct'],
          audits[s]['metrics']['poly']['correct']),reverse=True)
    ids=np.flatnonzero(changed)
    with (a.output/'all_26_K1_K2_specialist_audit.csv').open('w',newline='') as f:
        wr=csv.writer(f)
        wr.writerow(['native_id','piece','fold','true_K','S18_K','S29_K',
          'category','K1_K2_conflict','P_K1_audio_logistic',
          'P_K1_flow_logistic','P_K1_all335_logistic',
          'P_K1_audio_hgb','P_K1_flow_hgb','P_K1_all335_hgb',
          'promoted_by_K1_K2_policies'])
        for i in ids:
            promoted=[k for k,v in out.items()
                      if k.startswith('series34__') and v[i]==original[i]]
            wr.writerow([int(d['ids'][i]),str(d['pieces'][i]),
              int(d['fold'][i]),int(d['y'][i]),int(parent[i]),
              int(original[i]),'corrected' if original[i]==d['y'][i] else
              'regressed' if parent[i]==d['y'][i] else 'neutral',
              int(conflict[i]),*[round(float(est[k][i]),5) if
                 np.isfinite(est[k][i]) else '' for k in
                 ('audio58_logistic','flow245_logistic','all335_logistic',
                  'audio58_hgb','flow245_hgb','all335_hgb')],
              ';'.join(promoted)])
    meta=dict(status='completed',no_independent_unseen_music_validation=True,
        source_S29_run=37865173636,source_S31_run=37866124959,
        observed_failure_K1_K2_pattern=True,posthoc_hypothesis_on_exposed_cohort=True,
        labels_used_only_for_other_piece_fit_and_audit=True,
        K1_K2_specialist_models_saved=trained,model_count=len(trained),
        affected_K1_K2_transition_events=int(conflict.sum()),
        all_26_changes_preserved=True,policy_count=len(out),
        zero_loss_research_candidates=winners,
        best_global=ranking[0],audits=audits,
        elapsed_seconds=round(time.monotonic()-time0,2),
        production_promotion=False)
    (a.output/'report.json').write_text(json.dumps(meta,sort_keys=True,indent=2)+'\n')
    np.savez_compressed(a.output/'predictions.npz',
         global_index=d['ids'],true_K=d['y'],fold=d['fold'],
         variant_ids=np.asarray(list(out)),
         predictions=np.column_stack(list(out.values())))
    np.savez_compressed(a.output/'P_K1.npz',
         global_index=d['ids'],variant_ids=np.asarray(list(est)),
         probabilities=np.column_stack(list(est.values())))
    lines=['# Série34 — tête acoustique experte K1 ↔ K2',
       '', 'Apprentissage des classes K1/K2 sur autres morceaux ; aucune vérité du morceau évalué pendant la sélection.',
       'Même cohorte de développement exposée ; pas de validation externe ni promotion.',
       '', '| Variante | Exact global | Exact poly | Corrections vs S18 | Régressions vs S18 | Anciens 4 bloqués |',
       '|---|---:|---:|---:|---:|---:|']
    for name in ranking:
        m=audits[name];r=m['metrics']
        lines.append(f"| {name} | {r['exact']*100:.4f}% | "
             f"{r['poly']['exact']*100:.4f}% | {m['corrected']} | "
             f"{m['regressed']} | {4-m['regressed']} |")
    lines+=['',f'Zero old-correction losses research policies: {len(winners)}',
        f'Actual binary specialists trained: {len(trained)}',
        'See all 26 cases CSV, saved probabilities and per-K/fold paired audits.']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--features',type=Path)
    p.add_argument('--s18',type=Path)
    p.add_argument('--s29',type=Path)
    p.add_argument('--s31',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--self-test',action='store_true')
    x=p.parse_args()
    if x.self_test:selftest()
    else:
        require(x.features and x.s18 and x.s29 and
                x.s31 and x.output,'source evidence required')
        execute(x)
