"""S33 descriptive audit: uncover similarities/differences in S29 regression cases.

True K is used ONLY for auditing; no policy or future prediction derives from it.
All 26 changed predictions retained, including 5 neutrals.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from collections import Counter
import numpy as np
from scripts.train_v273_aba_recurrent import prepare
from scripts.yourmt3_exactk_common import require

GROUPS={'votes32':(0,32),'audio58':(32,90),'harmonic_flow245':(90,335),'full335':(0,335)}
KEY='series29__lambda2__threshold0.02'

def read(p):
    with np.load(p,allow_pickle=False) as f:
        return {k:f[k] for k in f.files}

def predictions(p,key):
    names=list(map(str,p['variant_ids']))
    require(key in names,'missing policy '+key)
    return p['predictions'][:,names.index(key)].astype(np.int8)

def audit(a):
    require(not a.output.exists(),'no overwriting prior research')
    d=prepare(a)
    s=read(a.s29)
    routes=read(a.routes)
    sc=read(a.s31)
    require(np.array_equal(s['global_index'],d['ids']) and
            np.array_equal(routes['global_index'],d['ids']) and
            np.array_equal(sc['global_index'],d['ids']),
            'all provenance IDs must match')
    parent=predictions(s,'series18_parent')
    prop=predictions(s,KEY)
    indices=np.flatnonzero(parent!=prop)
    require(len(indices)==26 and
        np.array_equal(parent,d['parent']),'S29 cases or parent changed')
    flags=np.where(prop[indices]==d['y'][indices],'corrected',
           np.where(parent[indices]==d['y'][indices],'regressed','neutral'))
    require(Counter(flags)==Counter(corrected=17,regressed=4,neutral=5),
            'all regression/neutral categories required')
    names=list(map(str,routes['variant_ids']))
    ri=names.index(KEY)
    rs=routes['actions'][:,ri,:]
    require(rs.shape==(59309,3),'3 stages expected')

    trust_names=list(map(str,sc['variant_ids']))
    require('all335_logistic' in trust_names and
            'flow245_logistic' in trust_names,'S31 risk score missing')
    fullscore=sc['probabilities'][:,trust_names.index('all335_logistic')]
    flowscore=sc['probabilities'][:,trust_names.index('flow245_logistic')]
    means=d['X'].mean(axis=0)
    std=d['X'].std(axis=0)
    zz=(d['X']-means)/np.maximum(std,1e-5)
    # Context features, descriptive normalization only (not deployed).
    category={}
    for label in ('corrected','regressed','neutral'):
        ix=indices[flags==label]
        category[label]=dict(count=len(ix),
          truth_K=Counter(map(int,d['y'][ix])),
          parent_to_proposed=Counter(f'{int(parent[j])}->{int(prop[j])}' for j in ix),
          folds=Counter(map(int,d['fold'][ix])),
          sources=dict(scores_all335_mean=float(np.mean(fullscore[ix])),
                       scores_flow245_mean=float(np.mean(flowscore[ix]))),
          feature_group={
            name:dict(median_normalized_l2=float(np.median(
                np.linalg.norm(zz[ix,lo:hi],axis=1))),
              mean_normalized_l2=float(np.mean(np.linalg.norm(
                zz[ix,lo:hi],axis=1))))
            for name,(lo,hi) in GROUPS.items()})
    near={}
    gains=indices[flags=='corrected']
    harmed=indices[flags=='regressed']
    for x in harmed:
        dist={}
        for name,(lo,hi) in GROUPS.items():
            xx=zz[gains,lo:hi]-zz[x,lo:hi]
            distances=np.sqrt(np.mean(xx*xx,axis=1))
            nearest=int(np.argmin(distances))
            t=int(gains[nearest])
            dist[name]=dict(nearest_corrected_native_id=int(d['ids'][t]),
                            nearest_corrected_true_K=int(d['y'][t]),
                            nearest_corrected_parent_K=int(parent[t]),
                            nearest_corrected_proposal_K=int(prop[t]),
                            standardized_rms_distance=float(distances[nearest]))
        near[str(int(d['ids'][x]))]=dist
    rows=[]
    for i,flag in zip(indices,flags):
        path=''.join('STOP' if v==0 else 'A' if v==1 else '(B-A)' for v in rs[i])
        record=dict(native_id=int(d['ids'][i]),fold=int(d['fold'][i]),
             piece=str(d['pieces'][i]),true_K=int(d['y'][i]),
             S18_K=int(parent[i]),S29_K=int(prop[i]),category=str(flag),
             route=path,score_all335=float(fullscore[i]),
             score_flow245=float(flowscore[i]),group_norms={
                name:float(np.linalg.norm(zz[i,lo:hi]))
                for name,(lo,hi) in GROUPS.items()})
        rows.append(record)
    result=dict(status='completed',audit_only_no_new_policy=True,
        independent_unseen_validation=False,
        labels_used_for_diagnostics_not_for_live_decisions=True,
        selection_cohort_previously_exposed=True,
        total_26_native_cases=True,
        counts={k:int(v) for k,v in Counter(flags).items()},
        categories={key:{
            **{k:v for k,v in val.items() if k not in ('truth_K','parent_to_proposed','folds')},
            'truth_K':dict(val['truth_K']),
            'parent_to_proposed':dict(val['parent_to_proposed']),
            'folds':dict(val['folds'])}
            for key,val in category.items()},
        nearest_corrected_event_per_regression=near,
        row_audit=rows,
        causal_proof=False,multiple_comparison_inference_not_allowed=True)
    a.output.mkdir(parents=True)
    (a.output/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    with (a.output/'all_26_state_flow_cases.csv').open('w',newline='') as f:
        wr=csv.writer(f)
        wr.writerow(['native_id','fold','piece','true_K','S18_K',
            'S29_K','category','route','all335_reliability','flow_reliability',
            'votes_l2','audio_l2','flow_l2'])
        for r in rows:
            wr.writerow([r['native_id'],r['fold'],r['piece'],r['true_K'],
                r['S18_K'],r['S29_K'],r['category'],r['route'],
                r['score_all335'],r['score_flow245'],
                r['group_norms']['votes32'],r['group_norms']['audio58'],
                r['group_norms']['harmonic_flow245']])
    lines=['# Série33 — inspection de la morphologie / flux de 4 régressions',
        '', 'Diagnostic seulement : aucun événement vrai K ne décide du résultat du modèle.',
        'Effectifs minuscules (4 erreurs) ; aucune différence décrite ci-dessous ne prouve une loi musicale.',
        '', '| Groupe | N | Véritables K | Transition parent→nouveau | Folds | Confiance acoustique moyenne |',
        '|---|---:|---|---|---|---:|']
    for k,v in category.items():
        lines.append(f"| {k} | {v['count']} | {dict(v['truth_K'])} | "
             f"{dict(v['parent_to_proposed'])} | {dict(v['folds'])} | "
             f"{v['sources']['scores_all335_mean']:.4f} |")
    lines+=['','## Quatre regressions exactes',
        '| ID natif | Fold | Pièce | Vrai K | S18 K | S29 K | Route | Score confiance |',
        '|---|---:|---|---:|---:|---:|---|---:|']
    for r in rows:
        if r['category']=='regressed':
            lines.append(f"| {r['native_id']} | {r['fold']} | {r['piece']} | "
              f"{r['true_K']} | {r['S18_K']} | {r['S29_K']} | {r['route']} | "
              f"{r['score_all335']:.4f} |")
    lines+=['','See report.json for corrected-event nearest neighbors separately by acoustic 58 and harmonic flow 245 and feature-group norms.',
        'Reference S18 unchanged. No promotion and no new-music validation.']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--features',type=Path)
    p.add_argument('--s18',type=Path)
    p.add_argument('--s29',type=Path)
    p.add_argument('--routes',type=Path)
    p.add_argument('--s31',type=Path)
    p.add_argument('--output',type=Path)
    x=p.parse_args()
    require(x.features and x.s18 and x.s29 and x.routes and
            x.s31 and x.output,'all producer sources required')
    audit(x)
