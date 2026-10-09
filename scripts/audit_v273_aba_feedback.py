"""S20 audit: quantify where B feedback helps or hurts, by every K and fold."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.yourmt3_exactk_common import FOLDS, metrics, paired, require


PAIRS=[
    ('series20__pass4__margin0.6','series20__no_B_pass4__margin0.6'),
    ('series20__pass4__raw','series20__no_B_pass4__raw'),
    ('series20__pass2__raw','series20__pass1__raw'),
    ('series20__pass4__raw','series20__pass2__raw'),
]


def read(p):
    with np.load(p,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def audit(a):
    require(not a.output.exists(),'cannot overwrite audit evidence')
    predictions=read(a.input/'predictions.npz')
    probabilities=read(a.input/'probabilities.npz')
    ids=predictions['global_index']
    y=predictions['true_K']
    fold=predictions['fold']
    n=len(ids)
    require(n==59309 and set(fold.tolist())==set(FOLDS),
            'native dataset/fold changed')
    require(np.array_equal(ids,probabilities['global_index']),
            'S20 feedback probabilities misaligned')
    variants=list(map(str,predictions['variant_ids']))
    def policy(name):
        require(name in variants,'missing policy '+name)
        return predictions['predictions'][:,variants.index(name)]
    parent=policy('series18_parent')
    noB=policy('series20__no_B_pass4__margin0.6')
    withB=policy('series20__pass4__margin0.6')
    require(metrics(y,parent)['correct']==49178 and
            metrics(y,noB)['correct']==49188 and
            metrics(y,withB)['correct']==49190,
            'published conditional test score mismatch')
    A=np.asarray(probabilities['A_probs'],np.float32)
    B=np.asarray(probabilities['B_compatibility'],np.float32)
    require(A.shape==(n,5,7) and B.shape==(n,4,7),
            'source communication matrix changed')
    explain={}
    for live,control in PAIRS:
        p=policy(live);q=policy(control)
        changed=p!=q
        correct=(p==y)&(q!=y)
        regressed=(p!=y)&(q==y)
        neutral=changed&~correct&~regressed
        matrix=np.zeros((7,7),np.int64)
        np.add.at(matrix,(q[changed],p[changed]),1)
        by_K={}
        for k in range(7):
            at=y==k
            by_K[str(k)]=dict(total=int(at.sum()),changed=int((at&changed).sum()),
                helped=int((at&correct).sum()),harmed=int((at&regressed).sum()),
                neutral=int((at&neutral).sum()),
                net=int((at&correct).sum()-(at&regressed).sum()))
        by_fold={}
        for f in FOLDS:
            at=fold==f
            by_fold[str(f)]=dict(changed=int((at&changed).sum()),
                helped=int((at&correct).sum()),harmed=int((at&regressed).sum()),
                neutral=int((at&neutral).sum()))
        explain[f'{live}__versus__{control}']=dict(
            paired=paired(y,q,p),by_true_K=by_K,by_fold=by_fold,
            changed=int(changed.sum()),neutral=int(neutral.sum()),
            transitions_from_control_to_B=matrix.tolist())
    # Interpret messages only descriptively. The saved compatibility B[t,k] is
    # not a known ground-truth musical rule or a calibrated probability.
    mask=withB!=noB
    correct_B=(withB==y)&(noB!=y)
    harmed_B=(withB!=y)&(noB==y)
    compatibility=[]
    for typ,m in [('helped',correct_B),('harmed',harmed_B),
                  ('neutral',mask&~correct_B&~harmed_B)]:
        inds=np.flatnonzero(m)
        compatibility.append(dict(type=typ,count=len(inds),
            B_at_true_K_mean=float(B[inds,0,y[inds]].mean()) if len(inds) else None,
            B_at_parent_K_mean=float(B[inds,0,parent[inds]].mean()) if len(inds) else None,
            B_max_mean=float(B[inds,0].max(1).mean()) if len(inds) else None))
    rows=[]
    for i in np.flatnonzero(mask):
        effect='helped' if correct_B[i] else 'harmed' if harmed_B[i] else 'neutral'
        rows.append(dict(native_id=int(ids[i]),fold=int(fold[i]),
            true_K=int(y[i]),series18_K=int(parent[i]),
            predicted_without_B=int(noB[i]),predicted_with_B=int(withB[i]),
            outcome=effect,
            B1_at_true_K=float(B[i,0,y[i]]),
            B1_at_S18_K=float(B[i,0,parent[i]]),
            A1_prediction=int(A[i,0].argmax()),
            A2_prediction=int(A[i,1].argmax()),
            A3_prediction=int(A[i,2].argmax()),
            A4_prediction=int(A[i,3].argmax())))
    a.output.mkdir(parents=True)
    with (a.output/'all-B-vs-noB-changed-cases.csv').open('w',newline='') as fd:
        fieldnames=list(rows[0]) if rows else ['native_id']
        w=csv.DictWriter(fd,fieldnames=fieldnames);w.writeheader();w.writerows(rows)
    result=dict(status='audited',training_is_recurrent=True,
        independent_validation=False,cohort_previously_exposed=True,
        not_causal_evidence_of_musical_truth=True,
        sources=dict(original_training_run=37858383973),
        pairwise=explain,B_score_groups=compatibility,
        per_event_changes_csv='all-B-vs-noB-changed-cases.csv')
    (a.output/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    key=f'{PAIRS[0][0]}__versus__{PAIRS[0][1]}'
    d=explain[key]
    lines=['# Série20 — audit concret de B face à un contrôle sans B','',
        'Le contrôle partage le même réseau entraîné : seul le message B est neutralisé.',
        'Cohorte de développement déjà exposée, ce n’est pas une validation indépendante.',
        '', '| Vrai K | Effet positif de B | Régressions dues à B | Net | Neutres |',
        '|---:|---:|---:|---:|---:|']
    for k,inf in d['by_true_K'].items():
        lines.append(f"| {k} | {inf['helped']} | {inf['harmed']} | "
            f"{inf['net']:+d} | {inf['neutral']} |")
    p=d['paired']['global']
    lines+=['',f"**Total** : B corrige {p['corrections']} erreurs du contrôle mais "
        f"casse {p['regressions']} décisions correctes (net {p['net']:+d}).",
        f'**Décisions changées** : {d["changed"]}; neutres : {d["neutral"]}.',
        '','## Audit par fold','',
        '| Fold | B corrige | B casse | Neutres |',
        '|---:|---:|---:|---:|']
    for k,inf in d['by_fold'].items():
        lines.append(f"| {k} | {inf['helped']} | {inf['harmed']} | {inf['neutral']} |")
    lines+=['','## Messagerie B, signatures observées',
        json.dumps(compatibility,sort_keys=True),
        '','Toutes les décisions distinctes sont dans all-B-vs-noB-changed-cases.csv.',
        'Ne pas promotionner un seuil sur la seule base de ces observations.']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    audit(p.parse_args())
