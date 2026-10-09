"""Independent numerical replay of series16's 108 K12/K23 decisions."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from scripts.yourmt3_exactk_common import FOLDS,metrics,paired,require
from scripts.loop_v273_k32_k34 import make_policies

WIN='series17__flow_logistic__k32__pbase_gt0.99'
def read(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def verify(a):
    require(not a.output.exists(),'no overwriting historical evidence')
    d=read(a.root/'analysis/evidence/v273-regression-loops/prepared/inputs.npz')
    ids,y,b,fold=[d[k] for k in (
        'native_global_index','native_truth','native_baseline','native_fold')]
    v=read(a.input/'predictions.npz')
    p=read(a.input/'probabilities.npz')
    s=json.loads((a.input/'report.json').read_text())
    for label,z in [('predictions',v),('probabilities',p)]:
        require(np.array_equal(z['global_index'],ids),label+' identity drift')
    require(np.array_equal(v['true_K'],y) and np.array_equal(v['fold'],fold),
            'native labels/fold alignment')
    require(len(ids)==59309 and set(fold.tolist())==set(FOLDS),
            'replayed cohort drift')
    policy_ids=list(map(str,v['variant_ids']))
    parent=v['predictions'][:,policy_ids.index('series16_parent')]
    heads=list(map(str,p['variant_ids']))
    require(len(heads)==12,'12 headed score inventory')
    probs={name:p['probabilities'][:,i] for i,name in enumerate(heads)}
    all_policies,mask=make_policies(b,parent,probs)
    require(len(all_policies)==110 and len(policy_ids)==110,'110 policy vectors')
    for key,vec in all_policies.items():
        require(key in policy_ids,key+' not in archived catalog')
        require(np.array_equal(vec,v['predictions'][:,policy_ids.index(key)]),
                key+' native replay mismatch')
    require(s['status']=='completed' and not s['independent_validation'],
            'safety report changed')
    folds_per_piece={}
    files={}
    for rec in s['fit_records']:
        if rec.get('status')!='trained':continue
        require(rec['piece'] not in rec['fit_pieces'],'piece training leakage')
        if rec['piece'] in folds_per_piece:
            require(folds_per_piece[rec['piece']]==rec['fold'],'piece fold drift')
        else:folds_per_piece[rec['piece']]=rec['fold']
        model=a.input/'models'/rec['file']
        require(model.is_file(),'fitted model missing: '+rec['file'])
        with model.open('rb') as fd:
            h=hashlib.sha256()
            for chunk in iter(lambda:fd.read(1<<20),b''):
                h.update(chunk)
        files[rec['file']]=h.hexdigest()
    candidate=all_policies[WIN]
    m=metrics(y,candidate)
    delta=paired(y,parent,candidate)
    freeze=paired(y,b,candidate)
    require(m['correct']==49174 and m['poly']['correct']==2994,
            'one native correction did not reproduce')
    require(delta['global']['corrections']==1 and delta['global']['regressions']==0,
            'no-loss claim was contradicted')
    require(freeze['global']['corrections']==2934 and
        freeze['global']['regressions']==2214,'historical corrections not preserved')
    rows=[]
    for i in np.flatnonzero(candidate!=parent):
        result=('correction' if candidate[i]==y[i] and parent[i]!=y[i]
                else 'regression' if candidate[i]!=y[i] and parent[i]==y[i]
                else 'neutral')
        rows.append(dict(global_id=int(ids[i]),fold=int(fold[i]),
            true_k=int(y[i]),freeze_k=int(b[i]),series15_k=int(parent[i]),
            series16_k=int(candidate[i]),prob_freeze=float(
              probs['k32__flow_logistic'][i]),outcome=result))
    outcome_counts={k:sum(r['outcome']==k for r in rows)
        for k in ('correction','regression','neutral')}
    require(outcome_counts['correction']==1 and outcome_counts['regression']==0,
            'one correct and zero regressed cases did not reproduce')
    report=dict(status='replayed',independent_validation=False,
        already_exposed_data=True,posthoc_selection=True,
        promotion=False,all_108_policies_bitwise_equal=True,
        model_sha256=files,candidate=WIN,
        metrics=m,versus_series15=delta,versus_freeze=freeze,
        by_fold={str(f):dict(
            vs_s15=paired(y[fold==f],parent[fold==f],candidate[fold==f]),
            metrics=metrics(y[fold==f],candidate[fold==f])) for f in FOLDS},
        changed_rows=rows,changed_outcome_counts=outcome_counts)
    a.output.mkdir(parents=True)
    (a.output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    with (a.output/'all-changed-decisions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# S17 independent arithmetic replay of development decisions','',
        'All 108 policies exactly reproduced from probabilities, with no truth in the decoder.',
        'This is not independent validation on new music.','',
        '| Candidate | Global correct | Poly correct | New corrections | New regressions | Remaining regressions |',
        '|---|---:|---:|---:|---:|---:|',
        f"| {WIN} | {m['correct']} | {m['poly']['correct']} | "
        f"{delta['global']['corrections']} | {delta['global']['regressions']} | "
        f"{freeze['global']['regressions']} |",'',
        f'Changed decision breakdown: {outcome_counts}', '',
        '| Native ID | Fold | Truth | Freeze | S16 | S17 | P(freeze) | Outcome |',
        '|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['global_id']} | {r['fold']} | {r['true_k']} | "
            f"{r['freeze_k']} | {r['series15_k']} | {r['series16_k']} | "
            f"{r['prob_freeze']:.6f} | {r['outcome']} |")
    lines+=['',f'Fitted models SHA256 verified: {len(files)}',
        'Not production promotion, not validation on unseen music.']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    verify(p.parse_args())
