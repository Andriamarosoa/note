"""S32: 28 fixed combinations of cross-piece S30/S31 reliability scores."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from scripts.yourmt3_exactk_common import FOLDS,metrics,paired,require

K='series29__lambda2__threshold0.02'
THRESH=(.25,.40,.55,.70)
TYPES=('all335_and_flow245','all335_and_audio_flow303',
    'all335_and_prior90','flow245_and_prior90',
    'at_least_two_of_three','all_three',
    'all335_and_prior90_and_audio303')

def load(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}

def select(z,k):
    ids=list(map(str,z['variant_ids']))
    require(k in ids,'missing score '+k)
    return z['probabilities'][:,ids.index(k)]

def decision(z,k):
    ids=list(map(str,z['variant_ids']))
    require(k in ids,'missing decision '+k)
    return z['predictions'][:,ids.index(k)]

def patterns(a,f,u,b,cut):
    yes=dict(all335=a>cut,flow245=f>cut,
             prior90=u>cut,audio303=b>cut)
    return {
        'all335_and_flow245':yes['all335']&yes['flow245'],
        'all335_and_audio_flow303':yes['all335']&yes['audio303'],
        'all335_and_prior90':yes['all335']&yes['prior90'],
        'flow245_and_prior90':yes['flow245']&yes['prior90'],
        'at_least_two_of_three':
           (yes['all335'].astype(np.int8)+yes['flow245'].astype(np.int8)+
            yes['prior90'].astype(np.int8))>=2,
        'all_three':yes['all335']&yes['flow245']&yes['prior90'],
        'all335_and_prior90_and_audio303':
            yes['all335']&yes['prior90']&yes['audio303']
    }

def selftest():
    a=np.array([.9,.6,.1]);f=np.array([.8,.2,.9])
    u=np.array([.4,.7,.1]);b=np.array([.6,.2,.4])
    z=patterns(a,f,u,b,.5)
    assert len(z)==7
    assert z['all335_and_flow245'].tolist()==[True,False,False]
    assert z['at_least_two_of_three'].tolist()==[True,True,False]
    assert z['all_three'].sum()==0
    print('PASS: 7 predeclared proof coalitions × 4 thresholds, no test truth')

def run(args):
    require(not args.output.exists(),'no overwrite')
    orig=load(args.s29)
    src30=load(args.s30)
    src31=load(args.s31)
    score30=load(args.gate30)
    score31=load(args.gate31)
    for z in (src30,src31,score30,score31):
        require(np.array_equal(z['global_index'],orig['global_index']),
            'event ID misaligned across source artifacts')
    truth=orig['true_K'];fold=orig['fold'];ids=orig['global_index']
    require(len(truth)==59309 and set(fold.tolist())==set(FOLDS),
        '59k native event cohort changed')
    parent=decision(orig,'series18_parent')
    s29=decision(orig,K)
    require(np.array_equal(parent,decision(src30,'series18_parent')) and
        np.array_equal(parent,decision(src31,'series18_parent')),
        'S18 baseline changed across gate fits')
    assert np.array_equal(s29,decision(src30,K)) and np.array_equal(s29,decision(src31,K))
    changed=np.flatnonzero(s29!=parent)
    require(len(changed)==26 and
        ((s29==truth)&(parent!=truth)).sum()==17 and
        ((s29!=truth)&(parent==truth)).sum()==4,'26 native decision cases drifted')
    a=select(score31,'all335_logistic')
    f=select(score31,'flow245_logistic')
    u=score30['quality_logistic']
    b=select(score31,'audio_plus_flow303_logistic')
    for q in (a,f,u,b):
        require(np.isfinite(q[changed]).all(),'not all held examples received trust scores')
    results={'series18_parent':parent.copy(),
             K:s29.copy(),
             'series31__all335_logistic__p_gt0.25':
                  decision(src31,'series31__all335_logistic__p_gt0.25').copy()}
    for t in THRESH:
        xx=patterns(a,f,u,b,t)
        require(set(xx)==set(TYPES),'pattern dictionary mismatch')
        for name,mask in xx.items():
            results[f'series32__{name}__p_gt{t:g}'] = np.where(
                mask,s29,parent).astype(np.int8)
    require(len(results)==31,'28 policies and three references required')
    audits={}
    for key,pred in results.items():
        mask=(pred!=parent)
        audits[key]=dict(metrics=metrics(truth,pred),
            vs_parent=paired(truth,parent,pred),
            corrected=int((mask&(pred==truth)&(parent!=truth)).sum()),
            regressed=int((mask&(pred!=truth)&(parent==truth)).sum()),
            neutral=int((mask&(pred!=truth)&(parent!=truth)).sum()),
            per_true_K={str(k):dict(
                corrected=int(((truth==k)&mask&(pred==truth)&(parent!=truth)).sum()),
                regressed=int(((truth==k)&mask&(pred!=truth)&(parent==truth)).sum()))
                for k in range(7)},
            per_fold={str(f):dict(metrics=metrics(truth[fold==f],pred[fold==f]),
               versus_S18=paired(truth[fold==f],parent[fold==f],pred[fold==f]))
               for f in FOLDS})
    candidates=[key for key in results if key.startswith('series32__')
                and audits[key]['corrected']>0 and audits[key]['regressed']==0 and
                audits[key]['metrics']['poly']['correct']>=2998]
    ranked=sorted(audits,key=lambda key:(
        audits[key]['metrics']['correct'],
        audits[key]['metrics']['poly']['correct']),reverse=True)
    args.output.mkdir(parents=True)
    report=dict(status='completed',study='28 fixed trust coalitions',
        original_26_events=True,source_models_trained_out_of_piece=True,
        observational_development_cohort_already_used=True,
        no_independent_unseen_music_validation=True,
        no_promotion=True,quality_sources=['S30 logistic 90',
            'S31 logistic flow245','S31 logistic all335',
            'S31 logistic audio+flow303'],
        zero_loss_research_candidates=candidates,
        best_global=ranked[0],audits=audits)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    np.savez_compressed(args.output/'predictions.npz',
        global_index=ids,true_K=truth,fold=fold,
        variant_ids=np.asarray(list(results)),
        predictions=np.column_stack(list(results.values())))
    with (args.output/'all_26_coalition_cases.csv').open('w',newline='') as handle:
        wr=csv.writer(handle)
        wr.writerow(['native_id','fold','true_K','parent_K','S29_candidate_K',
          'category','S30_prior90','S31_flow245','S31_all335','S31_audio303',
          'accepted_policies'])
        for i in changed:
            keep=[name for name,vec in results.items()
                  if name.startswith('series32__') and vec[i]==s29[i]]
            wr.writerow([int(ids[i]),int(fold[i]),int(truth[i]),int(parent[i]),int(s29[i]),
              'corrected' if s29[i]==truth[i] else
              'regressed' if parent[i]==truth[i] else 'neutral',
              *[round(float(q[i]),5) for q in (u,f,a,b)],';'.join(keep)])
    lines=['# Série32 — confirmation de preuves audio, harmoniques et ancienne porte','',
      'Development cohort already exposed, all 26 S29 changed events retained.',
      'Crosspiece source trust models, no truth in runtime gate. No promotion.',
      '', '| Policy | Exact global | Exact poly | Corrected | Regressed | Original four blocked |',
      '|---|---:|---:|---:|---:|---:|']
    for key in ranked:
        v=audits[key];m=v['metrics']
        lines.append(f"| {key} | {m['exact']*100:.4f}% | "
          f"{m['poly']['exact']*100:.4f}% | "
          f"{v['corrected']} | {v['regressed']} | {4-v['regressed']} |")
    lines+=['',f'Positive zero-regression research cases: {len(candidates)}',
      'Any positive result is hypothesis generation; must test unseen held compositions.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--s29',type=Path)
    p.add_argument('--s30',type=Path)
    p.add_argument('--s31',type=Path)
    p.add_argument('--gate30',type=Path)
    p.add_argument('--gate31',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        require(a.s29 and a.s30 and a.s31 and a.gate30 and a.gate31 and a.output,
                'pretrained held-piece source gates required')
        run(a)
