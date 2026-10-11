"""S45 independent replay of every family-specific piece-held risk head.

The source labels are used only for audit counters, never for source choice.
Checks all preserved model fit IDs and replays every S45 K0..K6 decision.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from scripts.train_v273_aba_recurrent import prepare
from scripts.train_v273_s45_specialized_risk_heads import make_inputs,choose_best,source_data,LAMS,CUTS,sha
from scripts.yourmt3_exactk_common import metrics,require

def run(a):
    require(not a.output.exists(),'no overwrite of prior replay')
    d=prepare(a)
    archive=source_data(a.input/'predictions.npz')
    riskdata=source_data(a.input/'risk_scores.npz')
    selected=source_data(a.input/'selected_family.npz')
    meta=json.loads((a.input/'report.json').read_text())
    n=len(d['ids']);old=d['parent']
    require(n==59309 and metrics(d['y'],old)['correct']==49178,
            'source S18 drift')
    for z in (archive,riskdata,selected):
        require(np.array_equal(z['global_index'],d['ids']),
                'source S45 ID mismatch')
    families=list(map(str,riskdata['family_ids']))
    require(len(families)==16 and
            not any('h9' in s.lower() or 'yourmt3' in s.lower() for s in families),
            'missing expert family or forbidden H9')
    proposal=riskdata['proposal_K']
    original=riskdata['risks']
    assert proposal.shape==(n,16) and original.shape==(n,16,2)
    agreements=riskdata['independent_votes']
    require(agreements.shape==(n,16),'corroboration labels misaligned')
    rerun=np.full_like(original,np.nan)
    checks=[]
    for entry in meta['model_provenance']:
        f=entry['family'];i=families.index(f)
        piece=entry['piece'];fold=int(entry['fold'])
        assert entry['file'].startswith('models/')
        p=a.input/entry['file']
        require(p.exists(),'missing individual risk model')
        z=joblib.load(p)
        require(z['family']==f and z['piece']==piece and z['fold']==fold,
                'model/fold/member differs from audit')
        expected_train=np.flatnonzero((d['fold']==fold)&
            (d['pieces']!=piece)&(proposal[:,i]!=old))
        expected_hold=np.flatnonzero((d['pieces']==piece)&(proposal[:,i]!=old))
        require(np.array_equal(z['train_native_ids'],d['ids'][expected_train]),
                'a source regression model saw held music piece labels')
        require(np.array_equal(z['held_native_ids'],d['ids'][expected_hold]),
                'held piece risk score provenance differs')
        require(entry['train_ids_SHA256']==sha(d['ids'][expected_train]),
                'train label provenance hash mismatch')
        if len(expected_hold):
            inp=make_inputs(d['X'][expected_hold],old[expected_hold],
                 proposal[expected_hold,i],agreements[expected_hold,i])
            with threadpool_limits(limits=2):
                score=z['model'].predict(np.clip(z['scaler'].transform(inp),-6,6))
            rerun[expected_hold,i]=np.clip(score,0,1)
        checks.append(entry['file'])
    require(len(checks)==16*19,'all original piece-held risk weights missing')
    changed=proposal!=old[:,None]
    require(np.isfinite(rerun[changed]).all(),
            'some changed source event was never verified')
    require(np.allclose(original[changed],rerun[changed],atol=1e-5),
            'reloaded family predictors do not match stored native risk')
    names=list(map(str,archive['variant_ids']))
    require(len(names)==1+16*12+12,'missing predeclared family policies')
    require(names[0]=='series18_parent' and
            np.array_equal(archive['predictions'][:,0],old),
            'unmodified S18 not preserved')
    for i,fam in enumerate(families):
        for lam in LAMS:
            utility=rerun[:,i,0]-lam*rerun[:,i,1]
            for cut in CUTS:
                k=f'series45__{fam}__lambda{lam:g}__cut{cut:g}'
                require(k in names,'missing per-head policy '+k)
                p=np.where(changed[:,i]&np.isfinite(utility)&(utility>cut),
                    proposal[:,i],old)
                require(np.array_equal(p,archive['predictions'][:,names.index(k)]),
                        'family-specific decision not reproduced: '+k)
    picked=list(map(str,selected['variant_ids']))
    require(len(picked)==12,'missing dynamic cross-family 12 comparisons')
    for lam in LAMS:
        for cut in CUTS:
            k=f'series45__ALLfamilies__lambda{lam:g}__cut{cut:g}'
            pp,j=choose_best(old,proposal,rerun,lam,cut)
            require(np.array_equal(pp,archive['predictions'][:,names.index(k)]),
                    'globally selected risk head not reproducible '+k)
            require(np.array_equal(j,selected['selected_family'][:,picked.index(k)]),
                    'globally selected family index differs '+k)
    for idx,k in enumerate(names):
        p=archive['predictions'][:,idx]
        fixed=int(((old!=d['y'])&(p==d['y'])).sum())
        broken=int(((old==d['y'])&(p!=d['y'])).sum())
        require(meta['audits'][k]['corrected']==fixed and
                meta['audits'][k]['regressed']==broken,
                'correction/regression count changed '+k)
    a.output.mkdir(parents=True)
    result=dict(status='independently_verified',all_304_piece_held_models_loaded=True,
        all_204_policies_reproduced=True,
        model_fit_excludes_evaluated_piece=True,H9_excluded=True,
        original_source_model_selection_posthoc=True,
        not_independent_fresh_music_validation=True,
        no_production_promotion=True,models_checked=len(checks))
    (a.output/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('PASS: 304 independent family-specific model forwards, 204 decisions and all K corrections reproduced')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--s18',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
