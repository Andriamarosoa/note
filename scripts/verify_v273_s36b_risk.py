"""S36b independent replay of every piece-held acoustic risk model and gate.

No fitting or threshold optimization occurs in this verification.
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import joblib
from threadpoolctl import threadpool_limits
from scripts.train_v273_aba_recurrent import prepare
from scripts.train_v273_s36b_crosspiece_risk import (
    BASES,VIEWS,LAMS,CUTS,build,get,read)
from scripts.yourmt3_exactk_common import require,metrics

SAFE='series36b__series35__lambda4__threshold0__full_path__lambda2__threshold0.005'

def audit(a):
    require(not a.output.exists(),'never overwrite prior independent replay')
    d=prepare(a)
    source=read(a.s35)
    paths=read(a.routes)
    saved=read(a.input/'predictions.npz')
    report=json.loads((a.input/'report.json').read_text())
    for z in (source,paths,saved):
        require(np.array_equal(z['global_index'],d['ids']),
                'not same original 59309 events')
    require(np.array_equal(saved['true_K'],d['y']) and
            np.array_equal(saved['fold'],d['fold']),'K/fold mismatch')
    require(report['status']=='completed' and report['model_count']==38,
            'incomplete learned Q models')
    action_names=list(map(str,paths['action_names']))
    require(len(action_names)==19 and
            'H9' not in action_names,'forbidden H9 input')
    parent=get(source,'series18_parent')
    require(np.array_equal(parent,d['parent']) and
            metrics(d['y'],parent)['correct']==49178,
            'S18 has changed from preserved baseline')
    labels=list(map(str,saved['variant_ids']))
    require(len(labels)==39,'36 learned + 3 source predictions')
    originpaths=list(map(str,paths['variant_ids']))
    out={'series18_parent':parent.copy()}
    for base in BASES:out[base]=get(source,base).copy()
    for name in labels:
        if name.startswith('series36b__'):
            out[name]=parent.copy()
    sha={}
    fit_reports={}
    for entry in report['models']:
        piece=entry['piece']
        view=entry['view'] if 'view' in entry else entry['file'].split('__')[-1].split('.joblib')[0]
        f=int(entry['fold'])
        ix=np.flatnonzero(d['pieces']==piece)
        target=(d['fold']==f)&(d['pieces']!=piece)
        modelpath=a.input/'models'/entry['file']
        require(modelpath.exists(),'missing saved Q model '+entry['file'])
        ckpt=joblib.load(modelpath)
        require(ckpt['piece']==piece and ckpt['fold']==f and ckpt['view']==view,
                'wrong model provenance')
        train_native=ckpt['train_native_ids']
        # Deduplicated training list may have repeated native IDs but
        # every one belongs to a different piece in the SAME fold.
        pos={int(k):j for j,k in enumerate(d['ids'])}
        samples=np.asarray([pos[int(k)] for k in train_native],np.int64)
        require(np.all(target[samples]),'piece leaked to risk fit')
        require(hashlib.sha256(np.asarray(train_native,dtype='<i8').tobytes()).hexdigest()==entry['fit_sha256'],
                'risk model source identity hash drift')
        sha[entry['file']]=hashlib.sha256(modelpath.read_bytes()).hexdigest()
        fit_reports[entry['file']]=dict(held_piece=piece,
             training_pair_rows=len(samples),train_fold=f)
        for base in BASES:
            original=get(source,base)
            hold=ix[original[ix]!=parent[ix]]
            if not len(hold):continue
            path=paths['selected_head'][hold,originpaths.index(base)]
            x=build(d['X'][hold],parent[hold],original[hold],path,view)
            with threadpool_limits(limits=2):
                q=np.clip(ckpt['model'].predict(np.clip(
                    ckpt['scaler'].transform(x),-6,6)),0,1)
            for lam in LAMS:
                gain=q[:,0]-lam*q[:,1]
                for cut in CUTS:
                    name=f'series36b__{base}__{view}__lambda{lam:g}__threshold{cut:g}'
                    yes=hold[gain>cut]
                    out[name][yes]=original[yes]
    require(len(sha)==38,'not all learned models verified')
    require(set(labels)==set(out),'source policy inventory mismatch')
    for name in labels:
        ii=labels.index(name)
        require(np.array_equal(saved['predictions'][:,ii],out[name]),
                'non-reproducible event-level policy '+name)
        m=metrics(d['y'],out[name])
        assert report['audits'][name]['metrics']['correct']==m['correct']
    validated=out[SAFE]
    corrected=np.flatnonzero((parent!=d['y'])&(validated==d['y']))
    regressed=np.flatnonzero((parent==d['y'])&(validated!=d['y']))
    require(len(corrected)==2 and len(regressed)==0,'advertised 2/0 not reproduced')
    require(int(((validated==d['y'])&(d['y']>=2)).sum())==3000,
            'poly 3000 / 7385 accuracy not reproduced')
    # The three cutoffs have the same 2/0 COUNT, not necessarily the
    # same event IDs. Preserve each corrected set to avoid a false claim.
    threshold_cases={}
    for cut in CUTS:
        name=f'series36b__series35__lambda4__threshold0__full_path__lambda2__threshold{cut:g}'
        v=out[name]
        fixes=np.flatnonzero((parent!=d['y'])&(v==d['y']))
        losses=np.flatnonzero((parent==d['y'])&(v!=d['y']))
        require(len(fixes)==2 and len(losses)==0,
                'expected score count differs in cutoff '+str(cut))
        threshold_cases[str(cut)]=[int(d['ids'][j]) for j in fixes]
    a.output.mkdir(parents=True)
    with (a.output/'two_confirmed_corrections.csv').open('w',newline='') as f:
        wr=csv.writer(f)
        wr.writerow(['global_id','fold','music_piece','real_K','S18_K',
                     'S35_K','S36b_K','risk_model_held_piece'])
        ref=get(source,BASES[1])
        for i in corrected:
            wr.writerow([int(d['ids'][i]),int(d['fold'][i]),
                str(d['pieces'][i]),int(d['y'][i]),int(parent[i]),
                int(ref[i]),int(validated[i]),str(d['pieces'][i])])
    res=dict(status='independently_replayed',all_38_fits_loaded=True,
        same_fold_other_pieces_only=True,H9_excluded=True,
        all_39_policy_vectors_identical=True,
        no_old_corrections_lost_on_seen_data=True,
        confirmed_corrections=2,confirmed_regressions=0,
        correct_global=int((validated==d['y']).sum()),
        correct_poly=int(((validated==d['y'])&(d['y']>=2)).sum()),
        previously_exposed_development_cohort=True,
        independent_unseen_validation=False,
        no_production_promotion=True,
        two_native_corrected_ids=[int(d['ids'][i]) for i in corrected],
        corrected_ids_by_cutoff=threshold_cases,
        model_SHA256=sha,fit_provenance=fit_reports)
    (a.output/'report.json').write_text(json.dumps(res,indent=2,sort_keys=True)+'\n')
    (a.output/'report.md').write_text(
        '# S36b independently reproduced all 38 risk gates\n\n'
        'All 39 policy vectors match event-for-event. '
        'The research candidate is **2 fixes, 0 regressions**, '
        '3,000/7,385 correct poly events (40.6229%). '
        'S18 unchanged; H9 excluded. '
        'Previously used cohort, so no new-music validation or promotion.\n')
    print(json.dumps({k:v for k,v in res.items()
       if k not in ('model_SHA256','fit_provenance')},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--s18',type=Path,required=True)
    p.add_argument('--s35',type=Path,required=True)
    p.add_argument('--routes',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    audit(p.parse_args())
