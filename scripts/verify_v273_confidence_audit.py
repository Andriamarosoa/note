"""Verify frozen replay archives and quantify confidence/generalization gaps."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.special import expit,logsumexp
from scripts.v273_confidence_metrics import summarize
from scripts.verify_v273_selector_repair_artifacts import digest,require,named_pairs

FOLDS={0,1,2,4}


def read(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def combine(items):
    return {k:np.concatenate([x[k] for x in items]) for k in items[0]}


def check_summary(a,b,path=''):
    require(type(a)==type(b) or isinstance(a,(int,float)) and isinstance(b,(int,float)), 'summary type '+path)
    if isinstance(a,dict):
        require(set(a)==set(b),'summary fields '+path)
        for k in a:check_summary(a[k],b[k],path+'/'+k)
    elif isinstance(a,list):
        require(len(a)==len(b),'summary length')
        for i,(x,y) in enumerate(zip(a,b)):check_summary(x,y,path+str(i))
    elif isinstance(a,(int,float)):
        require(np.isclose(a,b,rtol=1e-8,atol=1e-7),'summary numeric '+path)
    else:require(a==b,'summary value '+path)


def verify_snapshot(d):
    y,b=d['truth'],d['baseline'];n=len(y);row=np.arange(n);p=d['proposal']
    require(np.array_equal(p[:,0],b),'frozen action proposal')
    require(np.isin(p,[2,3,4,5,6]).all(),'proposal scope')
    available=np.stack([np.any(p==k,axis=1)&(b!=k) for k in range(7)],axis=1)
    logits=np.concatenate([np.where(available,d['pooled_class_logits'],-1e9),np.zeros((n,1))],1).astype(float)
    logq=logits-logsumexp(logits,axis=1,keepdims=True);q=np.exp(logq)
    z=d['baseline_logits'].astype(float);r=expit(z)
    expected=(1-r[:,None])*q[:,:7]+np.eye(7)[b]*r[:,None]
    require(np.allclose(r,d['baseline_correct'],atol=1e-6),'baseline sigmoid')
    require(np.allclose(expected,d['class_probability'],atol=1e-6),'categorical probability reconstruction')
    require(np.allclose((1-r)*q[:,7],d['other_probability'],atol=1e-6),'OTHER reconstruction')
    require(np.allclose(d['class_probability'].sum(1)+d['other_probability'],1,atol=1e-6),'joint simplex')
    target=np.where(available[row,y],y,7);B=y==b
    joint_log_probability=np.where(B,-np.logaddexp(0,-z),-np.logaddexp(0,z)+logq[row,target])
    s=summarize(d)
    require(np.isclose(s['nll']['event'],-joint_log_probability.mean(),atol=1e-10),'proper joint likelihood identity')
    # Decode directly from saved class probabilities, independent of pooled-logit reconstruction.
    cp=d['class_probability'].astype(float);rr=d['baseline_correct'].astype(float)
    scores=np.where(available,cp-rr[:,None],-np.inf);best=scores.argmax(1)
    take=scores[row,best]>0;prediction=np.where(take,best,b)
    pairs=named_pairs(y,b,prediction)['global']
    require(pairs['corrections']==s['corrections'] and pairs['regressions']==s['regressions'] and pairs['net']==s['net'],'independent class decoder')
    require(np.isclose(sum(s['selected_optimism_decomposition'][k] for k in ['baseline_component','conditional_component']),s['optimism'],atol=1e-5),'exact optimism decomposition')
    return s,prediction,available


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--original',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();manifest=json.loads((args.artifacts/'artifacts.json').read_text())
    original_proof=json.loads((Path(__file__).resolve().parents[1]/'analysis/evidence/v273-learned-selection/verification.json').read_text())
    evidence=dict(status='verified',run_id=manifest['run_id'],source_commit=manifest['source_commit'],
        trained_source_run=37789646292,trained_source_commit=original_proof['source_commit'],artifact_manifest=manifest,
        optimizer_used=False,weights_changed=False,independent_validation=False,promotion=False,arms={})
    table=[]
    for arm in ['control7','corrector8']:
        directory=args.artifacts/arm;item=manifest['arms'][arm]
        require(digest(args.artifacts/(arm+'.zip'))==item['sha256'],'replay archive SHA')
        original=args.original/arm;proof=original_proof['arms'][arm]
        require(digest(original/'predictions.npz')==proof['predictions_sha256'],'trained prediction SHA')
        archived=read(original/'predictions.npz');ids=archived['eligible_global_index'];lookup={v:i for i,v in enumerate(archived['global_index'])}
        pos=np.array([lookup[v] for v in ids]);y,b,f=(archived[k][pos] for k in ['true_K','frozen_baseline_K','fold'])
        elook={v:i for i,v in enumerate(ids)}
        report=json.loads((directory/'report.json').read_text());require(report['status']=='completed' and report['arm']==arm,'completed replay report')
        require(not report['optimizer_used'] and not report['weights_changed'],'frozen replay declarations')
        require(report['source_predictions_sha256']==proof['predictions_sha256'],'replayed model identity')
        groups={'train':[],'held':[],'votes2_audits3':[],'votes3_audits2':[],'votes2_audits2':[]}
        actual_common_inputs=True;actual_replay_identical=True;fold_summary={};checked=0
        for outer in sorted(FOLDS):
            fold_report=report['folds'][str(outer)]
            require(fold_report['weights_sha256']==proof['weights_sha256'][str(outer)],'restored weight identity')
            for role in ['train','held']:
                path=directory/f'fold-{outer}-{role}.npz';info=report['files'][path.name]
                require(digest(path)==info['sha256'],'snapshot SHA')
                d=read(path);take=f!=outer if role=='train' else f==outer
                require(np.array_equal(d['global_index'],ids[take]),'split IDs')
                for key,actual in [('truth',y[take]),('baseline',b[take]),('fold',f[take])]:
                    require(np.array_equal(d[key],actual),'split labels')
                s,pred,av=verify_snapshot(d);check_summary(s,fold_report[role]);checked+=1
                groups[role].append(d)
                if role=='held':
                    require(np.array_equal(pred,archived['predicted_K'][pos][take]),'all held decisions replay')
                    require(np.array_equal(d['proposal'],archived['group_proposal'][take]),'held proposals replay')
                    for k,ak in [('baseline_correct','baseline_correct_probability'),('pooled_class_logits','pooled_class_logits'),
                                 ('class_probability','class_probability'),('other_probability','other_probability')]:
                        actual_replay_identical &= bool(np.array_equal(d[k],archived[ak][take]))
                        require(np.allclose(d[k],archived[ak][take],atol=2e-5),'held output replay tolerance')
            fold_summary[str(outer)]={k:fold_report[k] for k in ['train_profile','held_profile']}
            fold_summary[str(outer)]['loss']={role:fold_report[role]['nll'] for role in ['train','held']}
            for diag in fold_report['regime_diagnostics']:
                regime=diag['regime'];omitted=diag['omitted_fold'];permitted=FOLDS-{outer,omitted}
                require(omitted in FOLDS-{outer},'valid omission')
                expected_votes=permitted if regime.startswith('votes2') else FOLDS-{outer}
                expected_refs=permitted if regime.endswith('audits2') else FOLDS-{outer}
                require(set(diag['vote_fit_folds'])==expected_votes and set(diag['audit_reference_folds'])==expected_refs,'regime exclusions')
                path=directory/f'fold-{outer}-{regime}-omit{omitted}.npz'
                require(digest(path)==report['files'][path.name]['sha256'],'regime snapshot SHA')
                d=read(path);take=f==outer
                require(np.array_equal(d['global_index'],ids[take]),'regime receiver IDs')
                require(np.array_equal(d['truth'],y[take]) and np.array_equal(d['baseline'],b[take]),'regime labels')
                s,_,_=verify_snapshot(d);check_summary(s,diag['summary']);checked+=1
                groups[regime].append(d)
            require(len(fold_report['regime_diagnostics'])==9,'all omissions and regimes included')
        require(checked==44 and len(report['files'])==44,'complete frozen replay inventory')
        merged={k:combine(v) for k,v in groups.items()}
        aggregate={k:summarize(v) for k,v in merged.items()}
        for name,d in merged.items():
            unique,cnt=np.unique(d['global_index'],return_counts=True)
            require(np.array_equal(unique,np.sort(ids)) and np.all(cnt==(1 if name=='held' else 3)),'event multiplicity')
            aggregate[name]['evaluations_per_native_eligible_event']=1 if name=='held' else 3
        # Same events and identical available destination sets across all four models.
        hd=merged['held'];td=merged['train'];hlook={v:i for i,v in enumerate(hd['global_index'])}
        def codes(d):
            return sum(((d['proposal']==k).any(1)&(d['baseline']!=k)).astype(np.int32)*(1<<k) for k in range(7))
        hc,tc=codes(hd),codes(td);same=np.ones(len(hd['truth']),bool)
        for gid,code in zip(td['global_index'],tc):same[hlook[gid]] &= code==hc[hlook[gid]]
        train_same=np.array([same[hlook[gid]] for gid in td['global_index']])
        common=dict(unique_events=int(same.sum()),definition='Identical available K sets in held model and all three models trained on that event.')
        if same.any():
            common['train']=summarize({k:v[train_same] for k,v in td.items()})
            common['held']=summarize({k:v[same] for k,v in hd.items()})
        entry=dict(archive_sha256=item['sha256'],report_sha256=digest(directory/'report.json'),
            held_outputs_bit_identical=actual_replay_identical,snapshots_verified=checked,
            aggregate=aggregate,common_available_destination_sets=common,folds=fold_summary)
        evidence['arms'][arm]=entry
        for role,s in aggregate.items():
            accepted=s['policies']['accepted_best']
            table.append(dict(arm=arm,regime=role,evaluations=s['rows'],changes=s['changes'],
                event_nll=s['nll']['event'],baseline_nll=s['nll']['baseline'],conditional_nll=s['nll']['conditional_on_baseline_wrong'],
                announced_net_per_100_changes=100*accepted['predicted_mean_gain'],
                observed_net_per_100_changes=100*accepted['observed_mean_gain'],
                observed_net_equivalent_7493=s['net']*7493/s['rows'],expected_net_equivalent_7493=s['expected_net']*7493/s['rows']))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    path=args.output.parent/'confidence-comparison.csv'
    with path.open('w',newline='') as handle:
        w=csv.DictWriter(handle,fieldnames=list(table[0]),lineterminator='\n');w.writeheader();w.writerows(table)
    evidence['comparison_file']=dict(path=path.name,rows=len(table),sha256=digest(path))
    args.output.write_text(json.dumps(evidence,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(dict(status='verified',summary=table)))


if __name__=='__main__':main()
