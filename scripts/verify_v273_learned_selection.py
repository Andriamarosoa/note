"""Independent NumPy verification of coverage, decisions and corrector provenance."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from scripts.verify_v273_selector_repair_artifacts import metrics, named_pairs, require, ids_digest
from scripts.verify_v273_group127 import digest, provenance

FOLDS = {0,1,2,4}


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def choose(gain, proposals, base, allowed=None):
    possible = proposals != base[:,None]
    if allowed is not None: possible &= np.asarray(allowed)[None]
    score = np.where(possible,gain,-np.inf); col = score.argmax(1); row = np.arange(len(base))
    accepted = score[row,col] > 0
    return np.where(accepted,proposals[row,col],base), np.where(accepted,col+1,0)


def proposals_from_votes(votes, base):
    result = []; row = np.arange(len(base))
    for mask in range(1,2**votes.shape[1]):
        members = [j for j in range(votes.shape[1]) if mask & (1<<j)]
        mean = votes[:,members].mean(1); best = mean.argmax(1)+2
        result.append(np.where(mean[row,best-2] > mean[row,base-2],best,base))
    return np.stack(result,1)


def verify_corrector(directory, manifest, yc, bc, fc, ids, features):
    cache = read_npz(directory/'correctors.npz')
    require(digest(directory/'correctors.npz') == manifest['cache_sha256'], 'corrector cache SHA')
    require(manifest['seed']==27403 and manifest['epochs']==30 and manifest['architecture']==[50,64,32,7], 'corrector frozen protocol')
    for k,v in [('truth',yc),('baseline',bc),('fold',fc),('eligible_global_index',ids)]:
        require(np.array_equal(cache[k],v),'corrector alignment '+k)
    rows = {}
    for file in features.rglob('rows.jsonl'):
        for line in file.open():
            r = json.loads(line)
            require(r['global_index'] not in rows, 'duplicate metadata')
            rows[r['global_index']] = r
    require(set(ids) == set(rows), 'feature coverage')
    recording_folds = {}
    for i,gid in enumerate(ids):
        r=rows[gid];name=r['recording_id']
        require((r['true_k'],r['baseline_k'],r['fold'])==(yc[i],bc[i],fc[i]),'feature label alignment')
        require(name[:2] in ('00','01','02','03','04'), 'player05 exclusion')
        require(name not in recording_folds or recording_folds[name]==r['fold'],'recording split')
        recording_folds[name]=r['fold']
    raw = np.array([[rows[i]['features'][n] for n in manifest['context_names']] for i in ids])
    producers = {}
    for record in manifest['producers']:
        key=tuple(record['train_folds']); inside=np.isin(fc,key)
        require(key not in producers and set(key) < FOLDS and key,'corrector producer set')
        require(record['train_rows']==int(inside.sum()) and record['train_id_sha256']==ids_digest(ids[inside]),'corrector fit hash')
        require(np.array_equal(record['train_global_ids'],ids[inside]),'corrector actual fit IDs')
        require(record['true_k_counts']==np.bincount(yc[inside],minlength=7).tolist(),'corrector training labels')
        require(record['parameters']==5575 and [h['epoch'] for h in record['training']]==[1,16,30],'corrector full training')
        require(digest(directory/record['weights'])==record['weights_sha256'],'corrector weights SHA')
        require(np.allclose(record['scaler_mean'],raw[inside].mean(0),atol=1e-10),'scaler excluded means')
        scale=raw[inside].std(0);scale=np.where(scale==0,1,scale)
        require(np.allclose(record['scaler_scale'],scale,atol=1e-10),'scaler excluded scales')
        p=cache['fit_'+'_'.join(map(str,key))]
        require(p.shape==(len(ids),7) and np.isnan(p[inside]).all() and np.isfinite(p[~inside]).all(),'cache excludes fit rows')
        require((p[~inside]>=0).all() and np.allclose(p[~inside].sum(1),1,atol=1e-6),'corrector distribution')
        producers[key]=record
    require(len(producers)==14 and set(producers)=={k for s in (1,2,3) for k in itertools.combinations(sorted(FOLDS),s)},'fourteen corrector producers')
    return cache, producers, rows


def verify_paths(report, producers, ids, fc):
    checked=0
    def path(record, forbidden):
        nonlocal checked
        key=tuple(record['train_folds']); p=producers[key]
        excluded=forbidden | {record['reference_prediction_fold']}
        require(not set(key)&excluded,'corrector forbidden fold path')
        require(not set(p['train_global_ids']) & set(ids[np.isin(fc,list(excluded))]),'corrector forbidden IDs')
        require(record['corrector_train_id_sha256']==p['train_id_sha256'],'corrector path digest')
        checked+=1
    for outer in report['provenance']:
        o=outer['outer_fold']
        for record in outer['train_oof_producers']:path(record,{o})
        for inner in outer['inner_audits']:
            for record in inner['reference_expert_producers']:path(record,{o,inner['receiver_fold']})
    require(checked==36,'corrector outer/nested path count')
    return dict(producer_sets=14, checked_crossfit_paths=checked, forbidden_overlap=0)


def verify_global_audits(data, count, experts, correctors, yc, bc, fc):
    """Rebuild every enabled held-out history from excluded cached producers."""
    for outer in sorted(FOLDS):
        train=np.flatnonzero(fc!=outer);held=np.flatnonzero(fc==outer)
        rv=np.empty((len(train),count,5),np.float32)
        for receiver in sorted(FOLDS-{outer}):
            at=np.flatnonzero(fc[train]==receiver);positions=train[at]
            key='fit_'+'_'.join(map(str,sorted(FOLDS-{outer,receiver})))
            p=experts[key][positions]
            rv[at,:6]=p;rv[at,6]=p[:,1:].mean(1)
            if count==8:
                q=correctors[key][positions];action=q[:,2:].copy()
                action[np.arange(len(at)),bc[positions]-2]+=q[:,:2].sum(1)
                rv[at,7]=action
        rp=proposals_from_votes(rv,bc[train]);hp=data['group_proposal'][held]
        cor=(rp==yc[train,None])&(bc[train,None]!=yc[train,None])
        reg=(rp!=bc[train,None])&(bc[train,None]==yc[train,None])
        audit=data['local_audit_features'][held]
        for source in (2,3,4):
            fit=bc[train]==source;receive=np.flatnonzero(bc[held]==source)
            for target in range(2,7):
                same=rp[fit]==target;used=same.sum(0)
                counts=np.stack([(same&cor[fit]).sum(0),(same&reg[fit]).sum(0),
                                 (same&~cor[fit]&~reg[fit]).sum(0)],1)
                prior=((counts+1)/(used[:,None]+3)).astype(np.float32)
                for local in receive:
                    take=hp[local]==target
                    require(np.array_equal(audit[local,take,:3],prior[take]),'global history outcome rates')
                    require(np.array_equal(audit[local,take,6],(used/fit.sum()).astype(np.float32)[take]),'global history support')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--archived-consensus',type=Path,required=True)
    p.add_argument('--archived-global',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--producer-cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    require(digest(args.reference)=='707fc1681e2b0b1ef01805710186c4119ad697a0b52ed3330698e48f34c2f77b','reference archive')
    require(digest(args.archived_consensus/'predictions.npz')=='2444d21599e60cdd0a03654fb304c25645b5397002f0e88688eb193ece99de34','CE archive')
    require(digest(args.archived_global/'predictions.npz')=='29758e90675918844c8eea01fcb9816f830dd3837bae02e79e56b0d57e6af038','global archive')
    ref=read_npz(args.reference); archived=read_npz(args.archived_global/'predictions.npz')
    ce=read_npz(args.archived_consensus/'predictions.npz')
    ids,y,b,f=(ref[k] for k in ['global_index','true_K','frozen_baseline_K','fold'])
    lookup={v:i for i,v in enumerate(ids)};eligible=ref['eligible_global_index'];pos=np.array([lookup[i] for i in eligible])
    yc,bc,fc=y[pos],b[pos],f[pos];row=np.arange(len(pos));active=np.zeros(len(y),bool);active[pos]=True
    require(len(y)==59309 and len(pos)==7493 and set(f)==FOLDS,'native cohort')
    require(np.array_equal(active,np.isin(b,[2,3,4])) and int((y==b).sum())==48454,'native baseline scope')
    manifest=json.loads((args.artifacts/'artifacts.json').read_text())
    for name,item in {**manifest['arms'],'corrector-cache':manifest['corrector_cache']}.items():
        require(digest(args.artifacts/(name+'.zip'))==item['sha256'],'artifact SHA '+name)
    directory=args.artifacts/'corrector-cache'
    cm=json.loads((directory/'manifest.json').read_text())
    cache,producers,metadata=verify_corrector(directory,cm,yc,bc,fc,eligible,args.features)
    require(digest(args.producer_cache/'producers.npz')=='836b07f9ae062160508c63a7c9ab4f72b65e8458c7e24c5e4e92f17567fde914','original producer cache')
    expert_cache=read_npz(args.producer_cache/'producers.npz')
    evidence=dict(status='verified', run_id=manifest['run_id'],source_commit=manifest['source_commit'],
        artifact_manifest=manifest,independent_validation=False,promotion=False,freeze_reference=metrics(y,b),
        source_reference_sha256=digest(args.reference),corrector_cache_sha256=cm['cache_sha256'],arms={})
    outputs={};new_cases=[];paired_cases=[]
    for arm,count in [('control7',7),('corrector8',8)]:
        directory=args.artifacts/arm;report=json.loads((directory/'report.json').read_text())
        data=read_npz(directory/'predictions.npz');outputs[arm]=data
        for key in ['global_index','true_K','frozen_baseline_K','fold','eligible_global_index']:
            require(np.array_equal(data[key],ref[key]),'alignment '+key)
        require(report['arm']==arm and report['configuration']['candidate_count']==count,'arm')
        require(report['configuration']['corrector_cache_sha256']==cm['cache_sha256'],'shared corrector cache')
        require(report['configuration']['seed']==27402 and report['configuration']['epochs']==30,'selector protocol')
        v=data['member_votes'];props=data['group_proposal'];g=data['group_expected_gain'];prob=data['group_outcome_probability']
        require(v.shape==(len(pos),count,5) and props.shape==(len(pos),2**count-1),'group dimensions')
        require(np.array_equal(v[:,:7],archived['member_votes']),'original votes unchanged')
        require(np.array_equal(props[:,:127],archived['group_proposal']),'original proposals unchanged')
        require(np.array_equal(data['local_audit_features'][:,:127],archived['local_audit_features']),'original audits unchanged')
        require(np.isfinite(v).all() and (v>=0).all() and np.allclose(v.sum(2),1,atol=1e-6),'vote normalization')
        if count==8:
            require(report['corrector_manifest']==cm,'corrector manifest identity')
            for outer in sorted(FOLDS):
                key=tuple(sorted(FOLDS-{outer}));take=fc==outer
                cp=cache['fit_'+'_'.join(map(str,key))][take]
                q=cp[:,2:].copy();q[np.arange(take.sum()),bc[take]-2]+=cp[:,:2].sum(1)
                require(np.array_equal(v[take,7],q),'held S8 from excluded producer')
            cp_path=verify_paths(report,producers,eligible,fc)
        else:cp_path=None
        require(np.array_equal(props,proposals_from_votes(v,bc)),'all full-group verdicts')
        verify_global_audits(data,count,expert_cache,cache,yc,bc,fc)
        require(np.isfinite(prob).all() and (prob>=-1e-7).all() and np.allclose(prob.sum(2),1,atol=1e-6),'outcome simplex')
        require(np.array_equal(g,prob[:,:,0]-prob[:,:,1]),'gain')
        cp=data['class_probability'];other=data['other_probability'];r=data['baseline_correct_probability'];logit=data['pooled_class_logits']
        available=np.stack([(props==k).any(1)&(bc!=k) for k in range(7)],1)
        full=np.concatenate([np.where(available,logit,-1.e9),np.zeros((len(pos),1))],1)
        full-=full.max(1,keepdims=True);q=np.exp(full);q/=q.sum(1,keepdims=True)
        require(np.allclose(cp,(1-r[:,None])*q[:,:7]+np.eye(7)[bc]*r[:,None],atol=1e-6),'class probabilities')
        require(np.allclose(other,(1-r)*q[:,7],atol=1e-6) and np.allclose(cp.sum(1)+other,1,atol=1e-6),'OTHER simplex')
        changing=props!=bc[:,None]
        require(np.array_equal(prob[:,:,0],np.take_along_axis(cp,props,1)*changing),'same K same correction probability')
        require(np.array_equal(prob[:,:,1],r[:,None]*changing),'shared regression risk')
        pred,mask=choose(g,props,bc)
        require(np.array_equal(pred,data['predicted_K'][pos]) and np.array_equal(mask,data['chosen_group_mask']),'decoder')
        require(np.array_equal(data['predicted_K'][~active],b[~active]),'frozen outside scope')
        for key,baseline in [('paired',b),('paired_vs_coherent',ref['predicted_K']),('paired_vs_archived_global',archived['predicted_K']),
                             ('paired_vs_archived_singletons',archived['singletons_only_K']),('paired_vs_archived_consensus',ce['predicted_K'])]:
            require(report[key]==named_pairs(y,baseline,data['predicted_K']),'score '+key)
        require(report['candidate']==metrics(y,data['predicted_K']),'native accuracy')
        sizes=np.array([m.bit_count() for m in range(1,2**count)])
        singleton,singlemask=choose(g,props,bc,sizes==1)
        require(np.array_equal(singleton,data['singletons_only_K'][pos]) and np.array_equal(singlemask,data['singleton_group_mask']),'singleton decoder')
        require(report['singletons_only']['paired']==named_pairs(y,b,data['singletons_only_K']),'singleton scores')
        for name,ablation in report['decode_ablations'].items():
            allowed={'old_groups_only':np.arange(len(sizes))<127,'old_groups_plus_new_singleton':np.arange(len(sizes))<128,
                     'new_singleton_gated':np.arange(len(sizes))==127}[name]
            decoded,_=choose(g,props,bc,allowed)
            require(np.array_equal(decoded,data[name+'_K'][pos]),'diagnostic decoder '+name)
            require(ablation['paired']==named_pairs(y,b,data[name+'_K']) and ablation['comparison_with_all']==named_pairs(y,data[name+'_K'],data['predicted_K']),'diagnostic score '+name)
        if count==8:
            require(np.array_equal(data['new_singleton_alone_K'][pos],props[:,127]),'S8 alone decoder')
            require(report['new_singleton_alone']['paired']==named_pairs(y,b,data['new_singleton_alone_K']),'S8 alone score')
        for fold in FOLDS:
            take=fc==fold;fr=report['folds'][str(fold)]
            require(fr['paired']==named_pairs(yc[take],bc[take],pred[take]),'fold score')
            require(fr['parameters']==12994+544*(count-7) and [h['epoch'] for h in fr['training']]==[1,16,30],'selector full training')
        selected=mask>0;cor=(pred==yc)&(bc!=yc);reg=(pred!=yc)&(bc==yc)
        announced_cor=float(cp[row,pred][selected].sum());announced_reg=float(r[selected].sum())
        announced=float(np.where(selected,g[row,np.maximum(mask-1,0)],0).sum())
        require(abs(announced-report['selected_expected_net'])<1e-5,'expected gain')
        old_reach=(bc!=yc)&(props[:,:127]==yc[:,None]).any(1)
        reach=(bc!=yc)&(props==yc[:,None]).any(1);new=reach&~old_reach
        extension=dict(old_reachable_errors=int(old_reach.sum()),new_reachable_errors=int(new.sum()),
            recovered_new=int((new&(pred==yc)).sum()),new_by_k={str(k):int((new&(yc==k)).sum()) for k in range(7)})
        if count==8:
            extension.update(new_reachable_by_singleton=int((new&(props[:,127]==yc)).sum()),
                new_reachable_only_by_combination=int((new&(props[:,127]!=yc)).sum()),
                recovered_combination_only=int((new&(props[:,127]!=yc)&(pred==yc)).sum()))
        require(extension==report['catalogue_extension'],'coverage extension accounting')
        require(int(reach.sum())==report['oracle']['correctable_errors_with_some_group'],'oracle count')
        singles=props[:,[2**j-1 for j in range(count)]]
        strong=reach&(singles!=yc[:,None]).all(1)
        entry={k:report[k] for k in ['candidate','paired','paired_vs_archived_consensus','paired_vs_archived_global','paired_vs_archived_singletons',
            'singletons_only','catalogue_extension','decode_ablations','oracle','control_reproduction','new_singleton_alone'] if k in report}
        entry.update(predictions_sha256=digest(directory/'predictions.npz'),fold_nets={str(k):report['folds'][str(k)]['paired']['global']['net'] for k in sorted(FOLDS)},
            expected_corrections=announced_cor,expected_regressions=announced_reg,expected_net=announced,
            changes=int(selected.sum()),neutral_changes=int((selected&~cor&~reg).sum()),
            all_singletons_wrong_but_group_correct=int(strong.sum()),recovered_strong_synergy=int((strong&(pred==yc)).sum()),
            corrector_provenance=cp_path,expert_provenance=provenance(report,eligible,yc,fc),
            coverage_recovered_by_k={str(k):int((new&(yc==k)&(pred==yc)).sum()) for k in range(7)},
            corrections_on_old_reachable=int((old_reach&(pred==yc)).sum()),
            weights_sha256={str(k):digest(directory/f'fold-{k}.weights.h5') for k in sorted(FOLDS)})
        evidence['arms'][arm]=entry
        if count==7:
            require(report['control_reproduction']['prediction_bit_identical']==bool(np.array_equal(data['predicted_K'],ce['predicted_K'])),'control replay declaration')
        else:
            for i in np.flatnonzero(new):
                correct_masks=(np.flatnonzero(props[i]==yc[i])+1).tolist();m=metadata[eligible[i]]
                new_cases.append(dict(global_index=int(eligible[i]),recording_id=m['recording_id'],start_sample=m['start_sample'],
                    fold=int(fc[i]),true_K=int(yc[i]),baseline_K=int(bc[i]),S8_K=int(props[i,127]),
                    final_K=int(pred[i]),recovered=bool(pred[i]==yc[i]),combination_only=bool(props[i,127]!=yc[i]),
                    min_correct_group_size=min(mask.bit_count() for mask in correct_masks),correct_masks='|'.join(map(str,correct_masks))))
    a=outputs['control7'];d=outputs['corrector8'];evidence['paired_extended_vs_control']=named_pairs(y,a['predicted_K'],d['predicted_K'])
    correct_a=a['predicted_K']==y;correct_d=d['predicted_K']==y
    old_cor=(y!=b)&correct_a;old_reg=(y==b)&~correct_a
    evidence['paired_decomposition']=dict(retained_corrections=int((old_cor&correct_d).sum()),
        lost_corrections=int((old_cor&~correct_d).sum()),new_corrections=int(((y!=b)&~correct_a&correct_d).sum()),
        repaired_regressions=int((old_reg&correct_d).sum()),persisting_regressions=int((old_reg&~correct_d).sum()),
        new_regressions=int(((y==b)&correct_a&~correct_d).sum()))
    evidence['control_all_archived_arrays_bit_identical']=all(np.array_equal(a[k],ce[k]) for k in ce)
    for i in np.flatnonzero((correct_a!=correct_d)[pos]):
        m=metadata[eligible[i]];at=pos[i]
        paired_cases.append(dict(global_index=int(eligible[i]),recording_id=m['recording_id'],start_sample=m['start_sample'],fold=int(fc[i]),
            true_K=int(yc[i]),baseline_K=int(bc[i]),control7_K=int(a['predicted_K'][at]),corrector8_K=int(d['predicted_K'][at]),
            effect='correction' if correct_d[at] else 'regression'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    evidence['case_files']={}
    for name,cases in [('new-reachable-errors.csv',new_cases),('extended-vs-control.csv',paired_cases)]:
        if not cases:continue
        path=args.output.parent/name
        with path.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(cases[0]),lineterminator='\n');writer.writeheader();writer.writerows(cases)
        evidence['case_files'][name]=dict(rows=len(cases),sha256=digest(path))
    args.output.write_text(json.dumps(evidence,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(dict(status='verified',nets={k:v['paired']['global']['net'] for k,v in evidence['arms'].items()},
        extension=evidence['arms']['corrector8']['catalogue_extension'],versus_control=evidence['paired_extended_vs_control']['global'])))


if __name__=='__main__':main()
