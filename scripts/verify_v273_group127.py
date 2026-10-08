"""Independently recompute native scores, full-group actions and ID provenance."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.verify_v273_selector_repair_artifacts import metrics, named_pairs, require, ids_digest

FOLDS = {0, 1, 2, 4}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(report, ids, y, folds):
    producers = {}
    for p in report['expert_producers']:
        key = tuple(p['train_folds']);take = np.isin(folds,key) & (y>=2)
        require(np.array_equal(p['train_global_ids'],ids[take]),'actual producer fit IDs')
        require(p['train_rows'] == int(take.sum()) and p['train_id_sha256'] == ids_digest(ids[take]),'producer count/hash')
        require(p['true_k_counts'] == np.bincount(y[take],minlength=7).tolist(),'producer class count')
        require(set(key) < FOLDS and key not in producers,'producer fold key')
        producers[key] = p
    require(len(producers)==14,'fourteen producer fold sets')

    def path(p,excluded):
        predicted = p['reference_prediction_fold'];fit = set(p['train_folds'])
        require(set(p['excluded_folds'])==excluded and fit == FOLDS-excluded-{predicted},'path fold exclusions')
        actual = producers[tuple(p['train_folds'])]
        forbidden = set(ids[np.isin(folds,list(excluded | {predicted}))])
        require(not forbidden.intersection(actual['train_global_ids']),'forbidden producer ID')
        require(p['train_id_sha256']==actual['train_id_sha256'] and p['forbidden_train_overlap']==0,'path manifest')

    checked = 0;outers=set()
    for outer in report['provenance']:
        o=outer['outer_fold'];outers.add(o)
        require(outer['train_ids_sha256']==ids_digest(ids[folds!=o]),'outer train IDs')
        require(outer['test_ids_sha256']==ids_digest(ids[folds==o]) and outer['forbidden_id_overlap']==0,'outer test IDs')
        require({p['reference_prediction_fold'] for p in outer['train_oof_producers']} == FOLDS-{o},'OOF coverage')
        for p in outer['train_oof_producers']:path(p,{o})
        require({r['receiver_fold'] for r in outer['inner_audits']} == FOLDS-{o},'audit receiver coverage')
        for r in outer['inner_audits']:
            receiver=r['receiver_fold'];permitted=FOLDS-{o,receiver};take=np.isin(folds,list(permitted))
            require(r['outer_fold']==o and set(r['reference_folds'])==permitted,'audit reference folds')
            require(r['reference_rows']==int(take.sum()) and r['reference_ids_sha256']==ids_digest(ids[take]),'audit/scaler reference IDs')
            require(r['receiver_ids_sha256']==ids_digest(ids[folds==receiver]) and r['forbidden_id_overlap']==0,'audit receiver IDs')
            require({p['reference_prediction_fold'] for p in r['reference_expert_producers']}==permitted,'nested reference coverage')
            for p in r['reference_expert_producers']:
                path(p,{o,receiver});checked+=1
    require(outers==FOLDS and checked==24,'nested path coverage')
    return dict(distinct_producers=14,nested_paths=24,forbidden_id_overlap=0)


def choose(gain,proposals,base,singletons=False):
    sizes=np.array([g.bit_count() for g in range(1,128)])
    allowed=proposals!=base[:,None]
    if singletons:allowed &= sizes[None]==1
    value=np.where(allowed,gain,-np.inf);selected=value.argmax(1);row=np.arange(len(base))
    take=value[row,selected]>0
    return np.where(take,proposals[row,selected],base),np.where(take,selected+1,0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--features',type=Path,help='Optional archived rows for identifiable synergy cases')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    manifest=json.loads((args.artifacts/'artifacts.json').read_text())
    previous=json.loads((Path(__file__).resolve().parents[1]/'analysis/evidence/v273-selector-repair/verification.json').read_text())
    require(digest(args.reference)==previous['arms']['coherent']['predictions_sha256'],'archived native reference digest')
    with np.load(args.reference,allow_pickle=False) as z:
        ref={k:z[k] for k in z.files}
    ids,y,b,folds=(ref[k] for k in ['global_index','true_K','frozen_baseline_K','fold'])
    lookup={v:i for i,v in enumerate(ids)};pos=np.array([lookup[v] for v in ref['eligible_global_index']])
    active=np.zeros(len(y),bool);active[pos]=True
    require(np.array_equal(active,np.isin(b,[2,3,4])),'native eligible scope')
    yc,bc=y[pos],b[pos];row=np.arange(len(pos))
    evidence=dict(status='verified',run_id=manifest['run_id'],source_commit=manifest['source_commit'],
        source_reference_sha256=digest(args.reference),artifact_manifest=manifest,
        independent_validation=False,promotion=False,freeze_reference=metrics(y,b),
        archived_coherent=metrics(y,ref['predicted_K']),arms={})
    source_data=None;input_manifests=None;parameter_count=None;outputs={};cases={};group_rows=[]
    cache_sha=None
    if 'producer_cache' in manifest:
        directory=args.artifacts/'producer-cache';item=manifest['producer_cache']
        require(digest(args.artifacts/'producer-cache.zip')==item['sha256'],'producer archive SHA-256')
        cached=json.loads((directory/'manifest.json').read_text())
        cache_sha=digest(directory/'producers.npz')
        require(cache_sha==cached['cache_sha256'],'producer array SHA-256')
        require((directory/'source-reference-sha256.txt').read_text().strip()==digest(args.reference),'producer source reference')
        with np.load(directory/'producers.npz',allow_pickle=False) as z:
            for key,actual in [('eligible_global_index',ids[pos]),('truth',yc),('baseline',bc),('fold',folds[pos])]:
                require(np.array_equal(z[key],actual),'producer cache alignment')
            require(len(cached['producers'])==14,'cached producer inventory')
            for producer in cached['producers']:
                train_folds=producer['train_folds'];inside=np.isin(folds[pos],train_folds)
                votes=z['fit_'+'_'.join(map(str,train_folds))]
                require(votes.shape==(7493,6,5),'producer cache shape')
                require(np.isnan(votes[inside]).all() and np.isfinite(votes[~inside]).all(),'only out-of-fit cache predictions')
        evidence['shared_producer_cache_sha256']=cache_sha
    for arm in ['direct','global','local']:
        item=manifest['arms'][arm]
        require(digest(args.artifacts/(arm+'.zip'))==item['sha256'],'archive SHA-256')
        directory=args.artifacts/arm
        report=json.loads((directory/'report.json').read_text())
        require(report['arm']==arm and report['configuration']['producer_cache_sha256']==cache_sha,'arm/cache provenance')
        with np.load(directory/'predictions.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
        for key in ['global_index','true_K','frozen_baseline_K','fold','eligible_global_index']:
            require(np.array_equal(data[key],ref[key]),'native alignment '+key)
        votes=data['member_votes'];proposals=data['group_proposal'];gain=data['group_expected_gain'];prob=data['group_outcome_probability']
        require(votes.shape==(7493,7,5) and np.isfinite(votes).all() and (votes>=0).all(),'candidate votes')
        require(np.allclose(votes.sum(2),1,atol=1e-6),'candidate normalization')
        require(np.array_equal(votes[:,0],np.eye(5,dtype=np.float32)[bc-2]),'frozen action encoding')
        require(np.array_equal(votes[:,6],votes[:,1:6].mean(1)),'derived seventh candidate')
        expected_proposals=[]
        for mask in range(1,128):
            members=[i for i in range(7) if mask & (1<<i)]
            average=votes[:,members].mean(1)
            best=average.argmax(1)+2
            expected_proposals.append(np.where(average[row,best-2]>average[row,bc-2],best,bc))
        require(np.array_equal(proposals,np.stack(expected_proposals,1)),'127 joint proposals')
        require(prob.shape==(7493,127,3) and np.isfinite(prob).all() and (prob>=-1e-7).all(),'group probabilities')
        require(np.allclose(prob.sum(2),1,atol=1e-6),'group normalization')
        require(np.array_equal(gain,prob[:,:,0]-prob[:,:,1]),'expected gain formula')
        change=proposals!=bc[:,None];r=data['baseline_correct_probability']
        require(np.array_equal(prob[:,:,1],r[:,None]*change),'one shared regression risk')
        require(np.array_equal(prob[~change],np.tile([0.,0.,1.],((~change).sum(),1))),'KEEP groups are neutral')
        pred,mask=choose(gain,proposals,bc);single,single_mask=choose(gain,proposals,bc,True)
        require(np.array_equal(pred,data['predicted_K'][pos]) and np.array_equal(mask,data['chosen_group_mask']),'full group decoder')
        require(np.array_equal(single,data['singletons_only_K'][pos]) and np.array_equal(single_mask,data['singleton_group_mask']),'singleton decoder')
        require(np.array_equal(data['predicted_K'][~active],b[~active]) and np.array_equal(data['singletons_only_K'][~active],b[~active]),'outside scope')
        require(report['candidate']==metrics(y,data['predicted_K']) and report['paired']==named_pairs(y,b,data['predicted_K']),'native score')
        require(report['paired_vs_coherent']==named_pairs(y,ref['predicted_K'],data['predicted_K']),'comparison to archived coherent')
        require(report['singletons_only']['metrics']==metrics(y,data['singletons_only_K']),'singleton score')
        require(report['singletons_only']['paired']==named_pairs(y,b,data['singletons_only_K']),'singleton pairs')
        require(report['singletons_only']['comparison_with_all_groups']==named_pairs(y,data['singletons_only_K'],data['predicted_K']),'group benefit accounting')
        cor=(proposals==yc[:,None]) & (bc[:,None]!=yc[:,None])
        reg=(proposals!=bc[:,None]) & (bc[:,None]==yc[:,None])
        singles=proposals[:,[2**i-1 for i in range(7)]]
        synergy=np.zeros((len(pos),127),bool)
        for m in range(1,128):
            members=[j for j in range(7) if m & (1<<j)]
            synergy[:,m-1]=(proposals[:,m-1]==yc) & (singles[:,members]!=yc[:,None]).all(1) & (len(members)>=2)
            g=report['groups'][m-1];selected=mask==m
            counts=dict(mask=m,size=len(members),corrections=int(cor[:,m-1].sum()),regressions=int(reg[:,m-1].sum()),
                neutral=int((~cor[:,m-1]&~reg[:,m-1]).sum()),selected=int(selected.sum()),
                selected_corrections=int((selected&cor[:,m-1]).sum()),selected_regressions=int((selected&reg[:,m-1]).sum()),
                synergy_rows=int(synergy[:,m-1].sum()))
            require(counts==g,'per group outcomes')
            group_rows.append(dict(arm=arm,**counts))
        oracle_fix=(bc!=yc)&(proposals==yc[:,None]).any(1)
        require(report['oracle']['correctable_errors_with_some_group']==int(oracle_fix.sum()),'oracle count')
        sy=dict(events_with_correct_group_and_all_its_members_wrong=int(synergy.any(1).sum()),
            among_initially_wrong=int((synergy.any(1)&(bc!=yc)).sum()),among_initially_correct=int((synergy.any(1)&(bc==yc)).sum()),
            selected_correct_joint_groups_with_all_members_wrong=int((synergy[row,np.maximum(mask-1,0)]&(mask>0)).sum()))
        require(sy==report['synergy'],'synergy outcomes')
        strong=(proposals==yc[:,None]).any(1)&(singles!=yc[:,None]).all(1)
        require(not (strong&(bc==yc)).any(),'all-seven-wrong implies initial error')
        strong_summary=dict(events=int(strong.sum()),recovered=int((strong&(pred==yc)).sum()),
            true_k_counts={str(k):int((strong&(yc==k)).sum()) for k in range(7)},
            global_ids=ids[pos][strong].tolist())
        for i in np.flatnonzero(strong):
            gid=int(ids[pos[i]]);correct_masks=(np.flatnonzero(proposals[i]==yc[i])+1).tolist()
            case=cases.setdefault(gid,dict(global_index=gid,fold=int(folds[pos[i]]),true_K=int(yc[i]),frozen_baseline_K=int(bc[i]),
                **{'S'+str(j+1)+'_K':int(singles[i,j]) for j in range(7)},
                correct_group_masks='|'.join(map(str,correct_masks)),
                min_correct_group_size=min(m.bit_count() for m in correct_masks)))
            case.update({arm+'_mask':int(mask[i]),arm+'_K':int(pred[i]),arm+'_correct':bool(pred[i]==yc[i])})
        for f in FOLDS:
            take=folds[pos]==f
            require(report['folds'][str(f)]['paired']==named_pairs(yc[take],bc[take],pred[take]),'fold score')
        expected_net=float(np.where(mask>0,gain[row,np.maximum(mask-1,0)],0).sum())
        require(abs(expected_net-report['selected_expected_net'])<1e-5,'announced gain')
        audit=data['local_audit_features']
        require(audit.shape==(7493,127,9) and np.isfinite(audit).all(),'raw audits')
        require(np.allclose(audit[...,:3].sum(2),1,atol=1e-6) and np.allclose(audit[...,3:6].sum(2),1,atol=1e-6),'audit rates')
        fingerprint={k:hashlib.sha256(data[k].tobytes()).hexdigest() for k in ['member_votes','group_proposal','local_audit_features']}
        if source_data is None:
            source_data=fingerprint;input_manifests=(report['provenance'],report['expert_producers'])
            parameter_count=[report['folds'][str(f)]['parameters'] for f in sorted(FOLDS)]
        else:
            require(fingerprint==source_data,'candidate/audit inputs differ between arms')
            require(input_manifests==(report['provenance'],report['expert_producers']),'input provenance differs between arms')
            require(parameter_count==[report['folds'][str(f)]['parameters'] for f in sorted(FOLDS)],'architecture count differs')
        entry={k:report[k] for k in ['candidate','paired','paired_vs_coherent','singletons_only','synergy','oracle','selected_group_sizes','keep_rows','selected_expected_net','configuration']}
        entry.update(artifact_id=item['id'],archive_sha256=item['sha256'],predictions_sha256=digest(directory/'predictions.npz'),
            all_seven_singletons_wrong=strong_summary,
            input_sha256=fingerprint,parameters=parameter_count,raw_audits_exported_before_arm_mask=True,
            provenance_verification=provenance(report,ref['eligible_global_index'],yc,folds[pos]),
            fold_nets={str(f):report['folds'][str(f)]['paired']['global']['net'] for f in sorted(FOLDS)})
        evidence['arms'][arm]=entry
        outputs[arm]=data['predicted_K']
    evidence['comparison_inputs_bit_identical']=True
    evidence['paired_between_arms']={a+'_to_'+b:named_pairs(y,outputs[a],outputs[b])
        for a,b in [('direct','global'),('direct','local'),('global','local')]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    groups_file=args.output.parent/'group-outcomes.csv'
    with groups_file.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(group_rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(group_rows)
    evidence['group_outcome_file']=dict(path=groups_file.name,rows=len(group_rows),sha256=digest(groups_file))
    if args.features is not None:
        found=set()
        for file in sorted(args.features.rglob('rows.jsonl')):
            for line in file.open():
                record=json.loads(line);gid=record['global_index']
                if gid not in cases:continue
                require(gid not in found,'duplicate synergy metadata')
                require((record['fold'],record['true_k'],record['baseline_k'])==
                    tuple(cases[gid][k] for k in ['fold','true_K','frozen_baseline_K']),'synergy metadata alignment')
                cases[gid].update(recording_id=record['recording_id'],start_sample=record['start_sample']);found.add(gid)
        require(found==set(cases),'missing synergy metadata')
        case_file=args.output.parent/'synergy-all-seven-wrong.csv'
        with case_file.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(cases[min(cases)]),lineterminator='\n')
            writer.writeheader();writer.writerows(cases[k] for k in sorted(cases))
        evidence['synergy_case_file']=dict(path=case_file.name,rows=len(cases),sha256=digest(case_file))
    args.output.write_text(json.dumps(evidence,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(dict(status='verified',nets={a:v['paired']['global']['net'] for a,v in evidence['arms'].items()})))


if __name__=='__main__':main()
