"""Verify contextual-risk outcomes and append durable candidates without replacing history."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.verify_v273_selector_repair_artifacts import require,digest,metrics,named_pairs,ids_digest
from scripts.verify_v273_learned_selection import read_npz,choose,proposals_from_votes,verify_corrector,verify_paths
from scripts.verify_v273_group127 import provenance
from scripts.v273_confidence_metrics import summarize
from scripts.v273_contextual_risk_contract import audit_profiles,retention,profile_retention


def check_numbers(actual,expected,path=''):
    if isinstance(expected,dict):
        require(set(actual)==set(expected),'keys '+path)
        for k,v in expected.items():check_numbers(actual[k],v,path+'/'+k)
    elif isinstance(expected,list):
        require(len(actual)==len(expected),'length '+path)
        for i,v in enumerate(expected):check_numbers(actual[i],v,path+'/'+str(i))
    elif isinstance(expected,float):require(np.isclose(actual,expected,atol=1e-7,rtol=1e-7),'number '+path)
    else:require(actual==expected,'value '+path)


def confidence(d,pos):
    return summarize(dict(truth=d['true_K'][pos],baseline=d['frozen_baseline_K'][pos],proposal=d['group_proposal'],
        baseline_correct=d['baseline_correct_probability'],pooled_class_logits=d['pooled_class_logits'],
        class_probability=d['class_probability'],other_probability=d['other_probability']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['results','parent','corrector-cache','features','memory','output']:
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    require(not args.output.exists(),'refusing to overwrite existing evidence and candidate memory')
    args.output.mkdir(parents=True)
    manifest=json.loads((args.results/'artifacts.json').read_text())
    item=manifest['artifact'];zipfile=args.results/item['filename']
    require(zipfile.stat().st_size==item['bytes'] and digest(zipfile)==item['sha256'],'download identity')
    directory=args.results/'model';d=read_npz(directory/'predictions.npz');old=read_npz(args.parent/'predictions.npz')
    report=json.loads((directory/'report.json').read_text())
    require(report['source_commit']==manifest['source_commit'],'executed source identity')
    require(report['predictions_sha256']==digest(directory/'predictions.npz'),'predictions identity')
    prior=json.loads((Path(__file__).resolve().parents[1]/'analysis/evidence/v273-audit-memory/verification.json').read_text())['arms']['corrector8']
    require(digest(args.parent/'predictions.npz')==prior['predictions_sha256']==report['parent_predictions_sha256'],'immutable parent')
    for key in ['global_index','true_K','frozen_baseline_K','fold','eligible_global_index','member_votes','group_proposal','local_audit_features']:
        require(np.array_equal(d[key],old[key]),'input identity '+key)
    ids,y,b,f,eligible=(d[k] for k in ['global_index','true_K','frozen_baseline_K','fold','eligible_global_index'])
    positions={int(v):i for i,v in enumerate(ids)};pos=np.array([positions[int(v)] for v in eligible])
    yc,bc,fc=y[pos],b[pos],f[pos];parent=old['predicted_K'];prediction=d['predicted_K']
    require(np.array_equal(d['parent_K'],parent),'archived parent decisions')
    require(len(ids)==59309 and len(pos)==7493 and set(f)=={0,1,2,4},'cohort')
    active=np.zeros(len(ids),bool);active[pos]=True;require(np.array_equal(active,np.isin(b,[2,3,4])),'eligible scope')
    props=d['group_proposal'];require(np.array_equal(props,proposals_from_votes(d['member_votes'],bc)),'all complete proposals')
    probability=d['group_outcome_probability'];gain=d['group_expected_gain'];r=d['baseline_correct_probability'];cp=d['class_probability']
    require(np.isfinite(probability).all() and (probability>=-1e-7).all() and np.allclose(probability.sum(2),1,atol=1e-6),'outcome simplex')
    change=props!=bc[:,None]
    require(np.array_equal(probability[:,:,0],np.take_along_axis(cp,props,1)*change),'shared destination correction probability')
    require(np.array_equal(probability[:,:,1],r[:,None]*change),'shared regression probability')
    require(np.array_equal(gain,probability[:,:,0]-probability[:,:,1]),'gain formula')
    predicted,mask=choose(gain,props,bc)
    require(np.array_equal(predicted,prediction[pos]) and np.array_equal(mask,d['chosen_group_mask']),'decoder')
    require(np.allclose(cp.sum(1)+d['other_probability'],1,atol=1e-6),'class simplex')
    available=np.stack([(props==k).any(1)&(bc!=k) for k in range(7)],1)
    logits=np.concatenate([np.where(available,d['pooled_class_logits'],-1e9),np.zeros((len(pos),1))],1)
    logits-=logits.max(1,keepdims=True);q=np.exp(logits);q/=q.sum(1,keepdims=True)
    require(np.allclose(cp,(1-r[:,None])*q[:,:7]+np.eye(7)[bc]*r[:,None],atol=1e-6),'conditional probabilities')
    require(np.allclose(d['other_probability'],(1-r)*q[:,7],atol=1e-6),'OTHER probability')
    require(report['paired']==named_pairs(y,b,prediction) and report['candidate']==metrics(y,prediction),'main native results')
    require(report['parent_paired']==named_pairs(y,b,parent),'parent native results')
    require(report['retention']==retention(y,b,parent,prediction),'all complementary cases')
    profiles=audit_profiles(props,d['local_audit_features'],parent[pos],bc)
    require(np.array_equal(d['parent_audit_profile'],profiles),'label-free profiles')
    require(report['parent_profile_retention']==profile_retention(profiles,yc,bc,parent[pos],prediction[pos]),'all profile outcomes')
    target=report['parent_profile_retention']['local_rates_favor_regression_for_all_groups']
    require(target['rows']==341 and target['parent_vs_freeze']['global']==dict(changed=341,corrections=92,regressions=126,net=-34),'fixed 126/92/123 cohort')
    check_numbers(report['confidence'],confidence(d,pos),'confidence')
    check_numbers(report['parent_confidence'],confidence(old,pos),'parent confidence')
    cm=json.loads((args.corrector_cache/'manifest.json').read_text())
    _,producers,metadata=verify_corrector(args.corrector_cache,cm,yc,bc,fc,eligible,args.features)
    expert_proof=provenance(report,eligible,yc,fc);corrector_proof=verify_paths(report,producers,eligible,fc)
    require(report['corrector_manifest']==cm,'corrector manifest identity')
    config=report['configuration'];require(config['seed']==27402 and config['epochs']==30 and config['parameters']==14562 and config['groups']==255,'fixed protocol')
    require(config['corrector_cache_sha256']==cm['cache_sha256'] and config['producer_cache_sha256']=='836b07f9ae062160508c63a7c9ab4f72b65e8458c7e24c5e4e92f17567fde914','frozen producer caches')
    require(config['local_audits_enabled'] and not config['individual_veto'] and not config['threshold_search'],'no individual or tuned veto')
    weight_hashes={}
    for fold in (0,1,2,4):
        take=fc==fold;fr=report['folds'][str(fold)];conn=fr['audit_connection']
        require(fr['train_ids_sha256']==ids_digest(eligible[~take]) and fr['held_ids_sha256']==ids_digest(eligible[take]),'train and held exclusions')
        require(fr['parent_replay_max_difference']<2e-5 and fr['parameters']==14562,'parent replay and model size')
        require([v['epoch'] for v in fr['training']]==[1,16,30],'full predeclared training')
        require(conn['held']['effective_audit_sha256']==hashlib.sha256(np.ascontiguousarray(d['local_audit_features'][take]).tobytes()).hexdigest(),'actual full held audits')
        require(conn['held']['rows_with_local_evidence']==int(take.sum()) and conn['train']['rows_with_local_evidence']==int((~take).sum()),'local evidence present')
        require(fr['paired']==named_pairs(yc[take],bc[take],prediction[pos[take]]),'fold results')
        require(fr['retention']==retention(yc[take],bc[take],parent[pos[take]],prediction[pos[take]]),'fold retention')
        weight=args.output/f'fold-{fold}.weights.h5';weight.write_bytes((directory/weight.name).read_bytes())
        require(digest(weight)==fr['weights_sha256'],'trained weights identity');weight_hashes[weight.name]=digest(weight)
    for name,prefix in [('zero_group_context_K','zero_context'),('zero_local_audit_K','zero_local')]:
        cr=d[prefix+'_baseline_correct'];ccp=d[prefix+'_class_probability'];other=d[prefix+'_other_probability']
        require(np.allclose(ccp.sum(1)+other,1,atol=1e-6),'intervention simplex')
        score=(np.take_along_axis(ccp,props,1)-cr[:,None])*change
        pr,_=choose(score,props,bc);require(np.array_equal(pr,d[name][pos]),'intervention replay')
        require(np.array_equal(d[name][~active],b[~active]),'intervention outside scope')
        expected=dict(paired_vs_freeze=named_pairs(y,b,d[name]),paired_main_vs_intervention=named_pairs(y,d[name],prediction),retrained=False)
        require(report['interventions'][name]==expected,'intervention metrics')
    require(np.array_equal(d['zero_context_pooled_class_logits'],d['pooled_class_logits']),'risk-only intervention keeps conditional distribution fixed')
    require(np.array_equal(prediction[~active],b[~active]),'main outside scope')
    require(not report['promotion'] and not report['independent_validation'],'development scope')
    singles=props[:,[2**j-1 for j in range(8)]];strong=(bc!=yc)&(props==yc[:,None]).any(1)&(singles!=yc[:,None]).all(1)
    require(report['combination_only']==dict(events=int(strong.sum()),parent_corrected=int((strong&(parent[pos]==yc)).sum()),candidate_corrected=int((strong&(prediction[pos]==yc)).sum())),'joint-only cases')
    # Append evidence; never rewrite the earlier 41-column memory.
    registry=json.loads((args.memory/'registry.json').read_text());history=read_npz(args.memory/'all-variant-decisions.npz')
    require(np.array_equal(history['global_index'],ids),'historical memory alignment')
    historical_hashes={v['prediction_values_sha256']:v['variant_id'] for v in reversed(registry['variants'])}
    fields=['predicted_K','zero_group_context_K','zero_local_audit_K']
    keys=['catalogue8_contextual_risk','catalogue8_contextual_risk__zero_group_context','catalogue8_contextual_risk__zero_local_audit']
    matrix=np.column_stack([d[k] for k in fields]).astype(np.int8);outcomes=(matrix==y[:,None]).astype(np.int8)-(b==y).astype(np.int8)[:,None]
    entries=[]
    for j,key in enumerate(keys):
        sha=hashlib.sha256(np.ascontiguousarray(matrix[:,j]).tobytes()).hexdigest()
        entry=dict(variant_id=key,parent_id='catalogue8_local' if j==0 else keys[0],
            candidate_kind='measured_neural_policy' if j==0 else 'fixed_weight_intervention',
            source_predictions_sha256=digest(directory/'predictions.npz'),source_field=fields[j],retained=True,
            retention_depends_on_global_gain=False,downstream_ready=False,
            required_for_downstream_activation='Regenerate nested producer predictions excluding every consumer receiver and forbidden fold.',
            metrics=metrics(y,matrix[:,j]),versus_freeze=named_pairs(y,b,matrix[:,j]),
            versus_parent=named_pairs(y,parent if j==0 else matrix[:,0],matrix[:,j]),prediction_values_sha256=sha,
            identical_prediction_alias_of=historical_hashes.get(sha))
        entries.append(entry);historical_hashes.setdefault(sha,key)
    memory_file=args.output/'preserved-candidates.npz'
    np.savez_compressed(memory_file,global_index=ids,true_K=y,frozen_baseline_K=b,fold=f,eligible_global_index=eligible,
        parent_K=parent,variant_ids=np.array(keys),predictions=matrix,outcomes_vs_freeze=outcomes)
    # Preserve compact probabilistic evidence too; group gains can be reconstructed
    # exactly from these probabilities and the shared complete-group proposals.
    compact=dict(eligible_global_index=eligible,group_proposal=props.astype(np.int8),
        true_K=yc,frozen_K=bc,chosen_group_mask=d['chosen_group_mask'],parent_profile=profiles)
    for field in ['baseline_correct_probability','class_probability','pooled_class_logits','other_probability']:
        compact['candidate_'+field]=d[field];compact['parent_'+field]=old[field]
    for prefix in ['zero_context','zero_local']:
        for field in ['baseline_correct','class_probability','pooled_class_logits','other_probability']:
            compact[prefix+'_'+field]=d[prefix+'_'+field]
    probability_file=args.output/'probability-evidence.npz';np.savez_compressed(probability_file,**compact)
    restored=read_npz(memory_file)
    require(np.array_equal(restored['predictions'],matrix) and np.array_equal(restored['outcomes_vs_freeze'],outcomes),'durable native decisions')
    restored=read_npz(probability_file)
    require(all(np.array_equal(restored[key],value) for key,value in compact.items()),'durable probability evidence')
    new_ids={v['variant_id'] for v in entries};require(not new_ids&{v['variant_id'] for v in registry['variants']},'never overwrite an existing candidate')
    combined=np.column_stack([history['predictions'],matrix]);union=(b!=y)&np.any(combined==y[:,None],axis=1)
    old_union=(b!=y)&np.any(history['predictions']==y[:,None],axis=1)
    evidence_registry=dict(schema='v273-appended-candidate-memory-v1',previous_registry='../v273-audit-memory/variants/registry.json',
        previous_registry_sha256=digest(args.memory/'registry.json'),previous_native_memory_sha256=digest(args.memory/'all-variant-decisions.npz'),
        previous_candidates=registry['retained_policy_candidates'],added_candidates=entries,
        total_candidates=registry['retained_policy_candidates']+len(entries),distinct_prediction_vectors=len(historical_hashes),
        candidate_file=memory_file.name,candidate_file_sha256=digest(memory_file),
        probability_file=probability_file.name,probability_file_sha256=digest(probability_file),
        context_join='../v273-audit-memory/variants/context-fold-*-part-*.npz; join by global_index',
        union_of_initial_errors_corrected=int(union.sum()),additional_errors_reached_by_appended_outputs=int((union&~old_union).sum()),
        oracle_union_not_prediction=True,promotion=False,independent_validation=False)
    (args.output/'candidate-registry.json').write_text(json.dumps(evidence_registry,indent=2,sort_keys=True)+'\n')
    casefile=args.output/'paired-cases.csv';cases=[]
    for i,g in enumerate(eligible):
        at=pos[i];meta=metadata[g]
        old_correct=parent[at]==y[at];new_correct=prediction[at]==y[at]
        cases.append(dict(global_index=int(g),fold=int(fc[i]),recording_id=meta['recording_id'],start_sample=meta['start_sample'],
            true_K=int(y[at]),frozen_K=int(b[at]),parent_K=int(parent[at]),candidate_K=int(prediction[at]),
            parent_profile=profiles[i],effect_vs_parent=int(new_correct)-int(old_correct),
            parent_baseline_correct_probability=float(old['baseline_correct_probability'][i]),
            candidate_baseline_correct_probability=float(r[i])))
    with casefile.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(cases[0]),lineterminator='\n');writer.writeheader();writer.writerows(cases)
    (args.output/'report.json').write_bytes((directory/'report.json').read_bytes())
    evidence=dict(status='verified',run_id=manifest['run_id'],source_commit=manifest['source_commit'],artifact_manifest=manifest,
        source_predictions_sha256=digest(directory/'predictions.npz'),report_sha256=digest(directory/'report.json'),
        raw_votes_proposals_audits_unchanged=True,expert_provenance=expert_proof,corrector_provenance=corrector_proof,
        paired=report['paired'],retention=report['retention'],target_profile=target,combination_only=report['combination_only'],
        confidence=report['confidence'],parent_confidence=report['parent_confidence'],interventions=report['interventions'],
        weight_hashes=weight_hashes,paired_cases=dict(file=casefile.name,rows=len(cases),sha256=digest(casefile)),
        preserved_candidates=keys,previous_registry_unchanged_sha256=digest(args.memory/'registry.json'),
        previous_native_memory_unchanged_sha256=digest(args.memory/'all-variant-decisions.npz'),
        total_candidates_after_neural_stage=evidence_registry['total_candidates'],independent_validation=False,promotion=False)
    (args.output/'verification.json').write_text(json.dumps(evidence,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(dict(status='verified',paired=report['paired']['global'],retention=report['retention'],target_profile=target,
        expected_net=report['confidence']['expected_net'],total_candidates=evidence_registry['total_candidates'])),flush=True)


if __name__=='__main__':main()
