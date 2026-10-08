"""Independently replay both factor crossovers and preserve them alongside all variants."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.verify_v273_learned_selection import read_npz,choose
from scripts.verify_v273_selector_repair_artifacts import require,digest,metrics,named_pairs
from scripts.v273_contextual_risk_contract import audit_profiles,retention,profile_retention
from scripts.v273_confidence_metrics import summarize
from scripts.verify_v273_contextual_risk import check_numbers


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['results','parent','candidate','memory-output','previous-memory']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--protocol-commit',required=True)
    args=p.parse_args();out=args.memory_output
    report=json.loads((args.results/'report.json').read_text());data=read_npz(args.results/'factor-swaps.npz')
    parent=read_npz(args.parent);candidate=read_npz(args.candidate)
    require(report['sources']==dict(parent_sha256=digest(args.parent),candidate_sha256=digest(args.candidate)),'factor source identities')
    require(report['predictions_sha256']==digest(args.results/'factor-swaps.npz'),'factor output identity')
    require(not report['labels_used_to_construct_predictions'] and not report['promotion'] and not report['independent_validation'],'diagnostic scope')
    ids,y,b,f,eligible=(data[k] for k in ['global_index','true_K','frozen_baseline_K','fold','eligible_global_index'])
    for key in ['global_index','true_K','frozen_baseline_K','fold','eligible_global_index']:
        require(np.array_equal(data[key],parent[key]) and np.array_equal(data[key],candidate[key]),'native alignment')
    lookup={int(v):i for i,v in enumerate(ids)};pos=np.array([lookup[int(g)] for g in eligible]);yc,bc=y[pos],b[pos]
    props=candidate['group_proposal'];require(np.array_equal(props,parent['group_proposal']),'same available groups')
    available=np.stack([(props==k).any(1)&(bc!=k) for k in range(7)],1);changing=props!=bc[:,None]
    profiles=audit_profiles(props,parent['local_audit_features'],parent['predicted_K'][pos],bc)
    entries=[];new_vectors=[]
    registry=json.loads((out/'candidate-registry.json').read_text());native=read_npz(out/registry['candidate_file'])
    require(len(registry['added_candidates'])==3 and registry['total_candidates']==44,'append once to verified initial candidates')
    require(digest(out/registry['candidate_file'])==registry['candidate_file_sha256'],'unchanged candidate memory before append')
    old_registry=json.loads((args.previous_memory/'registry.json').read_text())
    old_native=read_npz(args.previous_memory/'all-variant-decisions.npz')
    require(digest(args.previous_memory/'registry.json')==registry['previous_registry_sha256'],'previous registry unchanged')
    all_hashes={v['prediction_values_sha256']:v['variant_id'] for v in reversed(old_registry['variants'])}
    for v in registry['added_candidates']:all_hashes.setdefault(v['prediction_values_sha256'],v['variant_id'])
    variants=[('parent_risk_new_correction',parent,candidate),('new_risk_parent_correction',candidate,parent)]
    require(set(report['variants'])=={v[0] for v in variants},'both symmetric crossovers retained')
    for name,rs,qs in variants:
        r=rs['baseline_correct_probability'].astype(float)
        logits=np.column_stack([np.where(available,qs['pooled_class_logits'],-1e9),np.zeros(len(pos))]).astype(float)
        logits-=logits.max(1,keepdims=True);q=np.exp(logits);q/=q.sum(1,keepdims=True)
        cp=(1-r[:,None])*q[:,:7]+np.eye(7)[bc]*r[:,None];other=(1-r)*q[:,7]
        require(np.array_equal(data[name+'__baseline_correct'],r) and np.array_equal(data[name+'__pooled_class_logits'],qs['pooled_class_logits']),'fixed factor substitution')
        require(np.allclose(data[name+'__class_probability'],cp,atol=1e-12) and np.allclose(data[name+'__other_probability'],other,atol=1e-12),'independent probability reconstruction')
        prediction,_=choose((np.take_along_axis(cp,props,1)-r[:,None])*changing,props,bc)
        expected=b.copy();expected[pos]=prediction
        require(np.array_equal(expected,data[name+'_K']),'independent native decision reconstruction')
        entry=report['variants'][name]
        require(entry['paired']==named_pairs(y,b,expected) and entry['metrics']==metrics(y,expected),'native metrics')
        require(entry['versus_parent']==retention(y,b,parent['predicted_K'],expected),'parent paired retention')
        require(entry['versus_new']==retention(y,b,candidate['predicted_K'],expected),'new-model paired retention')
        require(entry['parent_profile_retention']==profile_retention(profiles,yc,bc,parent['predicted_K'][pos],prediction),'profile retention')
        summary=summarize(dict(truth=yc,baseline=bc,proposal=props,baseline_correct=r,pooled_class_logits=qs['pooled_class_logits'],class_probability=cp,other_probability=other))
        check_numbers(entry['confidence'],summary,'factor confidence '+name)
        vector=expected.astype(np.int8);sha=hashlib.sha256(np.ascontiguousarray(vector).tobytes()).hexdigest()
        key='catalogue8_'+name
        entries.append(dict(variant_id=key,parent_id='catalogue8_local',candidate_kind='fixed_neural_factor_composition',
            retained=True,retention_depends_on_global_gain=False,downstream_ready=False,
            required_for_downstream_activation='Regenerate both factors with every downstream receiver excluded from all producers and audits.',
            protocol_commit=args.protocol_commit,source_files=report['sources'],source_file='factor-swaps.npz',source_field=name+'_K',
            prediction_values_sha256=sha,identical_prediction_alias_of=all_hashes.get(sha),
            metrics=metrics(y,vector),versus_freeze=named_pairs(y,b,vector),versus_parent=named_pairs(y,parent['predicted_K'],vector)))
        all_hashes.setdefault(sha,key);new_vectors.append(vector)
    selected=candidate['predicted_K'][pos]!=bc
    fixed=dict(rows=int(selected.sum()),observed_baseline_correct=float((yc[selected]==bc[selected]).mean()),
        parent_risk_mean=float(parent['baseline_correct_probability'][selected].mean()),new_risk_mean=float(candidate['baseline_correct_probability'][selected].mean()))
    require(fixed==report['same_new_selected_cohort'],'same selected cohort risk comparison')
    previous_vectors=native['predictions'].copy();matrix=np.column_stack([previous_vectors,*new_vectors])
    keys=list(native['variant_ids'])+[e['variant_id'] for e in entries]
    require(len(set(keys))==5 and not set(keys)&set(old_native['variant_ids']),'unique appended identities')
    native.update(variant_ids=np.array(keys),predictions=matrix,
        outcomes_vs_freeze=(matrix==y[:,None]).astype(np.int8)-(b==y).astype(np.int8)[:,None])
    file=out/registry['candidate_file'];np.savez_compressed(file,**native)
    restored=read_npz(file);require(np.array_equal(restored['predictions'][:,:3],previous_vectors),'first three candidates unchanged')
    require(np.array_equal(restored['predictions'],matrix),'all appended decisions preserved')
    (out/'factor-swaps.npz').write_bytes((args.results/'factor-swaps.npz').read_bytes())
    (out/'factor-swaps-report.json').write_bytes((args.results/'report.json').read_bytes())
    combined=np.column_stack([old_native['predictions'],matrix]);union=(b!=y)&np.any(combined==y[:,None],axis=1)
    old_union=(b!=y)&np.any(old_native['predictions']==y[:,None],axis=1)
    registry['added_candidates']+=entries;registry.update(total_candidates=46,distinct_prediction_vectors=len(all_hashes),
        candidate_file_sha256=digest(file),factor_swap_file='factor-swaps.npz',factor_swap_file_sha256=digest(out/'factor-swaps.npz'),
        union_of_initial_errors_corrected=int(union.sum()),additional_errors_reached_by_appended_outputs=int((union&~old_union).sum()))
    (out/'candidate-registry.json').write_text(json.dumps(registry,indent=2,sort_keys=True)+'\n')
    evidence=dict(status='verified',protocol_commit=args.protocol_commit,source_files=report['sources'],
        predictions_sha256=digest(out/'factor-swaps.npz'),candidate_registry_sha256=digest(out/'candidate-registry.json'),
        both_crossovers_verified=True,all_previous_candidates_unchanged=True,total_candidates=46,
        variants=report['variants'],same_new_selected_cohort=fixed,independent_validation=False,promotion=False)
    (out/'factor-swaps-verification.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(status='verified',total_candidates=46,distinct_prediction_vectors=len(all_hashes),
        newly_covered_initial_errors=registry['additional_errors_reached_by_appended_outputs'],
        paired={k:v['paired']['global'] for k,v in report['variants'].items()})),flush=True)


if __name__=='__main__':main()
