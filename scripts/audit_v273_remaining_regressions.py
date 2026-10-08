"""Post-hoc descriptive profiles and three retained, label-free audit gates."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.evaluate_v273_regression_loops import read,dump,policy_stats
from scripts.verify_v273_selector_repair_artifacts import digest,require
from scripts.v273_contextual_risk_contract import retention

PARENTS=((3,'tree_group_audit__blend_0.25__cost_1'),(1,'group_audit__cost_1.15'),(2,'source_count90'))


def profiles(group_context,prediction,baseline):
    a=group_context[np.arange(len(baseline)),prediction];take=prediction!=baseline
    result=np.full(len(baseline),'keep',dtype='<U32');result[take]='tied'
    result[take&(a[:,9]<0)&(a[:,10]>0)]='mixed'
    result[take&(a[:,9]<0)&(a[:,10]<=0)]='regression_favored'
    result[take&(a[:,9]>=0)&(a[:,10]>0)]='correction_favored'
    result[take&(a[:,7]<=0)]='unsupported'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),'refusing to replace a post-hoc audit');a.output.mkdir(parents=True)
    data=read(a.evidence/'prepared/inputs.npz');prior=read(a.evidence/'all-candidate-decisions.npz')
    registry=json.loads((a.evidence/'candidate-registry.json').read_text());require(registry['total_candidates']==208,'prior memory count')
    y,b=data['truth'],data['baseline'];reports={};predictions=[];keys=[];rows=[]
    for series,key in PARENTS:
        d=read(a.evidence/f'series{series}'/'predictions.npz');at=list(d['variant_ids']).index(key)
        parent=d['predictions'][data['native_position'],at]
        profile=profiles(data['group_context'],parent,b)
        blocked=profile=='regression_favored'
        candidate=np.where(blocked,b,parent)
        # Independently express the gate through observable group extrema.
        g=data['group_context'][np.arange(len(b)),parent]
        direct=(parent!=b)&(g[:,7]>0)&(g[:,9]<0)&~(g[:,10]>0)
        require(np.array_equal(direct,blocked),'independent observable gate')
        require(not np.any((candidate!=b)&(candidate!=parent)),'only cancellation allowed')
        summary=policy_stats(data,candidate);paired=retention(y,b,parent,candidate)
        require(paired['corrections_added']==paired['regressions_added']==0,'no invented changes')
        identity='regression_loops_posthoc__'+key+'__no_favorable_group'
        keys.append(identity);predictions.append(candidate)
        profiles_table={}
        for name in np.unique(profile):
            if name=='keep':continue
            take=profile==name
            profiles_table[str(name)]=dict(rows=int(take.sum()),corrections=int((take&(parent==y)).sum()),
                regressions=int((take&(b==y)).sum()),neutral=int((take&(parent!=y)&(b!=y)).sum()))
        reports[identity]=dict(parent_series=series,parent_key=key,parent_profiles=profiles_table,
            gated=summary,versus_parent=paired,labels_used_to_construct_gate=False,posthoc=True,
            retained=True,downstream_ready=False)
        for i in np.flatnonzero(parent!=b):
            rows.append(dict(variant=identity,global_index=int(data['global_index'][i]),fold=int(data['fold'][i]),
                piece=str(data['piece'][i]),recording_id=str(data['recording_id'][i]),start_sample=int(data['start_sample'][i]),
                truth=int(y[i]),baseline=int(b[i]),parent=int(parent[i]),candidate=int(candidate[i]),
                profile=str(profile[i]),local_gain_min=float(g[i,9]),local_gain_max=float(g[i,10]),
                matching_support_mean=float(g[i,7]),blocked=bool(blocked[i]),
                effect_vs_parent=int(candidate[i]==y[i])-int(parent[i]==y[i])))
    with (a.output/'cases.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    matrix=np.broadcast_to(data['native_baseline'][:,None],(59309,3)).copy()
    matrix[data['native_position']]=np.column_stack(predictions)
    np.savez_compressed(a.output/'appended-candidates.npz',global_index=data['native_global_index'],true_K=data['native_truth'],
        frozen_baseline_K=data['native_baseline'],fold=data['native_fold'],eligible_global_index=data['global_index'],
        variant_ids=np.array(keys),predictions=matrix.astype(np.int8),
        outcomes_vs_freeze=(matrix==data['native_truth'][:,None]).astype(np.int8)-
                           (data['native_baseline']==data['native_truth'])[:,None].astype(np.int8))
    combined=np.column_stack([prior['predictions'],matrix]);distinct=len({hashlib.sha256(combined[:,j].astype(np.int8).tobytes()).hexdigest() for j in range(211)})
    output_registry=dict(previous_registry='../candidate-registry.json',previous_registry_sha256=digest(a.evidence/'candidate-registry.json'),
        previous_memory_sha256=digest(a.evidence/'all-candidate-decisions.npz'),previous_candidates=208,total_candidates=211,
        distinct_prediction_vectors=distinct,added_candidates=keys,retained=True,posthoc=True,
        candidate_file='appended-candidates.npz',candidate_file_sha256=digest(a.output/'appended-candidates.npz'),
        independent_validation=False,promotion=False,downstream_ready=False)
    dump(a.output/'candidate-registry.json',output_registry)
    dump(a.output/'report.json',dict(status='verified',scope='Post-hoc gate diagnostic after the four fitted series; not independent validation.',
        policies=reports,prior_208_columns_unchanged=True,total_candidates=211,distinct_prediction_vectors=distinct,
        all_decisions_replayed_from_label_free_observables=True,files={p.name:digest(p) for p in a.output.iterdir()}))
    print(json.dumps({k:dict(result=v['gated']['eligible'],saved=v['versus_parent']['regressions_avoided'],
        lost=v['versus_parent']['corrections_lost'],net_extra=v['versus_parent']['paired']['global']['net']) for k,v in reports.items()},indent=2),flush=True)


if __name__=='__main__':main()
