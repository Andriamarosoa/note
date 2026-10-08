"""Independently check that the durable memory loses no registered event or variant."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.verify_v273_selector_repair_artifacts import digest, metrics, named_pairs, require


def read(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--memory',type=Path,required=True)
    p.add_argument('--artifacts-root',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    r=json.loads((args.memory/'registry.json').read_text())
    for name, info in r['files'].items():
        require(digest(args.memory/name)==info['sha256'],'memory file identity '+name)
    d=read(args.memory/'all-variant-decisions.npz')
    ids,y,b,f,eligible=(d[k] for k in ['global_index','true_K','frozen_baseline_K','fold','eligible_global_index'])
    predictions=d['predictions'];entries=r['variants'];keys=[e['variant_id'] for e in entries]
    require(len(set(ids))==len(ids) and len(set(keys))==len(keys),'unique identities')
    require(np.array_equal(d['variant_ids'],keys),'candidate order')
    require(predictions.shape==(len(ids),len(entries)) and np.isin(predictions,range(7)).all(),'matrix shape and actions')
    require(np.array_equal(d['outcomes_vs_freeze'],(predictions==y[:,None]).astype(np.int8)-(b==y).astype(np.int8)[:,None]),'every outcome retained')
    source_columns=0
    for name, info in r['source_files'].items():
        source=args.artifacts_root/info['path'];require(digest(source)==info['sha256'],'source identity '+name)
        original=read(source)
        for key in ('global_index','true_K','frozen_baseline_K','fold'):
            require(np.array_equal(original[key],d[key]),'source native alignment')
        require(set(original['eligible_global_index'])==set(eligible),'eligible alignment')
        expected={'predicted_K'}|{key for key in original if key.endswith('_K') and key not in ('true_K','frozen_baseline_K','predicted_K','original_K') and original[key].shape==y.shape}
        actual=[e for e in entries if e['source']==name]
        require({e['source_field'] for e in actual}==expected and len(actual)==len(expected),'all native diagnostic outputs retained')
        for e in actual:
            require(np.array_equal(predictions[:,keys.index(e['variant_id'])],original[e['source_field']]),'unchanged source prediction')
        source_columns+=len(expected)
    require(source_columns==len(entries),'no undocumented column')
    aliases={}
    for j,e in enumerate(entries):
        v=predictions[:,j];parent=b if e['parent_id'] is None else predictions[:,keys.index(e['parent_id'])]
        sha=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
        require(e['prediction_values_sha256']==sha and e['identical_prediction_alias_of']==aliases.get(sha),'prediction alias and fingerprint')
        aliases.setdefault(sha,e['variant_id'])
        require(e['metrics']==metrics(y,v) and e['versus_freeze']==named_pairs(y,b,v) and e['versus_parent']==named_pairs(y,parent,v),'all paired metrics')
        require(e['retained'] and not e['retention_depends_on_global_gain'] and not e['downstream_ready'],'retention and activation contract')
    original_rows={};source_hashes={}
    for path in sorted(args.features.rglob('rows.jsonl')):
        for line in path.read_text().splitlines():
            row=json.loads(line);g=row['global_index'];require(g not in original_rows,'original context uniqueness');original_rows[g]=row
        source_hashes[int(row['fold'])]=digest(path)
    require({v['fold']:v['sha256'] for v in r['context_sources']}==source_hashes,'raw context source hashes')
    require(set(eligible)==set(original_rows),'raw context coverage')
    joined={};native_index={int(g):i for i,g in enumerate(ids)}
    for file in sorted(args.memory.glob('context-*.npz')):
        part=read(file)
        require(np.array_equal(part['feature_names'],r['context_features']),'feature name order')
        require(part['values'].dtype==np.float64,'no feature precision lost')
        require(len(part['global_index'])==r['files'][file.name]['rows'],'context row count')
        for i,g in enumerate(part['global_index']):
            g=int(g);require(g not in joined,'durable context uniqueness');row=original_rows[g];at=native_index[g]
            require(set(row['features'])==set(r['context_features']),'all primary features retained')
            require(np.array_equal(part['values'][i],np.array([row['features'][n] for n in r['context_features']])),'unchanged acoustic features')
            require((part['recording_id'][i],int(part['start_sample'][i]))==(row['recording_id'],row['start_sample']),'recording coordinates')
            require((y[at],b[at],f[at])==(row['true_k'],row['baseline_k'],row['fold']),'context label alignment')
            joined[g]=row
    require(set(joined)==set(eligible),'complete durable context')
    union=(b!=y)&np.any(predictions==y[:,None],axis=1)
    with (args.memory/'complementary-cases.csv').open() as handle:cases=list(csv.DictReader(handle))
    require(len(cases)==int(union.sum())==r['correctable_by_union'],'all complementary cases')
    require({int(c['global_index']) for c in cases}==set(map(int,ids[union])),'complementary identities')
    for c in cases:
        at=native_index[int(c['global_index'])]
        require(c['variants_correct']=='|'.join(map(str,np.flatnonzero(predictions[at]==y[at]))),'all successful candidates for each case')
        require(c['variants_wrong']=='|'.join(map(str,np.flatnonzero(predictions[at]!=y[at]))),'all failed candidates for each case')
    for c in r['paired_complements']:
        a,z=(predictions[:,keys.index(c[k])] for k in ('left','right'))
        ca,cz=(b!=y)&(a==y),(b!=y)&(z==y)
        require(c['paired']==named_pairs(y,a,z),'paired complement')
        require([c[k] for k in ['corrections_left_only','corrections_right_only','corrections_both','corrections_union']]==[int(x.sum()) for x in [ca&~cz,cz&~ca,ca&cz,ca|cz]],'complement counts')
    require(not r['promotion'] and not r['independent_validation'] and r['oracle_union_is_not_a_prediction'],'development evidence only')
    require(r['distinct_prediction_vectors']==len(aliases),'distinct predictions')
    family=['catalogue8_global','catalogue8_calibrated','catalogue8_local']
    family_columns=[keys.index(k) for k in family]
    family_evidence=dict(variants=family,union_of_initial_errors_corrected=int(((b!=y)&np.any(predictions[:,family_columns]==y[:,None],axis=1)).sum()),
        unique_corrections_within_family={k:int(((b!=y)&(predictions[:,j]==y)&~np.any(predictions[:,[other for other in family_columns if other!=j]]==y[:,None],axis=1)).sum()) for k,j in zip(family,family_columns)},
        oracle_diagnostic_not_prediction=True)
    proof=dict(status='verified',registry_sha256=digest(args.memory/'registry.json'),source_archives=len(r['source_files']),
        retained_policy_candidates=len(entries),distinct_prediction_vectors=len(aliases),native_rows=len(ids),
        context_rows=len(joined),context_features=r['context_feature_count'],unchanged_primary_feature_values=len(joined)*r['context_feature_count'],
        complementary_cases=len(cases),negative_net_candidates_preserved=[e['variant_id'] for e in entries if e['versus_freeze']['global']['net']<0],
        catalogue8_family_complements=family_evidence,
        all_source_and_diagnostic_outputs_preserved=True,all_outcomes_recomputed=True,all_primary_features_bitwise_equal=True,
        independent_validation=False,oracle_union_not_prediction=True,downstream_ready=False)
    args.output.write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n')
    print(json.dumps(proof),flush=True)


if __name__=='__main__':
    main()
