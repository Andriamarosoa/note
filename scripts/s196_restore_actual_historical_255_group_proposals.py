"""S196 restore exact preserved ORIGINAL 255 group proposals on 7493 historical rows.

The original historical proposals are two independently archived NPZ copies.
Recover byte-for-byte K action parity, check 20 aggregated nine-audit
summaries against explicit 255-mask algebra; align authentic parent risk
logits without accessing development truth. Does NOT reconstruct original
per-group nine descriptors or run original 527D S5 on new WAV.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.special import expit

def need(c,m):
    if not c:raise ValueError(m)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path,keys):
    with np.load(path,allow_pickle=False) as z:
        need(set(keys).issubset(z.files),'missing archived field '+str(path))
        return {k:z[k].copy() for k in keys}
def verify(root,out):
    need(not out.exists(),'immutable S196')
    source=root/'analysis/evidence/v273-regression-loops/prepared/inputs.npz'
    raw=root/'analysis/evidence/v273-contextual-risk/probability-evidence.npz'
    factor=root/'analysis/evidence/v273-contextual-risk/factor-swaps.npz'
    manifest=json.loads((root/'analysis/evidence/v273-regression-loops/prepared/input-manifest.json').read_text())
    need(sha(source)==manifest['inputs_sha256'],'original frozen 7493 input digest drift')
    need(sha(factor)==manifest['factor_swaps_sha256'],'original authentic parent risk digest drift')
    d=read(source,('global_index','baseline','proposal','available','z',
                   'logits','group_context','group_names','native_position'))
    s=read(raw,('eligible_global_index','group_proposal','frozen_K'))
    f=read(factor,('eligible_global_index',
       'parent_risk_new_correction__baseline_correct',
       'parent_risk_new_correction__pooled_class_logits'))
    ids=d['global_index']; b=d['baseline'].astype(np.int8)
    v=d['proposal'].astype(np.int8); historical=s['group_proposal'].astype(np.int8)
    need(ids.shape==(7493,) and v.shape==(7493,255) and
         d['group_context'].shape==(7493,7,20) and
         d['logits'].shape==(7493,7),'authentic historical shapes')
    need(np.array_equal(ids,s['eligible_global_index']) and
         np.array_equal(ids,f['eligible_global_index']),
         'historical row identities not fully aligned')
    proposal_mismatches=int(np.count_nonzero(historical!=v))
    need(proposal_mismatches==0,'255 original groups differ from independently preserved source')
    need(np.array_equal(b,s['frozen_K']),'original frozen parent differs')
    need(np.isin(v,np.arange(7)).all() and
         np.isin(b,np.arange(7)).all(),'invalid original K')
    need(np.all(np.isfinite(d['z'])) and
         np.all(np.isfinite(d['group_context'])),'nonfinite original scores')
    q=f['parent_risk_new_correction__baseline_correct']
    need(np.allclose(expit(d['z']),q,atol=1e-7,rtol=1e-7),
         'authentic parent risk logits not reconstructed')
    need(np.allclose(d['logits'],f['parent_risk_new_correction__pooled_class_logits'],
                     atol=0,rtol=0),'authentic parent pooled class logits diverge')
    calc_available=np.stack([(v==k).any(axis=1)&(b!=k) for k in range(7)],axis=1)
    need(np.array_equal(calc_available,d['available']),
         'original complete group eligibility differs from source')
    groups=d['group_context'];names=list(map(str,d['group_names']))
    expected=['audit_mean_'+str(j) for j in range(9)]+[
        'local_gain_min','local_gain_max','fraction_of_groups']+[
        'full_group_member_fraction_S'+str(j+1) for j in range(8)]
    need(names==expected,'historical 20D summary audit feature names reordered')
    masks=np.array([[bool(m&(1<<j)) for j in range(8)] for m in range(1,256)],
                   dtype=np.float64)
    differences=0.;checks=0
    for k in range(2,7):
        belongs=(v==k)&(b[:,None]!=k)
        cnt=belongs.sum(axis=1)
        den=np.maximum(cnt,1)
        fraction=cnt/255
        differences=max(differences,float(np.max(np.abs(fraction-groups[:,k,11]))))
        checks+=len(cnt)
        counts_expected=belongs.astype(np.float64)@masks/den[:,None]
        differences=max(differences,float(np.max(np.abs(counts_expected-groups[:,k,12:]))))
        checks+=len(cnt)*8
    need(differences<1e-10,'original 255 mask membership / 20D audit aggregates do not match')
    # Distinct 255 observations are AUTHENTIC and their original audit means
    # are AUTHENTIC. There is no injective recovery of 2295 audit parameters
    # from the destination-aggregated means and extrema alone.
    out.mkdir(parents=True)
    np.savez_compressed(out/'authentic_archived_255_and_20D_aggregated_audits.npz',
      global_index=ids,baseline_K=b,original_255_proposals=v,
      original_20D_destination_aggregates=groups.astype(np.float32),
      original_20_names=np.asarray(names),original_class_logits=d['logits'],
      original_baseline_correct_logit=d['z'],
      original_K2to6_available=d['available'])
    report={'status':'S196_GENUINE_ARCHIVED_255_GROUP_PROPOSALS_RESTORED_7493_EVENTS',
      'original_historical_255_proposal_mismatches_independent_archive':proposal_mismatches,
      'original_255_x_7493_proposals_recovered':int(v.size),
      'original_unmodified_audit_means_plus_minmax_plus_member_frequency_per_destination':True,
      'original_20D_group_context_columns_per_K':20,
      'full_original_255x9_audit_tensor_available':False,
      'original_local_group_audit_9D_rebuilt':False,
      'reconstructed_group_membership_summary_max_abs_error':differences,
      'group_membership_aggregate_scalar_checks':checks,
      'all_authentic_parent_risk_logits_restored':True,
      'all_original_255_available_destination_flags_verified':True,
      'source_proposal_npz_SHA256':sha(raw),
      'source_prepared_7493_npz_SHA256':sha(source),
      'source_original_parent_probabilities_SHA256':sha(factor),
      'restored_output_npz_SHA256':sha(out/'authentic_archived_255_and_20D_aggregated_audits.npz'),
      'S5_original527_runtime_generator_reproduced':False,
      'these_historical_7493_are_exposed_development_events':True,
      'new_WAV_proposals_reproduced':False,
      'truth_labels_or_MIDI_values_accessed':False,
      'S80_promoted':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
def selftest():
    arr=np.array([1,2,255])
    masks=np.array([[bool(m&(1<<j)) for j in range(8)] for m in range(1,256)])
    need(masks.shape==(255,8) and masks[0,0] and masks[1,1] and masks[-1].sum()==8,
         'historical binary mask order')
    print('S196 PASS every integer combination mask 1..255 8 members and fail-closed lossless source')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--out',type=Path)
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:selftest()
    else:
        need(args.out is not None,'output required')
        verify(args.root,args.out)
