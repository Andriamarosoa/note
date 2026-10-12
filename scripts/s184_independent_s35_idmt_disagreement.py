"""S184 independent label-free disagreement/OOD audit across S183/S179.

This is a DESCRIPTION, not a learned poly rescue or a new S80 source.
Only read the historic 58D acoustic features and derived external S35 arrays;
never read target K from either corpus.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
NAMES=('H1_spectral','H2_lifecycle','H3_harmonics','H4_fundamentals',
       'H6_log_energy_flow','H7_attack_morphology','H10_energy_source_coupling')
def need(x,m):
    if not x:raise ValueError(m)
def strict(p,keys):
    with np.load(p,allow_pickle=False) as z:
        need(set(z.files)==set(keys),'annotated source disallowed: '+str(p))
        return {k:z[k].copy() for k in z.files}
def run(s183,s179,old,out):
    need(not out.exists(),'immutable report')
    x=strict(s183,('row_id','summary58','flow245','morphology490','physics294',
                   'sequence42x49','source_head_ids','source_probabilities',
                   'source_predicted_K'))
    a=strict(s179,('row_id','eligible_row_id','eligible_source_positions',
                   'new_parent_K','new_poly_eligible_K','new_member_source_ids',
                   'source_vote_K2to6','new_255_group_proposal_K',
                   'new_255_group_direct_features','new_255_group_probabilities_K2to6'))
    need(np.array_equal(x['row_id'],a['row_id']) and len(x['row_id'])==100,
         'S183/S179 newly extracted 100 external WAV events')
    need(tuple(map(str,x['source_head_ids']))==NAMES,'source specialist model drift')
    p=x['source_probabilities'].astype(np.float64)
    votes=p.argmax(axis=2)
    hist=np.asarray(a['new_parent_K'],int)
    need(np.array_equal(hist,p[:,[4,5,6]].mean(axis=1).argmax(1)),
         'independent S179 parent decision mismatch')
    need(np.array_equal(a['eligible_source_positions'],
                        np.flatnonzero(np.isin(hist,[2,3,4]))),
         'S179 K2-4 eligible mask drift')
    with np.load(old,allow_pickle=False) as z:
        need('features' in z.files and 'feature_names' in z.files,
             'original historical audio-only descriptor tensors absent')
        # Intentional whitelist: historic y/truth/baseline are NOT read.
        ref=np.asarray(z['features'],np.float32)
    need(ref.shape==(7493,58) and x['summary58'].shape==(100,58),
         'original 58D acoustic reference population mismatch')
    lo=np.quantile(ref,.01,axis=0);hi=np.quantile(ref,.99,axis=0)
    outside=(x['summary58']<lo)|(x['summary58']>hi)
    rates=outside.mean(axis=0)
    # H1-H4 are TRAINED ONLY on K2-K6: their K2plus predictions are compulsory,
    # so they are NOT independent evidence that a sound is polyphonic.
    poly_specialists=votes[:,:4]
    allclass=votes[:,4:]
    agreement=int(np.sum(np.all(poly_specialists==poly_specialists[:,[0]],axis=1)))
    allclass_above=int(np.sum((allclass>=2).sum(axis=1)>=2))
    disagreement=int(np.sum((hist<=1)&((poly_specialists>=3).sum(axis=1)>=3)))
    native_groups=a['new_255_group_proposal_K']
    b=a['new_poly_eligible_K']
    need(native_groups.shape==(len(b),255),'new 255 group proposal completeness')
    supported=np.sum(native_groups!=b[:,None],axis=1) if len(b) else np.array([])
    report={'status':'S184_INDEPENDENT_S35_CROSS_HEAD_CONFLICT_DOMAIN_SHIFT_NO_LABELS',
      'native_unseen_audio_events':100,'actual_pretrained_S35_acoustic_expert_heads':7,
      'historical_poly_only_heads':['H1_spectral','H2_lifecycle',
                                     'H3_harmonics','H4_fundamentals'],
      'K2plus_predictions_by_first_four_forced_by_training_support':True,
      'poly_specialists_full_agreement_rows':agreement,
      'three_allclass_heads_K2plus_majority_rows':allclass_above,
      'three_allclass_mean_K0_or_K1_rows':int(np.sum(hist<=1)),
      'allclass_mean_K2_K3_K4_rows':int(np.sum(np.isin(hist,[2,3,4]))),
      'general_vs_three_poly_expert_K3plus_conflict_rows':disagreement,
      'historical_acoustic_reference_rows':len(ref),
      'mean_fraction_of_58D_features_outside_old_1to99percentile':
           float(rates.mean()),
      'highest_58D_feature_shift_fractions':sorted([float(v) for v in rates],reverse=True)[:10],
      'new_255_groups_on_K234_events':int(len(b)*255),
      'new_255_groups_nonparent_count_by_eligible_event':supported.astype(int).tolist(),
      'old_255_producer_audit_9dim_available':False,
      'original_S45_S46_S80_scores':None,
      'S183_exposed_for_unlabeled_development_diagnostics':True,
      'holdout_independent_evaluation_must_use_disjoint_unseen_tracks':True,
      'XML_MIDI_new_true_K_read':False,
      'weights_or_thresholds_retrained':False,'S80_promoted':False}
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
def selftest():
    z=np.array([2,3,0,1]);need(np.isin(z,[2,3,4]).sum()==2,'eligibility rule')
    print('S184 PASS no label access and strict three all-K expert versus four K2+ specialist distinction')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('s183','s179','old','out'):p.add_argument('--'+n,type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        need(all(getattr(a,n) for n in ('s183','s179','old','out')),
             'three genuine no-label sources needed')
        run(a.s183,a.s179,a.old,a.out)
