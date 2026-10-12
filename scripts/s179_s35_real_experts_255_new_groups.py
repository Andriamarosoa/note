"""S179 NEW DEPLOYMENT candidate adapter: 255 deterministic groups from real pretrained S35 expert outputs.

Not the historical 8 producer heads. K2-K4 eligible only; no 9 local
outcome audits, no inherited S1-S4 original model inference and no S80 scoring.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scripts.v273_catalogue_contract import joint_groups,group_masks,direct_dim
def need(b,s):
    if not b:raise ValueError(s)
HISTORICAL_HEADS=('H1_spectral','H2_lifecycle','H3_harmonics',
                  'H4_fundamentals','H6_log_energy_flow',
                  'H7_attack_morphology','H10_energy_source_coupling')
GENERAL=('H6_log_energy_flow','H7_attack_morphology','H10_energy_source_coupling')
def build(src,out):
    need(not out.exists(),'immutable S179 experiment')
    with np.load(src,allow_pickle=False) as z:
        need(set(z.files)=={'row_id','summary58','flow245','morphology490',
            'physics294','sequence42x49','source_head_ids',
            'source_probabilities','source_predicted_K'},
            'only original S178 WAV-derived data accepted')
        ids=z['row_id'].copy()
        names=tuple(map(str,z['source_head_ids']))
        values=z['source_probabilities'].copy().astype(np.float32)
    need(names==HISTORICAL_HEADS and values.shape==(len(ids),7,7),
         'original seven-source trained heads drifted')
    need(np.isfinite(values).all() and np.allclose(values.sum(2),1,atol=1e-4),
         'not calibrated probabilities')
    p={name:values[:,i] for i,name in enumerate(names)}
    majority=np.mean([p[h] for h in GENERAL],axis=0)
    base=majority.argmax(1).astype(np.int8)
    eligible=np.flatnonzero(np.isin(base,[2,3,4]))
    need(len(ids)>0 and len(set(map(str,ids)))==len(ids),'bad unseen audio identities')
    # 8 votes = deterministic K2-K4 parent + seven actual S35 pretrained audio experts.
    # The five-class normalization is a NEW adapter, not a historical source.
    vote=np.zeros((len(eligible),8,5),np.float32)
    if len(eligible):
        vote[np.arange(len(eligible)),0,base[eligible]-2]=1.
        for j,h in enumerate(names,1):
            raw=p[h][eligible,2:7]
            denom=raw.sum(1,keepdims=True)
            need(np.all(denom>0),'unrepresented polyphonic class mass')
            vote[:,j]=raw/denom
        proposals,direct,mean=joint_groups(vote,base[eligible])
        need(proposals.shape==(len(eligible),255) and
             direct.shape==(len(eligible),255,direct_dim(8)) and
             mean.shape==(len(eligible),255,5),'8 specialist 255 subset contract')
        need(np.isin(proposals,range(2,7)).all(),
             'new subset K support changed')
    else:
        proposals=np.empty((0,255),np.int8)
        direct=np.empty((0,255,direct_dim(8)),np.float32)
        mean=np.empty((0,255,5),np.float32)
    out.mkdir(parents=True)
    np.savez_compressed(out/'S179_new_8vote_255group_audio_only.npz',
        row_id=ids,eligible_row_id=ids[eligible],eligible_source_positions=eligible,
        new_parent_K=base,new_poly_eligible_K=base[eligible],
        new_member_source_ids=np.asarray(('new_H0_parent',)+names),
        source_vote_K2to6=vote,
        new_255_group_proposal_K=proposals.astype(np.int8),
        new_255_group_direct_features=direct,
        new_255_group_probabilities_K2to6=mean)
    hist=np.bincount(base,minlength=7).tolist()
    diversity=np.unique(proposals,axis=1).shape[1] if len(eligible) else 0
    report={'status':'S179_NEW_S35_AUDIO_POLY_255_GROUP_ADAPTER_RESEARCH_ONLY',
      'source_S178_SHA256':hashlib.sha256(src.read_bytes()).hexdigest(),
      'original_historical_eight_source_equivalence':False,
      'frozen_real_S35_audio_experts':7,
      'new_extra_H0_parent_K2K4_prior':True,
      'new_poly_prior_from_three_S35_allclass_experts':True,
      'native_external_audio_events':len(ids),
      'new_parent_prediction_histogram_K0to6':hist,
      'new_S35_expert_poly_eligible_rows':len(eligible),
      '255_group_proposals_on_poly_events':True,
      'mean_distinct_255_group_destinations_per_event':float(np.mean([len(set(row)) for row in proposals])) if len(eligible) else 0.,
      'old_255_source_model_proposals_reproduced':False,
      'historical_nine_audit_MN_statistics_rebuilt':False,
      'historical_source_risk_logits_available':False,
      'original_S1_S4_old_calibration_executable_on_new_inputs':False,
      'original_S45_source_16_families':False,
      'experiment_corpus_annotations_read':False,
      'S80_independent_ExactK_measured':False,
      'S80_promoted':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
def selftest():
    v=np.zeros((3,8,5),np.float32)
    v[:,:,0]=1.
    p,g,m=joint_groups(v,np.array([2,3,4],np.int8))
    need(p.shape==(3,255) and g.shape==(3,255,61) and m.shape==(3,255,5),
         'historical joint_groups pure formula changed')
    need(group_masks(8).shape==(255,8),'all eight head subsets')
    print('S179 PASS exact deterministic all 255 group formula; new-source identity explicit')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        need(a.source and a.out,'native S178 source and output mandatory')
        build(a.source,a.out)
