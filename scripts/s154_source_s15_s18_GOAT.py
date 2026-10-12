"""S154 original trained S15->S16->S17->S18 chain on GOAT, from adapted S9.

Original S13/S14 heldpiece risk checkpoints, original S15 safe_OR,
original S16/17/18 heldpiece risk checkpoints and selected original
source policy names. ONLY S9/S5 upstream is adapted (PR16#22).
Original 324 fixed gates, all score labels prohibited at runtime.
Comparison to S152 is WITHOUT GOAT MIDI. Original S80 not promoted.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scripts.s141_external_original_s40_cnn import load_input
from scripts.s152_original_s16_s18_cascade_GOAT import (
 probabilities,SOURCE_PARENT,DIRECTIONS)
from scripts.loop_v273_k12_k23 import make_policies as s16pol
from scripts.loop_v273_k32_k34 import make_policies as s17pol
from scripts.loop_v273_k43_k21 import make_policies as s18pol
def need(x,m):
    if not x:raise ValueError(m)
def run(s97,s85,s135,s15,s16,s17,s18,s152,out):
    need(not out.exists(),'immutable new experiment outputs')
    ids,X,_,originals=load_input(s97,s85,s135)
    b=originals['official_freeze'].astype(np.int8)
    with np.load(s15,allow_pickle=False) as z:
        need(set(z.files)=={'row_id','freeze_K','adapted_S9_K',
              'S13_safe_K','S14_safe_K','S15_safe_OR_K','S15_policy_ids',
              'S15_predictions','S13_policy_ids','S13_predictions',
              'S14_policy_ids','S14_predictions'},
              'strict unlabeled S153 saved input')
        need(np.array_equal(ids,z['row_id'].astype(str)) and
             np.array_equal(b,z['freeze_K']), 'S15 native alignment')
        s15parent=z['S15_safe_OR_K'].astype(np.int8)
        s9=z['adapted_S9_K'].astype(np.int8)
        need('series15__safe_OR' in set(map(str,z['S15_policy_ids'])),
             'S15 policy source fixed')
    with np.load(s152,allow_pickle=False) as z:
        need(set(z.files)=={'row_id','freeze_K','adapted_S11_initial_K',
           'S16_K','S17_K','S18_K'},'S152 baseline schema')
        need(np.array_equal(ids,z['row_id'].astype(str)) and
             np.array_equal(b,z['freeze_K']),'S152 native alignment')
        previous=z['S18_K'].astype(np.int8)
    Xviews={'summary':X[:,:90],'flow':X}
    folders={'S16':s16,'S17':s17,'S18':s18}
    decoders={'S16':s16pol,'S17':s17pol,'S18':s18pol}
    outvec={'freeze_K':b,'adapted_S9_K':s9,'adapted_S15_K':s15parent}
    parent=s15parent.copy()
    stats={};count=0;all_hashes=[]
    for stage in ('S16','S17','S18'):
        scores,records=probabilities(folders[stage],Xviews,DIRECTIONS[stage])
        policies,scope=decoders[stage](b,parent,scores)
        need(len(policies)==110 and SOURCE_PARENT[stage] in policies,
             'original 108 veto decisions missing')
        chosen=policies[SOURCE_PARENT[stage]].astype(np.int8)
        outvec[stage+'_K']=chosen.copy()
        count+=len(records);all_hashes.extend({'stage':stage,**r} for r in records)
        stats[stage]={
         'original_trained_checkpoint_count':len(records),
         'original_108_policies_executed':True,
         'chosen_original_historic_policy':SOURCE_PARENT[stage],
         'eligible_source_transitions':{k:int(v.sum()) for k,v in scope.items()},
         'changed_from_parent':int((chosen!=parent).sum()),
         'K0_to_K6':np.bincount(chosen,minlength=7).tolist()}
        print(json.dumps({'stage':stage,**stats[stage]}),flush=True)
        parent=chosen
    need(count>=12,'original S16/S17/S18 weights missing')
    out.mkdir(parents=True)
    np.savez_compressed(out/'original_S15_to_S18_trained_GOAT_adapted_S9.npz',
        row_id=ids,**outvec)
    report={
      'status':'S154_ORIGINAL_S13_S14_S15_S16_S17_S18_CONNECTED_ADAPTED_S9',
      'GOAT_new_audio_groups':57,
      'genuine_original_checkpoint_count_S16_to_S18':count,
      'trained_checkpoint_hashes':all_hashes,
      'stages':stats,
      'new_S15_to_S18_final_K_histogram':np.bincount(parent,minlength=7).tolist(),
      'new_S15_to_S18_delta_vs_frozen_reference':int((parent!=b).sum()),
      'new_S15_to_S18_changed_rows_vs_previous_S152':int((parent!=previous).sum()),
      'original_S13_S14_K0_chain_connected':True,
      'original_S15_S16_S17_S18_source_policy_names_applied':True,
      'only_remains_fundamentally_adapted_upstream_S9_S5':True,
      'authentic_historical_S80_equivalence':False,
      'S45_16_families_runtime_complete':False,
      'new_music_poly_ExactK_measured':False,
      'MIDI_truth_read':False,'promotion':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in (
      'status','GOAT_new_audio_groups','genuine_original_checkpoint_count_S16_to_S18',
      'new_S15_to_S18_final_K_histogram',
      'new_S15_to_S18_changed_rows_vs_previous_S152',
      'only_remains_fundamentally_adapted_upstream_S9_S5','promotion')},indent=2))
def selftest():
    need(SOURCE_PARENT['S16']=='series16__flow_logistic__k23__pbase_gt0.99',
         'S16 source parent name drift')
    need(SOURCE_PARENT['S17']=='series17__flow_logistic__k32__pbase_gt0.99',
         'S17 source parent drift')
    need(SOURCE_PARENT['S18']=='series18__flow_logistic__k21__pbase_gt0.95',
         'S18 source parent drift')
    print('S154 PASS original S15 input + chained S16/S17/S18 policy names')
if __name__=='__main__':
    a=argparse.ArgumentParser()
    for name in ('s97','s85','s135','s15','s16','s17','s18','s152','out'):
        a.add_argument('--'+name,type=Path)
    a.add_argument('--self-test',action='store_true')
    p=a.parse_args()
    if p.self_test:selftest()
    else:
        need(all(getattr(p,n) is not None for n in
             ('s97','s85','s135','s15','s16','s17','s18','s152','out')),
             'missing archived source')
        run(p.s97,p.s85,p.s135,p.s15,p.s16,p.s17,p.s18,p.s152,p.out)
