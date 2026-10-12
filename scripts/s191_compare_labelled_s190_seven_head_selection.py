"""S191 read-only audit frozen independently labeled S190: diagnose classification gate.

Never fit/select a winner on the S190 held-out labels, and do not replace
original S80, whose 527D and S45 16-producer dependencies remain unavailable.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
PRETRAINED_SPECIALISTS=('H1_spectral','H2_lifecycle','H3_harmonics','H4_fundamentals')
GENERAL=('H6_log_energy_flow','H7_attack_morphology','H10_energy_source_coupling')
def need(v,m):
    if not v:raise ValueError(m)
def run(inp,out):
    need(not out.exists(),'immutable evaluation diagnostic')
    r=json.loads(inp.read_text())
    need(r['status'].startswith('S190_FROZEN_PRELABEL_'),'source report not genuine S190')
    names=r['original_7_expert_individual_comparators']
    need(set(names)==set(PRETRAINED_SPECIALISTS)|set(GENERAL),'original seven source identities')
    parent=r['S179_new_parent_adapted_selection']
    need(parent['total']==62 and parent['poly_true_rows']==11 and
         parent['correct']==43 and parent['poly_ExactK']==0,
         'independent S190 adapted source benchmark drifted')
    audit={}
    for n,d in names.items():
        need(d['total']==62 and d['poly_true_rows']==11,'unlike-for-like Expert evidence')
        category='FORCED_K2to6_NO_K0K1_TRAINING' if n in PRETRAINED_SPECIALISTS else 'ALL_K0to6_TRAINED'
        audit[n]={'support_contract':category,
          'global_ExactK':d['global_ExactK'],'global_correct':d['correct'],
          'poly_ExactK':d['poly_ExactK'],
          'poly_correct':sum(d['per_K_correct'][2:]),
          'K1_correct':d['per_K_correct'][1],
          'predicted_polyphonic_K_count':sum(d['pred_K0toK6'][2:])}
    result={'status':'S191_LABELLED_S190_SEVEN_ORIGINAL_S35_HEADS_DIAGNOSE_GATE_NOT_TUNING',
       'frozen_S190_report_SHA256':hashlib.sha256(inp.read_bytes()).hexdigest(),
       'same_original_true_pitch_62_matched_events':True,
       'true_K1_count':51,'true_polyK2K3K4_count':11,
       'adapted_parent_ExactK_global':parent['global_ExactK'],
       'adapted_parent_ExactK_poly':parent['poly_ExactK'],
       'individual_checkpoint_diagnostic':audit,
       'all_seven_source_trainable_as_original_S80':False,
       'specialists_not_trained_for_K0K1':True,
       'source_test_labels_now_exposed':True,
       'S190_labels_used_for_model_training':False,
       'any_new_cutoff_optimized':False,
       'source_scorer_replaced':False,
       'requires_new_unexposed_independent_cohort_for_any_changed_gate':True,
       'original_freeze_local_combo_verified':False,
       'original_16_S45_families_executed':False,
       'S80_promoted':False}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
def selftest():
    need(len(PRETRAINED_SPECIALISTS)==4 and len(GENERAL)==3,'source head census')
    print('S191 PASS class-support distinction; output is diagnostic not learned gate')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        need(a.source is not None and a.out is not None,'true postfreeze S190 report required')
        run(a.source,a.out)
