"""S155: corrected fail-closed original S8/S9->S13/S14->S15->S16->S17->S18 chain; S11 parallel."""
import argparse,hashlib,json
from pathlib import Path
STATES={'verified_new_audio','adapted_new_audio','historical_only',
 'rejected_on_exposed_smoke','missing_new_audio','logic_ready_inputs_missing',
 'pending_independent','original_weights_new_audio_with_explicit_fold_aggregation'}
EXPECTED=(
 "S1_acoustic","S2_source","S3_repair","S4_repair",
 "historical_repair_coherent","S25_Bfirst","S24_Bfirst",
 "S40_temporal_CNN","S19_recurrent_ABA","S11_votes_time_audio",
 "S38_morphology","S12_audio_logistic","S20_repeated_ABA",
 "S27_early_stop","S41_neural_fusion","S35_multiroute")
def need(x,msg):
    if not x:raise ValueError(msg)
RECOVERED='original_weights_new_audio_with_explicit_fold_aggregation'
def verify_recovered_evidence(status,root):
    folder=root/'analysis/evidence/v273-recovered-s5-s18'
    s5=json.loads((folder/'s5_report.json').read_text())
    s18=json.loads((folder/'s18_report.json').read_text())
    need(status.get('original_S5_527D_reconstructed') is True and
         status.get('original_S5_forward_reproduced_on_new_audio') is True,
         'recovered node without matching top-level proof')
    need(s5['historical_constructor_exact'] and s5['historical_rows']==59309 and
         s5['historical_parent_decisions_checked']==118618 and
         s5['historical_parent_mismatches']==0 and s5['external_events']==57,
         'S5 historical/new-audio evidence incomplete')
    need({c['fold'] for c in s5['checkpoints']}=={0,1,2,4} and len(s5['checkpoints'])==4,
         'original four S5 checkpoints incomplete')
    stages=s18['historical_selected_stages_verified']
    need(len(stages)==6 and {s['stage'] for s in stages}==
         {'S8','S13','S14','S16','S17','S18'} and
         all(s['historical_rows']==59309 and s['historical_K_mismatches']==0 for s in stages),
         'selected historical cascade proof incomplete')
    need(s18['original_selected_checkpoints_executed']==133 and
         len(s18['checkpoints'])==133 and s18['external_events']==57 and
         s18['external_branches']==[0,1,2,4] and s18['new_piece_aggregation_is_new_policy'],
         'original cascade checkpoint/deployment contract mismatch')
    need(not s18['S135_or_PR16_substitutes_used'] and
         not s18['full_external_S80_executed'] and
         not s18['full_S45_16_families_complete'],
         'recovery falsely claims completed S80')
    for report in (s5,s18):
        need(not report['truth_labels_read'] and not report['models_retrained'] and
             not report['independent_accuracy_measured'] and not report['S80_promoted'],
             'recovery proof changed label/training/promotion scope')
    for name,expected in (
        ('external_s5.npz',s5['external_predictions_sha256']),
        ('external_s5_to_s18.npz',s18['external_prediction_sha256'])):
        need(hashlib.sha256((folder/name).read_bytes()).hexdigest()==expected,
             'saved external prediction digest mismatch: '+name)
    return True

def validate(d,status,root):
    families=d['families'];nodes=d['nodes']
    need(len(families)==16 and set(f['id'] for f in families)==set(EXPECTED),
         'historical family missing or duplicate')
    need(len(nodes)==28 and len({n['id'] for n in nodes})==28,
         'missing or duplicate source node')
    bank={n['id']:n for n in nodes}
    for n in nodes:
        need(n['state'] in STATES and bool(n['proof']),n['id']+' state/proof')
        need(all(x in bank for x in n['requires']),'unlisted dependency in '+n['id'])
    ordered=[];stack=[]
    def visit(key):
        need(key not in stack,'cycle in S80 ancestry '+key)
        if key in ordered:return
        stack.append(key)
        for parent in bank[key]['requires']:visit(parent)
        stack.pop()
        ordered.append(key)
    for key in bank:visit(key)
    need(ordered[-1]=='promote_S80' and
         bank['promote_S80']['requires']==['independent_ExactK'],
         'incomplete promotion chain')
    recovered=any(n['state']==RECOVERED for n in nodes)
    if recovered:verify_recovered_evidence(status,root)
    ancestry={
      'S8_original_parent':['S5_original527','wav_groups'],
      'S9_original_parent':['S8_original_parent'],
      'S13_K0_birth':['S9_original_parent','wav_groups'],
      'S14_K0_rollback':['S9_original_parent','wav_groups'],
      'S15_safe_OR':['S13_K0_birth','S14_K0_rollback'],
      'S16_K12_K23':['S15_safe_OR'],
      'S17_K32_K34':['S16_K12_K23'],
      'S18_parent':['S17_K32_K34']
    }
    for id,expected in ancestry.items():
        need(bank[id]['requires']==expected, id+' original ancestry drift')
        if id not in ('S8_original_parent','S9_original_parent'):
            need(bank[id]['state'] in ({'adapted_new_audio',RECOVERED} if recovered else {'adapted_new_audio'}),id+' original parents not proven')
    need('S11_original_votes' not in bank['S18_parent']['requires'],
         'S11 incorrectly represented as direct historical S18 ancestor')
    need('S11_original_votes' in bank['S43_S44_16_champions']['requires'],
         'S11 parallel champion missing from S45 provenance')
    need(bank['S80_K0_K2_tail']['state']=='logic_ready_inputs_missing',
         'S147 tail is not a complete S80 model')
    need(bank['S5_original527']['state'] in ({'historical_only',RECOVERED} if recovered else {'historical_only'}),
         'unverified S5 originally reconstructed')
    need(bank['S5_student135']['state']=='rejected_on_exposed_smoke',
         'negative GOAT evidence erased')
    need(d['official_promotion'] is False and
         d['independent_full_S80_accuracy_measured'] is False,
         'S80 erroneously promoted without unseen evaluation')
    need(status['reference']=='freeze_local_combo' and
         status['official_promotion'] is False and
         status['independent_S80_accuracy_measured'] is False,
         'official promotion status disagrees')
    need(status['validated_frozen_components']['s80_saved_tail']['exact_historical_rows']==59309,
         'historic S80 final stage evidence mismatch')
    paths=('scripts/s147_s80_tail_runtime.py',
      'scripts/s88_unlabeled_s46_runtime.py',
      'scripts/s89_label_blind_s69_runtime.py',
      'scripts/loop_v273_k43_k21.py',
      'scripts/loop_v273_k12_k23.py',
      'scripts/loop_v273_k32_k34.py',
      'scripts/loop_v273_conditional_k01.py',
      'scripts/verify_v273_series15_union.py',
      'scripts/audit_v273_series9_posthoc.py',
      'scripts/s153_original_s13_s15_GOAT.py',
      'scripts/s154_source_s15_s18_GOAT.py',
      'scripts/train_v273_s45_specialized_risk_heads.py')
    for path in paths:need((root/path).is_file(),'missing actual source '+path)
    return {'status':'S155_CORRECTED_28_STAGE_ORIGINAL_DEPENDENCY_GRAPH',
      'nodes':len(nodes),'families':len(families),
      'new_audio_verified_count':sum(n['state']=='verified_new_audio' for n in nodes),
      'not_fully_ready_count':sum(n['state']!='verified_new_audio' for n in nodes),
      'topological_order':ordered,
      'recovered_S5_S18_evidence_verified':recovered,
      'promotion':False,'full_external_score':None,
      'source_scripts_verified':list(paths)}
def selftest():
    assert len(set(EXPECTED))==16
    try:need(False,'expected negative check')
    except ValueError:pass
    else:raise AssertionError('fail-closed guard absent')
    print('S155 PASS: sixteen source families, corrected S8/S9 to S18 ancestry, fail-closed promotion guard')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--registry',type=Path)
    p.add_argument('--status',type=Path)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        need(a.registry and a.status and a.out,'input files required')
        r=validate(json.loads(a.registry.read_text()),
                   json.loads(a.status.read_text()),a.root)
        need(not a.out.exists(),'immutable audit output')
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(r,indent=2)+'\n')
        print(json.dumps(r,indent=2))
