"""S148: fail-closed, exhaustive S80 original/deploy-v1 dependency contract."""
import argparse,json
from pathlib import Path
STATES={'verified_new_audio','adapted_new_audio','historical_only',
 'rejected_on_exposed_smoke','missing_new_audio','logic_ready_inputs_missing',
 'pending_independent'}
EXPECTED=(
 "S1_acoustic","S2_source","S3_repair","S4_repair",
 "historical_repair_coherent","S25_Bfirst","S24_Bfirst",
 "S40_temporal_CNN","S19_recurrent_ABA","S11_votes",
 "S38_morphology","S12_logistic","S20_repeated_ABA",
 "S27_early_stop","S41_fusion","S35_multiroute")
def need(x,msg):
    if not x:raise ValueError(msg)
def validate(d,status,root):
    families=d['families'];nodes=d['nodes']
    need(len(families)==16 and set(f['id'] for f in families)==set(EXPECTED),
         'historical family missing or duplicate')
    need(len(nodes)==21 and len({n['id'] for n in nodes})==21,
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
    need(bank['S80_K0_K2_tail']['state']=='logic_ready_inputs_missing',
         'S147 tail is not a complete S80 model')
    need(bank['S5_original527']['state']=='historical_only',
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
      'scripts/train_v273_s45_specialized_risk_heads.py')
    for path in paths:need((root/path).is_file(),'missing actual source '+path)
    return {'status':'S148_ONE_VERIFIED_DEPENDENCY_GRAPH',
      'nodes':len(nodes),'families':len(families),
      'new_audio_verified_count':sum(n['state']=='verified_new_audio' for n in nodes),
      'not_fully_ready_count':sum(n['state']!='verified_new_audio' for n in nodes),
      'topological_order':ordered,
      'promotion':False,'full_external_score':None,
      'source_scripts_verified':list(paths)}
def selftest():
    assert len(set(EXPECTED))==16
    try:need(False,'expected negative check')
    except ValueError:pass
    else:raise AssertionError('fail-closed guard absent')
    print('S148 PASS: sixteen source families, immutable fail-closed promotion guard')
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
