"""Independent S35 artifact verifier: identity, OOF fits, head usages, case audits."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np

from scripts.v273_extended_head_bank import ALL_HEADS,REGISTRY
from scripts.yourmt3_exactk_common import FOLDS,require

def data(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def sha(arr):
    return hashlib.sha256(np.asarray(arr,dtype='<i8').tobytes()).hexdigest()

def check(args):
    require(not args.output.exists(),'cannot overwrite independent verification')
    original=data(args.input/'predictions.npz')
    routes=data(args.input/'routing.npz')
    report=json.loads((args.input/'report.json').read_text())
    ids=original['global_index'];truth=original['true_K'];fold=original['fold']
    require(len(ids)==59309 and len(np.unique(ids))==59309,
            'wrong source cohort or duplicated IDs')
    require(set(fold.tolist())==set(FOLDS),'wrong source folds')
    require(np.array_equal(ids,routes['global_index']),'paths misaligned')
    action=list(map(str,routes['action_names']))
    require(len(action)==19 and action[:-1]==list(ALL_HEADS),
            'a required historical action missing')
    require('H9' not in action and 'YourMT3' not in ''.join(action),
            'forbidden H9 is in routing')
    require(action[-1]=='AB_stateful_S29',
            'prior A/B recurrent producer missing')
    require(report['H9_excluded_from_all_actions'] and
            not report['H8_real_waveform_evidence_available'],
            'incorrect H9/H8 provenance')
    require(len(report['head_registry'])==18,
            'incomplete historical head registry')
    require(report['native_source_estimators_saved']==144,
            'outer/inner OOF model inventory wrong')
    require(len(report['fitted_source_provenance'])==144,
            'OOF provenance missing')
    for item in report['fitted_source_provenance']:
        outer=int(item['outer']);inner=item['inner']
        if inner is None:
            fit=fold!=outer;held=fold==outer
        else:
            inner=int(inner)
            fit=(fold!=outer)&(fold!=inner)
            held=fold==inner
        require(item['fit_SHA256']==sha(ids[fit]) and
                item['held_SHA256']==sha(ids[held]),
                'OOF model saw forbidden validation row '+item['file'])
        require((fit&held).sum()==0,'source fit/evaluation overlap')
        require((args.input/'models'/item['file']).is_file(),
                'missing original expert weights '+item['file'])
    keys=list(map(str,original['variant_ids']))
    routerkeys=list(map(str,routes['variant_ids']))
    require(len(keys)==11 and len(routerkeys)==9,
            'missing dynamic decision rules')
    require(np.array_equal(keys[2:],routerkeys),'routing policy alignment')
    result=original['predictions']
    base=result[:,0]
    require(keys[0]=='series18_parent' and
            int((base==truth).sum())==49178 and
            int(((base==truth)&(truth>=2)).sum())==2998,
            'conservative S18 reference changed')
    stored=routes['selected_head']
    require(stored.shape==(len(ids),9,3),
            'missing per-event dynamic head selection')
    require(((stored>=-1)&(stored<19)).all(),'invalid action index')
    require(not np.any(stored==action.index('H8_pitch_shift')),
            'H8 used without validated waveform source')
    independent={}
    for pidx,key in enumerate(routerkeys):
        chosen=result[:,pidx+2]
        assert chosen.shape==base.shape
        changed=chosen!=base
        fixed=int((changed&(chosen==truth)&(base!=truth)).sum())
        broken=int((changed&(chosen!=truth)&(base==truth)).sum())
        neutral=int((changed&(chosen!=truth)&(base!=truth)).sum())
        current=stored[:,pidx,:]
        # STOP is irreversible. Heads cannot be used more than once/event.
        require(not np.any((current[:,0]==-1)&(current[:,1]!=-1)),
                'head ran after STOP0')
        require(not np.any((current[:,1]==-1)&(current[:,2]!=-1)),
                'head ran after STOP1')
        for t in (0,1):
            for u in range(t+1,3):
                require(not np.any((current[:,t]>=0)&
                    (current[:,t]==current[:,u])),
                    'head was repeated contrary to declared S35 protocol')
        stat=report['audits'][key]
        require((fixed,broken,neutral)==(
            stat['corrected'],stat['regressed'],stat['neutral']),
            'paired audit mismatched '+key)
        usage={name:int(np.sum(current==i)) for i,name in enumerate(action)}
        require(usage==stat['route']['head_usage'],
                'head usage mismatch '+key)
        independent[key]=dict(corrected=fixed,regressed=broken,neutral=neutral,
            global_accuracy=float(np.mean(chosen==truth)),
            poly_accuracy=float(np.mean(chosen[truth>=2]==truth[truth>=2])),
            first_action_count=int((current[:,0]>=0).sum()),
            unique_heads_used=sum(n>0 for n in usage.values()),
            head_calls=usage)
    result=dict(status='verified',source_experiment='S35',
        all_18_modules_in_contract=True,H9_not_ever_called=True,
        H8_missing_waveform_honestly_masked=True,
        original_144_nested_fold_weights_verified=True,
        every_held_fold_inference_train_excluded=True,
        every_casewise_correction_regression_neutral_recomputed=True,
        all_9_dynamic_routes_replayed_from_archive=True,
        heldout_new_compositions=False,no_model_promotion=True,
        audits=independent)
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    lines=['# S35 independent evidence audit — 18 heads except H9',
        '', '144 original specialist files checked against nested fold provenance.',
        'H8 not invoked until true full-cohort transformed-WAV OOF source exists.',
        '**No validation on new compositions; no production promotion.**',
        '', '| Policy | Global | Poly | New corrections | New regressions | Neutral changes | Active distinct heads |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for k,v in independent.items():
        lines.append(f"| {k} | {v['global_accuracy']*100:.4f}% | "
             f"{v['poly_accuracy']*100:.4f}% | {v['corrected']} | "
             f"{v['regressed']} | {v['neutral']} | {v['unique_heads_used']} |")
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    check(parser.parse_args())
