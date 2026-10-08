"""Descriptive audit of frozen consensus predictions; no fit or rule changes.

All comparisons use verified archives. Labels are used only for this diagnostic.
History descriptors are averaged over EVERY group with the selected destination,
not over its arbitrary representative mask.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from scripts.verify_v273_selector_repair_artifacts import digest, require, metrics, named_pairs


def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def number(value):
    return float(value) if np.isfinite(value) else None


def binary_calibration(predicted,observed):
    p=np.asarray(predicted,float);y=np.asarray(observed,bool)
    result=dict(rows=len(y),predicted_mean=float(p.mean()),observed_rate=float(y.mean()),
        brier=float(np.mean((p-y)**2)),auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,bins=[])
    for j in range(10):
        take=(p>=j/10)&((p<(j+1)/10) if j<9 else (p<=1+1e-6));n=int(take.sum())
        result['bins'].append(dict(lower=j/10,upper=(j+1)/10,rows=n,
            predicted_mean=float(p[take].mean()) if n else None,observed_rate=float(y[take].mean()) if n else None))
    require(sum(x['rows'] for x in result['bins'])==len(y),'calibration bin coverage')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--consensus',type=Path,required=True)
    p.add_argument('--archived-global',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[1]
    proof=json.loads((root/'analysis/evidence/v273-group-consensus/verification.json').read_text())
    current_proof=proof['arms']['pooled_ce'];old_proof=proof['archived_global']
    require(digest(args.consensus/'predictions.npz')==current_proof['predictions_sha256'],'consensus archive mismatch')
    require(digest(args.archived_global/'predictions.npz')==old_proof['predictions_sha256'],'comparison archive mismatch')
    d=load_npz(args.consensus/'predictions.npz');old=load_npz(args.archived_global/'predictions.npz')
    report=json.loads((args.consensus/'report.json').read_text())
    require(report['configuration']['global_audits_enabled'] and not report['configuration']['local_audits_enabled'],'audited configuration changed')
    for key in ['global_index','true_K','frozen_baseline_K','fold','eligible_global_index','member_votes','group_proposal','local_audit_features']:
        require(np.array_equal(d[key],old[key]),'comparison alignment '+key)
    require(np.array_equal(d['baseline_correct_probability'],old['baseline_correct_probability']),'reference-risk branch changed')
    lookup={v:i for i,v in enumerate(d['global_index'])};at=np.array([lookup[v] for v in d['eligible_global_index']])
    y=d['true_K'][at];b=d['frozen_baseline_K'][at];pred=d['predicted_K'][at];fold=d['fold'][at]
    ids=d['eligible_global_index'];n=len(y);row=np.arange(n)
    r=d['baseline_correct_probability'].astype(float);cp=d['class_probability'].astype(float);props=d['group_proposal']
    changed=pred!=b;cor=changed&(pred==y);reg=changed&(b==y);neutral=changed&~cor&~reg
    possible=(props==y[:,None]).any(1)&(b!=y)
    available=np.stack([(props==k).any(1)&(b!=k) for k in range(7)],1)
    logits=np.concatenate([np.where(available,d['pooled_class_logits'],-1e9),np.zeros((n,1))],1)
    logits-=logits.max(1,keepdims=True);q=np.exp(logits);q/=q.sum(1,keepdims=True)
    require(np.allclose(cp,(1-r[:,None])*q[:,:7]+np.eye(7)[b]*r[:,None],atol=1e-6),'conditional reconstruction')
    gain=np.where(changed,cp[row,pred]-r,0.)
    native_pairs=named_pairs(d['true_K'],d['frozen_baseline_K'],d['predicted_K'])
    require(native_pairs==report['paired']==current_proof['paired'],'paired score mismatch')
    require(metrics(d['true_K'],d['predicted_K'])==current_proof['candidate'],'native metric mismatch')
    require((n,int(cor.sum()),int(reg.sum()),int(neutral.sum()))==(7493,582,522,696),'audited cohort changed')
    require(abs(gain.sum()-report['selected_expected_net'])<1e-4,'selected gain mismatch')
    require(np.all(r[changed]<.5),'categorical decision risk bound violated')

    states={
        'correct_reference_kept':(b==y)&~changed,
        'correct_reference_broken':reg,
        'wrong_reference_corrected':cor,
        'reachable_error_blocked_r_at_least_half':possible&(r>=.5),
        'reachable_error_kept_r_below_half':possible&(r<.5)&~changed,
        'reachable_error_changed_to_wrong_K':possible&changed&(pred!=y),
        'unreachable_error_kept':(b!=y)&~possible&~changed,
        'unreachable_error_changed':(b!=y)&~possible&changed,
    }
    require(np.array_equal(sum(x.astype(int) for x in states.values()),np.ones(n)),'state partition')
    expected_cor=float(cp[row,pred][changed].sum());expected_reg=float(r[changed].sum())
    cor_gap=expected_cor-int(cor.sum());reg_gap=int(reg.sum())-expected_reg
    result=dict(status='verified',source_run=37781750476,source_commit=proof['source_commit'],
        audit_type='fixed-prediction descriptive analysis; no training, threshold search or deployed change',
        independent_validation=False,promotion=False,source_sha256=dict(consensus=digest(args.consensus/'predictions.npz'),archived_global=digest(args.archived_global/'predictions.npz')),
        native_metrics=current_proof['candidate'],native_pairs=native_pairs,
        configuration=dict(global_history_used=True,local_history_used=False,representative_mask_not_an_attribution=True,
            local_history_columns_zeroed=[58,59,60,62,63],trainable_parameters=12994),
        state_counts={k:int(v.sum()) for k,v in states.items()},
        gain_accounting=dict(changed=int(changed.sum()),corrections=int(cor.sum()),regressions=int(reg.sum()),neutral=int(neutral.sum()),
            expected_corrections=expected_cor,expected_regressions=expected_reg,expected_net=expected_cor-expected_reg,
            observed_net=int(cor.sum()-reg.sum()),optimism=cor_gap+reg_gap,
            overestimated_corrections=cor_gap,underestimated_regressions=reg_gap,
            correction_share_of_optimism=cor_gap/(cor_gap+reg_gap)),
        calibration=dict(
            reference_all_eligible=binary_calibration(r,b==y),
            reference_selected=binary_calibration(r[changed],(b==y)[changed]),
            correction_selected=binary_calibration(cp[row,pred][changed],cor[changed]),
            conditional_selected_baseline_wrong=binary_calibration(q[row,pred][changed&(b!=y)],(pred==y)[changed&(b!=y)]),
            other_baseline_wrong=binary_calibration(q[:,7][b!=y],(~possible)[b!=y]),
            other_selected_baseline_wrong=binary_calibration(q[:,7][changed&(b!=y)],(~possible)[changed&(b!=y)])),
        availability_by_true_k={},transitions=[],folds={},comparisons={},gain_bins=[],histories={})
    for k in range(7):
        take=y==k;wrong=take&(b!=y)
        result['availability_by_true_k'][str(k)]=dict(eligible_rows=int(take.sum()),initial_errors=int(wrong.sum()),
            reachable=int((take&possible).sum()),unreachable=int((wrong&~possible).sum()),
            corrected=int((take&cor).sum()),missed_high_reference_probability=int((take&states['reachable_error_blocked_r_at_least_half']).sum()),
            missed_keep_low_reference_probability=int((take&states['reachable_error_kept_r_below_half']).sum()),
            missed_wrong_destination=int((take&states['reachable_error_changed_to_wrong_K']).sum()))
    for source in [2,3,4]:
        for target in range(2,7):
            take=changed&(b==source)&(pred==target)
            if not take.any():continue
            result['transitions'].append(dict(source=source,target=target,rows=int(take.sum()),corrections=int((take&cor).sum()),
                regressions=int((take&reg).sum()),neutral=int((take&neutral).sum()),net=int((take&cor).sum()-(take&reg).sum()),
                expected_net=float(gain[take].sum()),true_k_counts=np.bincount(y[take],minlength=7).tolist(),
                expected_regressions=float(r[take].sum()),expected_corrections=float(cp[row,pred][take].sum())))
    for f in sorted(set(fold)):
        take=changed&(fold==f)
        result['folds'][str(f)]=dict(changed=int(take.sum()),corrections=int((take&cor).sum()),regressions=int((take&reg).sum()),
            net=int((take&cor).sum()-(take&reg).sum()),expected_net=float(gain[take].sum()))
    for name,previous in [('archived_global',old['predicted_K'][at]),('archived_singletons',old['singletons_only_K'][at])]:
        pc=(previous==y)&(b!=y);pr=(previous!=y)&(b==y)
        result['comparisons'][name]=dict(paired=named_pairs(y,previous,pred),
            previous_corrections_retained=int((pc&cor).sum()),previous_corrections_lost=int((pc&~cor).sum()),
            new_corrections=int((cor&~pc).sum()),previous_regressions_repaired=int((pr&~reg).sum()),
            new_regressions=int((reg&~pr).sum()),previous_regressions_persist=int((pr&reg).sum()))
        require(int((cor&~pc).sum()+(pr&~reg).sum())==result['comparisons'][name]['paired']['global']['corrections'],'comparison correction decomposition')
        require(int((pc&~cor).sum()+(reg&~pr).sum())==result['comparisons'][name]['paired']['global']['regressions'],'comparison regression decomposition')
    edges=[0,.05,.1,.2,.3,.5,1.000001]
    for lo,hi in zip(edges[:-1],edges[1:]):
        take=changed&(gain>=lo)&(gain<hi)
        result['gain_bins'].append(dict(lower=lo,upper=min(hi,1.),rows=int(take.sum()),corrections=int((take&cor).sum()),
            regressions=int((take&reg).sum()),net=int((take&cor).sum()-(take&reg).sum()),expected_net=float(gain[take].sum())))
    require(sum(x['rows'] for x in result['gain_bins'])==1800,'gain bin coverage')

    same=props==pred[:,None]
    history=(d['local_audit_features']*same[:,:,None]).sum(1)/same.sum(1)[:,None]
    history_gains={'global':history[:,0]-history[:,1],'local_unused':history[:,3]-history[:,4]}
    informative=cor|reg
    result['gain_auc_corrections_vs_regressions']=float(roc_auc_score(cor[informative],gain[informative]))
    for name,value in history_gains.items():
        negative=changed&(value<0)
        result['histories'][name]=dict(auc_corrections_vs_regressions=float(roc_auc_score(cor[informative],value[informative])),
            negative_on_corrections=int((negative&cor).sum()),negative_on_regressions=int((negative&reg).sum()),negative_on_neutral=int((negative&neutral).sum()),
            descriptive_veto_net_difference=int((negative&reg).sum()-(negative&cor).sum()),
            counterfactual_not_validated=True,mean_on_corrections=float(value[cor].mean()),mean_on_regressions=float(value[reg].mean()))

    metadata={}
    for path in sorted(args.features.rglob('rows.jsonl')):
        for line in path.open():
            record=json.loads(line);gid=record['global_index']
            require(gid not in metadata,'duplicate metadata ID');metadata[gid]=record
    require(set(ids)==set(metadata),'metadata cohort mismatch')
    cases=[];states_by_row=np.empty(n,object)
    for name,take in states.items():states_by_row[take]=name
    for i,gid in enumerate(ids):
        record=metadata[gid]
        require((record['true_k'],record['baseline_k'],record['fold'])==(y[i],b[i],fold[i]),'case metadata alignment')
        cases.append(dict(global_index=int(gid),recording_id=record['recording_id'],start_sample=record['start_sample'],
            time_s=round(record['start_sample']/44100,6),fold=int(fold[i]),true_K=int(y[i]),frozen_K=int(b[i]),predicted_K=int(pred[i]),
            previous_global_K=int(old['predicted_K'][at[i]]),previous_singleton_K=int(old['singletons_only_K'][at[i]]),
            state=states_by_row[i],reference_correct_probability=round(float(r[i]),7),
            selected_correction_probability=round(float(cp[i,pred[i]]),7) if changed[i] else '',
            selected_gain=round(float(gain[i]),7),conditional_other_probability=round(float(q[i,7]),7),
            true_K_reachable=bool((props[i]==y[i]).any()),global_history_gain=round(float(history_gains['global'][i]),7) if changed[i] else '',
            local_history_gain_unused=round(float(history_gains['local_unused'][i]),7) if changed[i] else ''))
    args.output.mkdir(parents=True,exist_ok=True);files={}
    def write_cases(name,indices,columns=None):
        selected=[cases[int(i)] for i in indices]
        path=args.output/name
        fields=columns or list(cases[0])
        with path.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore',lineterminator='\n')
            writer.writeheader();writer.writerows(selected)
        files[name]=dict(rows=len(selected),sha256=digest(path))
    regression_order=np.flatnonzero(reg);regression_order=regression_order[np.argsort(-gain[regression_order],kind='stable')]
    write_cases('regressions-522.csv',regression_order)
    write_cases('corrections-582.csv',np.flatnonzero(cor))
    lost=(old['predicted_K'][at]==y)&(b!=y)&(pred!=y)
    write_cases('corrections-perdues-171.csv',np.flatnonzero(lost))
    missed=possible&(pred!=y)
    write_cases('opportunites-manquees-1042.csv',np.flatnonzero(missed),
        ['global_index','recording_id','start_sample','time_s','fold','true_K','frozen_K','predicted_K','state','reference_correct_probability'])
    result['files']=files;result['highest_confidence_regressions']=[cases[int(i)] for i in regression_order[:10]]
    result['limitations']=[
        'Already exposed folds 0/1/2/4; no unseen validation, significance or promotion.',
        'No fitted model, new threshold or inference rule: diagnostics use saved predictions only.',
        'History summaries are descriptive means over equal-destination groups, not a decomposition of the nonlinear network.',
        'Local history is exported but zeroed in this trained model; its descriptive comparison does not test a retrained local consensus model.',
        'AUC compares the 582 corrections with 522 regressions and omits 696 neutral changes.',
        'The original risk/score weights may be miscalibrated; output associations do not establish a missing physical acoustic signal.',
        'K0/K1 are not candidate destinations. Only initially predicted K2/K3/K4 events can be changed.',
        'The sign-veto counts are retrospective descriptions on the same data, not validated decision rules.',
    ]
    (args.output/'audit.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(dict(status='verified',gain_accounting=result['gain_accounting'],state_counts=result['state_counts'],files=files)))


if __name__=='__main__':main()
