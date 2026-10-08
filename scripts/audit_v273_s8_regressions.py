"""Audit saved S8 regressions without training, thresholds or model changes."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from scripts.audit_v273_consensus_failures import load_npz, binary_calibration
from scripts.verify_v273_selector_repair_artifacts import digest, require, named_pairs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['catalogue','control','features','output']:
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[1]
    proof=json.loads((root/'analysis/evidence/v273-learned-selection/verification.json').read_text())
    for name,directory in [('corrector8',args.catalogue),('control7',args.control)]:
        require(digest(directory/'predictions.npz')==proof['arms'][name]['predictions_sha256'],'verified predictions digest')
    d=load_npz(args.catalogue/'predictions.npz');a=load_npz(args.control/'predictions.npz')
    for k in ['global_index','true_K','frozen_baseline_K','fold','eligible_global_index']:
        require(np.array_equal(d[k],a[k]),'native alignment '+k)
    require(np.array_equal(d['member_votes'][:,:7],a['member_votes']),'original candidate identity')
    require(np.array_equal(d['group_proposal'][:,:127],a['group_proposal']),'original group identity')
    lookup={v:i for i,v in enumerate(d['global_index'])};ids=d['eligible_global_index'];pos=np.array([lookup[i] for i in ids])
    y,b,pred,old=(z[k][pos] for z,k in [(d,'true_K'),(d,'frozen_baseline_K'),(d,'predicted_K'),(a,'predicted_K')])
    row=np.arange(len(y));f=d['fold'][pos];s8=d['group_proposal'][:,127]
    change=pred!=b;cor=(pred==y)&(b!=y);reg=change&(b==y)
    new_reg=reg&(old==y);persistent=reg&(old!=y);repaired=(old!=y)&(b==y)&~reg
    lost=(old==y)&(b!=y)&(pred!=y)
    require((int(cor.sum()),int(reg.sum()),int(new_reg.sum()),int(persistent.sum()),int(repaired.sum()),int(lost.sum()))
            ==(584,546,168,378,144,163),'verified case partitions')
    require(named_pairs(y,old,pred)['global']==proof['paired_extended_vs_control']['global'],'paired comparator')
    r=d['baseline_correct_probability'].astype(float);ro=a['baseline_correct_probability'].astype(float)
    cp=d['class_probability'].astype(float);cpo=a['class_probability'].astype(float)
    gain=np.where(change,cp[row,pred]-r,0.)
    old_destination_gain=cpo[row,pred]-ro
    old_available=(a['group_proposal']==pred[:,None]).any(1)
    require((gain[reg]>0).all() and (old_destination_gain[new_reg&old_available]<=0).all(),'decision crossing')
    same=d['group_proposal']==pred[:,None]
    history=(d['local_audit_features']*same[:,:,None]).sum(1)/same.sum(1)[:,None]
    gh=history[:,0]-history[:,1];lh=history[:,3]-history[:,4]
    result=dict(status='verified',source_run=proof['run_id'],source_commit=proof['source_commit'],
        source_sha256={name:digest(path/'predictions.npz') for name,path in [('corrector8',args.catalogue),('control7',args.control)]},
        audit_type='saved-prediction descriptive audit; no training or inference rule changes',
        independent_validation=False,promotion=False,
        counts=dict(corrections=int(cor.sum()),regressions=int(reg.sum()),persistent=int(persistent.sum()),
            new_regressions=int(new_reg.sum()),repaired_regressions=int(repaired.sum()),lost_corrections=int(lost.sum())),
        regression_by_true_k={},transitions=[],folds={},history={},
        destination_and_s8=dict(regressions_with_old_destination=int((reg&old_available).sum()),
            new_regressions_with_old_destination=int((new_reg&old_available).sum()),
            new_regressions_with_new_destination=int((new_reg&~old_available).sum()),
            regressions_where_S8_alone_correct=int((reg&(s8==y)).sum()),
            corrections_despite_S8_voting_KEEP=int((cor&(s8==b)).sum()),
            new_regressions_where_S8_alone_correct=int((new_reg&(s8==y)).sum()),
            new_regressions_control_r_at_least_half=int((new_reg&(ro>=.5)).sum()),
            new_regressions_mean_r_control=float(ro[new_reg].mean()),new_regressions_mean_r_extended=float(r[new_reg].mean())),
        calibration=dict(reference_selected=binary_calibration(r[change],reg[change]),
            correction_selected=binary_calibration(cp[row,pred][change],cor[change]),
            predicted_corrections=float(cp[row,pred][change].sum()),predicted_regressions=float(r[change].sum()),
            expected_net=float(gain.sum()),observed_net=int(cor.sum()-reg.sum()),
            gain_auc_corrections_vs_regressions=float(roc_auc_score(cor[cor|reg],gain[cor|reg]))),
        decode_ablations=proof['arms']['corrector8']['decode_ablations'],files={})
    for k in [2,3,4]:
        t=y==k;result['regression_by_true_k'][str(k)]=dict(total=int((reg&t).sum()),new=int((new_reg&t).sum()),
            persistent=int((persistent&t).sum()),repaired=int((repaired&t).sum()),corrections=int((cor&t).sum()))
    for source in [2,3,4]:
        for target in range(2,7):
            t=change&(b==source)&(pred==target)
            if not t.any():continue
            result['transitions'].append(dict(source=source,target=target,changed=int(t.sum()),
                corrections=int((cor&t).sum()),regressions=int((reg&t).sum()),new_regressions=int((new_reg&t).sum()),
                net=int((cor&t).sum()-(reg&t).sum())))
    for fold in sorted(set(f)):
        t=f==fold;result['folds'][str(fold)]=dict(regressions=int((reg&t).sum()),new=int((new_reg&t).sum()),
            repaired=int((repaired&t).sum()),lost_corrections=int((lost&t).sum()))
    for name,h in [('global_enabled',gh),('local_unused',lh)]:
        result['history'][name]=dict(negative_on_corrections=int((cor&(h<0)).sum()),negative_on_regressions=int((reg&(h<0)).sum()),
            negative_on_new_regressions=int((new_reg&(h<0)).sum()),
            descriptive_veto_net=int((reg&(h<0)).sum()-(cor&(h<0)).sum()),
            auc_corrections_vs_regressions=float(roc_auc_score(cor[cor|reg],h[cor|reg])),
            validated_rule=False)
    metadata={}
    for file in args.features.rglob('rows.jsonl'):
        for line in file.open():
            m=json.loads(line);require(m['global_index'] not in metadata,'metadata duplicate');metadata[m['global_index']]=m
    require(set(metadata)==set(ids),'metadata coverage')
    cases=[]
    for i,gid in enumerate(ids):
        m=metadata[gid];require((m['true_k'],m['baseline_k'],m['fold'])==(y[i],b[i],f[i]),'case identity')
        cases.append(dict(global_index=int(gid),recording_id=m['recording_id'],start_sample=m['start_sample'],
            time_s=m['start_sample']/44100,fold=int(f[i]),true_K=int(y[i]),baseline_K=int(b[i]),
            control_K=int(old[i]),extended_K=int(pred[i]),S8_alone_K=int(s8[i]),
            regression_status='new' if new_reg[i] else 'persistent' if persistent[i] else 'lost_correction' if lost[i] else '',
            selected_destination_available_in_old_catalogue=bool(old_available[i]),
            selected_gain=float(gain[i]),selected_correction_probability=float(cp[i,pred[i]]),
            reference_correct_probability=float(r[i]),control_reference_correct_probability=float(ro[i]),
            control_score_for_same_destination=float(old_destination_gain[i]),
            global_history_gain=float(gh[i]),local_history_gain_unused=float(lh[i])))
    args.output.mkdir(parents=True,exist_ok=True)
    for name,mask in [('regressions-546.csv',reg),('corrections-perdues-163.csv',lost)]:
        indices=np.flatnonzero(mask);indices=indices[np.argsort(-gain[indices],kind='stable')]
        path=args.output/name
        with path.open('w',newline='') as handle:
            w=csv.DictWriter(handle,fieldnames=list(cases[0]),lineterminator='\n');w.writeheader();w.writerows(cases[i] for i in indices)
        with path.open() as handle:rows=list(csv.DictReader(handle))
        require([int(x['global_index']) for x in rows]==ids[indices].tolist(),'CSV case set/order')
        require(all(int(x['true_K'])==y[i] and int(x['extended_K'])==pred[i] and float(x['selected_gain'])==gain[i]
                    for x,i in zip(rows,indices)),'CSV labels/decisions/scores')
        result['files'][name]=dict(rows=len(rows),sha256=digest(path))
    top=np.flatnonzero(new_reg);top=top[np.argsort(-gain[top],kind='stable')][:5]
    result['highest_confidence_new_regressions']=[cases[i] for i in top]
    result['limitations']=[
        'Previously exposed recording folds; this is not unseen validation.',
        'The extended selector adds parameters and changes features/pooled scores; the comparison does not isolate a causal contributor.',
        'Equal-destination group logits are pooled; the selected representative mask is not causal attribution.',
        'History means summarize all groups proposing the selected K, not a neural importance score.',
        'Local history fields were zeroed during selector training. A retrospective sign veto is not a validated inference rule.',
        'AUC separates 584 corrections from 546 regressions and excludes 766 neutral changes.',
        'These prediction archives do not identify a missing acoustic feature or a physical cause.',
    ]
    (args.output/'audit.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['status','counts','regression_by_true_k','destination_and_s8','history']}))


if __name__=='__main__':main()
