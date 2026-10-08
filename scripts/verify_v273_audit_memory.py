"""Verify local-audit variants and describe remaining regression evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.verify_v273_selector_repair_artifacts import digest, metrics, named_pairs, require
from scripts.verify_v273_learned_selection import read_npz, choose, proposals_from_votes, verify_corrector, verify_paths
from scripts.verify_v273_group127 import provenance
from scripts.v273_confidence_metrics import summarize


def summary(data, pos):
    return summarize(dict(truth=data['true_K'][pos], baseline=data['frozen_baseline_K'][pos],
        proposal=data['group_proposal'], baseline_correct=data['baseline_correct_probability'],
        pooled_class_logits=data['pooled_class_logits'], class_probability=data['class_probability'],
        other_probability=data['other_probability']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--original',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); manifest=json.loads((args.results/'artifacts.json').read_text())
    for item in manifest['artifacts']:
        path=args.results/(item['name']+'.zip')
        require(digest(path)==item['sha256'] and path.stat().st_size==item['size'],'downloaded artifact identity')
    ref=read_npz(args.original/'corrector8/predictions.npz')
    ids,y,b,f,eligible=(ref[k] for k in ['global_index','true_K','frozen_baseline_K','fold','eligible_global_index'])
    lookup={int(v):i for i,v in enumerate(ids)};pos=np.array([lookup[int(v)] for v in eligible])
    yc,bc,fc=y[pos],b[pos],f[pos];active=np.zeros(len(y),bool);active[pos]=True
    cm=json.loads((args.original/'corrector-cache/manifest.json').read_text())
    _,producers,metadata=verify_corrector(args.original/'corrector-cache',cm,yc,bc,fc,eligible,args.features)
    proof=json.loads((Path(__file__).resolve().parents[1]/'analysis/evidence/v273-learned-selection/verification.json').read_text())
    evidence=dict(status='verified',run_id=manifest['run_id'],source_commit=manifest['source_commit'],
        artifact_manifest=manifest,independent_validation=False,promotion=False,arms={})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for arm,count in [('control7',7),('corrector8',8)]:
        directory=args.results/arm;data=read_npz(directory/'predictions.npz')
        report=json.loads((directory/'report.json').read_text())
        original_path=args.original/arm/'predictions.npz'
        require(digest(original_path)==proof['arms'][arm]['predictions_sha256'],'immutable original predictions')
        old=read_npz(original_path)
        for key in ['global_index','true_K','frozen_baseline_K','fold','eligible_global_index']:
            require(np.array_equal(data[key],ref[key]),'native alignment '+key)
        for key in ['member_votes','group_proposal','local_audit_features']:
            require(np.array_equal(data[key],old[key]),'all proposals and raw audits preserved '+key)
        config=report['configuration']
        require(config['audit_arm']=='local' and config['local_audits_enabled'] and config['global_audits_enabled'],'audits active')
        require(config['epochs']==30 and config['seed']==27402 and config['mode']=='pooled_ce','fixed learning protocol')
        props=data['group_proposal'];audits=data['local_audit_features'];gain=data['group_expected_gain']
        require(np.array_equal(props,proposals_from_votes(data['member_votes'],bc)),'all complete groups')
        pred,mask=choose(gain,props,bc)
        require(np.array_equal(pred,data['predicted_K'][pos]) and np.array_equal(mask,data['chosen_group_mask']),'main decoder')
        require(np.array_equal(data['predicted_K'][~active],b[~active]),'no action scope expansion')
        prob=data['group_outcome_probability'];cp=data['class_probability'];rr=data['baseline_correct_probability']
        require(np.allclose(prob.sum(2),1,atol=1e-6) and (prob>=-1e-7).all(),'outcome probabilities')
        require(np.array_equal(gain,prob[:,:,0]-prob[:,:,1]),'gain identity')
        changing=props!=bc[:,None]
        require(np.array_equal(prob[:,:,0],np.take_along_axis(cp,props,1)*changing),'shared destination correction probability')
        require(np.array_equal(prob[:,:,1],rr[:,None]*changing),'shared initial correctness probability')
        require(np.allclose(cp.sum(1)+data['other_probability'],1,atol=1e-6),'joint probability simplex')
        require(report['paired']==named_pairs(y,b,data['predicted_K']) and report['candidate']==metrics(y,data['predicted_K']),'native results')
        no_cp=data['no_local_class_probability'];no_r=data['no_local_baseline_correct_probability']
        no_gain=(np.take_along_axis(no_cp,props,1)-no_r[:,None])*changing
        no_pred,_=choose(no_gain,props,bc)
        require(np.array_equal(no_pred,data['no_local_audit_K'][pos]),'fixed-weight intervention decoder')
        require(report['local_influence_fixed_weights']['paired_local_vs_zero']==named_pairs(y,data['no_local_audit_K'],data['predicted_K']),'intervention paired results')
        fold_results={}
        for fold in (0,1,2,4):
            take=fc==fold;fr=report['folds'][str(fold)];conn=fr['audit_connection']
            expected_sha=hashlib.sha256(np.ascontiguousarray(audits[take]).tobytes()).hexdigest()
            require(conn['held']['effective_audit_sha256']==expected_sha,'actual full held audit inputs')
            require(conn['held']['rows_with_local_evidence']==int(take.sum()),'local held evidence present')
            require(conn['train']['rows_with_local_evidence']==int((~take).sum()),'local train evidence present')
            require(fr['parameters']==12994+544*(count-7) and [h['epoch'] for h in fr['training']]==[1,16,30],'parameter and epoch budgets')
            intervention=fr['local_influence_fixed_weights']
            require(intervention['decisions_changed']==int((pred[take]!=no_pred[take]).sum()),'intervention changed decisions')
            require(np.isclose(intervention['max_absolute_gain_change'],np.max(np.abs(gain[take]-no_gain[take])),atol=1e-6),'intervention gain change')
            fold_results[str(fold)]=dict(paired=fr['paired'],audit_connection=conn,local_influence=intervention,
                weights_sha256=digest(directory/f'fold-{fold}.weights.h5'))
        expert_proof=provenance(report,eligible,yc,fc)
        corrector_proof=verify_paths(report,producers,eligible,fc) if count==8 else None
        old_pred=old['predicted_K'][pos]
        old_cor=(bc!=yc)&(old_pred==yc);new_cor=(bc!=yc)&(pred==yc)
        old_reg=(bc==yc)&(old_pred!=yc);new_reg=(bc==yc)&(pred!=yc)
        decomposition=dict(corrections_retained=int((old_cor&new_cor).sum()),
            corrections_lost=int((old_cor&~new_cor).sum()),corrections_added=int((new_cor&~old_cor).sum()),
            regressions_retained=int((old_reg&new_reg).sum()),regressions_avoided=int((old_reg&~new_reg).sum()),
            regressions_added=int((new_reg&~old_reg).sum()),
            corrections_union=int((old_cor|new_cor).sum()),oracle_union_not_prediction=True)
        cases=[];all_cases=[];profile_counts=Counter();profile_for_existing=Counter();profile_effects={}
        for i in np.flatnonzero(pred != bc):
            same=props[i]==pred[i];a=audits[i,same]
            positive=a[:,3]>a[:,4];negative=a[:,4]>a[:,3]
            if not np.any(a[:,7]>0): profile='no_matching_local_support'
            elif positive.any() and negative.any(): profile='mixed_by_complete_group'
            elif negative.any(): profile='local_rates_favor_regression_for_all_groups'
            elif positive.any(): profile='local_rates_favor_correction_for_all_groups'
            else: profile='tied_local_rates'
            effect='correction' if new_cor[i] else 'regression' if new_reg[i] else 'neutral'
            profile_effects.setdefault(profile,Counter())[effect]+=1
            if new_reg[i]:
                profile_counts[profile]+=1
                if old_reg[i]:profile_for_existing[profile]+=1
            meta=metadata[eligible[i]]
            case=dict(global_index=int(eligible[i]),fold=int(fc[i]),recording_id=meta['recording_id'],
                start_sample=int(meta['start_sample']),true_K=int(yc[i]),baseline_K=int(bc[i]),
                global_K=int(old_pred[i]),local_K=int(pred[i]),existing_global_regression=bool(old_reg[i]),
                effect_vs_freeze=effect,local_audit_profile=profile,same_destination_groups=int(same.sum()),
                groups_favoring_correction=int(positive.sum()),groups_favoring_regression=int(negative.sum()),
                groups_tied=int((~positive&~negative).sum()),
                mean_local_correction_rate=float(a[:,3].mean()),mean_local_regression_rate=float(a[:,4].mean()),
                mean_local_neutral_rate=float(a[:,5].mean()),mean_local_support_fraction=float(a[:,7].mean()),
                mean_neighbor_distance=float(a[:,8].mean()),
                announced_gain=float(gain[i,np.flatnonzero(same)[0]]))
            all_cases.append(case)
            if new_reg[i]:cases.append(case)
        file=args.output.parent/(arm+'-remaining-regression-profiles.csv')
        if cases:
            with file.open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(cases[0]),lineterminator='\n');writer.writeheader();writer.writerows(cases)
        allfile=args.output.parent/(arm+'-all-changed-local-profiles.csv')
        with allfile.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(all_cases[0]),lineterminator='\n');writer.writeheader();writer.writerows(all_cases)
        singles=props[:,[2**j-1 for j in range(count)]]
        strong=(bc!=yc)&(props==yc[:,None]).any(1)&(singles!=yc[:,None]).all(1)
        evidence['arms'][arm]=dict(predictions_sha256=digest(directory/'predictions.npz'),report_sha256=digest(directory/'report.json'),
            original_predictions_sha256=digest(original_path),metrics=report['candidate'],paired=report['paired'],
            versus_original=named_pairs(y,old['predicted_K'],data['predicted_K']),decomposition=decomposition,
            old_confidence=summary(old,pos),local_confidence=summary(data,pos),folds=fold_results,
            expert_provenance=expert_proof,corrector_provenance=corrector_proof,
            raw_votes_proposals_and_audits_identical=True,all_local_channels_connected=True,
            local_influence_fixed_weights=report['local_influence_fixed_weights'],
            remaining_regression_profiles=dict(profile_counts),existing_regression_profiles=dict(profile_for_existing),
            all_changed_profile_effects={k:dict(corrections=v['correction'],regressions=v['regression'],neutral=v['neutral'],
                net=v['correction']-v['regression']) for k,v in profile_effects.items()},
            regression_profiles_are_descriptive_not_new_categories=True,
            regression_case_file=dict(path=file.name,sha256=digest(file),rows=len(cases)),
            all_changed_profile_file=dict(path=allfile.name,sha256=digest(allfile),rows=len(all_cases)),
            combination_only=dict(events=int(strong.sum()),old_corrected=int((strong&(old_pred==yc)).sum()),new_corrected=int((strong&(pred==yc)).sum())))
    args.output.write_text(json.dumps(evidence,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(dict(status='verified',results={a:dict(net=x['paired']['global']['net'],vs_original=x['versus_original']['global'],
        decomposition=x['decomposition'],regression_profiles=x['remaining_regression_profiles']) for a,x in evidence['arms'].items()})),flush=True)


if __name__=='__main__':
    main()
