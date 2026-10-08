"""Predeclared contextual regression-risk experiment; four exposed outer folds."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from scripts.audit_v273_selector_design import aligned_positions,load_feature_rows
from scripts.v273_selector_contract import matrices,FOLDS,require,id_digest
from scripts.prepare_v273_group127_producers import FrozenProducerPool
from scripts.prepare_v273_learned_corrector import FrozenCorrector,CataloguePool
from scripts.v273_catalogue_contract import assemble_catalogue_outer,direct_dim,choose_groups
from scripts.learn_v273_catalogue import CatalogueCritic,train_catalogue,predict
from scripts.learn_v273_contextual_risk import ContextualRiskCritic
from scripts.v273_audit_context import audit_context
from scripts.v273_contextual_risk_contract import audit_profiles,retention,profile_retention
from scripts.v273_confidence_metrics import summarize
from scripts.yourmt3_exactk_common import digest,metrics,paired


def confidence(out,proposals,y,b):
    return summarize(dict(truth=y,baseline=b,proposal=proposals,baseline_correct=out['baseline_correct'],
        pooled_class_logits=out['pooled_class_logits'],class_probability=out['class_probability'],other_probability=out['other_probability']))


def render(report):
    rows=['# V27.3 — Risque contextuel des groupes complets','',
        'Comparaison de développement ; aucune validation inédite ni promotion.','',
        '| Variante | Corrections | Régressions | Net face au freeze |','|---|---:|---:|---:|']
    for name,key in [('Parent local','parent_paired'),('Risque contextuel','paired')]:
        p=report[key]['global'];rows.append(f"| {name} | {p['corrections']} | {p['regressions']} | {p['net']:+d} |")
    d=report['retention'];rows+=['',f"Face au parent : {d['regressions_avoided']} régressions évitées, {d['corrections_lost']} corrections perdues, {d['corrections_added']} nouvelles corrections et {d['regressions_added']} nouvelles régressions.",'',
        '| Vrai K | Net face au freeze | Net face au parent |','|---|---:|---:|']
    for k in range(7):rows.append(f"| K{k} | {report['paired']['by_k'][str(k)]['net']:+d} | {d['paired']['by_k'][str(k)]['net']:+d} |")
    target=report['parent_profile_retention']['local_rates_favor_regression_for_all_groups'];d=target['retention']
    rows+=['',f"Profil défavorable du parent : {target['rows']} événements ; {d['regressions_avoided']} des 126 régressions évitées ; {d['corrections_retained']} des 92 corrections conservées.",'',
        f"Gain annoncé : {report['confidence']['expected_net']:.3f} ; gain observé : {report['paired']['global']['net']:+d}.",
        'Les variantes et les interventions sont conservées même si leur bilan global est inférieur.']
    return '\n'.join(rows)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['reference','features','producer-cache','corrector-cache','archive','output']:
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();require(not args.output.exists(),'refusing overwrite')
    root=Path(__file__).resolve().parents[1]
    proof=json.loads((root/'analysis/evidence/v273-audit-memory/verification.json').read_text())['arms']['corrector8']
    require(digest(args.archive/'predictions.npz')==proof['predictions_sha256'],'immutable local parent predictions')
    with np.load(args.archive/'predictions.npz') as z:archive={k:z[k] for k in z.files}
    require(digest(args.reference)=='707fc1681e2b0b1ef01805710186c4119ad697a0b52ed3330698e48f34c2f77b','native reference source')
    ids,y,b,f,eligible=(archive[k] for k in ['global_index','true_K','frozen_baseline_K','fold','eligible_global_index'])
    pos=aligned_positions(ids,eligible);rows,_=load_feature_rows(args.features,ids,y,b,f,pos)
    x,raw,names=matrices(rows);yc,bc,fc=y[pos],b[pos],f[pos]
    require(len(ids)==59309 and len(pos)==7493 and set(f)==set(FOLDS),'native cohort')
    require(all(row['recording_id'][:2] in ('00','01','02','03','04') for row in rows),'player05 excluded')
    experts=FrozenProducerPool(x,yc,bc,fc,eligible,args.producer_cache)
    corrector=FrozenCorrector(yc,bc,fc,eligible,args.corrector_cache)
    pool=CataloguePool(experts,corrector)
    args.output.mkdir(parents=True)
    native={'predicted_K':b.copy(),'zero_group_context_K':b.copy(),'zero_local_audit_K':b.copy()}
    buffers={};fold_results={};manifests=[]
    parent=archive['predicted_K'][pos]
    profiles=audit_profiles(archive['group_proposal'],archive['local_audit_features'],parent,bc)
    for outer in FOLDS:
        print(json.dumps(dict(fold=outer,stage='assemble_independent_inputs')),flush=True)
        train,held,tr,val,manifest=assemble_catalogue_outer(pool,raw,outer);manifests.append(manifest)
        for key,value in [('member_votes',held['member_votes']),('group_proposal',held['proposal']),
                          ('local_audit_features',held['group_features'][...,direct_dim(8):])]:
            require(np.array_equal(value,archive[key][val]),'parent input identity '+key)
        train,train_conn=audit_context(train,'local');held,held_conn=audit_context(held,'local')
        control=CatalogueCritic();control({k:v[:1] for k,v in held.items()},training=False)
        weights=args.archive/f'fold-{outer}.weights.h5'
        require(digest(weights)==proof['folds'][str(outer)]['weights_sha256'],'parent weights identity')
        control.load_weights(str(weights));control_out=predict(control,held)
        diff=max(float(np.max(np.abs(control_out[k]-archive[stored][val]))) for k,stored in
            [('baseline_correct','baseline_correct_probability'),('class_probability','class_probability'),
             ('other_probability','other_probability'),('pooled_class_logits','pooled_class_logits'),('expected_gain','group_expected_gain')])
        reproduced,_=choose_groups(control_out['expected_gain'],held['proposal'],bc[val])
        require(diff<2e-5 and np.array_equal(reproduced,parent[val]),'frozen parent replay')
        del control,control_out;tf.keras.backend.clear_session();gc.collect()
        print(json.dumps(dict(fold=outer,stage='train_contextual_risk',epochs=30,parent_replay_max_diff=diff)),flush=True)
        pred,mask,out,history,model=train_catalogue(train,yc[tr],bc[tr],held,model_factory=ContextualRiskCritic)
        require(model.count_params()==14562,'contextual risk parameter budget')
        native['predicted_K'][pos[val]]=pred
        model.save_weights(str(args.output/f'fold-{outer}.weights.h5'))
        train_out=predict(model,train)
        train_confidence=confidence(train_out,train['proposal'],yc[tr],bc[tr]);del train_out
        for key,value in out.items():
            if key not in buffers:buffers[key]=np.full((len(pos),*value.shape[1:]),np.nan,value.dtype)
            buffers[key][val]=value
        model.zero_group_risk_context=True
        without_context=predict(model,held);model.zero_group_risk_context=False
        require(np.array_equal(without_context['pooled_class_logits'],out['pooled_class_logits']),'risk intervention leaves conditional logits unchanged')
        no_context,_=choose_groups(without_context['expected_gain'],held['proposal'],bc[val]);native['zero_group_context_K'][pos[val]]=no_context
        global_only,_=audit_context(held,'global');without_local=predict(model,global_only)
        no_local,_=choose_groups(without_local['expected_gain'],held['proposal'],bc[val]);native['zero_local_audit_K'][pos[val]]=no_local
        for prefix,intervention in [('zero_context',without_context),('zero_local',without_local)]:
            for key in ['baseline_correct','class_probability','pooled_class_logits','other_probability']:
                name=prefix+'_'+key
                if name not in buffers:buffers[name]=np.full((len(pos),*intervention[key].shape[1:]),np.nan,intervention[key].dtype)
                buffers[name][val]=intervention[key]
        fold_results[str(outer)]=dict(parameters=model.count_params(),training=history,
            weights_sha256=digest(args.output/f'fold-{outer}.weights.h5'),parent_replay_max_difference=diff,
            train_ids_sha256=id_digest(eligible[tr]),held_ids_sha256=id_digest(eligible[val]),
            audit_connection=dict(train=train_conn,held=held_conn),paired=paired(yc[val],bc[val],pred),
            retention=retention(yc[val],bc[val],parent[val],pred),train_confidence=train_confidence,
            held_confidence=confidence(out,held['proposal'],yc[val],bc[val]),
            parent_profile_retention=profile_retention(profiles[val],yc[val],bc[val],parent[val],pred),
            fixed_weights_group_context=paired(yc[val],no_context,pred),fixed_weights_local_audit=paired(yc[val],no_local,pred))
        (args.output/f'fold-{outer}-result.json').write_text(json.dumps(fold_results[str(outer)],indent=2,sort_keys=True)+'\n')
        print(json.dumps(dict(fold=outer,stage='evaluated',paired=fold_results[str(outer)]['paired']['global'],
            versus_parent=fold_results[str(outer)]['retention']['paired']['global'],history=history)),flush=True)
        del train,held,model,out,without_context,without_local,global_only
        tf.keras.backend.clear_session();gc.collect()
    require(all(np.isfinite(v).all() for v in buffers.values()),'complete predictions')
    pred,mask=choose_groups(buffers['expected_gain'],archive['group_proposal'],bc)
    require(np.array_equal(pred,native['predicted_K'][pos]),'global decoder replay')
    active=np.zeros(len(ids),bool);active[pos]=True
    require(all(np.array_equal(v[~active],b[~active]) for v in native.values()),'action scope unchanged')
    arrays=dict(global_index=ids,true_K=y,frozen_baseline_K=b,fold=f,eligible_global_index=eligible,
        parent_K=archive['predicted_K'],parent_audit_profile=profiles,chosen_group_mask=mask,
        group_proposal=archive['group_proposal'],member_votes=archive['member_votes'],local_audit_features=archive['local_audit_features'],
        group_outcome_probability=buffers.pop('outcome_probability'),group_expected_gain=buffers.pop('expected_gain'),
        baseline_correct_probability=buffers.pop('baseline_correct'),**native,**buffers)
    np.savez_compressed(args.output/'predictions.npz',**arrays)
    singles=archive['group_proposal'][:,[2**j-1 for j in range(8)]]
    strong=(bc!=yc)&(archive['group_proposal']==yc[:,None]).any(1)&(singles!=yc[:,None]).all(1)
    report=dict(status='completed',source_commit=os.environ.get('GITHUB_SHA'),promotion=False,independent_validation=False,
        parent_run=37807124287,parent_predictions_sha256=digest(args.archive/'predictions.npz'),predictions_sha256=digest(args.output/'predictions.npz'),
        configuration=dict(candidate_count=8,groups=255,parameters=14562,parent_parameters=13538,added_parameters=1024,
            risk_context='mean of 32 nonlinear full-group states; appended to baseline-risk inputs',
            epochs=30,seed=27402,learning_rate=.002,batch_size=192,loss='unchanged baseline BCE plus conditional categorical CE',
            local_audits_enabled=True,individual_veto=False,threshold_search=False,context_names=names,
            producer_cache_sha256=experts.cache_sha256,corrector_cache_sha256=corrector.cache_sha256,
            folds=list(FOLDS),fold3_used=False,player05_used=False),
        freeze_reference=metrics(y,b),candidate=metrics(y,native['predicted_K']),paired=paired(y,b,native['predicted_K']),
        parent_paired=paired(y,b,archive['predicted_K']),retention=retention(y,b,archive['predicted_K'],native['predicted_K']),
        parent_profile_retention=profile_retention(profiles,yc,bc,parent,pred),
        confidence=confidence(dict(baseline_correct=arrays['baseline_correct_probability'],**buffers),arrays['group_proposal'],yc,bc),
        parent_confidence=confidence(dict(baseline_correct=archive['baseline_correct_probability'],
            pooled_class_logits=archive['pooled_class_logits'],class_probability=archive['class_probability'],other_probability=archive['other_probability']),arrays['group_proposal'],yc,bc),
        combination_only=dict(events=int(strong.sum()),parent_corrected=int((strong&(parent==yc)).sum()),candidate_corrected=int((strong&(pred==yc)).sum())),
        interventions={name:dict(paired_vs_freeze=paired(y,b,native[name]),paired_main_vs_intervention=paired(y,native[name],native['predicted_K']),retrained=False)
            for name in ['zero_group_context_K','zero_local_audit_K']},
        folds=fold_results,provenance=manifests,expert_producers=list(pool.manifests.values()),corrector_manifest=corrector.manifest,
        limitations=['Exposed development folds; no independent validation or automatic promotion.',
            'The added path has 1024 parameters and changes shared-trunk learning; not a parameter-matched causal ablation.',
            'Contextual group states are learned representations, not validated interpretable failure categories.',
            'The 126/92 profile is scored only after prediction; its labels do not gate, train or tune the model.',
            'No new independent information or proposal; this tests use of already available full-group evidence.'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    (args.output/'report.md').write_text(render(report));print(render(report),flush=True)


if __name__=='__main__':main()
