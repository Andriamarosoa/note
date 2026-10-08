"""Frozen-weight replay of train/held inputs and producer-regime diagnostics.

No optimizer, fit, early stopping, calibration or model selection is performed.
All held labels are read only when scoring already constructed predictions.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from scripts.audit_v273_selector_design import aligned_positions,load_feature_rows
from scripts.v273_selector_contract import matrices,FOLDS,require,id_digest
from scripts.prepare_v273_group127_producers import FrozenProducerPool
from scripts.prepare_v273_learned_corrector import FrozenCorrector,CataloguePool
from scripts.v273_catalogue_contract import assemble_catalogue_outer,joint_groups,direct_dim
from scripts.learn_v273_catalogue import CatalogueCritic
from scripts.v273_confidence_metrics import summarize,snapshot
from scripts.yourmt3_exactk_common import digest


def infer(model,x):
    keys=['baseline_logits','baseline_correct','pooled_class_logits','class_probability','other_probability','expected_gain']
    chunks=[]
    for start in range(0,len(x['baseline']),192):
        out=model({k:v[start:start+192] for k,v in x.items()},training=False)
        chunks.append({k:out[k].numpy() for k in keys})
    return {k:np.concatenate([c[k] for c in chunks]) for k in keys}


def global_audits(refp,y,b,query,qbase):
    """Only enabled fields; matches the original Laplace-smoothed global prior."""
    result=np.zeros((*query.shape,9),np.float32)
    correction=(refp==y[:,None])&(y[:,None]!=b[:,None])
    regression=(refp!=b[:,None])&(y[:,None]==b[:,None])
    for source in [2,3,4]:
        fit=b==source;receive=np.flatnonzero(qbase==source)
        require(fit.any(),'missing global reference source')
        for target in range(2,7):
            same=refp[fit]==target;used=same.sum(0)
            counts=np.stack([(same&correction[fit]).sum(0),(same&regression[fit]).sum(0),
                             (same&~correction[fit]&~regression[fit]).sum(0)],1)
            prior=((counts+1)/(used[:,None]+3)).astype(np.float32)
            support=(used/fit.sum()).astype(np.float32)
            for i in receive:
                take=query[i]==target;result[i,take,:3]=prior[take];result[i,take,6]=support[take]
    return result


def profile(inputs,y,b,offset):
    proposals=inputs['proposal'];active=proposals!=b[:,None]
    history=(inputs['group_features'][...,offset:]*active[:,:,None]).sum(1)/np.maximum(active.sum(1)[:,None],1)
    v=inputs['member_votes']
    return dict(rows=len(y),history_mean=history.mean(0).tolist(),history_std=history.std(0).tolist(),
        candidate_max_probability_mean=v.max(2).mean(0).tolist(),
        candidate_entropy_mean=(-(v*np.log(np.maximum(v,1e-12))).sum(2)).mean(0).tolist(),
        wrong_baseline_reachable_rate=float((proposals[b!=y]==y[b!=y,None]).any(1).mean()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['reference','features','producer-cache','corrector-cache','archive','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arm',choices=['control7','corrector8'],required=True)
    args=p.parse_args();require(not args.output.exists(),'refusing overwrite')
    root=Path(__file__).resolve().parents[1]
    proof=json.loads((root/'analysis/evidence/v273-learned-selection/verification.json').read_text())
    require(digest(args.archive/'predictions.npz')==proof['arms'][args.arm]['predictions_sha256'],'archive digest')
    require(digest(args.reference)==proof['source_reference_sha256'],'native reference digest')
    with np.load(args.archive/'predictions.npz') as z:archive={k:z[k] for k in z.files}
    ids,y,b,f,eligible=(archive[k] for k in ['global_index','true_K','frozen_baseline_K','fold','eligible_global_index'])
    pos=aligned_positions(ids,eligible);rows,_=load_feature_rows(args.features,ids,y,b,f,pos)
    x,raw,names=matrices(rows);yc,bc,fc=y[pos],b[pos],f[pos]
    experts=FrozenProducerPool(x,yc,bc,fc,eligible,args.producer_cache)
    corrector=FrozenCorrector(yc,bc,fc,eligible,args.corrector_cache)
    count=8 if args.arm=='corrector8' else 7;offset=direct_dim(count)
    pool=CataloguePool(experts,corrector if count==8 else None)
    args.output.mkdir(parents=True)
    report=dict(arm=args.arm,source_run=proof['run_id'],source_commit=proof['source_commit'],
        source_predictions_sha256=digest(args.archive/'predictions.npz'),
        optimizer_used=False,weights_changed=False,independent_validation=False,folds={},files={})
    for outer in FOLDS:
        train,held,tr,val,manifest=assemble_catalogue_outer(pool,raw,outer)
        for key,value in [('member_votes',held['member_votes']),('group_proposal',held['proposal']),
                          ('local_audit_features',held['group_features'][...,offset:])]:
            require(np.array_equal(value,archive[key][val]),'held input replay '+key)
        local_columns=[offset+j for j in [3,4,5,7,8]]
        train['group_features'][...,local_columns]=0;held['group_features'][...,local_columns]=0
        canonical=global_audits(train['proposal'],yc[tr],bc[tr],held['proposal'],bc[val])
        require(np.array_equal(canonical,held['group_features'][...,offset:]),'global-history reconstruction')
        model=CatalogueCritic();model({k:v[:1] for k,v in train.items()},training=False)
        weights=args.archive/f'fold-{outer}.weights.h5'
        require(digest(weights)==proof['arms'][args.arm]['weights_sha256'][str(outer)],'weights digest')
        model.load_weights(str(weights));model.trainable=False
        held_out=infer(model,held)
        maxdiff=max(float(np.max(np.abs(held_out[k]-archive[stored][val]))) for k,stored in
                    [('baseline_correct','baseline_correct_probability'),('class_probability','class_probability'),
                     ('other_probability','other_probability'),('pooled_class_logits','pooled_class_logits'),
                     ('expected_gain','group_expected_gain')])
        require(maxdiff<2e-5,'frozen held replay tolerance')
        train_out=infer(model,train)
        fold_result=dict(weights_sha256=digest(weights),held_replay_max_absolute_difference=maxdiff,
            parameters=model.count_params(),train_ids_sha256=id_digest(eligible[tr]),held_ids_sha256=id_digest(eligible[val]),
            train_profile=profile(train,yc[tr],bc[tr],offset),held_profile=profile(held,yc[val],bc[val],offset),
            train=summarize(snapshot(train_out,train,yc[tr],bc[tr],eligible[tr],fc[tr])),
            held=summarize(snapshot(held_out,held,yc[val],bc[val],eligible[val],fc[val])),regime_diagnostics=[])
        require(fold_result['held']['net']==proof['arms'][args.arm]['fold_nets'][str(outer)],'held replay decisions')
        def save(name,out,inp,at):
            path=args.output/f'fold-{outer}-{name}.npz'
            np.savez_compressed(path,**snapshot(out,inp,yc[at],bc[at],eligible[at],fc[at]))
            report['files'][path.name]=dict(sha256=digest(path),rows=len(at))
        save('train',train_out,train,tr);save('held',held_out,held,val)
        # Every possible omitted training fold is used, never selected by score.
        for omitted in sorted(set(FOLDS)-{outer}):
            permitted=set(FOLDS)-{outer,omitted};ref=np.flatnonzero(np.isin(fc,list(permitted)))
            require(not set(eligible[ref])&set(eligible[val]),'diagnostic reference overlap')
            votes2=pool.predict(permitted,val,{outer,omitted})
            pp2,dd2,_=joint_groups(votes2,bc[val])
            rp,_=pool.crossfit(ref,{outer,omitted});refp,_,_=joint_groups(rp,bc[ref])
            for regime,used_votes,proposals,direct,refs,refprops in [
                ('votes2_audits3',votes2,pp2,dd2,tr,train['proposal']),
                ('votes3_audits2',held['member_votes'],held['proposal'],held['group_features'][...,:offset],ref,refp),
                ('votes2_audits2',votes2,pp2,dd2,ref,refp)]:
                audits=global_audits(refprops,yc[refs],bc[refs],proposals,bc[val])
                inp=dict(context=held['context'],baseline=held['baseline'],member_votes=used_votes,
                         proposal=proposals,group_features=np.concatenate([direct,audits],2))
                out=infer(model,inp);summary=summarize(snapshot(out,inp,yc[val],bc[val],eligible[val],fc[val]))
                fold_result['regime_diagnostics'].append(dict(regime=regime,omitted_fold=omitted,
                    vote_fit_folds=sorted(permitted) if regime.startswith('votes2') else sorted(set(FOLDS)-{outer}),
                    audit_reference_folds=sorted(set(fc[refs])),reference_ids_sha256=id_digest(eligible[refs]),
                    summary=summary))
                save(f'{regime}-omit{omitted}',out,inp,val)
                del inp,out,audits
        require(digest(weights)==fold_result['weights_sha256'],'weights file changed during replay')
        report['folds'][str(outer)]=fold_result
        print(json.dumps(dict(arm=args.arm,fold=outer,replay_max_diff=maxdiff,
            train_nll=fold_result['train']['nll'],held_nll=fold_result['held']['nll'],
            train_gain=fold_result['train']['policies']['accepted_best'],held_gain=fold_result['held']['policies']['accepted_best'])),flush=True)
        del train,held,train_out,held_out,model
        tf.keras.backend.clear_session();gc.collect()
    report['status']='completed'
    (args.output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')


if __name__=='__main__':main()
