"""Both symmetric fixed-factor crossovers of parent and contextual-risk models.

No fit or label-dependent gate. Every factor is an outer-excluded prediction.
Both crossovers are kept; comparing them is a development diagnostic.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scripts.verify_v273_learned_selection import read_npz,choose
from scripts.verify_v273_selector_repair_artifacts import require,digest,metrics,named_pairs
from scripts.v273_confidence_metrics import summarize
from scripts.v273_contextual_risk_contract import audit_profiles,retention,profile_retention


def combine(r,logits,proposals,baseline):
    """The same coherent factorization as the trained critics, with fixed inputs."""
    r=np.asarray(r,float);logits=np.asarray(logits,float);b=np.asarray(baseline,int)
    require(logits.shape==(len(b),7) and r.shape==(len(b),),'factor dimensions')
    require(np.isfinite(r).all() and np.isfinite(logits).all() and ((r>=0)&(r<=1)).all(),'finite probability factors')
    available=np.stack([(proposals==k).any(1)&(b!=k) for k in range(7)],1)
    full=np.column_stack([np.where(available,logits,-1.e9),np.zeros(len(b))])
    q=np.exp(full-logsumexp(full,axis=1,keepdims=True))
    cp=(1-r[:,None])*q[:,:7]+np.eye(7)[b]*r[:,None]
    other=(1-r)*q[:,7]
    gain=(np.take_along_axis(cp,proposals,axis=1)-r[:,None])*(proposals!=b[:,None])
    prediction,mask=choose(gain,proposals,b)
    return dict(baseline_correct=r,pooled_class_logits=logits,class_probability=cp,other_probability=other,
        prediction=prediction,mask=mask)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['parent','candidate','output']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();require(not args.output.exists(),'refusing to overwrite factor audit');args.output.mkdir(parents=True)
    parent=read_npz(args.parent);candidate=read_npz(args.candidate)
    for key in ['global_index','true_K','frozen_baseline_K','fold','eligible_global_index','group_proposal','member_votes','local_audit_features']:
        require(np.array_equal(parent[key],candidate[key]),'identical factor context '+key)
    ids,y,b,f,eligible=(candidate[k] for k in ['global_index','true_K','frozen_baseline_K','fold','eligible_global_index'])
    idx={int(v):i for i,v in enumerate(ids)};pos=np.array([idx[int(g)] for g in eligible]);yc,bc=y[pos],b[pos]
    proposal=candidate['group_proposal'];profiles=audit_profiles(proposal,parent['local_audit_features'],parent['predicted_K'][pos],bc)
    for name,source in [('parent',parent),('candidate',candidate)]:
        replay=combine(source['baseline_correct_probability'],source['pooled_class_logits'],proposal,bc)
        require(np.array_equal(replay['prediction'],source['predicted_K'][pos]),'unchanged own-factor replay '+name)
        require(np.allclose(replay['class_probability'],source['class_probability'],atol=1e-6),'own-factor probability replay')
    arrays={k:candidate[k] for k in ['global_index','true_K','frozen_baseline_K','fold','eligible_global_index']}
    report=dict(scope='post-training development diagnostic; both fixed crossovers; no fitting or model selection',
        independent_validation=False,promotion=False,labels_used_to_construct_predictions=False,
        sources=dict(parent_sha256=digest(args.parent),candidate_sha256=digest(args.candidate)),variants={})
    for name,rsource,qsource in [('parent_risk_new_correction',parent,candidate),('new_risk_parent_correction',candidate,parent)]:
        out=combine(rsource['baseline_correct_probability'],qsource['pooled_class_logits'],proposal,bc)
        native=b.copy();native[pos]=out.pop('prediction');out.pop('mask')
        summary=summarize(dict(truth=yc,baseline=bc,proposal=proposal,**out))
        require(summary['net']==named_pairs(y,b,native)['global']['net'],'coherent summary decisions')
        arrays[name+'_K']=native
        for key,value in out.items():arrays[name+'__'+key]=value
        report['variants'][name]=dict(metrics=metrics(y,native),paired=named_pairs(y,b,native),
            versus_parent=retention(y,b,parent['predicted_K'],native),versus_new=retention(y,b,candidate['predicted_K'],native),
            parent_profile_retention=profile_retention(profiles,yc,bc,parent['predicted_K'][pos],native[pos]),
            confidence=summary,retained=True,retention_depends_on_global_gain=False,downstream_ready=False)
    selected=candidate['predicted_K'][pos]!=bc
    report['same_new_selected_cohort']=dict(rows=int(selected.sum()),observed_baseline_correct=float((yc[selected]==bc[selected]).mean()),
        parent_risk_mean=float(parent['baseline_correct_probability'][selected].mean()),
        new_risk_mean=float(candidate['baseline_correct_probability'][selected].mean()))
    np.savez_compressed(args.output/'factor-swaps.npz',**arrays)
    report['predictions_sha256']=digest(args.output/'factor-swaps.npz')
    (args.output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(dict(variants={k:dict(paired=v['paired']['global'],versus_parent=v['versus_parent']['paired']['global'],
        announced=v['confidence']['expected_net']) for k,v in report['variants'].items()},same_new_selected_cohort=report['same_new_selected_cohort'])),flush=True)


if __name__=='__main__':main()
