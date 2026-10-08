"""Event-weighted probability diagnostics, independent of TensorFlow."""
from __future__ import annotations
import numpy as np
from scipy.special import logsumexp, expit


def distributions(data):
    b=np.asarray(data['baseline'],int);p=data['proposal'];n=len(b)
    available=np.stack([(p==k).any(1)&(b!=k) for k in range(7)],1)
    logits=np.concatenate([np.where(available,data['pooled_class_logits'],-1.e9),np.zeros((n,1))],1).astype(float)
    logq=logits-logsumexp(logits,axis=1,keepdims=True)
    return available,logq,np.exp(logq)


def summarize(data):
    y=np.asarray(data['truth'],int);b=np.asarray(data['baseline'],int);row=np.arange(len(y))
    r=np.asarray(data['baseline_correct'],float);cp=np.asarray(data['class_probability'],float)
    available,logq,q=distributions(data);number=available.sum(1);has=number>0;B=b==y
    expected=(1-r[:,None])*q[:,:7]-r[:,None]
    actual=(y[:,None]==np.arange(7)).astype(float)-B[:,None]
    best=np.where(available,expected,-np.inf).argmax(1)
    best_gain=expected[row,best];accept=has&(best_gain>0)
    prediction=np.where(accept,best,b);right=prediction==y
    target=np.where(available[row,y],y,7)
    conditional_ce=-logq[row,target]
    if 'baseline_logits' in data:
        z=np.asarray(data['baseline_logits'],float)
        bce=np.logaddexp(0,z)-B*z
    else:
        bce=-np.log(np.maximum(np.where(B,r,1-r),1e-12))
    selected_q=q[row,prediction]
    base_component=(B.astype(float)-r)*(1+selected_q)
    conditional_component=(1-B)*(selected_q-right)
    observed_gain=right.astype(float)-B
    gain=np.where(accept,cp[row,prediction]-r,0)
    cor=accept&right;reg=accept&B
    def mean(x,t):return float(np.asarray(x)[t].mean()) if np.any(t) else None
    def policy(predicted,observed,t):
        return dict(rows=int(t.sum()),predicted_mean_gain=mean(predicted,t),observed_mean_gain=mean(observed,t),
            optimism_per_row=mean(predicted-observed,t))
    uniform_expected=(expected*available).sum(1)/np.maximum(number,1)
    uniform_actual=(actual*available).sum(1)/np.maximum(number,1)
    result=dict(rows=len(y),baseline_correct=int(B.sum()),baseline_accuracy=float(B.mean()),
        changes=int(accept.sum()),corrections=int(cor.sum()),regressions=int(reg.sum()),
        net=int(cor.sum()-reg.sum()),expected_corrections=float(cp[row,prediction][accept].sum()),
        expected_regressions=float(r[accept].sum()),expected_net=float(gain.sum()),
        optimism=float(gain.sum()-observed_gain.sum()),
        nll=dict(event=float((bce+(1-B)*conditional_ce).mean()),baseline=float(bce.mean()),
            conditional_weighted_by_all_events=float(((1-B)*conditional_ce).mean()),
            conditional_on_baseline_wrong=mean(conditional_ce,~B)),
        baseline_brier=float(np.mean((r-B)**2)),
        conditional_brier_on_baseline_wrong=mean(((q-np.eye(8)[target])**2).sum(1),~B),
        reference_all=dict(predicted=float(r.mean()),observed=float(B.mean())),
        reference_selected=dict(predicted=mean(r,accept),observed=mean(B,accept)),
        conditional_selected_baseline_wrong=dict(rows=int((accept&~B).sum()),
            predicted=mean(selected_q,accept&~B),observed=mean(right,accept&~B)),
        other_baseline_wrong=dict(predicted=mean(q[:,7],~B),observed=mean(~available[row,y],~B)),
        other_selected_baseline_wrong=dict(predicted=mean(q[:,7],accept&~B),observed=mean(~available[row,y],accept&~B)),
        selected_wrong_baseline_without_correct_alternative=int((accept&~B&~available[row,y]).sum()),
        selected_wrong_baseline_with_correct_alternative_but_wrong_choice=int((accept&~B&available[row,y]&~right).sum()),
        alternative_count={str(k):int((number==k).sum()) for k in range(5)},
        policies=dict(uniform_over_distinct_alternatives=policy(uniform_expected,uniform_actual,has),
            best_without_acceptance_gate=policy(best_gain,actual[row,best],has),
            accepted_best=policy(best_gain,actual[row,best],accept),
            rejected_best=policy(best_gain,actual[row,best],has&~accept)),
        selected_optimism_decomposition=dict(baseline_component=float(base_component[accept].sum()),
            conditional_component=float(conditional_component[accept].sum()),
            formula='(B-r)*(1+q) + (1-B)*(q-I(selected_K=true_K)); a fixed algebraic decomposition, not causal attribution'),
        best_gain_bins=[],selected_by_alternative_count={})
    edges=[-1.000001,-.5,-.2,-.1,0,.05,.1,.2,.3,.5,1.000001]
    for lo,hi in zip(edges[:-1],edges[1:]):
        t=has&(best_gain>=lo)&(best_gain<hi)
        result['best_gain_bins'].append(dict(lower=max(lo,-1),upper=min(hi,1),**policy(best_gain,actual[row,best],t)))
    for k in range(1,5):
        result['selected_by_alternative_count'][str(k)]=policy(best_gain,actual[row,best],accept&(number==k))
    return result


def snapshot(out,inputs,truth,baseline,ids,fold):
    data=dict(truth=np.asarray(truth),baseline=np.asarray(baseline),global_index=np.asarray(ids),fold=np.asarray(fold),
              proposal=inputs['proposal'])
    for k in ['baseline_logits','baseline_correct','pooled_class_logits','class_probability','other_probability']:
        data[k]=out[k]
    return data
