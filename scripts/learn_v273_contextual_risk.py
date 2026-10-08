"""Let the shared regression probability see learned full-group/audit interactions.

The original critic averages raw audits before its baseline-risk MLP. Here the
same MLP also receives the mean of the nonlinear 32-dimensional group states.
Group identity, members, destination, acoustics and audit rates interact before
this reduction. The original critic and all complete proposals remain available.
"""
from __future__ import annotations

import tensorflow as tf

from scripts.learn_v273_catalogue import CatalogueCritic
from scripts.learn_v273_group_consensus import consensus_outputs
from scripts.v273_catalogue_contract import direct_dim


class ContextualRiskCritic(CatalogueCritic):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Inference intervention only. No fit or threshold search uses this flag.
        self.zero_group_risk_context = False

    def call(self, x, training=False):
        features=tf.cast(x['group_features'],tf.float32)
        base=tf.cast(x['baseline'],tf.float32)
        context=tf.cast(x['context'],tf.float32)
        proposals=tf.cast(x['proposal'],tf.int32)
        votes=tf.cast(x['member_votes'],tf.float32)
        count=votes.shape[1];groups=2**count-1;n=tf.shape(features)[0]
        if features.shape[1:]!=(groups,direct_dim(count)+9):
            raise ValueError('full catalogue descriptors required')
        baseline=tf.argmax(base,axis=1,output_type=tf.int32)
        active=tf.cast(proposals!=baseline[:,None],tf.float32)
        cgroup=tf.broadcast_to(context[:,None],[n,groups,tf.shape(context)[1]])
        bgroup=tf.broadcast_to(base[:,None],[n,groups,7])
        state=self.group_hidden(tf.concat([features,cgroup,bgroup,tf.one_hot(proposals,7)],-1))
        state=self.group_interaction(state)
        conditional_logits=tf.squeeze(self.conditional_logit(state),-1)
        denominator=tf.maximum(tf.reduce_sum(active,axis=1,keepdims=True),1.)
        audit_summary=tf.reduce_sum(features[...,direct_dim(count):]*active[:,:,None],axis=1)/denominator
        group_risk_context=tf.reduce_sum(state*active[:,:,None],axis=1)/denominator
        if self.zero_group_risk_context:
            group_risk_context=tf.zeros_like(group_risk_context)
        row=self.baseline_hidden(tf.concat([context,base,tf.reshape(votes,[n,5*count]),audit_summary,group_risk_context],-1))
        baseline_logits=tf.squeeze(self.baseline_logit(row),-1)
        result=consensus_outputs(conditional_logits,proposals,baseline_logits,baseline,'pooled_ce')
        result['group_risk_context']=group_risk_context
        return result
