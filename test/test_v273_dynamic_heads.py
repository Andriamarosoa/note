"""Synthetic safeguards for per-event specialist-head routing."""
from __future__ import annotations
import unittest

import numpy as np

from scripts.evaluate_v273_dynamic_heads import (
    SoftHeadGate, EXPERTS, CLASSES, fixed_head, gate_inputs,
    policy_predictions, KEEP_BASE
)


class DynamicHeadTests(unittest.TestCase):
    def test_convex_mixture_and_variable_gate(self):
        # On a synthetic signal, one specialist is reliable in context A,
        # another in B. The gate cannot inspect the true class at inference.
        n=480
        rng=np.random.default_rng(27402)
        true=np.asarray([2+i%5 for i in range(n)],int)
        base=np.asarray([2+(i+1)%3 for i in range(n)],int)
        ctx=np.where(np.arange(n)%2==0,-3.,3.)
        p=np.full((n,len(EXPERTS)+1,5),.035,float)
        p[:,0,:]=fixed_head(base)
        p[:,1:,:]=.035
        p[np.arange(n),np.where(ctx<0,1,2),true-2]=.86
        # Wrong specialist tends to vote for adjacent class.
        p[np.arange(n),np.where(ctx<0,2,1),(true-1)%5]=.86
        # Other experts are uninformative but valid distributions.
        for k in range(3,len(EXPERTS)+1):
            p[:,k,:]=.2
        p/=p.sum(axis=2,keepdims=True)
        raw=ctx[:,None]
        z=gate_inputs(p,base,raw)
        gate=SoftHeadGate().fit(z,p,true)
        probs,weights=gate.predict(z,p)
        self.assertTrue(np.isfinite(weights).all())
        self.assertTrue(np.allclose(probs.sum(axis=1),1.,atol=1e-7))
        self.assertTrue(np.allclose(weights.sum(axis=1),1.,atol=1e-7))
        self.assertGreater(np.mean(weights[ctx<0,1]),np.mean(weights[ctx<0,2]))
        self.assertGreater(np.mean(weights[ctx>0,2]),np.mean(weights[ctx>0,1]))
        self.assertGreater(np.mean(CLASSES[np.argmax(probs,axis=1)]==true),.66)

    def test_all_policies_preserve_poly_classes(self):
        base=np.asarray([2,3,4,2])
        p=np.full((4,len(EXPERTS)+1,5),.2)
        for i in range(4):
            p[i,0]=fixed_head([base[i]])[0]
        probs=np.array([
            [.02,.02,.9,.03,.03],
            [.01,.9,.03,.03,.03],
            [.7,.1,.1,.05,.05],
            [.9,.025,.025,.025,.025]
        ])
        out,_,_,_=policy_predictions(probs,p,base)
        self.assertEqual(set(out),{"ungated","confidence_only","agreement_guard"})
        for x in out.values():
            self.assertTrue(np.isin(x,CLASSES).all())
            self.assertEqual(len(x),len(base))


if __name__=="__main__":
    unittest.main()
