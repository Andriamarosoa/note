from __future__ import annotations
import unittest
import numpy as np
from causal_note.strict_live import Selection,HOP,CausalSelection
from causal_note.s617_live_attack_bridge import NativeAttackBridge, BridgeParameters

class S617Tests(unittest.TestCase):
    def test_different_futures_cannot_change_emitted_past(self):
        rng=np.random.default_rng(617)
        p=rng.random((100,49,3)).astype(np.float32)
        k=np.broadcast_to(np.eye(7,dtype=np.float32)[2],(100,7))
        x=rng.random((100,49,126)).astype(np.float32)
        def play(p,x):
            engine=NativeAttackBridge(Selection(),BridgeParameters())
            out=[]
            for i in range(100):
                out.extend(engine.step(p[i],k[i],(i+1)*HOP,(i+1)*HOP,x[i]))
            return [e for e in out if e["decision_sample"]<=60*HOP]
        past=play(p,x)
        alteredp=p.copy(); alteredx=x.copy()
        alteredp[60:]*=0.01;alteredx[60:]*=50.
        self.assertEqual(past,play(alteredp,alteredx))

    def test_still_uses_native_attack_not_sustain_only(self):
        engine=NativeAttackBridge(Selection(onset=.6,active_on=.5),
                                  BridgeParameters(.75,.1,.05))
        p=np.zeros((49,3),np.float32)
        x=np.zeros((49,126),np.float32)
        p[12,1]=.95
        x[12,[0,8,16]]=1.
        self.assertEqual(engine.step(p,np.eye(7)[1],HOP,HOP,x),[])
        p[12,0]=.85;p[12,1]=.3
        x[12,[0,8,16]]=1.6
        self.assertIn(52,[v["pitch"] for v in engine.step(p,np.eye(7)[1],2*HOP,2*HOP,x)
                          if v["type"]=="note_on"])
    def test_invalid(self):
        with self.assertRaises(ValueError):BridgeParameters(.4,.3,.1)

if __name__=="__main__":unittest.main()
