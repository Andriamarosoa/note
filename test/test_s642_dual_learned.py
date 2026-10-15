"""S642: ensure original NOTE is preserved and learned delay is strictly causal."""
from __future__ import annotations
import unittest
import numpy as np

from causal_note.strict_live import HOP,SR,Selection
from causal_note.s642_dual_learned import (
    S642Policy,apply_learned_delay,decode_acoustic,evaluated_policy,
    training_delay_labels,fixed_policies
)


class ConstantDelay:
    def __init__(self,value):self.value=value
    def predict(self,x):return np.full(len(x),self.value,dtype=int)


class ConstantAttack:
    def predict_proba(self,x):
        return np.column_stack((np.zeros(len(x)),np.ones(len(x))))


def row_fixture(n=20):
    p=np.zeros((n,49,3),np.float32)
    p[:8,12,0]=.32;p[:8,12,1]=.75
    k=np.broadcast_to(np.eye(7,dtype=np.float32)[1],(n,7)).copy()
    return dict(ends=np.arange(1,n+1)*HOP,p=p,k=k,
                x=np.zeros((n,49,126),np.float32),n=n*HOP,
                selection=dict(onset=.6,active_on=.5,
                               release_frames=3,refractory_frames=10),
                truth=[dict(pitch=52,time=HOP/SR,duration=.08)])


def event(t,p,kind):
    return dict(type=kind,pitch=p,decision_sample=t,
                available_sample=t,time=t/SR,reason="original",
                velocity=96 if kind=="note_on" else 0)


class S642Tests(unittest.TestCase):
    def test_noop_byte_identical_original(self):
        from scripts.s613_train_strict_live import decode
        row=row_fixture()
        a,info=evaluated_policy(row,None,None,S642Policy())
        b=decode(row["ends"],row["p"],row["k"],
                 Selection(**row["selection"]),row["n"])
        self.assertEqual(a,b)
        self.assertEqual(info["rescued_frame_proposals"],0)
        self.assertEqual(info["held_noteons"],0)

    def test_delayed_noteon_matches_paired_noteoff_and_never_backdates(self):
        row=row_fixture()
        events=[event(HOP,52,"note_on"),event(5*HOP,52,"note_off")]
        output,info=apply_learned_delay(events,row,ConstantDelay(3),
                                        S642Policy(use_delay=True))
        self.assertEqual([e["available_sample"] for e in output],
                         [4*HOP,8*HOP])
        self.assertEqual([e["decision_sample"] for e in output],
                         [HOP,5*HOP])
        self.assertEqual(info["delays"],[3])
        self.assertTrue(all(e["decision_sample"]<=e["available_sample"] for e in output))

    def test_same_pitch_restrike_never_overlaps(self):
        row=row_fixture()
        events=[event(HOP,52,"note_on"),event(4*HOP,52,"note_off"),
                event(4*HOP,52,"note_on"),event(12*HOP,52,"note_off")]
        class PickFirst:
            def predict(self,x):return np.array([3,0],dtype=int)
        output,info=apply_learned_delay(events,row,PickFirst(),
                                        S642Policy(use_delay=True))
        self.assertEqual(info["delays"],[3,3])
        self.assertEqual([e["type"] for e in output],
                         ["note_on","note_off","note_on","note_off"])
        self.assertEqual(output[1]["available_sample"],output[2]["available_sample"])

    def test_generator_rescue_and_derived_train_labels(self):
        row=row_fixture()
        original,_=decode_acoustic(row,None,S642Policy())
        rescued,num=decode_acoustic(row,ConstantAttack(),
                                   S642Policy(use_candidate=True))
        self.assertEqual(sum(e["type"]=="note_on" for e in original),0)
        self.assertGreaterEqual(sum(e["type"]=="note_on" for e in rescued),1)
        self.assertGreater(num,0)
        row["p"][0,12,0]=.9
        row["truth"][0]["time"]=HOP/SR+2*HOP/SR
        features,target,count=training_delay_labels(row)
        self.assertEqual(features.shape[1],72)
        self.assertEqual(len(target),count)
        self.assertTrue(np.all((target>=0)&(target<=3)))

    def test_invalid_latency_and_original_policies(self):
        with self.assertRaises(ValueError):S642Policy(rescue_min_confidence=1.5)
        self.assertEqual([n for n,p in fixed_policies()],
                         ["S614_original","S642_candidate_only",
                          "S642_delay_only","S642_joint"])


if __name__=="__main__":unittest.main()
