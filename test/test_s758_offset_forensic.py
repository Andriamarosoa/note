"""S758 independent synthetic Hungarian reassignment falsification tests."""
import unittest
import numpy as np
from causal_note.strict_live import SR,HOP
from scripts.s758_forensic_offset_rematches import inspect_one

def ev(kind,pitch,t,reason="s708_native",extra=False):
    sample=int(round(t*SR))
    event=dict(type=kind,pitch=int(pitch),
        decision_sample=sample,available_sample=sample,
        time=sample/SR,reason=reason,
        velocity=90 if kind=="note_on" else 0)
    if extra:event["_s746_extra"]=True
    return event

class OffsetForensicTests(unittest.TestCase):
    def setUp(self):
        self.ends=np.arange(1,110,dtype=np.int64)*HOP
        self.truth=[dict(pitch=40,time=.19,duration=.035)]
        self.signed=[
            ev("note_on",40,.13),
            ev("note_off",40,.22,"s707_causal_hazard")]
        self.injected=[
            self.signed[0],self.signed[1],
            ev("note_on",40,.23,"s746_extra_on",True),
            ev("note_off",40,.31,"s746_extra_off",True)]

    def test_real_same_pitch_hungarian_rematch_can_lose_offset(self):
        cases,causes,old,new=inspect_one(
            "synthetic",self.signed,self.injected,self.truth,self.ends)
        self.assertEqual(len(cases),1)
        self.assertEqual(causes,{"reassigned_to_supplemental":1})
        self.assertEqual(cases[0]["cause"],"reassigned_to_supplemental")
        self.assertLess(old["offset_accuracy"],1.0000001)
        self.assertEqual(old["offset_correct"],1)
        self.assertEqual(new["offset_correct"],0)
        self.assertTrue(cases[0]["trial_on"]["is_extra"])

    def test_no_supplemental_no_loss(self):
        cases,causes,old,new=inspect_one(
            "synthetic",self.signed,self.signed,self.truth,self.ends)
        self.assertEqual(cases,[])
        self.assertEqual(causes,{})
        for name in ("offset_accuracy","full_note_f1","exact_k"):
            self.assertEqual(old[name],new[name])

    def test_reference_signature_protection_detects_tamper(self):
        bad=[dict(e) for e in self.injected]
        bad[0]["time"]+=.001
        bad[0]["decision_sample"]+=1
        bad[0]["available_sample"]+=1
        with self.assertRaises(AssertionError):
            inspect_one("synthetic",self.signed,bad,self.truth,self.ends)

if __name__=="__main__":
    unittest.main()
