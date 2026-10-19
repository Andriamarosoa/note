"""S756 adversarial note-instance safety and causal birth qualification."""
import unittest
import numpy as np

from causal_note.strict_live import SR,HOP
from causal_note.s746_instance_guard import InstanceGate,merge_events
from scripts.s756_source_protected_k_birth_mux import (
    GRID,RISK,pairs_from_events,select_novel_births)

def event(kind,pitch,at,reason="synthetic"):
    return dict(type=kind,pitch=pitch,decision_sample=int(at),
        available_sample=int(at),time=float(at/SR),
        reason=reason,velocity=90 if kind=="note_on" else 0)


class ProtectedBirthTests(unittest.TestCase):
    def setUp(self):
        self.ends=np.arange(1,11,dtype=np.int64)*4*HOP
        self.native=[event("note_on",40,self.ends[0],"s708_on"),
                     event("note_off",40,self.ends[3],"s707_causal_hazard")]
        self.post=np.zeros((len(self.ends),49,3),np.float32)
        self.post[:,:,0]=.1
        self.post[:,:,1]=.1
        self.post[1,1,:]=(.9,.85,.1)
        self.native_k=np.zeros((len(self.ends),7),np.float32)
        self.native_k[:,1]=.95
        self.native_k[:,0]=.05
        self.poly=self.native_k.copy()
        self.poly[1,:]=np.array([.02,.13,.85,0,0,0,0],np.float32)
        self.candidate=[
            event("note_on",40,self.ends[0],"s708_simulated"),
            event("note_on",41,self.ends[1],"count_supported_acoustic_attack"),
            event("note_off",40,self.ends[3],"s707_causal_hazard"),
            event("note_off",41,self.ends[6],"s707_causal_hazard")]

    def select(self,candidate=None,reference=None):
        return select_novel_births(
            self.native if reference is None else reference,
            self.candidate if candidate is None else candidate,
            self.ends,self.post,self.native_k,self.poly)

    def test_four_models_three_gates(self):
        self.assertEqual(len(GRID),12)
        self.assertEqual(RISK,(.4,.55,.7))

    def test_does_not_reemit_active_signed_pitch(self):
        proposals,score,indices,diag=self.select()
        self.assertEqual(len(proposals),1)
        self.assertEqual(int(proposals[0][0]["pitch"]),41)
        self.assertEqual(diag["blocked_source_active"],1)
        self.assertEqual(indices,[(1,1)])
        self.assertGreater(float(score[0]),.7)

    def test_signed_noteon_noteoff_identical_even_with_extra(self):
        pairs,prob,ix,_=self.select()
        result,counts=merge_events(
            self.native,pairs,prob,ix,self.poly,
            InstanceGate(risk=.55,k_tail=.1))
        self.assertEqual(counts["accepted"],1)
        expected=[tuple(e.items()) for e in self.native]
        actual=[tuple(e.items()) for e in result
                if not e.get("_s746_extra") and
                not str(e.get("reason","")).startswith("s746_")]
        self.assertEqual(expected,actual)

    def test_candidate_off_does_not_change_acceptance_score(self):
        _,short_prob,_,_=self.select()
        late=[dict(x) for x in self.candidate]
        late[-1]=event("note_off",41,self.ends[8],"s707_late_release")
        _,late_prob,_,_=self.select(late)
        np.testing.assert_array_equal(short_prob,late_prob)

    def test_distant_future_signed_note_not_seen_at_birth(self):
        original=self.select()
        # At t=2048, the future S708 NoteOn pitch 41 at t=8*1024
        # cannot affect the NOTE-on decision or acoustic ranking.
        late=self.native+[
            event("note_on",41,self.ends[7],"later_s708"),
            event("note_off",41,self.ends[8],"later_s707")]
        other=self.select(reference=late)
        self.assertEqual(len(original[0]),len(other[0]))
        np.testing.assert_array_equal(original[1],other[1])

    def test_future_signed_note_preempts_new_pitch_not_native(self):
        pairs,confidence,indices,_=self.select()
        signed=self.native+[
            event("note_on",41,self.ends[5],"s708_future_on"),
            event("note_off",41,self.ends[8],"s707_future_off")]
        stream,diagnostics=merge_events(
            signed,pairs,confidence,indices,self.poly,
            InstanceGate(risk=.55,k_tail=.1))
        self.assertEqual(diagnostics["accepted"],1)
        self.assertEqual(diagnostics["preempted"],1)
        self.assertIn("s746_source_priority",{
            e["reason"] for e in stream})
        self.assertEqual(
            [e["reason"] for e in stream if e["reason"] in
             {"s708_future_on","s707_future_off"}],
            ["s708_future_on","s707_future_off"])

    def test_bad_opaque_pair_timing_rejected(self):
        bad=[event("note_on",41,self.ends[1]),
             event("note_off",41,self.ends[0])]
        with self.assertRaises(AssertionError):
            self.select(bad)

    def test_orphan_candidate_noteoff_rejected(self):
        with self.assertRaises(AssertionError):
            pairs_from_events([event("note_off",41,self.ends[2])])

if __name__=="__main__":
    unittest.main()
