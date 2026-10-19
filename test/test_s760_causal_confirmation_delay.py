"""S760 verify delayed causal commitment with no event backdating.

Synthetic 23/46ms collisions mirror S758's three REAL GuitarSet CAL32
Offset losses. No CAL labels are read in either this test or inference.
"""
import unittest
import numpy as np
from causal_note.strict_live import SR,HOP
from scripts.s760_causal_confirmation_delay import (
    DELAYS,GRID,select_novel_births)

BLOCK=4*HOP

def e(kind,pitch,at,reason="signed"):
    return dict(type=kind,pitch=int(pitch),
        decision_sample=int(at),available_sample=int(at),
        time=float(at/SR),reason=reason,
        velocity=96 if kind=="note_on" else 0)

class ConfirmationDelayTests(unittest.TestCase):
    def case(self,*,collision_blocks=1,with_collision=True):
        ends=np.arange(1,20,dtype=np.int64)*BLOCK
        base=[e("note_on",40,ends[0],"s708_reference"),
              e("note_off",40,ends[1],"s707_release")]
        proposed_at=int(ends[4])
        candidate=[e("note_on",41,proposed_at,"extra_proposal"),
                   e("note_off",41,ends[14],"extra_closure")]
        if with_collision:
            future_on=int(ends[4+collision_blocks])
            base.extend((
                e("note_on",41,future_on,"future_signed_on"),
                e("note_off",41,ends[13],"future_signed_off")))
            # A real candidate would be preempted at the future
            # native NoteOn, so the provisional source stream closes.
            candidate[-1]=e("note_off",41,future_on,"s707_preempt")
        post=np.zeros((len(ends),49,3),np.float32)
        post[:,1,:]=[.9,.95,.0]
        native_k=np.zeros((len(ends),7),np.float32)
        native_k[:,0]=1.
        fused=native_k.copy()
        fused[:,2]=.85
        fused[:,0]=.10
        fused[:,1]=.05
        return base,candidate,ends,post,native_k,fused

    def select(self,lag,*,collision=1,with_collision=True):
        r,c,ends,p,k,f=self.case(
            collision_blocks=collision,with_collision=with_collision)
        return select_novel_births(
            r,c,ends,p,k,f,confirmation_blocks=lag)

    def test_four_models_times_three_delays(self):
        self.assertEqual(DELAYS,(1,2,3))
        self.assertEqual(len(GRID),12)

    def test_single_block_23ms_preemption_cancels_extra(self):
        for lag in DELAYS:
            proposals,score,indices,diag=self.select(lag,collision=1)
            self.assertEqual(len(proposals),0)

    def test_two_block_46ms_preemption_cancels_extra(self):
        for lag in (2,3):
            proposals,score,indices,diag=self.select(lag,collision=2)
            self.assertEqual(len(proposals),0)

    def test_extra_without_future_native_is_emitted_at_current_time(self):
        for lag in DELAYS:
            src,candidate,ends,p,k,f=self.case(with_collision=False)
            pairs,prob,indices,_=select_novel_births(
                src,candidate,ends,p,k,f,confirmation_blocks=lag)
            self.assertEqual(len(pairs),1)
            on,off=pairs[0]
            self.assertEqual(on["available_sample"],int(ends[4+lag]))
            self.assertEqual(on["decision_sample"],on["available_sample"])
            self.assertEqual(on["time"],on["available_sample"]/SR)
            self.assertEqual(off["available_sample"],ends[14])
            self.assertEqual(indices[0][0],4+lag)

    def test_future_native_outside_confirm_window_does_not_change_prefix(self):
        src,candidate,ends,p,k,f=self.case(with_collision=False)
        future=src+[
            e("note_on",41,ends[12],"late_native_on"),
            e("note_off",41,ends[13],"late_native_off")]
        plain=select_novel_births(
            src,candidate,ends,p,k,f,confirmation_blocks=2)
        extra=select_novel_births(
            future,candidate,ends,p,k,f,confirmation_blocks=2)
        self.assertEqual(len(plain[0]),len(extra[0]))
        np.testing.assert_array_equal(plain[1],extra[1])

    def test_future_noteoff_date_cannot_affect_birth_confidence(self):
        src,candidate,ends,p,k,f=self.case(with_collision=False)
        a=select_novel_births(
            src,candidate,ends,p,k,f,confirmation_blocks=2)
        later=[dict(event) for event in candidate]
        later[1]=e("note_off",41,ends[17],"later_close")
        b=select_novel_births(
            src,later,ends,p,k,f,confirmation_blocks=2)
        np.testing.assert_array_equal(a[1],b[1])

    def test_candidate_closed_before_confirm_is_rejected(self):
        src,candidate,ends,p,k,f=self.case(with_collision=False)
        candidate[-1]=e("note_off",41,ends[5],"closed_before_decision")
        pairs,_,_,diag=select_novel_births(
            src,candidate,ends,p,k,f,confirmation_blocks=2)
        self.assertEqual(len(pairs),0)
        self.assertEqual(diag["blocked_candidate_already_closed"],1)

    def test_invalid_delay_rejected(self):
        src,candidate,ends,p,k,f=self.case(with_collision=False)
        with self.assertRaises(ValueError):
            select_novel_births(src,candidate,ends,p,k,f,
                                confirmation_blocks=4)

if __name__=="__main__":
    unittest.main()
