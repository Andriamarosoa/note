"""S747 independent native NOTE head states, frozen source and causality."""
import copy
import unittest
import numpy as np
from causal_note.s747_independent_heads import isolated_onsets, IsolatedBirthPolicy
from causal_note.strict_live import SR


def event(ty,pitch,sample):
    return dict(type=ty,pitch=pitch,decision_sample=sample,
                available_sample=sample,time=sample/SR,
                reason="frozen_s708",velocity=96 if ty=="note_on" else 0)


class IndependentHeadsTests(unittest.TestCase):
    def setUp(self):
        n=6
        self.ends=np.arange(1,n+1,dtype=np.int64)*1024
        self.post=np.zeros((n,49,3),np.float32)
        self.k=np.full((n,7),.16,np.float32)
        self.heads=np.zeros((n,49,6),np.float32)
        self.ctx=np.zeros((n,49,16),np.float32)
        self.post[:,:,2]=.1
        self.post[:,:,1]=.05
        self.post[:,2,0]=.10
        self.post[:,2,1]=.75
        self.heads[:,2,0]=.10
        self.heads[:,2,1]=.80
        self.ctx[:,2,4]=.20
        self.ctx[:,2,7]=.50
        self.reference=[event("note_on",40,1024),
                        event("note_off",40,6144)]

    def read(self,reference=None,**overrides):
        return isolated_onsets(
            self.ends,overrides.get("post",self.post),
            overrides.get("k",self.k),
            overrides.get("heads",self.heads),
            overrides.get("context",self.ctx),
            self.reference if reference is None else reference)

    def test_independent_head_can_add_unoccupied_pitch(self):
        original=copy.deepcopy(self.reference)
        proposed,diag=self.read()
        self.assertTrue(proposed)
        self.assertEqual(proposed[0]["pitch"],42)
        self.assertEqual(proposed[0]["s747_head"],1)
        self.assertEqual(self.reference,original)
        self.assertTrue(all((e["s747_mask"] & 1) for e in proposed))
        self.assertGreaterEqual(len(diag["mask_histogram"]),1)

    def test_never_proposes_current_native_note(self):
        self.reference=[event("note_on",42,1024),
                        event("note_off",42,8192)]
        proposed,diag=self.read()
        self.assertFalse(proposed)
        self.assertGreater(diag["rejected_source"],0)

    def test_future_audio_does_not_change_earlier_candidates(self):
        before,_=self.read()
        future=self.heads.copy()
        future[-1,:,:]=1
        after,_=self.read(heads=future)
        initial=lambda rows:[(e["pitch"],e["decision_sample"])
                            for e in rows if e["decision_sample"]<self.ends[-1]]
        self.assertEqual(initial(before),initial(after))

    def test_source_future_events_do_not_modify_earlier_candidates(self):
        before,_=self.read()
        changed=copy.deepcopy(self.reference)
        changed[-1]["available_sample"]=self.ends[-1]+2048
        after,_=self.read(reference=changed)
        prefix=lambda rows:[(e["pitch"],e["decision_sample"])
                             for e in rows if e["decision_sample"]<self.ends[-1]]
        self.assertEqual(prefix(before),prefix(after))

    def test_masks_preserve_independent_head_eligibility(self):
        p=self.post.copy()
        h=self.heads.copy()
        p[0,4,:]=(.10,.90,.10)
        h[0,4,5]=.95
        proposals,diag=self.read(post=p,heads=h)
        self.assertTrue(any(e["s747_head"]==5 for e in proposals))
        self.assertTrue(any(e["s747_head"]==1 for e in proposals))

    def test_adjacent_heads_cannot_retrigger_same_pitch_immediately(self):
        # Independent experts may alternate but the MIDI mux must not
        # dispatch repeated NoteOn for a sustained pitch within 100ms.
        self.heads[:,2,5]=.90
        proposals,diag=self.read()
        pitch_42=[e for e in proposals if e["pitch"]==42]
        times=[e["decision_sample"] for e in pitch_42]
        self.assertTrue(all(b-a >= .10*SR for a,b in zip(times,times[1:])))
        self.assertGreater(diag["rejected_cross_head_restrike"],0)

    def test_native_release_tail_is_not_a_fresh_attack(self):
        self.reference=[event("note_on",42,1024),
                        event("note_off",42,2048)]
        proposed,diag=self.read()
        self.assertTrue(all(e["decision_sample"]-2048 >= .07*SR
                            for e in proposed))
        self.assertGreater(diag["rejected_native_tail"],0)

    def test_invalid_schema_or_policy_fails_closed(self):
        with self.assertRaises(ValueError):
            self.read(k=np.zeros((6,6),np.float32))
        with self.assertRaises(ValueError):
            IsolatedBirthPolicy(min_tail=1.5).validate()


if __name__=="__main__":
    unittest.main()
