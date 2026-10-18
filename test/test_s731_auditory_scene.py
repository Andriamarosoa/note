"""S731 human-inspired grouping/old-plus-new causal checks across all notes."""
import unittest
import numpy as np
from causal_note.s731_auditory_scene import (
    AuditoryScene,OWNERSHIP,harmonic_ownership,scene_scores,reweight_onsets)


def frame():
    x=np.zeros((49,288),np.float32)
    x[0,192:216]=4.  # lower source E2
    x[12,192:216]=4. # old octave-related bins (same frequencies)
    return x


class S731AuditoryTests(unittest.TestCase):
    def test_octave_and_fifth_ownership_not_hardcoded_E2(self):
        owned=harmonic_ownership()
        for src in (0,5,11,16,25):
            if src+12<49:
                self.assertTrue(np.all(owned[src+12,src,:]))
                self.assertFalse(np.any(owned[src+12,src+12,:]))
            if src+19<49:
                self.assertTrue(np.all(owned[src+19,src,:]))
        self.assertEqual(owned.shape,(49,49,8))
        # Third, fourth, fifth, ... parent overtones can also overlap.
        self.assertTrue(owned[24,0,0])
        self.assertTrue(owned[28,0,0])

    def test_new_independent_E3_changes_shared_harmonics_not_parent_gain(self):
        base=frame()
        global_gain=base.copy()
        global_gain[0,192:216]+=1.
        global_gain[12,192:216]+=1.
        actual_child=base.copy()
        source=actual_child[0,192:216].reshape(3,8)
        source[:,[1,3,5,7]]+=1.
        actual_child[12,192:216]+=1.
        active=np.zeros(49,bool);active[0]=True
        k=np.zeros(7,np.float32);k[2]=1.
        a=AuditoryScene();b=AuditoryScene()
        for agent in (a,b):
            agent.step(base,active,k)
            agent.step(base,active,k)
        flat=a.step(global_gain,active,k)
        independent=b.step(actual_child,active,k)
        self.assertGreater(float(independent.novelty_shared[12]),.4)
        self.assertAlmostEqual(float(flat.novelty_shared[12]),0,places=5)
        self.assertGreater(float(independent.novelty[12]),float(flat.novelty[12])+.15)
        self.assertGreater(float(independent.old_ownership[12]),.99)
        self.assertEqual(independent.count_room,1.)

    def test_gain_invariant_event_scoring_with_distinct_physical_excitation(self):
        x=frame()
        active=np.zeros(49,bool);active[0]=True
        k=np.zeros(7,np.float32);k[2]=.93;k[1]=.07
        m=AuditoryScene()
        m.step(x,active,k);m.step(x,active,k)
        onset=x.copy()
        onset[12,192:216]+=1.
        score=m.step(onset,active,k)
        p=np.zeros((49,3),np.float32);p[:,0]=.20;p[:,1]=.70
        trial=reweight_onsets(p,score,gain=.5)
        self.assertGreater(trial[12,0],p[12,0])
        np.testing.assert_array_equal(trial[:,1:],p[:,1:])
        off=reweight_onsets(p,score,gain=0)
        np.testing.assert_array_equal(off,p)

    def test_K_cannot_invent_source_when_count_confidently_stays_one(self):
        x=frame()
        k=np.zeros(7,np.float32);k[1]=1.
        active=np.zeros(49,bool);active[0]=True
        m=AuditoryScene()
        m.step(x,active,k)
        hotter=x.copy();hotter[12,192:216]+=2.
        evidence=m.step(hotter,active,k)
        p=np.zeros((49,3),np.float32);p[:,0]=.20
        no=reweight_onsets(p,evidence,gain=1.,require_k=True)
        np.testing.assert_array_equal(no,p)
        yes=reweight_onsets(p,evidence,gain=1.,require_k=False)
        self.assertGreater(yes[12,0],no[12,0])

    def test_online_stream_does_not_read_future_decisions(self):
        rng=np.random.default_rng(731)
        track=rng.uniform(0,3,(14,49,288)).astype(np.float32)
        ends=np.arange(1,15)*1024
        k=np.zeros((14,7),np.float32);k[:,2]=1.
        base_events=[dict(type="note_on",pitch=40,decision_sample=int(ends[2])),
                     dict(type="note_off",pitch=40,decision_sample=int(ends[9]))]
        first=scene_scores(track,None,k,base_events,ends)
        changed=track.copy();changed[10:]*=100.
        second=scene_scores(changed,None,k,base_events,ends)
        for t in range(10):
            np.testing.assert_array_equal(first[t].novelty,second[t].novelty)
            np.testing.assert_array_equal(first[t].old_ownership,second[t].old_ownership)
        self.assertEqual(first[2].old_ownership[12],0.)
        self.assertEqual(first[3].old_ownership[12],1.)
        self.assertEqual(first[10].old_ownership[12],0.)

    def test_invalid_nonfinite_or_changed_original_abi_fails(self):
        k=np.zeros(7,np.float32);a=np.zeros(49,bool)
        agent=AuditoryScene()
        with self.assertRaisesRegex(ValueError,"288D"):
            agent.step(np.zeros((49,320),np.float32),a,k)
        x=np.zeros((49,288),np.float32);x[3,204]=np.nan
        with self.assertRaisesRegex(ValueError,"finite"):
            agent.step(x,a,k)
        with self.assertRaises(ValueError):
            reweight_onsets(np.zeros((49,3),np.float32),
                agent.step(frame(),a,k),gain=2.)


if __name__=="__main__":
    unittest.main()
