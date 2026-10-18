"""S716 independent causal S707-style NoteOff tests; no external microphone needed."""
import unittest
from unittest.mock import patch
import numpy as np

from causal_note.strict_live import HOP, SR
from causal_note.s707_causal_release import received_history
from causal_note.s716_s707_online import S716S707OnlineRelease, DECISION
from scripts.s672_causal_offset_timing_grid import (
    frame_matrix, native_rhythm_clock,
)

class FixedHazard:
    n_features_in_=75
    def __init__(self, prob):
        self.prob=float(prob)
        self.rows=[]
    def predict_proba(self, x):
        a=np.asarray(x)
        assert a.ndim==2 and a.shape[1]==75 and np.isfinite(a).all()
        self.rows.extend(a.copy())
        return np.tile([1.-self.prob,self.prob],(len(a),1))


def source(kind,step,pitch=52,reason="joint_causal_attack"):
    end=(step+1)*DECISION
    return dict(type=kind,pitch=pitch,decision_sample=end,
                available_sample=end,time=end/SR,reason=reason,
                velocity=96 if kind=="note_on" else 0)


def sample(step):
    rng=np.random.default_rng(1000+step)
    return (rng.normal(0,.2,(49,126)).astype(np.float32),
            rng.random((49,3),dtype=np.float32),
            rng.random(7,dtype=np.float32))


class S716Tests(unittest.TestCase):
    def call(self,release,step,events=()):
        f,p,k=sample(step)
        return release.step((step+1)*DECISION,list(events),f,p,k,
                            (step+1)*DECISION)

    def test_hazard_only_after_s707_original_minimum_age_and_4hop_decision(self):
        release=S716S707OnlineRelease(FixedHazard(.99))
        self.assertEqual([x["type"] for x in self.call(release,0,[source("note_on",0)])],["note_on"])
        self.assertEqual(self.call(release,1),[])
        events=self.call(release,2)
        self.assertEqual([(e["type"],e["pitch"],e["reason"]) for e in events],
                         [("note_off",52,"s716_causal_hazard")])
        self.assertEqual(events[0]["decision_sample"],3*DECISION)
        self.assertEqual(release.finish(3*DECISION),[])

    def test_source_native_off_is_deferred_until_forward_only_fallback(self):
        release=S716S707OnlineRelease(FixedHazard(.001),gate=.80,grace=24)
        self.call(release,0,[source("note_on",0)])
        self.assertEqual(self.call(release,1,[source("note_off",1,reason="joint_causal_release")]),[])
        for step in range(2,7):
            self.assertEqual(self.call(release,step),[])
        events=self.call(release,7)
        self.assertEqual([(e["type"],e["reason"]) for e in events],
                         [("note_off","s716_forward_fallback")])
        self.assertEqual(events[0]["available_sample"],8*DECISION)

    def test_restrike_always_closes_old_midi_instance_first(self):
        release=S716S707OnlineRelease(FixedHazard(.0))
        self.call(release,0,[source("note_on",0)])
        events=self.call(release,1,[
            source("note_off",1,reason="joint_restrike"),
            source("note_on",1)])
        self.assertEqual([(e["type"],e["reason"]) for e in events],
                         [("note_off","s716_restrike"),
                          ("note_on","joint_causal_attack")])
        self.assertEqual(len(release.active),1)

    def test_six_note_budget_and_clean_end(self):
        release=S716S707OnlineRelease(FixedHazard(.0))
        events=self.call(release,0,[source("note_on",0,pitch=p)
                                    for p in range(40,47)])
        self.assertEqual(len(events),6)
        self.assertEqual(len(release.active),6)
        self.assertEqual(release.stats["note_on_skipped_due_to_six_note_cap"],1)
        final=release.finish(DECISION,reason="shutdown")
        self.assertEqual(len(final),6)
        self.assertTrue(all(e["reason"]=="shutdown" for e in final))
        self.assertEqual(release.finish(DECISION),[])

    def test_partial_callback_timestamp_never_backdated(self):
        release=S716S707OnlineRelease(FixedHazard(.9))
        f,p,k=sample(0)
        events=release.step(DECISION,[source("note_on",0)],f,p,k,DECISION+33)
        self.assertEqual(len(events),1)
        self.assertGreaterEqual(events[0]["available_sample"],events[0]["decision_sample"])

    def test_exact_s707_received_history_and_rhythm_abi(self):
        release=S716S707OnlineRelease(FixedHazard(.0))
        f,p,k=[],[],[]
        original=[]
        for step in range(22):
            x,y,z=sample(step)
            f.append(x);p.append(y);k.append(z)
            onset=[source("note_on",step,pitch=40+step%8)] if step in (0,3,9,15) else []
            original.extend(onset)
            release.step((step+1)*DECISION,onset,x,y,z,(step+1)*DECISION)
            n=(step+1)*4
            for actual,history in ((f,release.virtual_features),
                                   (p,release.virtual_post),(k,release.virtual_k)):
                offline=received_history(np.stack(actual),n)
                np.testing.assert_allclose(np.stack(history),offline[-len(history):],
                                           atol=0,rtol=0)
            clock=native_rhythm_clock(original,np.arange(1,n+1)*HOP)
            np.testing.assert_allclose(release._clock(n-1),clock[-1],atol=1e-6)

    def test_hazard_features_match_pinned_offline_s707_at_same_decision(self):
        model=FixedHazard(.001)
        release=S716S707OnlineRelease(model)
        from causal_note.s707_causal_release import received_history
        all_features=[];all_posts=[];all_ks=[]
        for step in range(22):
            f,p,k=sample(step)
            all_features.append(f);all_posts.append(p);all_ks.append(k)
            source_events=[source("note_on",0,pitch=52)] if step==0 else []
            release.step((step+1)*DECISION,source_events,f,p,k,(step+1)*DECISION)
        self.assertGreater(len(model.rows),4)
        ends=np.arange(1,22*4+1)*HOP
        f=received_history(np.stack(all_features),len(ends))
        p=received_history(np.stack(all_posts),len(ends))
        k=received_history(np.stack(all_ks),len(ends))
        rhythm=native_rhythm_clock([source("note_on",0,pitch=52)],ends)
        original_start=3
        measured_frames=np.arange(11,len(ends),4)
        original=frame_matrix(f,p,k,ends,52,original_start,measured_frames,rhythm)
        np.testing.assert_allclose(np.stack(model.rows),original,atol=5e-6,rtol=5e-6)


if __name__=="__main__":
    unittest.main()
