"""S719 counterexamples for evidence-gated retrigger selection."""
import unittest
import numpy as np

from causal_note.s708_cardinality_selection import GuidedPolicy,GuidedSelector
from causal_note.s719_shadow_reattack import ShadowReattackSelector,decode_shadow
from causal_note.strict_live import HOP


def policy():
    return GuidedPolicy(onset=.75,sustain_on=.5,sustain_off=.3,
        release=.85,release_steps=1,count_confidence=1.1,blend=.7,
        guide_confidence=.5,weak_onset=.15,weak_sustain=.55,rise=.04,trim=False)


def decisions(step,on):
    post=np.zeros((49,3),np.float32)
    post[12]=[on,.90,.05]
    k=np.zeros(7,np.float32);k[1]=1.
    return post,k,(step+1)*4*HOP


class S719RearmTests(unittest.TestCase):
    def test_frozen_never_retriggers_but_audio_rise_can(self):
        old=GuidedSelector(policy())
        trial=ShadowReattackSelector(policy(),rise=.20,min_gap_ms=120)
        frozen,shadow=[],[]
        for step,on in enumerate([.96,*([.55]*6),.98,.90,.90]):
            p,k,end=decisions(step,on)
            frozen.extend(old.step(p,k,end))
            shadow.extend(trial.step(p,k,end))
        self.assertEqual([e["type"] for e in frozen],["note_on"])
        self.assertEqual([e["type"] for e in shadow],
                         ["note_on","note_off","note_on"])
        self.assertEqual([e["reason"] for e in shadow[1:]],
                         ["joint_restrike","joint_causal_attack"])
        self.assertEqual(trial.trial_reattack_emissions,1)

    def test_no_second_onset_for_stationary_harmonic_plateau(self):
        trial=ShadowReattackSelector(policy(),rise=.20,min_gap_ms=120)
        events=[]
        for step,on in enumerate([.95]*18):
            p,k,end=decisions(step,on)
            events.extend(trial.step(p,k,end))
        self.assertEqual([e["type"] for e in events],["note_on"])
        self.assertEqual(trial.trial_reattack_emissions,0)

    def test_early_rise_within_gap_is_blocked(self):
        trial=ShadowReattackSelector(policy(),rise=.20,min_gap_ms=120)
        events=[]
        for step,on in enumerate([.95,.50,.98,.90]):
            p,k,end=decisions(step,on)
            events.extend(trial.step(p,k,end))
        self.assertEqual([e["type"] for e in events],["note_on"])

    def test_decoder_source_exactly_contiguous_and_bounded(self):
        p=[];k=[];ends=[]
        for step,on in enumerate([.95,.5,.5,.5,.5,.5,.5,.95]):
            a,b,c=decisions(step,on);p.append(a);k.append(b);ends.append(c)
        events,stats=decode_shadow(ends,np.stack(p),np.stack(k),policy(),
                                   ends[-1],rise=.20,min_gap_ms=120)
        self.assertEqual(stats["actual_retriggers"],1)
        self.assertEqual([e["type"] for e in events],
                         ["note_on","note_off","note_on","note_off"])
        self.assertEqual(events[-1]["reason"],"stream_end")
        self.assertTrue(all(e["available_sample"]>=e["decision_sample"] for e in events))

if __name__=="__main__":
    unittest.main()
