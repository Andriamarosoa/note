"""S646 confirms NOTE-native octave candidate selection and real-time clock."""
import unittest
import numpy as np
from causal_note.strict_live import HOP,SR,Selection
from scripts.s646_native_note_octave_head import (
    right_edge_signature,octave_features,steps_for_delay,sample_times,
    replay_with_gate,candidate_labels,DELAYS_MS,
)


def event(pitch,seconds):
    samples=round(seconds*SR/HOP)*HOP
    return dict(type="note_on",pitch=pitch,time=samples/SR,
                decision_sample=samples)


class AlwaysAccept:
    def predict_proba(self,features):
        return np.tile([0.,1.],(len(features),1))


class CandidateOctaveTests(unittest.TestCase):
    def test_all_candidate_windows_look_only_at_real_available_audio(self):
        n=100*HOP
        t=np.arange(n)/SR
        base=np.asarray(.2*np.sin(2*np.pi*164.814*t),np.float32)
        altered=base.copy()
        altered[85*HOP:]=np.asarray(
            .21*np.sin(2*np.pi*330*t[85*HOP:]),np.float32)
        onset=40*HOP
        for lag in DELAYS_MS:
            early,late=sample_times(onset,lag)
            self.assertLessEqual(early,late)
            self.assertLessEqual(late,85*HOP)
            fa=octave_features(base,52,onset,lag)
            fb=octave_features(altered,52,onset,lag)
            self.assertEqual(fa.shape,(173,))
            np.testing.assert_array_equal(fa,fb)
            self.assertTrue(np.isfinite(fa).all())
        self.assertEqual(steps_for_delay(0),0)
        self.assertGreaterEqual(steps_for_delay(40)*HOP/SR,.040)
        self.assertGreaterEqual(steps_for_delay(80)*HOP/SR,.080)
        self.assertGreaterEqual(steps_for_delay(190)*HOP/SR,.190)

    def test_raw_spectral_subharmonic_vs_octave_independent(self):
        n=100*HOP
        t=np.arange(n)/SR
        e2=440*2**((40-69)/12)
        waveform=np.asarray(.14*np.sin(2*np.pi*e2*t)
                            +.045*np.sin(4*np.pi*e2*t),np.float32)
        from_frequency_e2=right_edge_signature(waveform,40,65*HOP)
        added=np.asarray(waveform+.09*np.sin(4*np.pi*e2*t),np.float32)
        duo=right_edge_signature(added,40,65*HOP)
        self.assertEqual(len(from_frequency_e2),58)
        self.assertGreater(float(np.linalg.norm(duo-from_frequency_e2)),0.03)
        with self.assertRaises(ValueError):
            right_edge_signature(waveform,65,65*HOP)

    def test_groundtruth_labels_only_score_native_noteon_emissions(self):
        true=[dict(pitch=40,time=1.,duration=.5),
              dict(pitch=52,time=1.01,duration=.4)]
        pred=[event(40,1.),event(55,1.25)]
        labeled=candidate_labels(pred,true)
        self.assertEqual(len(labeled),2)
        self.assertEqual(labeled[0]["label"],1)
        self.assertEqual(labeled[0]["pitch"],40)
        self.assertEqual(labeled[1]["label"],0) # extra spurious NOTE low is a hard negative
        # A false native low must NOT become positive just because
        # another true higher harmonic exists.
        false=candidate_labels([event(41,1.)],true)
        self.assertEqual(false[0]["label"],0)

    def test_shadow_decoder_schedules_real_non_backdated_octave(self):
        cfg=Selection(onset=.60,active_on=.50)
        n=40
        ends=np.arange(1,n+1)*HOP
        p=np.zeros((n,49,3),np.float32)
        k=np.zeros((n,7),np.float32)
        k[:,0]=1.
        low=52-40
        high=64-40
        p[4,low,:2]=[.95,.90]
        pcm=np.zeros(n*HOP,np.float32)
        gates={d:dict(head=AlwaysAccept(),threshold=.5) for d in DELAYS_MS}
        events,audit=replay_with_gate(pcm,ends,p,k,None,cfg,gates)
        original=[e for e in events["S614_original"] if e["type"]=="note_on"]
        self.assertEqual(len(original),1)
        self.assertEqual(original[0]["pitch"],52)
        for delay in DELAYS_MS:
            ons=[e for e in events[f"head_{delay}ms"] if e["type"]=="note_on"]
            self.assertEqual({e["pitch"] for e in ons},{52,64})
            lower=next(e for e in ons if e["pitch"]==52)
            upper=next(e for e in ons if e["pitch"]==64)
            self.assertEqual(upper["decision_sample"]-lower["decision_sample"],
                             steps_for_delay(delay)*HOP)
            self.assertEqual(upper["available_sample"],upper["decision_sample"])
            self.assertGreaterEqual(upper["time"],lower["time"])
        self.assertEqual(audit["admitted"][0],1)
        self.assertNotIn("64",str(original))


if __name__=="__main__":
    unittest.main()
