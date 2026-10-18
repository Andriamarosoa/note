import unittest
import numpy as np
from unittest.mock import patch
from causal_note.strict_live import SR,HOP
from causal_note.s704_joint_heads import acoustic_features,JointPolicy,JointSelector,decode_joint
from scripts.s704_train_joint_loops import targets,causal_s675_events,POLICIES
from scripts.s704_joint_metrics import metric


class JointTests(unittest.TestCase):
    def test_acoustic_prefix_causality(self):
        rng=np.random.default_rng(704)
        feat=rng.random((110,49,126),dtype=np.float32)
        post=rng.random((110,49,3),dtype=np.float32)
        k=rng.random((110,7),dtype=np.float32)
        k/=k.sum(1,keepdims=True)
        full=acoustic_features(feat,post,k)
        for stop in (1,7,33,79):
            np.testing.assert_array_equal(full[:stop],acoustic_features(feat[:stop],post[:stop],k[:stop]))
        self.assertEqual(full.shape,(110,49,192))

    def test_events_prefix_and_single_active_state(self):
        rng=np.random.default_rng(705)
        p=rng.random((100,49,3),dtype=np.float32)
        k=np.ones((100,7),np.float32)/7
        ends=np.arange(1,101)*HOP*4
        full=decode_joint(ends,p,k,JointPolicy(),int(ends[-1]))
        prefix=decode_joint(ends[:37],p[:37],k[:37],JointPolicy(),int(ends[36]))
        self.assertEqual([e for e in full if e['decision_sample']<=ends[36] and e['reason']!='stream_end'],
                         [e for e in prefix if e['reason']!='stream_end'])
        active=set()
        for e in full:
            self.assertEqual(e['available_sample'],e['decision_sample'])
            self.assertEqual(e['time'],e['available_sample']/SR)
            if e['type']=='note_on':
                self.assertNotIn(e['pitch'],active);active.add(e['pitch'])
            else:
                self.assertIn(e['pitch'],active);active.remove(e['pitch'])
            self.assertLessEqual(len(active),6)
        self.assertFalse(active)

    def test_discontinuity_rejected(self):
        selector=JointSelector(JointPolicy())
        with self.assertRaises(ValueError):
            selector.step(np.zeros((49,3)),np.ones(7)/7,8*HOP)

    def test_no_eof_offset_credit_and_one_stream_k(self):
        ends=np.arange(1,21)*4*HOP
        p=np.zeros((20,49,3));p[:,:,1]=.8
        p[0,0,0]=.9
        k=np.ones((20,7))/7
        events=decode_joint(ends,p,k,JointPolicy(),int(ends[-1]))
        truth=[dict(pitch=40,time=4*HOP/SR,duration=10*HOP/SR)]
        m=metric(events,truth,ends)
        self.assertEqual(m['tp'],1)
        self.assertEqual(m['offset_correct'],0)
        self.assertEqual(m['full_note_f1'],0)
        # Three true-active frames plus the final silent frame after panic.
        self.assertEqual(m['k_correct'],4)

    def test_current_truth_not_delayed_and_not_used_by_inference(self):
        ends=np.arange(1,10)*4*HOP
        start,stop=ends[1]/SR,ends[5]/SR
        y,k=targets(ends,[dict(pitch=44,time=start,duration=stop-start)])
        np.testing.assert_array_equal(k,[0,1,1,1,1,0,0,0,0])
        np.testing.assert_array_equal(y[:,4,1],k)

    def test_causal_fallback_waits_and_does_not_look_for_later_hazard(self):
        n=100; ends=np.arange(1,n+1)*HOP
        feat=np.zeros((n,49,126));post=np.zeros((n,49,3));k=np.ones((n,7))/7
        def ev(kind,frame):
            return dict(type=kind,pitch=40,decision_sample=int(ends[frame]),
                available_sample=int(ends[frame]),time=ends[frame]/SR,reason='causal',velocity=96)
        native=[ev('note_on',1),ev('note_off',20)]
        frames=[np.arange(7,100)]; prob=[np.zeros(93)]
        # A hazard at frame 70 must not erase the known fallback at 20+24.
        prob[0][70-7]=.9
        with patch('scripts.s704_train_joint_loops.hazard_predictions',return_value=(frames,prob)):
            events=causal_s675_events(ends,feat,post,k,native,int(ends[-1]),None)
        off=events[1]
        self.assertEqual(off['reason'],'s675_causal_fallback_grace')
        self.assertEqual(off['decision_sample'],45*HOP)
        self.assertEqual(off['available_sample'],48*HOP)

    def test_preregistered_budget(self):
        self.assertEqual(len(POLICIES),96)


if __name__=='__main__':
    unittest.main()
