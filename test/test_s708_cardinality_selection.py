import unittest
import numpy as np
from causal_note.strict_live import HOP
from causal_note.s708_cardinality_selection import GuidedPolicy,GuidedSelector,decode_guided
from causal_note.s704_joint_heads import decode_joint,JointPolicy


class GuidedTests(unittest.TestCase):
    def test_k_alone_cannot_create_notes(self):
        p=np.zeros((49,3));k=np.zeros(7);k[4]=1
        s=GuidedSelector(GuidedPolicy())
        self.assertEqual(s.step(p,k,4*HOP),[])

    def test_rising_acoustic_candidate_is_required(self):
        p=np.zeros((49,3));p[4]=[.3,.8,.1]
        k=np.zeros(7);k[1]=1
        s=GuidedSelector(GuidedPolicy())
        events=s.step(p,k,4*HOP)
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]['pitch'],44)
        self.assertEqual(events[0]['reason'],'count_supported_acoustic_attack')

    def test_disabled_guide_is_exact_original_decoder(self):
        rng=np.random.default_rng(708)
        p=rng.random((90,49,3));k=rng.dirichlet(np.ones(7),size=90)
        ends=np.arange(1,91)*4*HOP
        a=decode_guided(ends,p,k,GuidedPolicy(guide_confidence=1.1),int(ends[-1]))
        b=decode_joint(ends,p,k,JointPolicy(),int(ends[-1]))
        self.assertEqual(a,b)

    def test_prefix_and_budget_are_preserved(self):
        rng=np.random.default_rng(709)
        p=rng.random((100,49,3));k=np.eye(7)[rng.integers(0,7,100)]
        ends=np.arange(1,101)*4*HOP
        policy=GuidedPolicy(trim=True,guide_confidence=.5)
        a=decode_guided(ends,p,k,policy,int(ends[-1]))
        b=decode_guided(ends[:33],p[:33],k[:33],policy,int(ends[32]))
        self.assertEqual([e for e in a if e['decision_sample']<=ends[32] and e['reason']!='stream_end'],
                         [e for e in b if e['reason']!='stream_end'])
        active=set()
        for e in a:
            if e['type']=='note_on':
                self.assertNotIn(e['pitch'],active);active.add(e['pitch'])
            else:
                self.assertIn(e['pitch'],active);active.remove(e['pitch'])
            self.assertLessEqual(len(active),6)


if __name__=='__main__':
    unittest.main()
