import unittest
import numpy as np
from causal_note.strict_live import HOP
from causal_note.s704_joint_heads import JointPolicy,decode_joint
from causal_note.s707_causal_release import prepare_hazards,render_release


class DummyHazard:
    def predict_proba(self,x):
        p=1/(1+np.exp(-(.3*x[:,0]-.15*x[:,1]+.3*x[:,-6]-2.0)))
        return np.column_stack((1-p,p))


class WholeReleasePrefixTests(unittest.TestCase):
    def test_whole_replay_prefix_and_exact_eof_onset(self):
        rng=np.random.default_rng(710);n=100
        ends=np.arange(1,4*n+1)*HOP;decision=ends[3::4]
        post=rng.random((n,49,3),dtype=np.float32)
        k=rng.dirichlet(np.ones(7),size=n).astype(np.float32)
        feat=rng.random((n,49,126),dtype=np.float32)
        def emit(limit):
            es=ends[:4*limit];ds=decision[:limit]
            item=dict(ends=es,native_features=feat[:limit],native_post=post[:limit],native_k=k[:limit])
            ev=decode_joint(ds,post[:limit],k[:limit],JointPolicy(),int(es[-1]))
            plan=prepare_hazards(item,ev,DummyHazard())
            return render_release(plan,es,int(es[-1]),gate=.6,grace=12)[0]
        full=emit(n)
        for limit in (17,43,71):
            short=emit(limit);cut=int(ends[4*limit-1])
            self.assertEqual([e for e in full if e['available_sample']<=cut and e['reason']!='stream_end'],
                             [e for e in short if e['reason']!='stream_end'])
        active=set()
        for e in full:
            if e['type']=='note_on':
                self.assertNotIn(e['pitch'],active);active.add(e['pitch'])
            else:
                self.assertIn(e['pitch'],active);active.remove(e['pitch'])
        self.assertFalse(active)


if __name__=='__main__':unittest.main()
