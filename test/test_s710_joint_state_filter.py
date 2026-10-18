import unittest
import numpy as np
from causal_note.s710_joint_state_filter import StatePolicy,StateSelector,decode_state
from causal_note.s704_joint_heads import JointPolicy,decode_joint

class StateTests(unittest.TestCase):
    def fixture(self,n=8):
        p=np.zeros((n,49,3),np.float32);p[:,:,1]=.01
        k=np.full((n,7),.001,np.float32);k[:,1]=.994
        return np.arange(1,n+1)*1024,p,k

    def test_present_pitch_can_open_without_strong_attack(self):
        ends,p,k=self.fixture()
        p[:,4,0]=.10;p[:,4,1]=.95
        new=decode_state(ends,p,k,StatePolicy(),int(ends[-1]))
        old=decode_joint(ends,p,k,JointPolicy(onset=.6,sustain_on=.5),int(ends[-1]))
        self.assertEqual([e["pitch"] for e in new if e["type"]=="note_on"],[44])
        self.assertFalse(any(e["type"]=="note_on" for e in old))

    def test_count_alone_cannot_invent_pitch(self):
        ends,p,k=self.fixture();k[:]=.001;k[:,6]=.994
        self.assertFalse(any(e["type"]=="note_on" for e in
            decode_state(ends,p,k,StatePolicy(count_weight=2),int(ends[-1]))))

    def test_fall_closes_before_eof(self):
        ends,p,k=self.fixture()
        p[:3,5,1]=.95
        k[3:]=.001;k[3:,0]=.994
        events=decode_state(ends,p,k,StatePolicy(close_steps=1),int(ends[-1]))
        offs=[e for e in events if e["type"]=="note_off"]
        self.assertEqual(len(offs),1)
        self.assertEqual(offs[0]["available_sample"],int(ends[3]))
        self.assertEqual(offs[0]["reason"],"joint_state_release")

    def test_capacity_and_duplicate_state(self):
        ends,p,k=self.fixture(40)
        p[:,:12,1]=.99;p[:,:12,0]=.9;k[:]=.001;k[:,6]=.994
        events=decode_state(ends,p,k,StatePolicy(opening_penalty=0),int(ends[-1]))
        active=set()
        for e in events:
            if e["type"]=="note_on":
                self.assertNotIn(e["pitch"],active);active.add(e["pitch"])
            else:active.discard(e["pitch"])
            self.assertLessEqual(len(active),6)
        self.assertFalse(active)

    def test_prefix_and_eof_order(self):
        rng=np.random.default_rng(710)
        ends=np.arange(1,71)*1024
        p=rng.uniform(.05,.95,(70,49,3));k=rng.dirichlet(np.ones(7),70)
        policy=StatePolicy(opening_penalty=.5,close_steps=2)
        full=decode_state(ends,p,k,policy,int(ends[-1])+107)
        for length in (1,13,39):
            cutoff=int(ends[length-1])
            prefix=decode_state(ends[:length],p[:length],k[:length],policy,cutoff)
            ordinary=[e for e in prefix if e["reason"]!="stream_end"]
            expected=[e for e in full if e["available_sample"]<=cutoff and e["reason"]!="stream_end"]
            self.assertEqual(ordinary,expected)
            active=set()
            for e in prefix:
                self.assertGreaterEqual(e["available_sample"],e["decision_sample"])
                if e["type"]=="note_on":active.add(e["pitch"])
                else:active.discard(e["pitch"])
            self.assertFalse(active)

    def test_noncontiguous_blocks_rejected(self):
        d=StateSelector(StatePolicy())
        with self.assertRaises(ValueError):d.step(np.zeros((49,3)),np.ones(7)/7,2048)

if __name__=="__main__":unittest.main()
