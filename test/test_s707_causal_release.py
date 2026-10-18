import unittest
import numpy as np
from causal_note.strict_live import HOP,SR
from causal_note.s707_causal_release import received_history,render_release


class ReleaseTests(unittest.TestCase):
    def test_block_history_has_no_early_sample(self):
        a=np.array([[1],[2],[3]],np.float32)
        h=received_history(a,12)[:,0]
        np.testing.assert_array_equal(h,[0,0,0,1,1,1,1,2,2,2,2,3])

    def test_first_available_timer_beats_future_hazard(self):
        ends=np.arange(1,101)*HOP
        on=dict(type='note_on',pitch=40,decision_sample=4*HOP,available_sample=4*HOP,
                time=4*HOP/SR,reason='test',velocity=96)
        frames=np.arange(11,100,4);p=np.zeros(len(frames));p[-1]=1
        plan=dict(notes=[dict(start=3,next_start=100,native_end=19,pitch=40)],
                  onsets=[on],frames=[frames],probabilities=[p])
        events,_=render_release(plan,ends,int(ends[-1]),grace=12)
        self.assertEqual(events[1]['reason'],'s707_forward_fallback')
        self.assertEqual(events[1]['available_sample'],32*HOP)

    def test_six_note_budget_counts_actual_emitted_offsets(self):
        ends=np.arange(1,101)*HOP
        ons=[dict(type='note_on',pitch=40+i,decision_sample=(4+4*i)*HOP,
             available_sample=(4+4*i)*HOP,time=(4+4*i)*HOP/SR,reason='test',velocity=96) for i in range(7)]
        plan=dict(notes=[dict(start=3+4*i,next_start=100,native_end=None,pitch=40+i) for i in range(7)],
            onsets=ons,frames=[np.array([],int) for _ in range(7)],probabilities=[np.array([]) for _ in range(7)])
        events,skipped=render_release(plan,ends,int(ends[-1]))
        self.assertEqual(skipped,1)
        self.assertEqual(sum(e['type']=='note_on' for e in events),6)
        self.assertEqual(sum(e['type']=='note_off' for e in events),6)


if __name__=='__main__':
    unittest.main()
