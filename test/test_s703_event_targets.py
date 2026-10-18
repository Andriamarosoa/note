import unittest
from scripts.s613_train_strict_live import event_score
from scripts.s703_event_targets import component_targets


class S703Targets(unittest.TestCase):
    def event(self, t, native):
        return dict(type='note_on', pitch=40, time=t, native=native)

    def test_closer_proposal_cannot_negate_correct_native_label(self):
        rows=[self.event(.201,False),self.event(.240,True)]
        truth=[dict(pitch=40,time=.2,duration=.1)]
        _, old_pairs=event_score(rows,truth)
        self.assertEqual(old_pairs,[(0,0)])
        self.assertEqual(component_targets(rows,truth).tolist(),[0,1])

    def test_rescue_targets_missing_notes_without_native_competition(self):
        rows=[self.event(.21,True),self.event(.23,False),self.event(.51,False)]
        truth=[dict(pitch=40,time=t,duration=.1) for t in (.2,.5)]
        self.assertEqual(component_targets(rows,truth).tolist(),[1,0,1])

    def test_native_false_positive_remains_negative(self):
        rows=[self.event(.21,True),self.event(.5,True),self.event(.8,False)]
        truth=[dict(pitch=40,time=.2,duration=.1)]
        self.assertEqual(component_targets(rows,truth).tolist(),[1,0,0])


if __name__=='__main__':
    unittest.main()
