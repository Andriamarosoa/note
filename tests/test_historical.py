import unittest
from historical.scorer import onset_score
import numpy as np
class HistoricalH0(unittest.TestCase):
    def test_independent_onset_score(self):
        events=[dict(type="note_on",pitch=40,time=.10,decision_sample=4410,available_sample=4410,reason="source"),
                dict(type="note_off",pitch=40,time=.5,decision_sample=22050,available_sample=22050,reason="source")]
        gt=[dict(pitch=40,time=.10,duration=.40)]
        self.assertEqual(onset_score(events,gt),dict(tp=1,fp=0,fn=0))
    def test_does_not_expose_foreign_decoder(self):
        import historical
        self.assertEqual(historical.__all__,["onset_score"])
if __name__=="__main__":unittest.main()
