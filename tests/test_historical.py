import unittest
from historical.scorer import onset_score,offset_score
import numpy as np
class HistoricalH0(unittest.TestCase):
    def test_independent_offset_score(self):
        events=[dict(type="note_on",pitch=40,time=.10,decision_sample=4410,available_sample=4410,reason="source"),
                dict(type="note_off",pitch=40,time=.5,decision_sample=22050,available_sample=22050,reason="source")]
        gt=[dict(pitch=40,time=.10,duration=.40)]
        self.assertEqual(offset_score(events,gt,np.arange(256,44100+1,256)),dict(offset_eligible=1,offset_correct=1))
    def test_does_not_expose_foreign_decoder(self):
        import historical
        self.assertEqual(historical.__all__,["offset_score"])
if __name__=="__main__":unittest.main()
