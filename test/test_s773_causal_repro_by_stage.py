import unittest
import numpy as np
from scripts.s773_causal_repro_by_stage import fingerprint,event_fingerprint,stable_stages
class S773StageTests(unittest.TestCase):
    def test_array_byte_fingerprint(self):
        x=np.array([[1,2],[3,4]],np.float32)
        self.assertEqual(fingerprint(x),fingerprint(x.copy()))
        y=x.copy();y[1,1]=5
        self.assertNotEqual(fingerprint(x)["sha256"],fingerprint(y)["sha256"])
    def test_empty_and_exact_midi(self):
        a=[dict(type="note_on",pitch=40,available_sample=1024)]
        b=[dict(type="note_on",pitch=40,available_sample=2048)]
        self.assertEqual(event_fingerprint(a),event_fingerprint(a.copy()))
        self.assertNotEqual(event_fingerprint(a)["sha256"],event_fingerprint(b)["sha256"])
        self.assertEqual(event_fingerprint([])["count"],0)
    def test_stage_differences(self):
        rows=[{"stages":{"audio_feature":{"sha256":"a"},"emitted_midi":{"sha256":"b"}}},
              {"stages":{"audio_feature":{"sha256":"a"},"emitted_midi":{"sha256":"c"}}}]
        r=stable_stages(rows)
        self.assertTrue(r["audio_feature"]["same"])
        self.assertFalse(r["emitted_midi"]["same"])
if __name__=="__main__":
    unittest.main()
