"""S675 guards for authentic-TRAIN augmentation and causal threshold search."""
import unittest,io,zipfile
import numpy as np
from causal_note.strict_live import HOP
from scripts.s675_expanded_rhythm_timing import (
    expanded_train_specs,GATES,PERSISTENCES,choose_offset,native_rhythm_clock,
)

class S675Tests(unittest.TestCase):
    def test_no_reserved_players_in_training(self):
        source=[dict(name="01_Funk2-108-Eb_comp",composition="Funk2",
                     audio="01_Funk2-108-Eb_comp_mic.wav",
                     annotation="01_Funk2-108-Eb_comp.jams")]
        blob=io.BytesIO();anno=io.BytesIO()
        with zipfile.ZipFile(blob,"w") as z1,zipfile.ZipFile(anno,"w") as z2:
            for i in range(6):
                name=f"{i:02d}_Funk2-108-Eb_comp"
                z1.writestr(name+"_mic.wav",b"123")
                z2.writestr(name+".jams",b"abc")
        blob.seek(0);anno.seek(0)
        with zipfile.ZipFile(blob) as z1,zipfile.ZipFile(anno) as z2:
            result=expanded_train_specs(source,z1,z2)
        self.assertEqual(len(result),4)
        self.assertTrue(all(x["name"][:2] in ("00","01","02","03") for x in result))

    def test_fixed_timing_search_and_no_future_onsets(self):
        self.assertEqual(PERSISTENCES,(1,2,3,4))
        self.assertIn(.6,GATES)
        ends=np.arange(1,120)*HOP
        a=native_rhythm_clock([dict(type="note_on",pitch=40,
                                      decision_sample=10*HOP)],ends)
        b=native_rhythm_clock([dict(type="note_on",pitch=40,
                                      decision_sample=10*HOP),
                               dict(type="note_on",pitch=40,
                                      decision_sample=90*HOP)],ends)
        np.testing.assert_array_equal(a[:80],b[:80])
        item=dict(ends=ends,notes=[dict(start=10,pitch=40,
                          native_end=100,next_start=110)])
        frames=np.arange(20,105)
        p=np.ones(len(frames),np.float32)
        time=choose_offset(item,0,frames,p,.6,"hazard",batch_hops=4,persistence=1)
        self.assertGreaterEqual(time,ends[20]/44100.)

if __name__=="__main__":
    unittest.main()
