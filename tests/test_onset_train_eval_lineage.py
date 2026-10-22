"""Guard against falsely promoting Onset on tracks seen during training/selection."""
import json,unittest
from pathlib import Path
from onset256.lineage_audit import analyze

ROOT=Path(__file__).resolve().parents[1]
LEDGER=json.loads((ROOT/"series/ONSET_S701_256/training_ledger.json").read_text())

class OnsetTrainTestLineage(unittest.TestCase):
    def factory(self):
        tracks=[]
        for name in LEDGER["fit80_names"]:
            tracks.append({"name":name,"counts":{"tp":1,"fp":0,"fn":0},
                           "shadow_counts":{"tp":1,"fp":0,"fn":0}})
        return dict(complete=True,decision_hop_samples=256,tracks=tracks)
    def test_overlapping_historic_data_cannot_validate_f1_even_perfect(self):
        a=analyze(self.factory(),LEDGER)
        self.assertEqual(a["counts"]["s701_train"]["count"],48)
        self.assertEqual(a["counts"]["s701_held"]["count"],32)
        self.assertEqual(a["counts"]["s613_train"]["count"],20)
        self.assertEqual(a["textures"]["comp"]["tracks"],40)
        self.assertEqual(a["textures"]["solo"]["tracks"],40)
        self.assertFalse(a["promotion_allowed"])
        self.assertFalse(a["independent_outer_test"])
        self.assertTrue(a["s701_fit80_all_previously_seen"])
    def test_wrong_cohort_detected(self):
        bad=self.factory()
        bad["tracks"][0]["name"]="not-a-signed-recording"
        with self.assertRaises(ValueError):analyze(bad,LEDGER)
    def test_wrong_hop_detected(self):
        bad=self.factory();bad["decision_hop_samples"]=1024
        with self.assertRaises(ValueError):analyze(bad,LEDGER)
if __name__=="__main__":unittest.main()
