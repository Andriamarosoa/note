"""Enforce honest signed-fit80 science, not a green smoke from weaker scores."""
import unittest
from onset256.audit import audit,statistics,REFERENCE,COMPOSITIONS

class ScientificOnsetAudit(unittest.TestCase):
    def factory(self):
        rows=[]
        for comp in COMPOSITIONS:
            for i in range(16):
                rows.append(dict(name=f"{comp}_{i}",composition=comp,
                    frames=306536 if i==0 and comp==COMPOSITIONS[0] else 0,
                    counts=dict(tp=0,fp=0,fn=0),
                    shadow_counts=dict(tp=0,fp=0,fn=0)))
        rows[0]["counts"]=dict(tp=8099,fp=4600,fn=3168)
        rows[0]["shadow_counts"]=dict(tp=8113,fp=4603,fn=3154)
        return dict(complete=True,tracks=rows,decision_hop_samples=256,
            feature_hop_samples=256,s613_training_hop_samples=256,
            s701_training_hop_samples=256,no_s708_temporal_checkpoint=True,
            s701_preselected_policy="native",shadow_promoted=False,
            totals=dict(tp=8099,fp=4600,fn=3168),
            s701_shadow_totals=dict(tp=8113,fp=4603,fn=3154))
    def test_score_and_rejection_are_true(self):
        s=audit(self.factory())
        self.assertEqual(s["scientific_verdict"],"REJECT_NONREGRESSION")
        self.assertEqual(s["changed_tracks"],1)
        self.assertEqual(s["delta_s701_minus_native"]["tp"],14)
        self.assertEqual(s["delta_s701_minus_native"]["fp"],3)
        self.assertGreater(s["reference"]["onset_f1"],s["totals"]["native"]["onset_f1"])
        self.assertFalse(s["accepted_for_promotion"])
    def test_inference_not_trained256_rejected(self):
        r=self.factory()
        r["s701_training_hop_samples"]=1024
        with self.assertRaisesRegex(ValueError,"genuine"):
            audit(r)
    def test_posthoc_promotion_rejected(self):
        r=self.factory()
        r["shadow_promoted"]=True
        with self.assertRaisesRegex(ValueError,"post-selection"):
            audit(r)
    def test_incomplete_cohort_rejected(self):
        r=self.factory()
        r["tracks"].pop()
        with self.assertRaises(ValueError):audit(r)
    def test_s708_weight_loading_rejected(self):
        r=self.factory()
        r["no_s708_temporal_checkpoint"]=False
        with self.assertRaisesRegex(ValueError,"S708"):
            audit(r)
if __name__=="__main__":unittest.main()
