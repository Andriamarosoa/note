"""S769 independent native Onset quality: event isolation and FIT label firewall."""
import unittest
import numpy as np
from test.test_s765_fit_independent_arbiter import ArbiterTests
from scripts.s768_independent_onset_lattice_fit80 import feature_grid
from scripts.s769_source_onset_precision_fit80 import (
    DIM,THRESHOLDS,source_rows,kept_native,group_oof,scores,fit)

class OnsetPrecisionTests(unittest.TestCase):
    def fixture(self):
        item,post,k,source,shadow,ctx=ArbiterTests().fixture()
        features=feature_grid(item,post,k,ctx)
        return item,post,k,source,features

    def test_source_rows_causal_last_observed_audio(self):
        item,post,k,source,features=self.fixture()
        ons,X=source_rows(source,item["decision_ends"],features)
        self.assertEqual(len(ons),1)
        self.assertEqual(X.shape,(1,DIM))
        self.assertEqual(ons[0]["pitch"],40)

    def test_changed_future_feature_cannot_change_past_source_decision(self):
        item,post,k,source,features=self.fixture()
        _,x=source_rows(source,item["decision_ends"],features)
        altered=features.copy()
        altered[1:]=100.
        _,b=source_rows(source,item["decision_ends"],altered)
        np.testing.assert_array_equal(x,b)

    def test_last_seen_frame_no_anticipatory_peek(self):
        item,post,k,source,features=self.fixture()
        first=source[0].copy()
        first["available_sample"]=int(item["decision_ends"][0]+256)
        first["decision_sample"]=first["available_sample"]
        first["time"]=first["available_sample"]/44100.
        original,X=source_rows([first],item["decision_ends"],features)
        self.assertEqual(len(original),1)
        np.testing.assert_array_equal(X[0,:87],features[0,0])

    def test_threshold_never_changes_signed_source(self):
        item,post,k,source,features=self.fixture()
        ons,X=source_rows(source,item["decision_ends"],features)
        saved=[e.copy() for e in source]
        filtered=kept_native(ons,np.array([.28]),threshold=.3)
        self.assertEqual(filtered,[])
        retained=kept_native(ons,np.array([.28]),threshold=.2)
        self.assertEqual(retained,ons)
        self.assertEqual(saved,source)
        with self.assertRaises(ValueError):
            kept_native(ons,np.array([.28]),threshold=.33)

    def test_grouped_holdout_hides_each_held_composition(self):
        item,post,k,source,features=self.fixture()
        rows=[]
        for i,composition in enumerate(("BN1","Jazz1","Funk2","Rock3","SS1")):
            events=[]
            X=np.zeros((20,DIM),np.float32)
            Y=np.zeros(20,np.int8);Y[:10]=1
            X[:,0]=np.arange(20,dtype=np.float32)+i
            for ix in range(20):
                t=(ix+1)*1024
                events.append(dict(type="note_on",pitch=40,
                                   available_sample=t,decision_sample=t,
                                   time=t/44100.,reason="synthetic",velocity=96))
            truth=[dict(time=e["time"],pitch=40,duration=.1) for e in events[:10]]
            rows.append(dict(composition=composition,X=X,y=Y,events=events,truth=truth))
        model,threshold,summaries,folds,counts=group_oof(rows)
        self.assertIsNotNone(model)
        self.assertIn(threshold,THRESHOLDS)
        self.assertEqual(len(folds),5)
        self.assertEqual(counts["total"],100)
        self.assertEqual(counts["matched"],50)
        self.assertEqual(len(summaries),len(THRESHOLDS))

if __name__=="__main__":
    unittest.main()
