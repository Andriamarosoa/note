"""Negative controls for S805 causal crossfold heads (no network)."""
import unittest
import numpy as np
from scripts.s805_fit_crossfold_onset_risk import (
    train_risk,feature_matrix,FEATURES_ACOUSTIC,FEATURES_JOINT,gate,SCHEMA)
from scripts.s802_retrigger_forensics import SCHEMA as F_SCHEMA
from causal_note.s795_activek_experts import HEADS

class CausalOnsetRiskTests(unittest.TestCase):
    def fake_forensic(self):
        rows=[]
        for c in ("BN1","Funk2","Jazz1","Rock3","SS1"):
            for i in range(36):
                x={"composition":c,"synthetic_matched_truth":i%3!=0,"pitch":40+i%40,
                   "on":i/36,"sustain":.3+(i%5)/10,"off":(36-i)/36,
                   "on_rise":i/50,"sustain_rise":(i%3)/10,
                   "harmonic_parent":i/100}
                for name in HEADS:x[name+"_p"]=.4+(i%6)/11
                rows.append(x)
        return {"schema":F_SCHEMA,"complete":True,
                "observations":{"consensus":rows,"adaptive":rows}}
    def test_train_excludes_current_composition(self):
        f=self.fake_forensic()
        for c in ("BN1","Funk2","Jazz1","Rock3","SS1"):
            for source,cols in (("consensus",FEATURES_ACOUSTIC),
                                ("adaptive",FEATURES_JOINT)):
                model,proof=train_risk(f,c,source,cols)
                self.assertNotIn(c,proof["training_compositions"])
                self.assertEqual(len(proof["training_compositions"]),4)
                self.assertEqual(proof["training_rows"],144)
                self.assertEqual(len(model.classes_),2)
    def test_future_frame_mutation_does_not_change_causal_prefix(self):
        X=np.zeros((4,49,23),np.float32)
        X[:,:,0]=.5; X[:,:,1]=.7; X[:,:,2]=.2; X[:,:,5]=.3
        P={h:np.full((4,49),.99,np.float32) for h in HEADS}
        a=feature_matrix(X,P,FEATURES_JOINT)
        mutated=X.copy();mutated[2:]=1.
        changedP={h:v.copy() for h,v in P.items()}
        for h in HEADS:changedP[h][2:]=.1
        b=feature_matrix(mutated,changedP,FEATURES_JOINT)
        np.testing.assert_array_equal(a.reshape(4,49,-1)[:2],
                                      b.reshape(4,49,-1)[:2])
    def test_eligibility_guard_is_current_frame(self):
        X=np.zeros((3,49,23),np.float32)
        X[:,:,0]=.6;X[:,:,1]=.7;X[:,:,2]=.2;X[:,:,5]=.3
        P={h:np.full((3,49),.99,np.float32) for h in HEADS}
        a,_=gate(X,P,"spectral_balance")
        future=X.copy();future[2,:,:]=1.
        b,_=gate(future,P,"spectral_balance")
        for h in HEADS:
            np.testing.assert_array_equal(a[h][:2],b[h][:2])

if __name__=="__main__":unittest.main()
