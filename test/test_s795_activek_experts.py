"""S795 causal, no retroactive events, no latency and head diversity tests."""
import unittest
import numpy as np
from causal_note.strict_live import SR
from causal_note.s795_activek_experts import (
    HEADS,MASKS,CUTS,FEATURES,causal_features,regions,
    onset_targets,sample_head,decode
)
from causal_note.s793_zero_added_latency import (
    SOURCE_ID,veto_added_event_delay,mark_reference,LatencyViolation
)

def event(kind,pitch,t):
    return dict(type=kind,pitch=pitch,decision_sample=t,
        available_sample=t,time=t/SR,reason="native",
        velocity=96 if kind=="note_on" else 0)

class S795IndependentCausalExpertTests(unittest.TestCase):
    def fixture(self):
        t=3;d=np.array([1024,2048,3072],np.int64)
        X=np.zeros((t,49,FEATURES),np.float32)
        X[0,3,0]=.80
        X[0,3,1]=.72
        X[0,3,5]=.50
        X[0,3,6]=.35
        X[0,3,12]=0
        X[0,3,15]=1./6
        X[0,0,12]=1
        X[0,3,10]=.44
        originals=[event("note_on",40,1024),event("note_off",40,2048)]
        P={name:np.zeros((3,49),np.float32) for name in HEADS}
        P["general"][0,3]=.99
        return d,X,originals,P

    def test_specialists_have_distinct_regions_and_mask_count_not_fixed(self):
        d,X,ev,P=self.fixture()
        source=X[:,:,12]>.5
        e=regions(X,source)
        self.assertEqual(len(HEADS),5)
        self.assertEqual(len(MASKS),13)
        self.assertTrue(e["general"][0,3])
        self.assertFalse(e["bass"][0,3])
        self.assertTrue(e["polyphonic"][0,3])
        self.assertTrue(e["transient"][0,3])
        self.assertTrue(e["harmonic"][0,3])

    def test_identity_preserves_every_source_event_with_same_timestamp(self):
        d,X,original,P=self.fixture()
        candidate,changes=decode(original,d,X,P,0,0.,3072)
        self.assertEqual([(z["type"],z["pitch"],z["available_sample"]) for z in candidate],
            [(z["type"],z["pitch"],z["available_sample"]) for z in original])
        self.assertEqual(changes["new_onsets"],0)
        self.assertEqual(changes["max_added_native_delay_samples"],0)

    def test_new_missing_voice_has_immediate_noteon_and_first_evidence_release(self):
        d,X,original,P=self.fixture()
        out,changes=decode(original,d,X,P,1,.93,3072)
        self.assertEqual(changes["new_onsets"],1)
        added=[e for e in out if e.get(SOURCE_ID) is None]
        self.assertEqual([(e["type"],e["pitch"],e["available_sample"]) for e in added],
                         [("note_on",43,1024),("note_off",43,2048)])
        self.assertEqual(added[0]["first_eligible_sample"],1024)
        self.assertEqual(added[1]["first_eligible_sample"],2048)
        self.assertEqual(changes["extra_confirmation_samples"],0)
        self.assertEqual([z["available_sample"] for z in out if z.get(SOURCE_ID) is not None],
                         [1024,2048])

    def test_capacity_and_native_priority(self):
        d,X,original,P=self.fixture()
        original=[event("note_on",p,1024) for p in range(40,46)]
        original += [event("note_off",p,2048) for p in range(40,46)]
        out,changes=decode(original,d,X,P,1,.85,3072)
        self.assertEqual(changes["new_onsets"],0)
        self.assertEqual(len(out),len(original))
        self.assertEqual(sum(e["type"]=="note_on" for e in out),6)

    def test_native_restrike_closes_synthetic_at_current_clock(self):
        d,X,original,P=self.fixture()
        original=[event("note_on",40,1024),event("note_off",40,2048),
                  event("note_on",43,2048),event("note_off",43,3072)]
        out,changes=decode(original,d,X,P,1,.85,3072)
        early=[e for e in out if e["pitch"]==43 and e["available_sample"]==2048]
        self.assertEqual([e["type"] for e in early],["note_off","note_on"])
        self.assertEqual(changes["new_onsets"],1)

    def test_audio_future_mutation_not_affect_past(self):
        d,X,orig,P=self.fixture()
        first,_=decode(orig,d,X,P,1,.93,3072)
        XX=X.copy();XX[2:]=.999
        PP={name:p.copy() for name,p in P.items()}
        for name in PP:PP[name][2:,:]=.99
        second,_=decode(orig,d,XX,PP,1,.93,3072)
        def before(rows):
            return [(e["type"],e["pitch"],e["available_sample"])
                    for e in rows if e["available_sample"]<3072]
        self.assertEqual(before(first),before(second))

    def test_end_of_stream_synthetic_noteoff_not_late(self):
        d,X,orig,P=self.fixture()
        X[1:,3,1]=.7
        out,c=decode(orig,d,X,P,1,.93,3072)
        self.assertTrue(any(e["type"]=="note_off" and e["pitch"]==43
                            and e["available_sample"]==3072 for e in out))
        self.assertEqual(c["new_offsets"],c["new_onsets"])

    def test_training_uses_supervised_targets_only_in_fit_function(self):
        d,X,orig,P=self.fixture()
        y=onset_targets([dict(pitch=43,time=900/SR,duration=.3)],d,
                        X[:,:,12]>.5)
        self.assertTrue(y[0,3])
        mask=regions(X,X[:,:,12]>.5)["general"]
        data,labels=sample_head(X,y,mask,795)
        self.assertEqual(data.shape[1],23)
        self.assertEqual(int(labels.sum()),1)

    def test_invalid_candidate_mask_cut_and_prediction_rejected(self):
        d,X,orig,P=self.fixture()
        for mask,cut in ((128,.85),(1,.12),(0,.93)):
            if mask==0:continue
            with self.assertRaises(ValueError):
                decode(orig,d,X,P,mask,cut,3072)
        with self.assertRaises(ValueError):
            decode(orig,d,X,{k:P[k] for k in HEADS if k!="bass"},1,.93,3072)

    def test_any_direct_source_delay_vetoed(self):
        d,X,orig,P=self.fixture()
        baseline=mark_reference(orig)
        late=[dict(z) for z in baseline]
        late[1]["decision_sample"]+=1;late[1]["available_sample"]+=1
        with self.assertRaises(LatencyViolation):
            veto_added_event_delay(baseline,late)

if __name__=="__main__":
    unittest.main()
