"""Causal native256 Onset harmonic forensics: no changed decisions."""
import io, json, unittest
import numpy as np
from onset256.acoustics import HOP,SR
from onset256.runtime import Stream,Intent,Decision
from onset256.harmonic_evidence import (
    current_pitch_strength,onset_event_evidence,HARMONIC_FIELDS)
from scripts.onset_harmonic_forensics256 import annotate,aggregate,score

class NativeDummy:
    metadata={"selection":{}}
    def __init__(self):self.calls=0
    def predict(self,feat):
        self.calls+=1
        p=np.zeros((49,3),np.float32)
        p[0]=(.25,.85,.01)
        return p,np.ones(7,np.float32)/7

class SpecialistDummy:
    def __init__(self):self.calls=0
    def predict_proba(self,x):
        self.calls+=1
        if x.shape!=(49,39):raise AssertionError("wrong trained 39D schema")
        return np.tile(np.array([.2,.8],np.float32),(49,1))

class Harmonic256Tests(unittest.TestCase):
    def test_parent_harmonics_use_only_current_pitch_features(self):
        f=np.zeros((49,126),np.float32)
        f[0,[0,8,16]]=2.
        f[12,[0,8,16]]=.25
        p=np.zeros((49,3),np.float32)
        p[0,1]=.9;p[12]=(.4,.7,.1)
        k=np.ones(7,np.float32)/7
        s=np.full(49,.75,np.float32)
        parent=onset_event_evidence(f,p,k,s,52,HOP)
        self.assertEqual(parent["fundamentals_parent_pitch"],40)
        self.assertTrue(parent["harmonic_parent_stronger"])
        self.assertAlmostEqual(parent["context"]["strongest_lower_fundamental"],2.)
        self.assertGreater(parent["context"]["lower_fundamental_ratio"],4.)
        self.assertEqual(set(parent["context"]),set(HARMONIC_FIELDS))
        low=onset_event_evidence(f,p,k,s,40,HOP)
        self.assertIsNone(low["fundamentals_parent_pitch"])
        self.assertFalse(low["harmonic_parent_stronger"])
        with self.assertRaises(ValueError):
            onset_event_evidence(f,p,k,s,52,HOP+1)
    def test_synthetic_fragmentation_and_audit_do_not_change_onset(self):
        pcm=np.zeros(4*HOP,np.float32)
        n1,m1=NativeDummy(),SpecialistDummy()
        n2,m2=NativeDummy(),SpecialistDummy()
        baseline=Stream(n1,m1,audit_events=False)
        forensic=Stream(n2,m2,audit_events=True)
        a=baseline.push(pcm,start_sample=0)
        b=tuple(x for i in range(4) for x in
                forensic.push(pcm[i*HOP:(i+1)*HOP],start_sample=i*HOP))
        self.assertEqual([(d.sample,d.intents,d.candidate_intents) for d in a],
                         [(d.sample,d.intents,d.candidate_intents) for d in b])
        self.assertEqual((n1.calls,m1.calls,n2.calls,m2.calls),(4,4,4,4))
        self.assertFalse(any(d.evidence or d.candidate_evidence for d in a))
        self.assertEqual(len(b[0].candidate_evidence),1)
        self.assertEqual(b[0].candidate_evidence[0]["sample"],256)
        self.assertTrue(all(d.sample%256==0 for d in b))
    def test_labels_added_only_after_stream_audio_decisions(self):
        f=np.zeros((49,126),np.float32)
        p=np.zeros((49,3),np.float32)
        k=np.ones(7,np.float32)/7
        prob=np.zeros(49,np.float32)
        evid=onset_event_evidence(f,p,k,prob,40,256)
        d=[Decision(256,
            (Intent("note_on",40,256,.8,"test"),),
            (Intent("note_on",40,256,.8,"test"),),
            (evid,),(evid,))]
        writer=io.StringIO()
        counts,rows=annotate(d,[{"pitch":40,"time":256/SR}],
                           "native","00_BN1-foo_solo","BN1",writer)
        self.assertEqual(counts,dict(tp=1,fp=0,fn=0))
        self.assertEqual(rows[0]["label"],"TP")
        self.assertFalse(rows[0]["harmonic_lower_pitch_in_truth_80ms"])
        self.assertEqual(json.loads(writer.getvalue())["label"],"TP")
        writer=io.StringIO()
        counts,rows=annotate(d,[{"pitch":41,"time":256/SR}],
                           "native","00_BN1-foo_solo","BN1",writer)
        self.assertEqual(counts,dict(tp=0,fp=1,fn=1))
        self.assertEqual([v["label"] for v in map(json.loads,writer.getvalue().splitlines())],
                         ["FP","FN"])
    def test_harmonic_stronger_does_not_act_as_veto(self):
        f=np.zeros((49,126),np.float32)
        f[0,[0,8,16]]=3
        f[12,[0,8,16]]=.05
        p=np.zeros((49,3),np.float32)
        k=np.ones(7,np.float32)/7
        e=onset_event_evidence(f,p,k,np.ones(49,np.float32),52,256)
        self.assertTrue(e["harmonic_parent_stronger"])
        self.assertNotIn("accepted",e)
        self.assertNotIn("rejected",e)
        self.assertNotIn("ground_truth",e)
    def test_score_formula_unmodified(self):
        self.assertAlmostEqual(score(8099,4600,3168)["f1"],
                              2*8099/(2*8099+4600+3168))
if __name__=="__main__":
    unittest.main()
