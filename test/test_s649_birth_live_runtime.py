"""S649 real original NOTE native true blocked PCM parity and causality."""
from __future__ import annotations
import unittest
import numpy as np

from causal_note.strict_live import feature_track,HOP,load_heads
from causal_note.s648_event_birth import BirthPolicy,decode_birth
from causal_note.s649_birth_live_runtime import S649CausalBirthStream
from scripts.s616_live_dev_ablation import MODEL
from scripts.s638_note_midi_same_guitarset import S614_SHA

class SimpleHead:
    def predict_proba(self,x):
        q=np.where(x[:,0]>.3,.92,.01)
        return np.column_stack((1-q,q))


class S649Tests(unittest.TestCase):
    def test_real_checkpoint_parity_at_all_supported_policies(self):
        original=load_heads(MODEL,expected_sha256=S614_SHA)
        rng=np.random.default_rng(649)
        samples=rng.normal(0,.02,HOP*28).astype(np.float32)
        ends,x=feature_track(samples)
        p,k=original.predict(x)
        row=dict(ends=ends,x=x,p=p,k=k,n=len(samples),
                 selection=dict(original.metadata["selection"]))
        for policy in (BirthPolicy(),BirthPolicy(.7,.10),BirthPolicy(.85,.15)):
            expected,_=decode_birth(row,SimpleHead(),policy)
            live=S649CausalBirthStream(original,SimpleHead(),policy)
            observed=[]
            for pos in range(0,len(samples),HOP):
                observed.extend(live.push(samples[pos:pos+HOP],start_sample=pos))
            observed.extend(live.finish())
            self.assertEqual(expected,observed)
            self.assertEqual(sum(e["type"]=="note_on" for e in observed),
                             sum(e["type"]=="note_off" for e in observed))
            self.assertTrue(all(e["decision_sample"]<=e["available_sample"]
                                for e in observed))

    def test_future_mutation_cannot_change_prefix(self):
        original=load_heads(MODEL,expected_sha256=S614_SHA)
        rng=np.random.default_rng(649)
        samples=rng.normal(0,.025,HOP*24).astype(np.float32)
        mutated=samples.copy()
        mutated[17*HOP:]=rng.normal(0,.2,len(samples)-17*HOP)
        def outputs(audio):
            live=S649CausalBirthStream(original,SimpleHead(),BirthPolicy(.7,.1))
            events=[]
            for pos in range(0,len(audio),HOP):
                events.extend(live.push(audio[pos:pos+HOP],start_sample=pos))
            return [e for e in events if e["available_sample"]<=17*HOP]
        self.assertEqual(outputs(samples),outputs(mutated))
if __name__=="__main__":unittest.main()
