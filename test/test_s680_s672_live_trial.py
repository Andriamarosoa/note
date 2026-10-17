"""S680 regression: opt-in S672 NoteOff specialist on original batched S630."""
import io,json,tempfile,unittest
from contextlib import redirect_stderr,redirect_stdout
from dataclasses import asdict
from pathlib import Path
import numpy as np

from causal_note.strict_live import HOP,Selection
from causal_note.batched_live import BatchedLiveStream
from causal_note.s672_live_offset import S672TrialNoteOff
from scripts.live_strict_notes import main

class FrozenHeads:
    def __init__(self):
        self.metadata={"selection":asdict(Selection(onset=.5,active_on=.5,
                        active_off=.25,release_frames=3))}
        self.i=0
    def predict(self,x):
        n=len(x) if np.asarray(x).ndim==3 else 1
        p=np.zeros((n,49,3),np.float32)
        p[:,:,1]=.02
        k=np.zeros((n,7),np.float32);k[:,1]=1.
        for j in range(n):
            t=self.i+j
            p[j,0,0]=.99 if t==2 else .02
            p[j,0,1]=.90 if 2<=t<20 else .01
            p[j,0,2]=.02
        self.i+=n
        return (p,k) if np.asarray(x).ndim==3 else (p[0],k[0])
class SimpleHazard:
    n_features_in_=75
    def predict_proba(self,X):
        # First valid after onset+six source hops, no future audio.
        assert X.shape[1]==75
        score=np.full(len(X),.98,np.float64)
        return np.column_stack((1-score,score))

def play(stream):
    out=[]
    for i in range(45):
        out+=stream.push(np.zeros(HOP,np.float32),start_sample=i*HOP)
    out+=stream.finish()
    return out

class S680Checks(unittest.TestCase):
    def test_disabled_default_native_preserved(self):
        a=play(BatchedLiveStream(FrozenHeads(),batch_hops=4))
        b=play(BatchedLiveStream(FrozenHeads(),batch_hops=4,
                                 offset_adapter=None))
        self.assertEqual([(e["type"],e["pitch"],e["decision_sample"])
                          for e in a],
                         [(e["type"],e["pitch"],e["decision_sample"])
                          for e in b])

    def test_optin_preserves_noteon_and_exact_K(self):
        default=play(BatchedLiveStream(FrozenHeads(),batch_hops=4))
        adapter=S672TrialNoteOff(SimpleHazard(),gate=.6,persistence=1)
        candidate=play(BatchedLiveStream(FrozenHeads(),batch_hops=4,
                                         offset_adapter=adapter))
        orig_on=[(e["pitch"],e["decision_sample"]) for e in default
                 if e["type"]=="note_on"]
        candidate_on=[(e["pitch"],e["decision_sample"]) for e in candidate
                      if e["type"]=="note_on"]
        self.assertEqual(orig_on,candidate_on)
        offs=[e for e in candidate if e["type"]=="note_off"]
        self.assertEqual(len(offs),1)
        self.assertEqual(offs[0]["reason"],"s680_causal_hazard")
        for e in candidate:
            self.assertLessEqual(e["decision_sample"],e["available_sample"])
        self.assertEqual(adapter.stats["native_note_on_preserved"],len(orig_on))

    def test_cli_guard_rejects_bad_combinations(self):
        # The argument rejection is before model loading.
        with tempfile.TemporaryDirectory() as tmp:
            sample=Path(tmp)/"s672_hazard.joblib"
            sample.write_bytes(b"placeholder")
            cases=[
                ["--inference-batch-hops","1"],
                ["--inference-batch-hops","4","--lookahead-ms","5"],
                ["--inference-batch-hops","4","--s667-active-off",".15"],
                ["--inference-batch-hops","4","--hop-size","128"],
            ]
            for case in cases:
                with self.subTest(case=case),redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as err:
                        main(["--status","--s672-hazard-model",str(sample),*case])
                self.assertEqual(err.exception.code,2)

if __name__=="__main__":unittest.main()
