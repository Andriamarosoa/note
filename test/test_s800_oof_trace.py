"""S800 FIT OOF trace integrity and no effect on native source MIDI."""
import tempfile,unittest
from pathlib import Path
import numpy as np
from causal_note.strict_live import SR
from causal_note.s795_activek_experts import HEADS,FEATURES
from scripts.s800_oof_trace import snapshot,verify_snapshot

class FITTraceTests(unittest.TestCase):
    def case(self):
        d=np.array([1024,2048,3072],np.int64)
        X=np.zeros((3,49,FEATURES),np.float16)
        y=np.zeros((3,49),bool);y[0,3]=True
        def event(kind,t):
            return {"type":kind,"pitch":40,"available_sample":t,
                    "decision_sample":t,"time":t/SR,
                    "reason":"source_native","velocity":96}
        ev=[event("note_on",1024),event("note_off",2048)]
        row={"name":"synthetic","composition":"foldA","texture":"solo",
             "X":X,"y":y,"decision_ends":d,"ends":d,
             "samples":3072,"events":ev,
             "truth":[dict(pitch=40,time=1024/SR,duration=1024/SR)]}
        preds={h:np.zeros((3,49),np.float32) for h in HEADS}
        training={h:dict(trained=True,compositions=["foldB","foldC"]) for h in HEADS}
        return row,preds,training
    def test_round_trip_integrity_and_immutable_midi(self):
        record,P,training=self.case()
        original_events=[dict(x) for x in record["events"]]
        with tempfile.TemporaryDirectory() as td:
            m=snapshot(0,record,P,training,"foldA",td)
            detail=verify_snapshot(m,td)
            self.assertEqual(detail["head_order"],list(HEADS))
            self.assertEqual(detail["source_events"],original_events)
            self.assertEqual(record["events"],original_events)
            with self.assertRaises(FileExistsError):
                snapshot(0,record,P,training,"foldA",td)
            (Path(td)/"fit_000.json").write_text("altered")
            with self.assertRaises(AssertionError):
                verify_snapshot(m,td)
    def test_oof_composition_held_out(self):
        record,P,training=self.case()
        training["bass"]["compositions"].append("foldA")
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(AssertionError):
                snapshot(0,record,P,training,"foldA",td)
    def test_trace_rejects_corrupt_probas_and_shape(self):
        record,P,training=self.case()
        P["general"][0,0]=np.nan
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(AssertionError):
                snapshot(0,record,P,training,"foldA",td)
        P["general"][0,0]=0
        record["X"]=record["X"].astype(np.float32)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(AssertionError):
                snapshot(0,record,P,training,"foldA",td)
    def test_no_cal_or_borrowed_fold(self):
        record,P,training=self.case()
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(AssertionError):
                snapshot(0,record,P,training,"CAL",td)
            with self.assertRaises(AssertionError):
                snapshot(80,record,P,training,"foldA",td)
if __name__=="__main__":unittest.main()
