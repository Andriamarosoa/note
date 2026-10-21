import unittest
import numpy as np
import torch
from pathlib import Path
from full_note.acoustics import HOP,load_native_heads
from full_note.temporal import TemporalHeads
from full_note.runtime import Stream
class OnlyFullNote(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.native=load_native_heads(Path(__file__).resolve().parents[1]/"model/live/strict_note/NOTE_strict_live.npz")
        cls.model=TemporalHeads(np.zeros(288,np.float32),np.ones(288,np.float32),width=8).eval()
    def test_pcm256(self):
        s=Stream(self.native,self.model,.5)
        v=s.push(np.zeros(4*HOP,np.float32),start_sample=0)
        self.assertEqual([x.sample for x in v],[256,512,768,1024])
        self.assertEqual(s.calls,4)
        with self.assertRaises(ValueError):s.push(np.zeros(HOP,np.float32),start_sample=0)
    def test_no_other_axis_decoder(self):
        from full_note import runtime as r
        for name in ("OnsetDecoder","OffsetDecoder","ExactKDecoder","ActiveKDecoder","SixIndependentAxes","FramePitchDecoder"):
            self.assertFalse(hasattr(r,name))
if __name__=="__main__":unittest.main()
