"""Windows microphone should not stop on arbitrary CPU time budgets by default.

No real capture device needed: a fake PortAudio stream feeds one valid
block, deliberately ages it beyond the legacy two-block deadline, and
tests both default and opt-in strict modes.
"""
from __future__ import annotations
from contextlib import redirect_stderr
import io
import sys
import time
import types
import unittest
from unittest.mock import patch
import numpy as np
from causal_note.strict_live import HOP,SR
from scripts import live_strict_notes


class AbortStream(Exception):
    pass


class FakeInputStream:
    latency = .017
    active = True
    def __init__(self, *, samplerate, blocksize, channels, dtype, device, latency, callback):
        if (samplerate,channels,dtype)!=(SR,1,"float32"):
            raise AssertionError("NOTE opened unexpected mic input format")
        self.blocksize=blocksize
        self.callback=callback
    def __enter__(self):
        self.callback(np.zeros((self.blocksize,1),np.float32),self.blocksize,None,None)
        # Legacy mode with 256-sample blocks would exceed 11.6ms.
        time.sleep(.032)
        return self
    def __exit__(self,*args):
        self.active=False


class FakeNotes:
    live_instance=None
    def __init__(self,heads):
        self.times=0
        if FakeNotes.live_instance is None:
            FakeNotes.live_instance=self
        else:
            FakeNotes.live_instance=self
    def push(self,samples,*,start_sample):
        self.times+=1
        # First instance: warmup is never a real mic; second is the mic.
        if self.times<9:
            return []
        raise KeyboardInterrupt()
    def finish(self,reason):
        return []


class MicDeadlineTests(unittest.TestCase):
    def test_default_disables_outdated_two_block_deadline(self):
        self.assertFalse(live_strict_notes.capture_deadline_exceeded(-100.,100.,HOP,0))
        self.assertTrue(live_strict_notes.capture_deadline_exceeded(0.,.05,HOP,2))
        self.assertFalse(live_strict_notes.capture_deadline_exceeded(0.,.005,HOP,2))
    def test_old_deadline_disabled_in_actual_capture_loop(self):
        class Stream:
            instances=[]
            def __init__(self,heads):
                self.i=len(Stream.instances)
                Stream.instances.append(self)
            def push(self,samples,*,start_sample):
                if self.i==1:raise KeyboardInterrupt()
                return []
            def finish(self,reason):return []
        fake_device=types.SimpleNamespace(InputStream=FakeInputStream,CallbackAbort=AbortStream)
        Stream.instances=[]
        with patch.dict(sys.modules,{"sounddevice":fake_device}), \
             patch.object(live_strict_notes,"load_heads",return_value=object()), \
             patch.object(live_strict_notes,"StrictLiveStream",Stream), \
             redirect_stderr(io.StringIO()) as output:
            live_strict_notes.main(["--mic","--device","1","--block-size","256"])
            self.assertIn('"capture_deadline_blocks": 0',output.getvalue())
            self.assertIn('"queue_blocks": 32',output.getvalue())
        self.assertEqual(len(Stream.instances),2)
    def test_strict_two_block_check_opt_in(self):
        class Stream:
            def __init__(self,heads):pass
            def push(self,samples,*,start_sample):return []
            def finish(self,reason):return []
        fake_device=types.SimpleNamespace(InputStream=FakeInputStream,CallbackAbort=AbortStream)
        with patch.dict(sys.modules,{"sounddevice":fake_device}), \
             patch.object(live_strict_notes,"load_heads",return_value=object()), \
             patch.object(live_strict_notes,"StrictLiveStream",Stream), \
             redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(BufferError,"Optional capture deadline exceeded"):
                live_strict_notes.main(["--mic","--device","1","--block-size","256",
                                        "--capture-deadline-blocks","2"])
    def test_invalid_queue_and_deadline_values_rejected(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                live_strict_notes.main(["--queue-blocks","0","--mic"])
            with self.assertRaises(SystemExit):
                live_strict_notes.main(["--capture-deadline-blocks","-1","--mic"])


if __name__=="__main__":
    unittest.main()
