"""NOTE microphone Live runtime contract tests; no microphone hardware required."""
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from scripts import live_note_onset256 as live
from onset256.live_event_stream import OnsetDecision
from onset256.runtime import Intent


class FakeOnset:
    def __init__(self):
        self.starts = []
        self.closed = False

    def push(self, pcm, *, start_sample):
        self.starts.append(start_sample)
        assert pcm.shape == (256,)
        return OnsetDecision(start_sample + 256,
                             (Intent("note_on",40,start_sample+256,.9,"test"),))

    def finish(self):
        self.closed = True
        return ()


class FakePortAudio:
    def __init__(self, overflow_at=None):
        self.overflow_at = overflow_at
        self.n = 0
        self.open_kwargs = None

    def query_devices(self, index=None):
        ds = [
            {"name":"External Microphone (Realtek Audio)","max_input_channels":1,
             "hostapi":0,"default_samplerate":44100},
            {"name":"Output Only","max_input_channels":0,
             "hostapi":0,"default_samplerate":44100}]
        return ds[index] if index is not None else ds

    def query_hostapis(self, index):
        return {"name":"WASAPI"}

    def InputStream(self, **kw):
        self.open_kwargs=kw
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        return None

    def read(self, count):
        self.n += 1
        assert count == 256
        return np.zeros((256,1),np.float32), self.n == self.overflow_at


class LiveNotePCM256Tests(unittest.TestCase):
    def test_device_listing_does_not_load_neural_model(self):
        p=FakePortAudio()
        with patch.object(live,"load_engine",side_effect=AssertionError("loaded")),redirect_stdout(StringIO()) as output:
            result=live.run(live.make_parser().parse_args(["--list-devices"]),sd=p)
        self.assertEqual(result,0)
        self.assertIn("Realtek",output.getvalue())

    def test_exact_256_boundaries_no_extra_hold(self):
        audio=FakePortAudio()
        engine=FakeOnset()
        args=live.make_parser().parse_args([
            "--fold","BN1","--duration-s","0.012","--input-device","0"])
        with patch.object(live,"load_engine",return_value=engine),redirect_stdout(StringIO()) as output:
            code=live.run(args,sd=audio)
        self.assertEqual(code,0)
        self.assertEqual(engine.starts,[0,256,512])
        self.assertTrue(engine.closed)
        self.assertEqual(audio.open_kwargs["blocksize"],256)
        self.assertEqual(audio.open_kwargs["samplerate"],44100)
        self.assertIn('"pitch": 40',output.getvalue())

    def test_overflow_stops_without_fabricating_missing_audio(self):
        audio=FakePortAudio(overflow_at=2)
        engine=FakeOnset()
        args=live.make_parser().parse_args(["--fold","Rock3","--duration-s","0.02"])
        with patch.object(live,"load_engine",return_value=engine),redirect_stdout(StringIO()):
            with self.assertRaisesRegex(BufferError,"overflow"):
                live.run(args,sd=audio)
        self.assertEqual(engine.starts,[0])

    def test_sample_timestamp_must_equal_current_boundary(self):
        class Delayed:
            def push(self, pcm, *, start_sample):
                return OnsetDecision(start_sample+512,())
        with self.assertRaisesRegex(AssertionError,"delayed"):
            live.checked_hop(Delayed(),np.zeros(256,np.float32),0)

    def test_reject_invalid_blocks_and_duplicate_onsets(self):
        with self.assertRaisesRegex(ValueError,"PCM256"):
            live.checked_hop(FakeOnset(),np.zeros(255,np.float32),0)
        class Duplicates:
            def push(self,pcm,*,start_sample):
                return OnsetDecision(start_sample+256,(
                    Intent("note_on",40,start_sample+256,.8,"a"),
                    Intent("note_on",40,start_sample+256,.9,"b")))
        with self.assertRaisesRegex(AssertionError,"Repeated pitch"):
            live.checked_hop(Duplicates(),np.zeros(256,np.float32),0)


if __name__=="__main__":
    unittest.main()
