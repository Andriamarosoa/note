"""S635 simulated PortAudio interface tests: no real microphone hardware in CI."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from causal_note.strict_live import SR
from scripts.s635_three_view_mic import capture_pcm, run


class FakeCapture:
    def __init__(self,*, error=None, invalid=False):
        self.error=error
        self.invalid=invalid
        self.parameters=None

    @contextmanager
    def stream_factory(self,**kwargs):
        self.parameters=kwargs
        callback=kwargs["callback"]
        block=kwargs["blocksize"]
        # Same signature sounddevice.InputStream uses. ALL input data is
        # simulated, but it goes through the real NOTE callback/collector.
        t=0
        for idx in range(50):
            chunk=.08*np.sin(2*np.pi*110*np.arange(t,t+block)/SR)
            input=np.asarray(chunk,dtype=np.float32).reshape(-1,1)
            if self.error and idx==1:
                callback(input,block,None,self.error)
                break
            if self.invalid and idx==1:
                callback(input,block,None,False)
                break
            callback(input,block,None,False)
            t+=block
        class Info:
            device_name="Fake WASAPI input"
            host_api="WASAPI"
            latency_ms=12.3
        yield None,Info()


class S635MicrophoneTests(unittest.TestCase):
    def test_in_memory_microphone_44k_samples_no_wav(self):
        fake=FakeCapture()
        samples,details=capture_pcm(seconds=1.,block_size=1024,
                                   stream_factory=fake.stream_factory)
        self.assertEqual(samples.shape,(SR,))
        self.assertGreater(np.sqrt(np.mean(samples**2)),.04)
        self.assertEqual(details["received"],SR)
        self.assertEqual(details["input_host_api"],"WASAPI")
        self.assertEqual(fake.parameters["dtype"],"float32")
        self.assertEqual(fake.parameters["samplerate"],SR)

    def test_unsupported_capture_errors_never_silently_dropped(self):
        with self.assertRaisesRegex(BufferError,"overload"):
            capture_pcm(seconds=1.,block_size=1024,
                        stream_factory=FakeCapture(error="overload").stream_factory)
        for sec,block in ((0,1024),(130,1024),(8,15)):
            with self.assertRaises(ValueError):
                capture_pcm(seconds=sec,block_size=block,
                            stream_factory=FakeCapture().stream_factory)

    def test_three_view_json_from_microphone_and_original_K_kept(self):
        fake=FakeCapture()
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/"mic_three.json"
            report=run(seconds=1.,block_size=1024,device="auto",
                       confidence=.6,out=target,
                       stream_factory=fake.stream_factory)
            self.assertTrue(target.exists())
            data=json.loads(target.read_text())
            self.assertEqual(data["source"],"microphone_capture_in_ram")
            self.assertTrue(data["not_live_midi_output"])
            self.assertEqual(data["original_samples"],SR)
            self.assertEqual(data["all_three_native_K_source"],"original")
            self.assertIn("collision_audit",data)
            self.assertTrue(data["collision_audit"]["never_use_future_native_for_live_decisions"])
            self.assertEqual(data["frames_analyzed"],SR//256)
            self.assertIn("original_candidates",data["candidate_counts"]) if data["candidate_counts"].get("original_candidates") else None
            self.assertEqual(data["capture"]["received"],SR)
            self.assertEqual(report["reference_sha256"],data["reference_sha256"])
            with self.assertRaises(FileExistsError):
                run(seconds=1.,block_size=1024,device=None,
                    confidence=.6,out=target,stream_factory=fake.stream_factory)

    def test_command_line_help_no_audio_device_needed(self):
        from scripts.s635_three_view_mic import main
        with self.assertRaises(SystemExit) as err:
            main(["--help"])
        self.assertEqual(err.exception.code,0)


if __name__=="__main__":
    unittest.main()
