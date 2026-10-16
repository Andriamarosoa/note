"""Execute real python -m NOTE CLI with S630 flags, including frozen checkpoint.

This test specifically catches hop_request used before assignment.
Unlike the old purely model-unit tests it goes through argparse, model
loading, warm-up, stdout/stderr and full WAV -> NOTE event generation.
It does NOT emulate a real Windows iRig/PortAudio interface.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
import wave
import numpy as np

REPO=pathlib.Path(__file__).resolve().parents[1]


def cli(*args):
    return subprocess.run(
        [sys.executable,"-m","scripts.live_strict_notes",*map(str,args)],
        cwd=REPO,capture_output=True,text=True,timeout=50,
    )


def write_wave(destination,duration=.14):
    sr=44100
    times=np.arange(round(duration*sr))/sr
    pcm=.05*np.sin(2*np.pi*82.4069*times)+.025*np.sin(2*np.pi*164.8138*times)
    data=(pcm*32767).astype("<i2")
    with wave.open(str(destination),"wb") as f:
        f.setnchannels(1);f.setframerate(sr);f.setsampwidth(2)
        f.writeframes(data.tobytes())


class S630RealCliSmokeTests(unittest.TestCase):
    def test_real_cli_status_accepts_original_and_batch4(self):
        for batch in (1,4):
            with self.subTest(batch=batch):
                result=cli("--status","--hop-size","256",
                           "--inference-batch-hops",batch)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertNotIn("UnboundLocalError",result.stderr)
                self.assertEqual(json.loads(result.stdout)["hop"],256)

    def test_real_cli_wav_full_checkpoint_and_batched_onnx_prediction(self):
        with tempfile.TemporaryDirectory() as td:
            td=pathlib.Path(td)
            source=td/"fixture.wav";write_wave(source)
            for batch in (1,4):
                dest=td/f"events{batch}.jsonl"
                with self.subTest(batch=batch):
                    result=cli("--wav",source,
                        "--hop-size","256","--block-size","256",
                        "--lookahead-ms","0","--inference-batch-hops",batch,
                        "--out",dest)
                    self.assertEqual(result.returncode,0,result.stderr)
                    self.assertTrue(dest.exists())
                    for line in dest.read_text().splitlines():
                        event=json.loads(line)
                        self.assertIn(event["type"],("note_on","note_off"))
                        self.assertLessEqual(event["decision_sample"],
                                             event["available_sample"])
            self.assertEqual(
                [(json.loads(l)["type"],json.loads(l)["pitch"],
                  json.loads(l)["decision_sample"])
                 for l in (td/"events1.jsonl").read_text().splitlines()],
                [(json.loads(l)["type"],json.loads(l)["pitch"],
                  json.loads(l)["decision_sample"])
                 for l in (td/"events4.jsonl").read_text().splitlines()],
            )

    def test_full_mic_and_soundfont_command_with_simulated_capture(self):
        # Exercise the REAL CLI main(), PortAudio callback -> bounded input
        # queue -> frozen S613 ONNX batch4 -> MIDI output -> shutdown.
        # Hardware callback and soundfont engine are intentionally mocked:
        # the CI runners do not have the user's iRig or their SF2.
        import io
        import types
        from contextlib import contextmanager
        from unittest.mock import patch
        from scripts import live_strict_notes

        audio=np.zeros((256,1),dtype=np.float32)
        seen=[]

        class SimulatedPiano:
            def __init__(self,*a,**kw):
                self.events=[]
                seen.append(self)
            def send(self,event):
                self.events.append(event)
            def health_error(self):
                return None
            def diagnostics(self):
                return {"simulated":True}
            def close(self):
                pass

        class FakeCapture:
            @property
            def active(self):
                # Exit the production mic processing loop naturally after
                # draining six already-captured test blocks.
                raise KeyboardInterrupt

        @contextmanager
        def fake_microphone(**kwargs):
            for _ in range(6):
                kwargs["callback"](audio,256,None,None)
            yield FakeCapture(),types.SimpleNamespace(
                device_name="EXTERNAL_MIC_SIMULATED",
                host_api="MME",device_index=1,latency_ms=17.4,
            )

        with tempfile.TemporaryDirectory() as td:
            sf2=pathlib.Path(td)/"mock.sf2"
            sf2.write_bytes(b"PLACEHOLDER_FOR_REAL_SF2_ENGINE")
            with (
                patch.object(live_strict_notes,"managed_input_stream",fake_microphone),
                patch.dict(sys.modules,{"sounddevice":types.SimpleNamespace(
                    CallbackAbort=RuntimeError)}),
                patch("scripts.live_piano_output.PianoOutput",SimulatedPiano),
                patch.object(sys,"stdout",io.StringIO()) as out,
                patch.object(sys,"stderr",io.StringIO()) as err,
            ):
                live_strict_notes.main([
                    "--mic","--device","MME:External Microphone",
                    "--soundfont",str(sf2),"--hop-size","256",
                    "--block-size","256","--lookahead-ms","0",
                    "--inference-batch-hops","4",
                ])
                stderr=err.getvalue()
                self.assertNotIn("Traceback",stderr)
                self.assertIn('"s630_input_queue_diagnostics"',stderr)
                self.assertIn('"callbacks_received": 6',stderr)
                self.assertIn('"callbacks_processed": 6',stderr)
                self.assertTrue(seen)

    def test_invalid_hop_gives_argument_error_not_unbound_variable(self):
        x=cli("--status","--hop-size","128","--inference-batch-hops","4")
        self.assertNotEqual(x.returncode,0)
        self.assertIn("requires --hop-size 256",x.stderr)
        self.assertNotIn("UnboundLocalError",x.stderr)

    def test_true_mic_command_reaches_device_selection_instead_of_hop_crash(self):
        # A nonexistent input device must fail only during PortAudio device
        # selection, not during argparse or warmup. Some CI images without
        # PortAudio have a missing sounddevice module, also acceptable.
        # We deliberately do NOT open the user's real hardware remotely.
        x=cli("--mic","--device","___S630_NONEXISTENT_MIC___",
              "--hop-size","256","--block-size","256",
              "--lookahead-ms","0","--inference-batch-hops","4")
        self.assertNotEqual(x.returncode,0)
        self.assertNotIn("UnboundLocalError",x.stderr)
        self.assertTrue(
            "Peripherique audio ambigu ou absent" in x.stderr
            or "ModuleNotFoundError" in x.stderr
            or "Error querying device" in x.stderr
            or "No module named 'sounddevice'" in x.stderr
            or "PortAudio library not found" in x.stderr,
            x.stderr,
        )


if __name__=="__main__":
    unittest.main()
