"""Regression tests for NOTE's copied MIDI live audio transport (no sound card required)."""
from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace
import unittest

from scripts.live_audio_io import open_input_stream, open_output_stream


class FakeAudioBuffer:
    def fill(self, _value):
        pass


class FakeStream:
    def __init__(self, driver, direction, **kwargs):
        self.driver = driver
        self.direction = direction
        self.kwargs = kwargs
        self.index = kwargs["device"]
        if self.index is None:
            self.index = driver.default.device[0 if direction == "input" else 1]
        self.latency = driver.negotiated_latency[self.index]
        self.active = False
        self.closed = False

    def start(self):
        if self.index in self.driver.failing:
            raise OSError("simulated driver unavailable")
        self.active = True
        # The production selector requires at least 16 well-formed probe callbacks.
        for _ in range(16):
            self.kwargs["callback"](
                FakeAudioBuffer(),
                self.kwargs["blocksize"],
                None,
                False,
            )

    def stop(self):
        self.active = False

    def close(self):
        self.closed = True


class FakeSoundDevice:
    def __init__(self):
        self.default = SimpleNamespace(device=(0, 2))
        self.hosts = [
            {"name": "MME", "default_input_device": 0, "default_output_device": 2},
            {"name": "Windows WASAPI", "default_input_device": 1, "default_output_device": 3},
            {"name": "Windows WDM-KS", "default_input_device": 5, "default_output_device": 4},
        ]
        self.devices = [
            self._device("Microphone Array", 0, 1, 0),
            self._device("Microphone Array", 1, 1, 0),
            self._device("Speakers High Definition Audio", 0, 0, 2),
            self._device("Speakers High Definition Audio", 1, 0, 2),
            self._device("Speakers High Definition Audio", 2, 0, 2),
            self._device("Microphone Array", 2, 1, 0),
        ]
        self.negotiated_latency = {0: .022, 1: .008, 2: .025, 3: .010, 4: .002, 5: .001}
        self.failing = set()
        self.created = []
        self.settings = []

    def _device(self, name, hostapi, ins, outs):
        return {
            "name": name,
            "hostapi": hostapi,
            "max_input_channels": ins,
            "max_output_channels": outs,
            "default_samplerate": 44100.0,
        }

    def query_hostapis(self):
        return self.hosts

    def query_devices(self, index=None, kind=None):
        if kind is not None:
            index = self.default.device[0 if kind == "input" else 1]
        if index is None:
            return self.devices
        return dict(self.devices[index], index=index)

    def WasapiSettings(self, **kwargs):
        self.settings.append(kwargs)
        return kwargs

    def InputStream(self, **kwargs):
        stream = FakeStream(self, "input", **kwargs)
        self.created.append(stream)
        return stream

    def OutputStream(self, **kwargs):
        stream = FakeStream(self, "output", **kwargs)
        self.created.append(stream)
        return stream


class AudioSelectionTests(unittest.TestCase):
    def setUp(self):
        self.sd = FakeSoundDevice()

    def open(self, direction, selected=None):
        return (open_input_stream if direction == "input" else open_output_stream)(
            preferred_device=selected, samplerate=44100, blocksize=256,
            channels=1 if direction == "input" else 2,
            dtype="float32", latency="low", callback=lambda *args: None,
            _sd=self.sd,
        )

    def test_input_automatically_prefers_wasapi_not_wdm_ks(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            stream, info = self.open("input")
        self.assertEqual(info.device_index, 1)
        self.assertTrue(info.wasapi_shared)
        self.assertFalse(info.kernel_streaming)
        self.assertAlmostEqual(info.latency_ms, 8)
        self.assertEqual(stdout.getvalue(), "")  # JSONL stdout must stay clean.
        stream.close()

    def test_output_selects_lowest_healthy_wdm_ks(self):
        with contextlib.redirect_stderr(io.StringIO()):
            stream, info = self.open("output")
        self.assertEqual(info.device_index, 4)
        self.assertTrue(info.kernel_streaming)
        self.assertAlmostEqual(info.latency_ms, 2)
        stream.close()

    def test_output_falls_back_from_failed_ks_to_wasapi(self):
        self.sd.failing.add(4)
        with contextlib.redirect_stderr(io.StringIO()):
            stream, info = self.open("output")
        self.assertEqual(info.device_index, 3)
        self.assertTrue(info.wasapi_shared)
        stream.close()

    def test_explicit_mme_device_does_not_override_user_choice(self):
        with contextlib.redirect_stderr(io.StringIO()):
            stream, info = self.open("output", selected=2)
        self.assertEqual(info.device_index, 2)
        self.assertFalse(info.wasapi_shared)
        stream.close()

    def test_explicit_wasapi_uses_shared_auto_convert(self):
        with contextlib.redirect_stderr(io.StringIO()):
            stream, info = self.open("output", selected=3)
        self.assertEqual(info.device_index, 3)
        self.assertTrue(info.wasapi_shared)
        self.assertEqual(self.sd.settings[-1], {"exclusive": False, "auto_convert": True})
        stream.close()


if __name__ == "__main__":
    unittest.main()
