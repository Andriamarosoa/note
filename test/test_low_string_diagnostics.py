"""Synthetic-signal checks for the iRig diagnostic, independent of hardware."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import unittest

import numpy as np

from scripts.diagnose_low_strings import SR, db, diagnostics


class LowStringCaptureTests(unittest.TestCase):
    def make_tone(self, note_midi=40, amplitude=0.25):
        t = np.arange(SR, dtype=np.float64) / SR
        f0 = 440.0 * 2 ** ((note_midi - 69) / 12)
        return (amplitude * np.sin(2 * np.pi * f0 * t)).astype(np.float32)

    def test_e2_fundamental_visible_in_raw_spectrum(self):
        s = io.StringIO()
        with redirect_stdout(s):
            diagnostics(self.make_tone(), "E2")
        self.assertIn("Requested=E2 MIDI=40", s.getvalue())
        self.assertIn("H1:", s.getvalue())
        self.assertIn("82.4Hz", s.getvalue())

    def test_synthetic_silence_reported_as_low_level(self):
        s = io.StringIO()
        with redirect_stdout(s):
            diagnostics(np.zeros(SR, dtype=np.float32), "E2")
        self.assertIn("very low capture level", s.getvalue())

    def test_e2_not_confused_with_pitch_outside_note_range(self):
        self.assertAlmostEqual(db(0.5), -6.0205999, places=3)
        s = io.StringIO()
        with redirect_stdout(s):
            diagnostics(self.make_tone(38), "D2")
        self.assertIn("Requested=D2 MIDI=38", s.getvalue())


if __name__ == "__main__":
    unittest.main()
