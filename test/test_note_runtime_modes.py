"""The new mode router must delegate and NEVER mutate the old Live engines."""
from contextlib import redirect_stdout
from io import StringIO
import json
import unittest
from unittest.mock import patch

from scripts.note_runtime import main


class RuntimeModesTests(unittest.TestCase):
    def test_native_notes_delegates_to_NOTE_heads_with_explicit_upstream_K(self):
        args=["--wav","source.wav","--decisions","original_S80.jsonl","--out","replay"]
        with patch("scripts.live_native_notes.main") as native:
            main(["--mode","native-notes",*args])
            native.assert_called_once_with(args)

    def test_native_status_is_not_consumed_as_status_profile_abbreviation(self):
        with patch("scripts.live_native_notes.main") as native:
            main(["--mode","native-notes","--status"])
            native.assert_called_once_with(["--status"])

    def test_legacy_live_model_and_thresholds_forwarded_byte_for_byte(self):
        forwarded=["--model","original.keras","--stdin-pcm16",
                   "--chunk-size","512","--onset-threshold","0.55"]
        with patch("scripts.detect_live.main") as original:
            main(["--mode","live-boundaries",*forwarded])
            original.assert_called_once_with(forwarded)

    def test_v8_live_interface_is_not_replaced_by_midi_decoder(self):
        forwarded=["--model","old_v8.keras","--stdin-pcm16",
                   "--receptive-field","4093"]
        with patch("scripts.detect_v8_live.main") as original:
            main(["--mode","live-v8",*forwarded])
            original.assert_called_once_with(forwarded)

    def test_live_midi_forwards_source_profile_and_output_unmodified(self):
        forwarded=["--stdin-jsonl","--profile","/tmp/profile.json",
                   "--midi-file","/tmp/music.mid"]
        with patch("scripts.stream_k_midi_live.main") as bridge:
            main(["--mode","live-midi",*forwarded])
            bridge.assert_called_once_with(forwarded)

    def test_status_does_not_misrepresent_offline_as_live(self):
        with redirect_stdout(StringIO()) as out:
            main(["--mode","status"])
        obj=json.loads(out.getvalue())
        self.assertTrue(obj["legacy_live_untouched"])
        self.assertFalse(obj["live_accuracy_certified"])
        self.assertFalse(obj["S80_causal_live_frontend_currently_integrated"])
        self.assertEqual(obj["S80_ExactK_reference"],"S80_S397_K5_GUARDED")
        self.assertEqual(obj["S422_noncausal_lookahead_ms"],5000)

    def test_status_does_not_accept_model_inference_flags(self):
        with self.assertRaises(SystemExit):
            main(["--mode","status","--model","wrong.keras"])


if __name__=="__main__":
    unittest.main()
