"""Regression checks for S667 mic/WAV release controls; default engine stays native."""
import unittest
from dataclasses import asdict
from causal_note.strict_live import Selection
from scripts.live_strict_notes import s667_live_selection, main


class S667LiveControls(unittest.TestCase):
    def setUp(self):
        self.original = Selection(onset=.60, active_on=.5, active_off=.25,
                                  onset_reset=.2, release_frames=5)
        self.metadata = {"selection": asdict(self.original)}

    def test_default_does_not_override_native_selection(self):
        self.assertIsNone(s667_live_selection(self.metadata))
        self.assertEqual(self.original.release_frames,5)
        self.assertEqual(self.original.active_off,.25)

    def test_research_flags_only_modify_release_decision(self):
        out = s667_live_selection(self.metadata,5,.15)
        self.assertEqual(out.release_frames,5)
        self.assertAlmostEqual(out.active_off,.15)
        self.assertEqual(out.onset,self.original.onset)
        self.assertEqual(out.active_on,self.original.active_on)
        self.assertEqual(out.refractory_frames,self.original.refractory_frames)
        self.assertEqual(out.max_notes,self.original.max_notes)
        self.assertEqual(self.original.active_off,.25)

    def test_invalid_configuration_rejected_early(self):
        with self.assertRaises(ValueError):
            s667_live_selection(self.metadata,2,1.5)

    def test_cli_help_advertises_opt_in_s667(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["--help"])
        self.assertEqual(ctx.exception.code,0)

    def test_status_does_not_require_microphone(self):
        main(["--status"])

if __name__=="__main__":
    unittest.main()
