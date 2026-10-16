"""S636 prove causal collision decisions cannot consult native future."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest

from scripts.s636_three_view_collision import audit_three_view_report,main


def event(view,pitch,time):
    return dict(view=view,detected_pitch=pitch,time_s=time,
                attack_confidence=.8,sustain_confidence=.7,native_active_K=2)


class CollisionTests(unittest.TestCase):
    def setUp(self):
        self.report=dict(kind="S634_three_view_shadow_candidates_only",
                         model_audio_seconds=8,
                         candidates=[
                             event("original",49,1.),
                             event("original",67,2.),
                             event("original",49,3.053),
                             event("original",76,4.0116),
                             event("bass_x2",49,1.02),
                             event("high_half",79,2.005),
                             event("bass_x2",61,3.),
                             event("high_half",76,4.),
                             event("high_half",77,4.4),
                         ])

    def test_same_pitch_octave_unmatched_with_past_and_future_distinct(self):
        a=audit_three_view_report(self.report)
        self.assertEqual(a["original_candidate_count"],4)
        self.assertEqual(a["rescue_candidate_count"],5)
        self.assertEqual(a["retrospective_counts"],
                         {"same_pitch":2,"octave_relation":2,"unmatched":1})
        self.assertEqual(a["strictly_causal_counts"],
                         {"same_pitch_already_known":1,
                          "octave_already_known":1,"requires_more_evidence":3})
        rows=a["details"]
        self.assertEqual(rows[2]["retrospective"],"octave_relation")
        self.assertEqual(rows[2]["live_causal"],"requires_more_evidence")
        self.assertGreater(rows[2]["coincidences"][0]["native_minus_rescue_ms"],0)
        self.assertEqual(rows[3]["earlier_than_native_same_pitch_ms"],11.6)
        self.assertTrue(a["never_use_future_native_for_live_decisions"])

    def test_after_absolute_cutoff_no_ahead_of_time_matches(self):
        a=audit_three_view_report(self.report,tolerance_ms=10)
        self.assertEqual(a["strictly_causal_counts"],
                         {"octave_already_known":1,"requires_more_evidence":4})
        with self.assertRaises(ValueError):
            audit_three_view_report(self.report,tolerance_ms=-1)
        with self.assertRaises(ValueError):
            audit_three_view_report(self.report,tolerance_ms=300)

    def test_json_cli_creates_new_file_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"input.json"
            out=Path(d)/"collision.json"
            path.write_text(json.dumps(self.report))
            main(["--input",str(path),"--out",str(out)])
            r=json.loads(out.read_text())
            self.assertEqual(r["rescue_candidate_count"],5)
            with self.assertRaises(SystemExit):
                main(["--input",str(path),"--out",str(out)])

    def test_invalid_or_unbounded_rejected(self):
        with self.assertRaises(ValueError):
            audit_three_view_report({"kind":"wrong","candidates":[]})
        bad=dict(self.report,candidates=[event("fake",56,2)])
        with self.assertRaises(ValueError):
            audit_three_view_report(bad)
        bad=dict(self.report,candidates=[event("original",99,2)])
        with self.assertRaises(ValueError):
            audit_three_view_report(bad)


if __name__=="__main__":
    unittest.main()
