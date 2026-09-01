import subprocess
import sys
import unittest

from causal_note.guitarset import NoteBoundary
from scripts.audit_end_of_stream_finalization import (
    aggregate_split,
    replay_oracle_track,
)


def _slots(*slot_notes):
    values = [tuple(notes) for notes in slot_notes]
    values.extend([tuple()] * (6 - len(values)))
    return tuple(values)


class EndOfStreamAuditTests(unittest.TestCase):
    def test_import_does_not_import_tensorflow(self):
        code = (
            "import scripts.audit_end_of_stream_finalization, sys; "
            "raise SystemExit(int('tensorflow' in sys.modules))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_oracle_keeps_internal_offset_and_finalizes_eof_multiplicity(self):
        slots = _slots(
            (NoteBoundary(100, 700), NoteBoundary(800, 1000)),
            (NoteBoundary(200, 1000),),
        )

        result = replay_oracle_track(
            slots,
            frame_count=1000,
            member="00_test_comp.jams",
            player="00",
        )

        self.assertEqual(result.notes, 3)
        self.assertEqual(result.internal_reference_offsets, 1)
        self.assertEqual(result.terminal_reference_offsets, 2)
        self.assertEqual(result.open_events_before_finalization, 2)
        self.assertEqual(result.terminal_control_offsets_emitted, 2)
        self.assertEqual(result.missing_terminal_offsets, 0)
        self.assertEqual(result.extra_terminal_offsets, 0)
        self.assertEqual(result.open_events_after_finalization, 0)
        self.assertEqual(result.last_real_sample_active_terminal_notes, 2)

    def test_aggregate_reports_affected_tracks_and_multiplicity(self):
        affected = replay_oracle_track(
            _slots(
                (NoteBoundary(10, 100),),
                (NoteBoundary(20, 100),),
            ),
            frame_count=100,
            member="00_a_comp.jams",
            player="00",
        )
        unaffected = replay_oracle_track(
            _slots((NoteBoundary(10, 50),)),
            frame_count=100,
            member="01_b_solo.jams",
            player="01",
        )

        report = aggregate_split((affected, unaffected))

        self.assertEqual(report["tracks_with_terminal_offsets"], 1)
        self.assertEqual(report["tracks_without_terminal_offsets"], 1)
        self.assertEqual(report["terminal_multiplicity_per_affected_track"], {"2": 1})
        self.assertEqual(report["before"]["open_events_at_eof"], 2)
        self.assertEqual(report["after"]["terminal_control_offsets_emitted"], 2)
        self.assertTrue(report["all_tracks_exact"])


if __name__ == "__main__":
    unittest.main()
