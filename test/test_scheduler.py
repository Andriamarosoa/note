import unittest

from causal_note.detector import (
    BoundaryEvent,
    BoundaryScoreChunk,
    BoundaryType,
    LiveBoundaryScoreDecoder,
)
from causal_note.scheduler import RestartScheduler, SchedulerError


class RestartSchedulerTests(unittest.TestCase):
    def test_model_events_drive_both_overlapping_restart_frames(self) -> None:
        decoder = LiveBoundaryScoreDecoder(slot_count=2)
        scheduler = RestartScheduler()

        onset_rows = [[0.0, 0.0] for _ in range(512)]
        offset_rows = [[0.0, 0.0] for _ in range(512)]
        onset_rows[100][0] = 1.0
        onset_rows[200][1] = 1.0
        scheduler.append_audio((0.0,) * 512)
        for event in decoder.process_chunk(
            BoundaryScoreChunk(0, tuple(onset_rows), tuple(offset_rows))
        ):
            scheduler.accept_event(event)

        onset_rows = [[0.0, 0.0] for _ in range(512)]
        offset_rows = [[0.0, 0.0] for _ in range(512)]
        offset_rows[700 - 512][0] = 1.0
        offset_rows[820 - 512][1] = 1.0
        scheduler.append_audio((0.0,) * 512)
        events = decoder.process_chunk(
            BoundaryScoreChunk(512, tuple(onset_rows), tuple(offset_rows))
        )
        scheduler.accept_event(events[0])
        self.assertEqual(scheduler.open_event_ids(), (events[1].event_id,))
        scheduler.accept_event(events[1])

        self.assertEqual(scheduler.open_event_ids(), ())
        self.assertEqual(
            tuple(frame.frame_start for frame in scheduler.restart_frames()),
            (700, 820),
        )

    def test_a_700_then_b_820_keeps_b_open_and_renews_both_frames(self) -> None:
        scheduler = RestartScheduler()
        scheduler.append_audio((0.0,) * 200)
        scheduler.accept_event(BoundaryEvent(BoundaryType.ONSET, "A", 100))
        scheduler.accept_event(BoundaryEvent(BoundaryType.ONSET, "B", 200))

        scheduler.append_audio((0.0,) * 500)
        frame_700 = scheduler.accept_event(
            BoundaryEvent(BoundaryType.OFFSET, "A", 700)
        )
        self.assertIsNotNone(frame_700)
        self.assertFalse(frame_700.ready)
        self.assertEqual(scheduler.open_event_ids(), ("B",))

        scheduler.append_audio((0.0,) * 120)
        frame_820 = scheduler.accept_event(
            BoundaryEvent(BoundaryType.OFFSET, "B", 820)
        )
        self.assertIsNotNone(frame_820)
        self.assertFalse(frame_820.ready)
        self.assertEqual(scheduler.open_event_ids(), ())
        self.assertEqual(
            tuple(frame.frame_start for frame in scheduler.restart_frames()),
            (700, 820),
        )

        scheduler.append_audio((0.0,) * 392)
        ready = scheduler.ready_frames()
        self.assertEqual(tuple(frame.frame_start for frame in ready), (700,))
        self.assertEqual(len(ready[0].samples), 512)
        scheduler.complete_frame(700)
        self.assertEqual(scheduler.prune_completed(), 820)
        self.assertEqual(
            tuple(frame.frame_start for frame in scheduler.restart_frames()),
            (820,),
        )

        scheduler.append_audio((0.0,) * 120)
        ready = scheduler.ready_frames()
        self.assertEqual(tuple(frame.frame_start for frame in ready), (820,))
        self.assertEqual(len(ready[0].samples), 512)
        scheduler.complete_frame(820)
        self.assertEqual(scheduler.prune_completed(), 1332)
        self.assertEqual(scheduler.audio_start_sample, scheduler.audio_end_sample)
        self.assertEqual(scheduler.restart_frames(), ())

    def test_offset_must_be_received_and_after_onset(self) -> None:
        scheduler = RestartScheduler()
        scheduler.open_event("A", onset_sample=100)
        scheduler.append_audio((0.0,) * 200)
        for offset in (100, 201):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                scheduler.close_event("A", offset)

    def test_incomplete_frame_cannot_be_completed(self) -> None:
        scheduler = RestartScheduler()
        scheduler.open_event("A", onset_sample=100)
        scheduler.append_audio((0.0,) * 700)
        scheduler.close_event("A", 700)
        with self.assertRaises(SchedulerError):
            scheduler.complete_frame(700)

    def test_terminal_events_close_complete_set_without_restart_frames(self) -> None:
        scheduler = RestartScheduler()
        scheduler.append_audio((0.0,) * 1000)
        scheduler.open_event("A", 100)
        scheduler.open_event("B", 200)
        scheduler.open_event("already-closed", 300)
        scheduler.close_event("already-closed", 500)
        existing_frames = scheduler.restart_frames()

        scheduler.accept_terminal_events(
            (
                BoundaryEvent(BoundaryType.OFFSET, "B", 1000),
                BoundaryEvent(BoundaryType.OFFSET, "A", 1000),
            )
        )

        self.assertEqual(scheduler.open_event_ids(), ())
        self.assertEqual(scheduler.restart_frames(), existing_frames)
        with self.assertRaises(RuntimeError):
            scheduler.accept_terminal_events(())
        with self.assertRaises(RuntimeError):
            scheduler.append_audio((0.0,))

    def test_invalid_terminal_sets_fail_atomically(self) -> None:
        invalid_sets = (
            (BoundaryEvent(BoundaryType.OFFSET, "A", 1000),),
            (
                BoundaryEvent(BoundaryType.OFFSET, "A", 1000),
                BoundaryEvent(BoundaryType.OFFSET, "A", 1000),
                BoundaryEvent(BoundaryType.OFFSET, "B", 1000),
            ),
            (
                BoundaryEvent(BoundaryType.OFFSET, "A", 1000),
                BoundaryEvent(BoundaryType.OFFSET, "unknown", 1000),
            ),
            (
                BoundaryEvent(BoundaryType.ONSET, "A", 1000),
                BoundaryEvent(BoundaryType.OFFSET, "B", 1000),
            ),
            (
                BoundaryEvent(BoundaryType.OFFSET, "A", 999),
                BoundaryEvent(BoundaryType.OFFSET, "B", 1000),
            ),
        )
        for terminal_events in invalid_sets:
            with self.subTest(terminal_events=terminal_events):
                scheduler = RestartScheduler()
                scheduler.append_audio((0.0,) * 1000)
                scheduler.open_event("A", 100)
                scheduler.open_event("B", 200)
                before_frames = scheduler.restart_frames()

                with self.assertRaises((ValueError, SchedulerError)):
                    scheduler.accept_terminal_events(terminal_events)

                self.assertEqual(scheduler.open_event_ids(), ("A", "B"))
                self.assertEqual(scheduler.restart_frames(), before_frames)

                scheduler.accept_terminal_events(
                    (
                        BoundaryEvent(BoundaryType.OFFSET, "A", 1000),
                        BoundaryEvent(BoundaryType.OFFSET, "B", 1000),
                    )
                )
                self.assertEqual(scheduler.open_event_ids(), ())


if __name__ == "__main__":
    unittest.main()
