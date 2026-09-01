import unittest

from causal_note.detector import BoundaryScoreChunk, LiveModelDetector
from causal_note.pipeline import LiveOnsetOffsetPipeline


class _BoundaryPredictor:
    slot_count = 2

    def predict_chunk(self, samples, *, start_sample):
        onset = [[0.0, 0.0] for _ in samples]
        offset = [[0.0, 0.0] for _ in samples]
        end = start_sample + len(samples)
        for sample, slot in ((100, 0), (200, 1)):
            if start_sample <= sample < end:
                onset[sample - start_sample][slot] = 1.0
        for sample, slot in ((700, 0), (820, 1)):
            if start_sample <= sample < end:
                offset[sample - start_sample][slot] = 1.0
        return BoundaryScoreChunk(
            start_sample,
            tuple(onset),
            tuple(offset),
        )


class LiveOnsetOffsetPipelineTests(unittest.TestCase):
    def test_finalize_stream_closes_open_ids_without_restart_frames(self):
        pipeline = LiveOnsetOffsetPipeline(
            LiveModelDetector(_BoundaryPredictor())
        )
        first = pipeline.process_chunk((0.0,) * 512, start_sample=0)

        terminal = pipeline.finalize_stream()

        self.assertEqual(
            tuple((event.event_id, event.sample) for event in terminal.events),
            tuple((event.event_id, 512) for event in first.events),
        )
        self.assertEqual(terminal.requested_frames, ())
        self.assertEqual(pipeline.scheduler.restart_frames(), ())
        self.assertEqual(pipeline.scheduler.open_event_ids(), ())
        with self.assertRaises(RuntimeError):
            pipeline.process_chunk((0.0,))
        with self.assertRaises(RuntimeError):
            pipeline.finalize_stream()

    def test_empty_stream_finalizes_without_events(self):
        pipeline = LiveOnsetOffsetPipeline(
            LiveModelDetector(_BoundaryPredictor())
        )

        result = pipeline.finalize_stream()

        self.assertEqual(result.events, ())
        self.assertEqual(result.requested_frames, ())

    def test_failed_finalization_does_not_finalize_the_detector(self):
        pipeline = LiveOnsetOffsetPipeline(
            LiveModelDetector(_BoundaryPredictor())
        )
        pipeline.process_chunk((0.0,) * 512, start_sample=0)
        pipeline.scheduler.open_event("external-desynchronization", 300)

        with self.assertRaises(ValueError):
            pipeline.finalize_stream()

        follow_up = pipeline.process_chunk((0.0,) * 189)
        self.assertEqual(follow_up.events[0].sample, 700)

    def test_detects_renews_and_compacts_a_700_b_820(self):
        pipeline = LiveOnsetOffsetPipeline(
            LiveModelDetector(_BoundaryPredictor())
        )

        first = pipeline.process_chunk((0.0,) * 512, start_sample=0)
        second = pipeline.process_chunk((0.0,) * 512)

        self.assertEqual(
            tuple(event.sample for event in first.events + second.events),
            (100, 200, 700, 820),
        )
        self.assertEqual(
            tuple(frame.frame_start for frame in second.requested_frames),
            (700, 820),
        )
        self.assertEqual(second.ready_frames, ())

        third = pipeline.process_chunk((0.0,) * 188)
        self.assertEqual(
            tuple(frame.frame_start for frame in third.ready_frames),
            (700,),
        )
        pipeline.complete_frame(700)
        self.assertEqual(pipeline.scheduler.audio_start_sample, 820)

        fourth = pipeline.process_chunk((0.0,) * 120)
        self.assertEqual(
            tuple(frame.frame_start for frame in fourth.ready_frames),
            (820,),
        )
        pipeline.complete_frame(820)
        self.assertEqual(
            pipeline.scheduler.audio_start_sample,
            pipeline.scheduler.audio_end_sample,
        )
        self.assertEqual(pipeline.scheduler.restart_frames(), ())


if __name__ == "__main__":
    unittest.main()
