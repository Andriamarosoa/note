import tempfile
from pathlib import Path
import unittest

from causal_note.detector import BoundaryScoreChunk
from scripts.trace_hysteresis_closures import (
    CONTROL,
    TREATMENT,
    OffsetOnlyClosureTracer,
    create_argument_parser,
    refuse_output_overwrite,
    trace_offset_only_stream,
)


class _ScriptedPredictor:
    slot_count = 1

    def __init__(self, onset, offset):
        self.onset = tuple(onset)
        self.offset = tuple(offset)
        self.calls = []

    def predict_chunk(self, samples, *, start_sample):
        self.calls.append((start_sample, len(samples)))
        end_sample = start_sample + len(samples)
        return BoundaryScoreChunk(
            start_sample,
            tuple((self.onset[index],) for index in range(start_sample, end_sample)),
            tuple((self.offset[index],) for index in range(start_sample, end_sample)),
        )


def _run(onset, offset, chunk_lengths):
    predictor = _ScriptedPredictor(onset, offset)
    chunks = []
    start = 0
    for length in chunk_lengths:
        chunks.append((start, (0.0,) * length))
        start += length
    result = trace_offset_only_stream(
        predictor,
        tuple(chunks),
        track="synthetic.jams",
    )
    return predictor, result


class OffsetOnlyClosureTraceTests(unittest.TestCase):
    def test_suppressed_closure_is_recovered_later_with_exact_latency(self):
        predictor, result = _run(
            onset=(0.0, 0.60, 0.0, 0.0, 0.0),
            # High while empty, control-only re-arm, control close, treatment
            # re-arm, then delayed treatment close.
            offset=(0.60, 0.52, 0.60, 0.49, 0.60),
            chunk_lengths=(2, 1, 2),
        )

        self.assertEqual(predictor.calls, [(0, 2), (2, 1), (3, 2)])
        self.assertEqual(
            tuple((event.kind.value, event.sample) for event in result.events[CONTROL]),
            (("onset", 1), ("offset", 2)),
        )
        self.assertEqual(
            tuple((event.kind.value, event.sample) for event in result.events[TREATMENT]),
            (("onset", 1), ("offset", 4)),
        )
        answer = result.trace.summary
        self.assertEqual(
            answer["same_event_offset_closure_opportunities_suppressed"], 1
        )
        self.assertEqual(answer["suppressed_but_event_closed_later"], 1)
        self.assertEqual(
            answer["suppressed_and_event_remained_open_at_track_end"], 0
        )
        self.assertEqual(answer["recovery_latency"]["p50_samples"], 2.0)
        self.assertEqual(answer["recovery_latency"]["p90_samples"], 2.0)
        self.assertEqual(answer["recovery_latency"]["max_samples"], 2)

    def test_permanent_event_never_rearms(self):
        _, result = _run(
            onset=(0.0, 0.60, 0.0),
            offset=(0.60, 0.52, 0.60),
            chunk_lengths=(1, 2),
        )

        answer = result.trace.summary
        self.assertEqual(
            answer["suppressed_and_event_remained_open_at_track_end"], 1
        )
        self.assertEqual(
            answer["permanent_never_rearmed_after_suppression"], 1
        )
        self.assertEqual(answer["permanent_rearmed_but_no_later_offset"], 0)
        self.assertIsNone(
            result.trace.permanent_examples[0]["treatment_rearmed_sample"]
        )

    def test_permanent_event_rearms_but_has_no_later_offset(self):
        _, result = _run(
            onset=(0.0, 0.60, 0.0, 0.0),
            offset=(0.60, 0.52, 0.60, 0.49),
            chunk_lengths=(3, 1),
        )

        answer = result.trace.summary
        self.assertEqual(
            answer["suppressed_and_event_remained_open_at_track_end"], 1
        )
        self.assertEqual(
            answer["permanent_never_rearmed_after_suppression"], 0
        )
        self.assertEqual(answer["permanent_rearmed_but_no_later_offset"], 1)
        self.assertEqual(
            result.trace.permanent_examples[0]["treatment_rearmed_sample"],
            3,
        )

    def test_chunk_boundaries_do_not_reset_latches_or_identity(self):
        _, single = _run(
            onset=(0.0, 0.60, 0.0, 0.0, 0.0),
            offset=(0.60, 0.52, 0.60, 0.49, 0.60),
            chunk_lengths=(5,),
        )
        _, fragmented = _run(
            onset=(0.0, 0.60, 0.0, 0.0, 0.0),
            offset=(0.60, 0.52, 0.60, 0.49, 0.60),
            chunk_lengths=(1, 1, 1, 1, 1),
        )

        self.assertEqual(single.events, fragmented.events)
        self.assertEqual(single.trace.summary, fragmented.trace.summary)
        self.assertEqual(
            single.trace.permanent_examples,
            fragmented.trace.permanent_examples,
        )

    def test_offset_is_processed_before_onset_at_the_same_sample(self):
        _, result = _run(
            # Offset is already high at sample 0.  Because offsets run first,
            # it cannot close the onset allocated later at sample 0.
            onset=(0.60, 0.0),
            offset=(0.60, 0.60),
            chunk_lengths=(1, 1),
        )

        for candidate in (CONTROL, TREATMENT):
            self.assertEqual(
                tuple(
                    (event.kind.value, event.sample)
                    for event in result.events[candidate]
                ),
                (("onset", 0),),
            )
            self.assertEqual(
                result.trace.summary["candidate_counts"][candidate][
                    "predicted_incomplete_events"
                ],
                1,
            )

    def test_instrumented_public_sequences_equal_official_decoder(self):
        _, result = _run(
            onset=(0.60, 0.52, 0.60, 0.0, 0.70, 0.0),
            offset=(0.0, 0.60, 0.52, 0.60, 0.0, 0.90),
            chunk_lengths=(2, 2, 2),
        )

        self.assertTrue(result.official_public_sequences_equal)
        for candidate in (CONTROL, TREATMENT):
            event_ids = [event.event_id for event in result.events[candidate]]
            self.assertTrue(all(value.startswith("event-") for value in event_ids))

    def test_identity_divergence_is_not_counted_as_same_event_suppression(self):
        _, result = _run(
            # The treatment keeps the first event open at sample 2.  The
            # control opens a new onset at 3 and closes it at 5, when the
            # treatment still holds the old event: this second closure is a
            # cascade, not another same-event suppression.
            onset=(0.0, 0.60, 0.0, 0.60, 0.0, 0.0),
            offset=(0.60, 0.52, 0.60, 0.52, 0.49, 0.60),
            chunk_lengths=(3, 3),
        )

        answer = result.trace.summary
        self.assertEqual(
            answer["same_event_offset_closure_opportunities_suppressed"], 1
        )
        self.assertEqual(
            answer["control_closure_identity_divergence_or_cascade_not_attributed"],
            1,
        )


class OffsetOnlyClosureTraceCliTests(unittest.TestCase):
    def test_cli_requires_source_and_output_and_exposes_no_player_or_seed_switch(self):
        parser = create_argument_parser()
        arguments = parser.parse_args(
            ["--source-report", "source.json", "--output", "trace.json"]
        )
        self.assertEqual(arguments.chunk_size, 512)
        self.assertFalse(hasattr(arguments, "players"))
        self.assertFalse(hasattr(arguments, "seed"))
        with self.assertRaises(SystemExit):
            parser.parse_args(["--output", "trace.json"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--source-report", "source.json"])

    def test_existing_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.json"
            output.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                refuse_output_overwrite(output)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep")

    def test_tracer_rejects_noncontiguous_chunks(self):
        tracer = OffsetOnlyClosureTracer("synthetic.jams", 1)
        tracer.process_chunk(BoundaryScoreChunk(0, ((0.0,),), ((0.0,),)))
        with self.assertRaisesRegex(ValueError, "expected contiguous"):
            tracer.process_chunk(BoundaryScoreChunk(2, ((0.0,),), ((0.0,),)))


if __name__ == "__main__":
    unittest.main()
