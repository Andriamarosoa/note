import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from causal_note.detector import BoundaryScoreChunk, BoundaryType
from scripts.audit_boundary_candidate_relations import (
    CandidateRelationObserver,
    DEFAULT_SOURCE_REPORT,
    IdentifiedReference,
    TracedCandidate,
    analyze_head,
    decode_relation_stream,
    gap_bin,
    load_source_expectations,
    main,
    note_support_counts,
)


def chunk(start, onset, offset=None):
    onset_rows = tuple((float(value),) for value in onset)
    if offset is None:
        offset_rows = tuple((0.0,) for _ in onset_rows)
    else:
        offset_rows = tuple((float(value),) for value in offset)
    return BoundaryScoreChunk(start, onset_rows, offset_rows)


def traced(kind, sample, channel=0, episode_id=0):
    return TracedCandidate(
        kind=kind,
        sample=sample,
        channel=channel,
        entry_score=0.6,
        preceding_low_run_samples=None,
        previous_same_channel_gap_samples=None,
        episode_id=episode_id,
        survives_n16=True,
    )


class ScriptedPredictor:
    slot_count = 1

    def __init__(self, chunks):
        self._chunks = iter(chunks)

    def predict_chunk(self, samples, *, start_sample):
        value = next(self._chunks)
        if value.start_sample != start_sample or value.sample_count != len(samples):
            raise AssertionError("scripted score chunk differs")
        return value


class CandidateRelationObserverTests(unittest.TestCase):
    def test_shared_stream_reproduces_both_public_decoders(self):
        values = [0.6, 0.4, 0.6] + [0.4] * 16 + [0.6]
        chunks = (chunk(0, values[:10]), chunk(10, values[10:]))
        predictor = ScriptedPredictor(chunks)
        audio_chunks = (
            (0, tuple(0.0 for _ in range(10))),
            (10, tuple(0.0 for _ in range(len(values) - 10))),
        )
        decoded = decode_relation_stream(predictor, audio_chunks)
        self.assertEqual(
            [item.sample for item in decoded.control_candidates], [0, 2, 19]
        )
        self.assertEqual(
            [item.sample for item in decoded.diagnostic_candidates], [0, 19]
        )
        self.assertEqual(decoded.chunks, 2)

    def test_short_chatter_stays_in_one_n16_episode_across_chunks(self):
        values = [0.6, 0.4, 0.6] + [0.4] * 16 + [0.6]
        observer = CandidateRelationObserver(1, threshold=0.55, n16_low_samples=16)
        observer.process_chunk(chunk(0, values[:8]))
        observer.process_chunk(chunk(8, values[8:]))

        self.assertEqual(
            [item.sample for item in observer.n1_candidates], [0, 2, 19]
        )
        self.assertEqual(
            [item.sample for item in observer.n16_candidates], [0, 19]
        )
        first, second = observer.n16_candidates
        self.assertEqual(observer.episode_front_counts[first.episode_id], 2)
        self.assertEqual(observer.episode_front_counts[second.episode_id], 1)
        self.assertEqual(second.preceding_low_run_samples, 16)
        self.assertEqual(second.previous_same_channel_gap_samples, 19)

    def test_heads_and_channels_are_independent(self):
        observer = CandidateRelationObserver(2)
        scores = BoundaryScoreChunk(
            100,
            onset=((0.6, 0.7),),
            offset=((0.0, 0.8),),
        )
        observer.process_chunk(scores)
        self.assertEqual(
            [(item.kind, item.channel, item.sample) for item in observer.n16_candidates],
            [
                (BoundaryType.OFFSET, 1, 100),
                (BoundaryType.ONSET, 0, 100),
                (BoundaryType.ONSET, 1, 100),
            ],
        )

    def test_non_contiguous_chunks_are_rejected(self):
        observer = CandidateRelationObserver(1)
        observer.process_chunk(chunk(0, [0.0]))
        with self.assertRaisesRegex(ValueError, "contiguous"):
            observer.process_chunk(chunk(2, [0.0]))


class CandidateRelationAnalysisTests(unittest.TestCase):
    def test_main_prints_revised_relation_classes(self):
        head = {
            "official_metrics": {"prediction_count": 3, "false_positive": 2},
            "false_positive_relations": {
                "isolated": {"count": 1},
                "same_channel_same_reference_successor_excess": {"count": 1},
                "single_reference_near_excess": {"count": 0},
                "ambiguous_near": {"count": 0},
            },
        }
        result = {"aggregates": {"global": {"heads": {"onset": head, "offset": head}}}}
        output = StringIO()
        with patch(
            "scripts.audit_boundary_candidate_relations.run_relation_audit",
            return_value=result,
        ), redirect_stdout(output):
            self.assertEqual(main([]), 0)
        self.assertIn("same_channel_same_reference_successor_excess=1", output.getvalue())

    def test_false_positive_partition_single_reference_and_isolated(self):
        references = (
            IdentifiedReference(BoundaryType.ONSET, 100, 0, 0),
        )
        candidates = (
            traced(BoundaryType.ONSET, 100, episode_id=0),
            traced(BoundaryType.ONSET, 110, channel=1, episode_id=1),
            traced(BoundaryType.ONSET, 1000, episode_id=2),
        )
        raw, _ = analyze_head(
            references,
            candidates,
            candidates,
            {0: 1, 1: 1, 2: 1},
            tolerance_samples=50,
            proximity_tolerances={"1": 1, "50": 50},
        )
        self.assertEqual(raw.true_positive, 1)
        self.assertEqual(raw.false_positive, 2)
        self.assertEqual(raw.fp_partition["single_reference_near_excess"], 1)
        self.assertEqual(raw.fp_partition["isolated"], 1)
        self.assertEqual(raw.fp_partition["ambiguous_near"], 0)

    def test_strong_repeat_requires_same_channel_successor_and_reference(self):
        references = (
            IdentifiedReference(BoundaryType.ONSET, 100, 0, 0),
        )
        candidates = (
            traced(BoundaryType.ONSET, 100, channel=0, episode_id=0),
            traced(BoundaryType.ONSET, 110, channel=0, episode_id=1),
        )
        raw, _ = analyze_head(
            references,
            candidates,
            candidates,
            {0: 1, 1: 1},
            tolerance_samples=50,
            proximity_tolerances={"50": 50},
        )
        self.assertEqual(
            raw.fp_partition["same_channel_same_reference_successor_excess"],
            1,
        )
        self.assertEqual(raw.fp_partition["single_reference_near_excess"], 0)

    def test_false_positive_partition_keeps_ambiguous_occurrence(self):
        references = (
            IdentifiedReference(BoundaryType.ONSET, 100, 0, 0),
            IdentifiedReference(BoundaryType.ONSET, 140, 1, 0),
        )
        candidates = (
            traced(BoundaryType.ONSET, 100, 0, 0),
            traced(BoundaryType.ONSET, 120, 0, 1),
            traced(BoundaryType.ONSET, 140, 1, 2),
            traced(BoundaryType.ONSET, 1000, 0, 3),
        )
        raw, _ = analyze_head(
            references,
            candidates,
            candidates,
            {0: 1, 1: 1, 2: 1, 3: 1},
            tolerance_samples=50,
            proximity_tolerances={"50": 50},
        )
        self.assertEqual(raw.true_positive, 2)
        self.assertEqual(raw.fp_partition["ambiguous_near"], 1)
        self.assertEqual(raw.fp_partition["isolated"], 1)

    def test_note_support_requires_same_private_channel(self):
        references = {
            "onset": (
                IdentifiedReference(BoundaryType.ONSET, 100, 0, 0),
                IdentifiedReference(BoundaryType.ONSET, 200, 1, 0),
            ),
            "offset": (
                IdentifiedReference(BoundaryType.OFFSET, 300, 0, 0),
                IdentifiedReference(BoundaryType.OFFSET, 400, 1, 0),
            ),
        }
        candidates = {
            "onset": (
                traced(BoundaryType.ONSET, 100, 0, 0),
                traced(BoundaryType.ONSET, 200, 0, 1),
            ),
            "offset": (traced(BoundaryType.OFFSET, 300, 0, 2),),
        }
        result = note_support_counts(references, candidates, tolerance_samples=10)
        self.assertEqual(result["notes"], 2)
        self.assertEqual(result["both"], 1)
        self.assertEqual(result["neither"], 1)

    def test_note_support_does_not_reuse_one_candidate_for_two_notes(self):
        references = {
            "onset": (
                IdentifiedReference(BoundaryType.ONSET, 100, 0, 0),
                IdentifiedReference(BoundaryType.ONSET, 170, 0, 1),
            ),
            "offset": (
                IdentifiedReference(BoundaryType.OFFSET, 300, 0, 0),
                IdentifiedReference(BoundaryType.OFFSET, 370, 0, 1),
            ),
        }
        candidates = {
            "onset": (traced(BoundaryType.ONSET, 135, 0, 0),),
            "offset": (traced(BoundaryType.OFFSET, 335, 0, 1),),
        }
        result = note_support_counts(references, candidates, tolerance_samples=50)
        self.assertEqual(result["notes"], 2)
        self.assertEqual(result["both"], 1)
        self.assertEqual(result["neither"], 1)
        self.assertEqual(result["onset_matched_supports"], 1)
        self.assertEqual(result["offset_matched_supports"], 1)

    def test_gap_bins_are_fixed(self):
        expected = {
            0: "0",
            1: "1-15",
            15: "1-15",
            16: "16-63",
            63: "16-63",
            64: "64-255",
            255: "64-255",
            256: "256-511",
            511: "256-511",
            512: "512-2205",
            2205: "512-2205",
            2206: "2206_plus",
        }
        self.assertEqual({value: gap_bin(value) for value in expected}, expected)

    def test_completed_exp07_source_is_locked(self):
        source = load_source_expectations(DEFAULT_SOURCE_REPORT)
        self.assertEqual(len(source.tracks), 60)
        self.assertEqual(
            source.global_metrics["treatment"]["onset"]["prediction_count"],
            158032,
        )
        self.assertEqual(
            source.global_metrics["treatment"]["offset"]["false_positive"],
            250647,
        )


if __name__ == "__main__":
    unittest.main()
