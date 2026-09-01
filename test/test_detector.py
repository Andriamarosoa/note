import unittest

from causal_note.detector import (
    BoundaryCandidate,
    BoundaryScoreChunk,
    BoundaryEvent,
    BoundaryType,
    LiveBoundaryPeakDecoder,
    LiveBoundaryScoreDecoder,
    LiveEnergyDetector,
    LiveEventTracker,
    LiveModelDetector,
)


class LiveEventTrackerTests(unittest.TestCase):
    def test_a_and_b_are_closed_by_their_own_opaque_ids(self) -> None:
        tracker = LiveEventTracker()
        onset_a = tracker.start_event(100, representation=(1.0, 0.0))
        onset_b = tracker.start_event(200, representation=(0.0, 1.0))

        offset_a = tracker.finish_event(onset_a.event_id, 700)
        self.assertEqual(
            tuple(event.event_id for event in tracker.active_events()),
            (onset_b.event_id,),
        )
        offset_b = tracker.finish_event(onset_b.event_id, 820)

        self.assertEqual(offset_a.kind, BoundaryType.OFFSET)
        self.assertEqual(offset_a.event_id, onset_a.event_id)
        self.assertEqual(offset_a.sample, 700)
        self.assertEqual(offset_b.event_id, onset_b.event_id)
        self.assertEqual(offset_b.sample, 820)
        self.assertEqual(tracker.active_events(), ())

    def test_unknown_or_early_offset_is_rejected(self) -> None:
        tracker = LiveEventTracker()
        onset = tracker.start_event(100)
        with self.assertRaises(ValueError):
            tracker.finish_event(onset.event_id, 100)
        with self.assertRaises(KeyError):
            tracker.finish_event("missing", 200)

    def test_finish_all_is_atomic_ordered_and_one_shot(self) -> None:
        tracker = LiveEventTracker()
        later = tracker.start_event(200)
        earlier = tracker.start_event(100)

        with self.assertRaises(ValueError):
            tracker.finish_all(100)
        self.assertEqual(
            tuple(event.event_id for event in tracker.active_events()),
            (earlier.event_id, later.event_id),
        )

        terminal = tracker.finish_all(300)
        self.assertEqual(
            terminal,
            (
                BoundaryEvent(BoundaryType.OFFSET, earlier.event_id, 300),
                BoundaryEvent(BoundaryType.OFFSET, later.event_id, 300),
            ),
        )
        self.assertEqual(tracker.active_events(), ())
        with self.assertRaises(RuntimeError):
            tracker.finish_all(300)
        with self.assertRaises(RuntimeError):
            tracker.start_event(301)


class LiveEnergyDetectorTests(unittest.TestCase):
    def test_emits_onset_then_associated_offset_across_chunks(self) -> None:
        detector = LiveEnergyDetector(release_samples=3)

        first = detector.process_chunk(
            (0.0,) * 4 + (0.5,) * 4,
            start_sample=700,
        )
        second = detector.process_chunk((0.5,) * 2 + (0.0,) * 3)

        self.assertEqual(
            first,
            (BoundaryEvent(BoundaryType.ONSET, "event-000001", 704),),
        )
        self.assertEqual(
            second,
            (BoundaryEvent(BoundaryType.OFFSET, "event-000001", 710),),
        )
        self.assertEqual(detector.active_events(), ())

    def test_short_silence_does_not_create_a_false_offset(self) -> None:
        detector = LiveEnergyDetector(release_samples=3)
        first = detector.process_chunk((0.5, 0.0), start_sample=0)
        second = detector.process_chunk((0.0, 0.5))

        self.assertEqual(first[0].kind, BoundaryType.ONSET)
        self.assertEqual(second, ())
        self.assertEqual(len(detector.active_events()), 1)

    def test_chunks_must_be_contiguous(self) -> None:
        detector = LiveEnergyDetector()
        detector.process_chunk((0.0,) * 10, start_sample=100)
        with self.assertRaises(ValueError):
            detector.process_chunk((0.0,), start_sample=111)

    def test_finalize_stream_closes_open_event_at_exclusive_end(self) -> None:
        detector = LiveEnergyDetector()
        onset = detector.process_chunk((0.0, 0.5, 0.5), start_sample=0)[0]

        with self.assertRaises(ValueError):
            detector.finalize_stream(2)
        self.assertEqual(len(detector.active_events()), 1)

        self.assertEqual(
            detector.finalize_stream(),
            (BoundaryEvent(BoundaryType.OFFSET, onset.event_id, 3),),
        )
        with self.assertRaises(RuntimeError):
            detector.process_chunk((0.0,))
        with self.assertRaises(RuntimeError):
            detector.finalize_stream()


def score_chunk(start, length, *, onsets=(), offsets=(), slots=6):
    onset_rows = [[0.0] * slots for _ in range(length)]
    offset_rows = [[0.0] * slots for _ in range(length)]
    for sample, slot in onsets:
        onset_rows[sample - start][slot] = 1.0
    for sample, slot in offsets:
        offset_rows[sample - start][slot] = 1.0
    return BoundaryScoreChunk(start, tuple(onset_rows), tuple(offset_rows))


class LiveBoundaryScoreDecoderTests(unittest.TestCase):
    def test_finalize_stream_closes_all_slots_without_padding(self) -> None:
        decoder = LiveBoundaryScoreDecoder(slot_count=2)
        onsets = decoder.process_chunk(
            score_chunk(0, 4, slots=2, onsets=((0, 1), (2, 0)))
        )

        terminal = decoder.finalize_stream()

        self.assertEqual(
            terminal,
            (
                BoundaryEvent(BoundaryType.OFFSET, onsets[0].event_id, 4),
                BoundaryEvent(BoundaryType.OFFSET, onsets[1].event_id, 4),
            ),
        )
        self.assertEqual(decoder.active_events(), ())
        with self.assertRaises(RuntimeError):
            decoder.process_chunk(score_chunk(4, 1, slots=2))
        with self.assertRaises(RuntimeError):
            decoder.finalize_stream()

    def test_zero_threshold_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            LiveBoundaryScoreDecoder(onset_threshold=0.0)

    def test_release_thresholds_default_to_the_entry_thresholds(self) -> None:
        decoder = LiveBoundaryScoreDecoder(
            onset_threshold=0.55,
            offset_threshold=0.6,
        )

        self.assertEqual(decoder.onset_threshold, 0.55)
        self.assertEqual(decoder.offset_threshold, 0.6)
        self.assertEqual(decoder.onset_release_threshold, 0.55)
        self.assertEqual(decoder.offset_release_threshold, 0.6)

    def test_release_thresholds_must_be_positive_and_not_exceed_entry(self) -> None:
        invalid_configurations = (
            {"onset_threshold": 0.55, "onset_release_threshold": 0.0},
            {"onset_threshold": 0.55, "onset_release_threshold": 0.56},
            {"offset_threshold": 0.55, "offset_release_threshold": 0.0},
            {"offset_threshold": 0.55, "offset_release_threshold": 0.56},
        )
        for configuration in invalid_configurations:
            with self.subTest(configuration=configuration), self.assertRaises(
                ValueError
            ):
                LiveBoundaryScoreDecoder(**configuration)

    def test_release_equal_to_entry_preserves_historical_edges(self) -> None:
        default_decoder = LiveBoundaryScoreDecoder(
            slot_count=1,
            onset_threshold=0.55,
            offset_threshold=0.55,
        )
        explicit_decoder = LiveBoundaryScoreDecoder(
            slot_count=1,
            onset_threshold=0.55,
            offset_threshold=0.55,
            onset_release_threshold=0.55,
            offset_release_threshold=0.55,
        )
        chunks = (
            BoundaryScoreChunk(
                0,
                ((0.55,), (0.54,), (0.55,)),
                ((0.0,), (0.0,), (0.0,)),
            ),
            BoundaryScoreChunk(
                3,
                ((0.55,), (0.54,), (0.55,)),
                ((0.55,), (0.54,), (0.0,)),
            ),
        )

        default_events = tuple(
            event
            for chunk in chunks
            for event in default_decoder.process_chunk(chunk)
        )
        explicit_events = tuple(
            event
            for chunk in chunks
            for event in explicit_decoder.process_chunk(chunk)
        )

        self.assertEqual(explicit_events, default_events)
        self.assertEqual(
            default_events,
            (
                BoundaryEvent(BoundaryType.ONSET, "event-000001", 0),
                BoundaryEvent(BoundaryType.OFFSET, "event-000001", 3),
                BoundaryEvent(BoundaryType.ONSET, "event-000002", 5),
            ),
        )

    def test_onset_dip_inside_hysteresis_band_does_not_rearm_across_chunks(
        self,
    ) -> None:
        decoder = LiveBoundaryScoreDecoder(
            slot_count=1,
            onset_threshold=0.55,
            onset_release_threshold=0.5,
        )

        first = decoder.process_chunk(
            BoundaryScoreChunk(
                0,
                ((0.6,), (0.53,)),
                ((0.0,), (0.6,)),
            )
        )
        second = decoder.process_chunk(
            BoundaryScoreChunk(
                2,
                ((0.6,), (0.49,), (0.6,)),
                ((0.0,), (0.0,), (0.0,)),
            )
        )

        self.assertEqual(
            first + second,
            (
                BoundaryEvent(BoundaryType.ONSET, "event-000001", 0),
                BoundaryEvent(BoundaryType.OFFSET, "event-000001", 1),
                BoundaryEvent(BoundaryType.ONSET, "event-000002", 4),
            ),
        )

    def test_offset_dip_inside_hysteresis_band_does_not_rearm_across_chunks(
        self,
    ) -> None:
        decoder = LiveBoundaryScoreDecoder(
            slot_count=1,
            offset_threshold=0.55,
            offset_release_threshold=0.5,
        )

        first = decoder.process_chunk(
            BoundaryScoreChunk(
                0,
                ((0.6,), (0.0,), (0.6,)),
                ((0.0,), (0.6,), (0.53,)),
            )
        )
        second = decoder.process_chunk(
            BoundaryScoreChunk(
                3,
                ((0.0,), (0.0,), (0.0,)),
                ((0.6,), (0.49,), (0.6,)),
            )
        )

        self.assertEqual(
            first + second,
            (
                BoundaryEvent(BoundaryType.ONSET, "event-000001", 0),
                BoundaryEvent(BoundaryType.OFFSET, "event-000001", 1),
                BoundaryEvent(BoundaryType.ONSET, "event-000002", 2),
                BoundaryEvent(BoundaryType.OFFSET, "event-000002", 5),
            ),
        )

    def test_overlapping_a_and_b_keep_their_ids_across_chunks(self) -> None:
        decoder = LiveBoundaryScoreDecoder(slot_count=2)

        first = decoder.process_chunk(
            score_chunk(
                0,
                512,
                slots=2,
                onsets=((100, 0), (200, 1)),
            )
        )
        second = decoder.process_chunk(
            score_chunk(
                512,
                512,
                slots=2,
                offsets=((700, 0), (820, 1)),
            )
        )

        self.assertEqual(
            first,
            (
                BoundaryEvent(BoundaryType.ONSET, "event-000001", 100),
                BoundaryEvent(BoundaryType.ONSET, "event-000002", 200),
            ),
        )
        self.assertEqual(
            second,
            (
                BoundaryEvent(BoundaryType.OFFSET, "event-000001", 700),
                BoundaryEvent(BoundaryType.OFFSET, "event-000002", 820),
            ),
        )
        self.assertEqual(decoder.active_events(), ())

    def test_offset_precedes_retrigger_in_the_same_slot(self) -> None:
        decoder = LiveBoundaryScoreDecoder(slot_count=1)
        events = decoder.process_chunk(
            score_chunk(
                0,
                5,
                slots=1,
                onsets=((0, 0), (3, 0)),
                offsets=((3, 0),),
            )
        )

        self.assertEqual(
            events,
            (
                BoundaryEvent(BoundaryType.ONSET, "event-000001", 0),
                BoundaryEvent(BoundaryType.OFFSET, "event-000001", 3),
                BoundaryEvent(BoundaryType.ONSET, "event-000002", 3),
            ),
        )

    def test_sustained_high_score_emits_only_one_boundary(self) -> None:
        decoder = LiveBoundaryScoreDecoder(slot_count=1)
        scores = BoundaryScoreChunk(
            20,
            ((0.9,), (0.9,), (0.9,)),
            ((0.0,), (0.0,), (0.0,)),
        )
        self.assertEqual(len(decoder.process_chunk(scores)), 1)


class LiveBoundaryPeakDecoderTests(unittest.TestCase):
    def test_candidate_and_configuration_validation(self) -> None:
        with self.assertRaises(ValueError):
            BoundaryCandidate("onset", 0)
        with self.assertRaises(ValueError):
            BoundaryCandidate(BoundaryType.ONSET, -1)

        invalid_configurations = (
            {"slot_count": 0},
            {"slot_count": True},
            {"rearm_low_samples": 0},
            {"rearm_low_samples": -1},
            {"rearm_low_samples": True},
            {"rearm_low_samples": 1.0},
            {"consolidation_samples": -1},
            {"consolidation_samples": True},
            {"consolidation_samples": 1.0},
            {"onset_threshold": 0.0},
            {"offset_threshold": 0.0},
            {"onset_threshold": 0.55, "onset_release_threshold": 0.0},
            {"onset_threshold": 0.55, "onset_release_threshold": 0.56},
            {"offset_threshold": 0.55, "offset_release_threshold": 0.0},
            {"offset_threshold": 0.55, "offset_release_threshold": 0.56},
        )
        for configuration in invalid_configurations:
            with self.subTest(configuration=configuration), self.assertRaises(
                ValueError
            ):
                LiveBoundaryPeakDecoder(**configuration)

        self.assertEqual(
            LiveBoundaryPeakDecoder(rearm_low_samples=16).rearm_low_samples,
            16,
        )
        self.assertEqual(
            LiveBoundaryPeakDecoder(consolidation_samples=2205).consolidation_samples,
            2205,
        )

    def test_chunks_must_match_slot_count_and_be_contiguous(self) -> None:
        decoder = LiveBoundaryPeakDecoder(slot_count=2)
        with self.assertRaises(ValueError):
            decoder.process_chunk("not scores")
        with self.assertRaises(ValueError):
            decoder.process_chunk(score_chunk(0, 1, slots=1))

        decoder.process_chunk(score_chunk(100, 4, slots=2))
        with self.assertRaises(ValueError):
            decoder.process_chunk(score_chunk(105, 1, slots=2))

    def test_two_onsets_in_one_channel_do_not_require_an_offset(self) -> None:
        decoder = LiveBoundaryPeakDecoder(slot_count=1)

        first = decoder.process_chunk(
            score_chunk(0, 512, slots=1, onsets=((100, 0),))
        )
        second = decoder.process_chunk(
            score_chunk(512, 512, slots=1, onsets=((700, 0),))
        )

        self.assertEqual(
            first + second,
            (
                BoundaryCandidate(BoundaryType.ONSET, 100),
                BoundaryCandidate(BoundaryType.ONSET, 700),
            ),
        )

    def test_offset_is_emitted_without_a_prior_onset(self) -> None:
        decoder = LiveBoundaryPeakDecoder(slot_count=1)

        candidates = decoder.process_chunk(
            score_chunk(512, 512, slots=1, offsets=((820, 0),))
        )

        self.assertEqual(
            candidates,
            (BoundaryCandidate(BoundaryType.OFFSET, 820),),
        )

    def test_simultaneous_multiplicity_and_offset_first_order_are_preserved(
        self,
    ) -> None:
        decoder = LiveBoundaryPeakDecoder(slot_count=3)
        candidates = decoder.process_chunk(
            score_chunk(
                0,
                8,
                slots=3,
                onsets=((4, 0), (4, 2)),
                offsets=((4, 0), (4, 1)),
            )
        )

        self.assertEqual(
            candidates,
            (
                BoundaryCandidate(BoundaryType.OFFSET, 4),
                BoundaryCandidate(BoundaryType.OFFSET, 4),
                BoundaryCandidate(BoundaryType.ONSET, 4),
                BoundaryCandidate(BoundaryType.ONSET, 4),
            ),
        )

    def test_hysteresis_state_is_preserved_across_chunk_boundaries(self) -> None:
        decoder = LiveBoundaryPeakDecoder(
            slot_count=1,
            onset_threshold=0.55,
            offset_threshold=0.55,
            onset_release_threshold=0.5,
            offset_release_threshold=0.5,
        )

        first = decoder.process_chunk(
            BoundaryScoreChunk(
                0,
                ((0.6,), (0.53,)),
                ((0.0,), (0.6,)),
            )
        )
        second = decoder.process_chunk(
            BoundaryScoreChunk(
                2,
                ((0.6,), (0.49,), (0.6,), (0.0,)),
                ((0.53,), (0.6,), (0.49,), (0.6,)),
            )
        )

        self.assertEqual(
            first + second,
            (
                BoundaryCandidate(BoundaryType.ONSET, 0),
                BoundaryCandidate(BoundaryType.OFFSET, 1),
                BoundaryCandidate(BoundaryType.ONSET, 4),
                BoundaryCandidate(BoundaryType.OFFSET, 5),
            ),
        )

    def test_one_low_sample_reproduces_historical_rearming(self) -> None:
        default_decoder = LiveBoundaryPeakDecoder(
            slot_count=2,
            onset_threshold=0.55,
            offset_threshold=0.55,
            onset_release_threshold=0.5,
            offset_release_threshold=0.5,
        )
        explicit_decoder = LiveBoundaryPeakDecoder(
            slot_count=2,
            onset_threshold=0.55,
            offset_threshold=0.55,
            onset_release_threshold=0.5,
            offset_release_threshold=0.5,
            rearm_low_samples=1,
        )
        chunks = (
            BoundaryScoreChunk(
                0,
                ((0.6, 0.0), (0.53, 0.6), (0.49, 0.53)),
                ((0.0, 0.6), (0.6, 0.53), (0.49, 0.49)),
            ),
            BoundaryScoreChunk(
                3,
                ((0.6, 0.6), (0.0, 0.49), (0.0, 0.6)),
                ((0.6, 0.6), (0.49, 0.0), (0.6, 0.0)),
            ),
        )

        expected = (
            (
                BoundaryCandidate(BoundaryType.OFFSET, 0),
                BoundaryCandidate(BoundaryType.ONSET, 0),
                BoundaryCandidate(BoundaryType.OFFSET, 1),
                BoundaryCandidate(BoundaryType.ONSET, 1),
            ),
            (
                BoundaryCandidate(BoundaryType.OFFSET, 3),
                BoundaryCandidate(BoundaryType.OFFSET, 3),
                BoundaryCandidate(BoundaryType.ONSET, 3),
                BoundaryCandidate(BoundaryType.OFFSET, 5),
                BoundaryCandidate(BoundaryType.ONSET, 5),
            ),
        )

        default_result = tuple(
            default_decoder.process_chunk(chunk) for chunk in chunks
        )
        explicit_result = tuple(
            explicit_decoder.process_chunk(chunk) for chunk in chunks
        )
        self.assertEqual(default_result, expected)
        self.assertEqual(explicit_result, expected)

    def test_short_low_dip_across_chunks_does_not_rearm(self) -> None:
        decoder = LiveBoundaryPeakDecoder(
            slot_count=1,
            onset_threshold=0.55,
            onset_release_threshold=0.5,
            rearm_low_samples=3,
        )

        first = decoder.process_chunk(
            BoundaryScoreChunk(
                0,
                ((0.6,), (0.49,)),
                ((0.0,), (0.0,)),
            )
        )
        second = decoder.process_chunk(
            BoundaryScoreChunk(
                2,
                ((0.49,), (0.6,), (0.49,), (0.49,), (0.6,)),
                ((0.0,),) * 5,
            )
        )

        self.assertEqual(
            first + second,
            (BoundaryCandidate(BoundaryType.ONSET, 0),),
        )

    def test_required_consecutive_lows_rearm_the_channel(self) -> None:
        decoder = LiveBoundaryPeakDecoder(
            slot_count=1,
            onset_threshold=0.55,
            onset_release_threshold=0.5,
            rearm_low_samples=3,
        )
        candidates = decoder.process_chunk(
            BoundaryScoreChunk(
                20,
                ((0.6,), (0.49,), (0.49,), (0.49,), (0.6,)),
                ((0.0,),) * 5,
            )
        )

        self.assertEqual(
            candidates,
            (
                BoundaryCandidate(BoundaryType.ONSET, 20),
                BoundaryCandidate(BoundaryType.ONSET, 24),
            ),
        )

    def test_two_onsets_need_only_rearming_not_an_offset(self) -> None:
        decoder = LiveBoundaryPeakDecoder(
            slot_count=1,
            rearm_low_samples=16,
        )
        onset_rows = [[0.0] for _ in range(821)]
        offset_rows = [[0.0] for _ in range(821)]
        onset_rows[100][0] = 1.0
        onset_rows[700][0] = 1.0

        candidates = decoder.process_chunk(
            BoundaryScoreChunk(0, tuple(onset_rows), tuple(offset_rows))
        )

        self.assertEqual(
            candidates,
            (
                BoundaryCandidate(BoundaryType.ONSET, 100),
                BoundaryCandidate(BoundaryType.ONSET, 700),
            ),
        )

    def test_rearming_preserves_simultaneous_multiplicity(self) -> None:
        decoder = LiveBoundaryPeakDecoder(
            slot_count=3,
            rearm_low_samples=2,
        )
        candidates = decoder.process_chunk(
            score_chunk(
                0,
                8,
                slots=3,
                onsets=((1, 0), (1, 2), (4, 0), (4, 2)),
            )
        )

        self.assertEqual(
            candidates,
            (
                BoundaryCandidate(BoundaryType.ONSET, 1),
                BoundaryCandidate(BoundaryType.ONSET, 1),
                BoundaryCandidate(BoundaryType.ONSET, 4),
                BoundaryCandidate(BoundaryType.ONSET, 4),
            ),
        )

    def test_onset_and_offset_rearm_independently(self) -> None:
        decoder = LiveBoundaryPeakDecoder(
            slot_count=1,
            onset_threshold=0.55,
            offset_threshold=0.55,
            onset_release_threshold=0.5,
            offset_release_threshold=0.5,
            rearm_low_samples=3,
        )
        candidates = decoder.process_chunk(
            BoundaryScoreChunk(
                0,
                (
                    (0.6,),
                    (0.49,),
                    (0.49,),
                    (0.6,),
                    (0.49,),
                    (0.49,),
                    (0.49,),
                    (0.6,),
                ),
                (
                    (0.6,),
                    (0.49,),
                    (0.49,),
                    (0.49,),
                    (0.6,),
                    (0.49,),
                    (0.49,),
                    (0.6,),
                ),
            )
        )

        self.assertEqual(
            candidates,
            (
                BoundaryCandidate(BoundaryType.OFFSET, 0),
                BoundaryCandidate(BoundaryType.ONSET, 0),
                BoundaryCandidate(BoundaryType.OFFSET, 4),
                BoundaryCandidate(BoundaryType.ONSET, 7),
            ),
        )

    def test_fixed_consolidation_boundary_is_inclusive(self) -> None:
        suppressed = LiveBoundaryPeakDecoder(
            slot_count=1,
            rearm_low_samples=1,
            consolidation_samples=2205,
        )
        retained = LiveBoundaryPeakDecoder(
            slot_count=1,
            rearm_low_samples=1,
            consolidation_samples=2205,
        )

        suppressed_values = suppressed.process_chunk(
            score_chunk(100, 2206, slots=1, onsets=((100, 0), (2305, 0)))
        )
        retained_values = retained.process_chunk(
            score_chunk(100, 2207, slots=1, onsets=((100, 0), (2306, 0)))
        )

        self.assertEqual(
            suppressed_values,
            (BoundaryCandidate(BoundaryType.ONSET, 100),),
        )
        self.assertEqual(
            retained_values,
            (
                BoundaryCandidate(BoundaryType.ONSET, 100),
                BoundaryCandidate(BoundaryType.ONSET, 2306),
            ),
        )

    def test_suppressed_candidate_does_not_extend_consolidation(self) -> None:
        decoder = LiveBoundaryPeakDecoder(
            slot_count=1,
            rearm_low_samples=1,
            consolidation_samples=2205,
        )

        candidates = decoder.process_chunk(
            score_chunk(
                100,
                2901,
                slots=1,
                onsets=((100, 0), (2305, 0), (3000, 0)),
            )
        )

        self.assertEqual(
            candidates,
            (
                BoundaryCandidate(BoundaryType.ONSET, 100),
                BoundaryCandidate(BoundaryType.ONSET, 3000),
            ),
        )

    def test_consolidation_state_crosses_chunks_without_cross_channel_blocking(self) -> None:
        decoder = LiveBoundaryPeakDecoder(
            slot_count=2,
            rearm_low_samples=1,
            consolidation_samples=2205,
        )

        first = decoder.process_chunk(
            score_chunk(
                0,
                512,
                slots=2,
                onsets=((100, 0), (100, 1)),
            )
        )
        second = decoder.process_chunk(
            score_chunk(
                512,
                1794,
                slots=2,
                onsets=((2305, 0), (2305, 1)),
                offsets=((2305, 0), (2305, 1)),
            )
        )

        self.assertEqual(
            first + second,
            (
                BoundaryCandidate(BoundaryType.ONSET, 100),
                BoundaryCandidate(BoundaryType.ONSET, 100),
                BoundaryCandidate(BoundaryType.OFFSET, 2305),
                BoundaryCandidate(BoundaryType.OFFSET, 2305),
            ),
        )


class _ScriptedPredictor:
    slot_count = 2

    def predict_chunk(self, samples, *, start_sample):
        end = start_sample + len(samples)
        onsets = tuple(
            item for item in ((100, 0), (200, 1)) if start_sample <= item[0] < end
        )
        offsets = tuple(
            item for item in ((700, 0), (820, 1)) if start_sample <= item[0] < end
        )
        return score_chunk(
            start_sample,
            len(samples),
            slots=self.slot_count,
            onsets=onsets,
            offsets=offsets,
        )


class LiveModelDetectorTests(unittest.TestCase):
    def test_raw_audio_api_decodes_scripted_model_scores(self) -> None:
        detector = LiveModelDetector(_ScriptedPredictor())
        first = detector.process_chunk((0.0,) * 512, start_sample=0)
        second = detector.process_chunk((0.0,) * 512)

        self.assertEqual(
            tuple((event.kind, event.sample) for event in first + second),
            (
                (BoundaryType.ONSET, 100),
                (BoundaryType.ONSET, 200),
                (BoundaryType.OFFSET, 700),
                (BoundaryType.OFFSET, 820),
            ),
        )
        self.assertEqual(first[0].event_id, second[0].event_id)
        self.assertEqual(first[1].event_id, second[1].event_id)

    def test_propagates_resolved_release_thresholds_to_live_decoder(self) -> None:
        detector = LiveModelDetector(
            _ScriptedPredictor(),
            onset_threshold=0.55,
            offset_threshold=0.6,
            onset_release_threshold=0.5,
            offset_release_threshold=0.45,
        )

        self.assertEqual(detector.onset_threshold, 0.55)
        self.assertEqual(detector.offset_threshold, 0.6)
        self.assertEqual(detector.onset_release_threshold, 0.5)
        self.assertEqual(detector.offset_release_threshold, 0.45)

    def test_finalize_stream_delegates_without_predicting_future_audio(self) -> None:
        predictor = _ScriptedPredictor()
        detector = LiveModelDetector(predictor)
        onsets = detector.process_chunk((0.0,) * 512, start_sample=0)

        terminal = detector.finalize_stream()

        self.assertEqual(
            terminal,
            tuple(
                BoundaryEvent(BoundaryType.OFFSET, event.event_id, 512)
                for event in onsets
            ),
        )
        with self.assertRaises(RuntimeError):
            detector.process_chunk((0.0,))
        with self.assertRaises(RuntimeError):
            detector.finalize_stream()


if __name__ == "__main__":
    unittest.main()
