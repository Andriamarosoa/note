import unittest

from causal_note.v8_runtime import (
    AnonymousBoundary,
    AnonymousEventAssociator,
    BoundaryKind,
    V8BoundaryDecoder,
    V8ScoreChunk,
)


def chunk(start, onset_p, offset_p, onset_m=None, offset_m=None):
    n = len(onset_p)
    if onset_m is None:
        onset_m = [(1.0, 0.0, 0.0)] * n
    if offset_m is None:
        offset_m = [(1.0, 0.0, 0.0)] * n
    return V8ScoreChunk(
        start,
        tuple(onset_p),
        tuple(offset_p),
        tuple(tuple(x) for x in onset_m),
        tuple(tuple(x) for x in offset_m),
    )


class V8RuntimeTests(unittest.TestCase):
    def test_presence_plateau_emits_one_boundary(self):
        decoder = V8BoundaryDecoder()
        events = decoder.process_chunk(
            chunk(0, [0.1, 0.8, 0.9, 0.7, 0.1], [0.0] * 5)
        )
        self.assertEqual(events, (AnonymousBoundary(BoundaryKind.ONSET, 1, 1),))

    def test_multiplicity_is_read_only_on_rising_edge(self):
        decoder = V8BoundaryDecoder()
        events = decoder.process_chunk(
            chunk(
                0,
                [0.1, 0.8, 0.9],
                [0.0, 0.0, 0.0],
                onset_m=[
                    (1.0, 0.0, 0.0),
                    (0.1, 0.2, 0.7),
                    (1.0, 0.0, 0.0),
                ],
            )
        )
        self.assertEqual(events, (AnonymousBoundary(BoundaryKind.ONSET, 1, 3),))

    def test_chunk_continuity_preserves_edge_state(self):
        decoder = V8BoundaryDecoder()
        first = decoder.process_chunk(chunk(0, [0.8, 0.9], [0.0, 0.0]))
        second = decoder.process_chunk(chunk(2, [0.9, 0.1, 0.8], [0.0] * 3))
        self.assertEqual(len(first), 1)
        self.assertEqual(
            second, (AnonymousBoundary(BoundaryKind.ONSET, 4, 1),)
        )

    def test_offset_processed_before_same_sample_onset(self):
        decoder = V8BoundaryDecoder()
        decoder.process_chunk(chunk(0, [0.8, 0.1], [0.0, 0.0]))
        events = decoder.process_chunk(
            chunk(2, [0.8], [0.8], onset_m=[(0.0, 1.0, 0.0)])
        )
        self.assertEqual(
            events,
            (
                AnonymousBoundary(BoundaryKind.OFFSET, 2, 1),
                AnonymousBoundary(BoundaryKind.ONSET, 2, 2),
            ),
        )

    def test_fifo_association_preserves_multiplicity(self):
        associator = AnonymousEventAssociator()
        opened = associator.process(
            (
                AnonymousBoundary(BoundaryKind.ONSET, 10, 2),
                AnonymousBoundary(BoundaryKind.ONSET, 20, 1),
            )
        )
        self.assertEqual([event.event_id for event in opened], [
            "event-000001", "event-000002", "event-000003"
        ])
        closed = associator.process(
            (AnonymousBoundary(BoundaryKind.OFFSET, 30, 2),)
        )
        self.assertEqual(
            [event.event_id for event in closed],
            ["event-000001", "event-000002"],
        )
        self.assertEqual(
            [event.event_id for event in associator.active_events()],
            ["event-000003"],
        )


if __name__ == "__main__":
    unittest.main()
