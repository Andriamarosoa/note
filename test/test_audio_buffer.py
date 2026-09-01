import unittest

from causal_note.audio_buffer import (
    AudioDiscardedError,
    CausalAudioBuffer,
    FrameAvailability,
    InsufficientAudioError,
    NonContiguousAudioError,
)


class CausalAudioBufferTests(unittest.TestCase):
    def test_discard_compacts_storage_without_changing_absolute_indices(self) -> None:
        buffer = CausalAudioBuffer(frame_size=4)
        buffer.append(range(10))
        buffer.discard_before(6)

        self.assertEqual(buffer.start_sample, 6)
        self.assertEqual(buffer.end_sample, 10)
        self.assertEqual(len(buffer), 4)
        self.assertEqual(buffer.frame_at(6), (6.0, 7.0, 8.0, 9.0))
        with self.assertRaises(AudioDiscardedError):
            buffer.frame_at(5)

        buffer.append((10.0, 11.0))
        self.assertEqual(buffer.end_sample, 12)

    def test_internal_offset_reuses_suffix_without_loss(self) -> None:
        buffer = CausalAudioBuffer(frame_size=512)
        buffer.append(range(512))

        availability = buffer.availability(300)
        self.assertEqual(availability.frame_start, 300)
        self.assertEqual(availability.frame_end, 812)
        self.assertEqual(availability.reusable_samples, 212)
        self.assertEqual(availability.stream_samples_until_ready, 300)
        self.assertFalse(availability.ready)

        buffer.append(range(512, 812))
        frame = buffer.frame_at(300)
        self.assertEqual(len(frame), 512)
        self.assertEqual(frame, tuple(float(value) for value in range(300, 812)))

    def test_offset_after_current_frame_waits_for_contiguous_stream(self) -> None:
        buffer = CausalAudioBuffer(frame_size=512)
        buffer.append(range(512))

        availability = buffer.availability(700)
        self.assertEqual(availability.reusable_samples, 0)
        self.assertEqual(availability.stream_samples_until_ready, 700)

        buffer.append(range(512, 1212))
        self.assertEqual(
            buffer.frame_at(700),
            tuple(float(value) for value in range(700, 1212)),
        )

    def test_offset_equal_to_t2_starts_at_exclusive_frame_end(self) -> None:
        buffer = CausalAudioBuffer(frame_size=512)
        buffer.append(range(1024))
        frame = buffer.frame_at(512)
        self.assertEqual(frame[0], 512.0)
        self.assertEqual(frame[-1], 1023.0)

    def test_multiple_offset_requests_do_not_overwrite_each_other(self) -> None:
        buffer = CausalAudioBuffer(frame_size=512)
        buffer.append(range(1200))
        frame_a = buffer.frame_at(100)
        frame_b = buffer.frame_at(250)
        self.assertEqual(frame_a, tuple(float(value) for value in range(100, 612)))
        self.assertEqual(frame_b, tuple(float(value) for value in range(250, 762)))

    def test_incomplete_frame_is_explicit_and_never_zero_padded(self) -> None:
        buffer = CausalAudioBuffer(frame_size=512)
        buffer.append(range(500))
        self.assertIsNone(buffer.try_frame_at(0))
        with self.assertRaises(InsufficientAudioError) as raised:
            buffer.frame_at(0)
        self.assertEqual(
            raised.exception.availability.stream_samples_until_ready,
            12,
        )

    def test_append_rejects_gap_and_overlap(self) -> None:
        buffer = CausalAudioBuffer(frame_size=4)
        buffer.append([0.0, 1.0])
        with self.assertRaises(NonContiguousAudioError):
            buffer.append([2.0], start_sample=3)
        with self.assertRaises(NonContiguousAudioError):
            buffer.append([2.0], start_sample=1)

    def test_invalid_chunk_is_rejected_atomically(self) -> None:
        buffer = CausalAudioBuffer(frame_size=4)
        buffer.append([0.0, 1.0])
        for invalid in (float("nan"), float("inf"), 1e39, 10**400, object()):
            with self.subTest(invalid=type(invalid).__name__):
                with self.assertRaises(ValueError):
                    buffer.append([2.0, invalid])
                self.assertEqual(len(buffer), 2)
                self.assertEqual(buffer.start_sample, 0)
                self.assertEqual(buffer.end_sample, 2)

    def test_frame_availability_rejects_inconsistent_public_values(self) -> None:
        with self.assertRaises(ValueError):
            FrameAvailability(0, 512, 211, 300)
        with self.assertRaises(ValueError):
            FrameAvailability(0, 0, 0, 0)


if __name__ == "__main__":
    unittest.main()
