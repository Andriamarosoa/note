import unittest
import numpy as np

from scripts.v273_residual_map import (
    ResidualStream, innovation, time_frequency_map, ownership_input, onset_coverage,
    FRAME_ENDS, FREQUENCIES)
from scripts.v273_ownership_context import owners_at_samples


class ResidualMapTest(unittest.TestCase):
    def test_silence_and_map_shape(self):
        x = np.zeros(5000, np.float32)
        residual, _ = innovation(x)
        np.testing.assert_array_equal(residual, 0)
        image = time_frequency_map(x, residual, 2000)
        self.assertEqual(image.shape, (31, 64, 4))
        np.testing.assert_array_equal(image, 0)

    def test_arbitrary_chunks_and_partial_prefix(self):
        rng = np.random.default_rng(93031)
        x = rng.normal(0, .1, 15000).astype(np.float32)
        full, _ = innovation(x)
        stream = ResidualStream(); chunks = []; position = 0
        while position < len(x):
            length = (1, 127, 257, 4096)[len(chunks) % 4]
            chunks.append(stream.process(x[position:position+length])); position += length
        np.testing.assert_array_equal(np.concatenate(chunks), full)
        prefix, _ = innovation(x[:13217])
        np.testing.assert_array_equal(prefix, full[:13217])

    def test_unseen_suffix_cannot_change_map(self):
        rng = np.random.default_rng(93032); origin = 10500; horizon = origin+2788
        a = rng.normal(0, .1, 16000).astype(np.float32)
        b = a.copy(); b[horizon:] = rng.normal(0, .8, len(b)-horizon)
        ar, _ = innovation(a); br, _ = innovation(b)
        np.testing.assert_array_equal(time_frequency_map(a, ar, origin), time_frequency_map(b, br, origin))
        prefix, _ = innovation(a[:horizon])
        np.testing.assert_array_equal(time_frequency_map(a[:horizon], prefix, origin),
                                      time_frequency_map(a, ar, origin))

    def test_every_admissible_onset_has_positive_short_hann_support(self):
        support = onset_coverage(np.arange(-882, 2647))
        self.assertTrue(np.all(support.any(axis=1)))
        self.assertEqual(int(FRAME_ENDS[-1]), 2788)

    def test_ownership_exact_ties_prefix_and_refusal(self):
        groups = [np.array([5000, 6764]), np.array([8300])]
        watermark = 8528; decision = watermark+5
        packed, fractions = ownership_input(groups, 0, complete_through=watermark, decision_sample=decision)
        q = 5000-1308+np.arange(4096)
        expected = owners_at_samples(groups, q) == 0
        np.testing.assert_array_equal(np.unpackbits(packed, bitorder='little').astype(bool), expected)
        augmented = groups + [np.array([20000])]
        second, _ = ownership_input(augmented, 0, complete_through=watermark, decision_sample=decision)
        np.testing.assert_array_equal(packed, second)
        for complete, ready in [(watermark-1, decision), (watermark, decision-1)]:
            with self.assertRaises(ValueError):
                ownership_input(groups, 0, complete_through=complete, decision_sample=ready)
        self.assertTrue(np.all((fractions >= 0) & (fractions <= 1)))

    def test_equal_energy_different_frequencies_remain_distinct(self):
        # Information-preservation test: these are tones, not musical K labels.
        t = np.arange(12000)/44100.
        a = np.sin(2*np.pi*440*t).astype(np.float32)
        b = np.sin(2*np.pi*1300*t).astype(np.float32)
        b *= np.sqrt(np.mean(a.astype(float)**2)/np.mean(b.astype(float)**2))
        self.assertAlmostEqual(float(np.mean(a*a)), float(np.mean(b*b)), places=6)
        am = time_frequency_map(a, a, 8000)[:, :, 3].mean(axis=0)
        bm = time_frequency_map(b, b, 8000)[:, :, 3].mean(axis=0)
        self.assertLess(abs(FREQUENCIES[np.argmax(am)]-440), 40)
        self.assertLess(abs(FREQUENCIES[np.argmax(bm)]-1300), 100)
        self.assertNotEqual(int(np.argmax(am)), int(np.argmax(bm)))

    def test_equal_energy_different_times_remain_distinct(self):
        a = np.zeros(12000, np.float32); b = a.copy(); origin = 7000
        a[origin-700:origin-572] = np.hanning(128)
        b[origin+2200:origin+2328] = np.hanning(128)
        self.assertEqual(float(np.sum(a*a)), float(np.sum(b*b)))
        am = time_frequency_map(a, a, origin)[:, :, 1].sum(axis=1)
        bm = time_frequency_map(b, b, origin)[:, :, 1].sum(axis=1)
        self.assertLess(int(np.argmax(am)), int(np.argmax(bm)))

    def test_no_annotation_argument_and_invalid_audio(self):
        import inspect
        self.assertEqual(list(inspect.signature(time_frequency_map).parameters), ['observed', 'residual', 'origin'])
        with self.assertRaises(ValueError):
            ResidualStream().process(np.array([np.nan]))


if __name__ == '__main__':
    unittest.main()
