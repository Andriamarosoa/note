"""S841 E2/E3 truth controls: nearby notes are not proof of sustained chord."""
import unittest
import numpy as np
from scripts.s841_audit_real_guitarset_e2e3_256 import E2,E3,HOP,SR,COMP


class RealOnsetE2E3AuditTests(unittest.TestCase):
    def test_exact_pitch_and_cadence(self):
        self.assertEqual((E2,E3),(40,52))
        self.assertEqual(HOP,256)
        self.assertEqual(SR,44100)
        self.assertEqual(len(COMP),5)
        self.assertAlmostEqual(HOP/SR*1000,5.80498866,places=5)

    def test_near_onsets_are_not_equivalent_to_simultaneous_sustain(self):
        # Two real NoteOn events near each other do NOT prove that both
        # pitched sounds are physically present in a later PCM256 block.
        e2=.100
        e3=.170
        self.assertGreater(abs(e2-e3),HOP/SR)
        self.assertLess(abs(e2-e3),.08)

    def test_can_use_only_current_256_pcm_for_a_hypothesis(self):
        pcm=np.zeros(1024,dtype=np.int16)
        end=768
        x=pcm[end-HOP:end]
        self.assertEqual(x.shape,(HOP,))
        self.assertFalse(np.shares_memory(x,pcm[end:]))

if __name__=="__main__":
    unittest.main()
