import unittest
import numpy as np
from causal_note.s704_joint_heads import JointPolicy, JointSelector
from causal_note.s712_attack_memory import AttackMemory, MemoryPolicy, decode_memory


class MemoryTests(unittest.TestCase):
    def make(self, policy):
        return AttackMemory(JointSelector(JointPolicy(onset=.6, sustain_on=.5,
                            sustain_off=.3, release=.85)), policy)

    def test_aligns_received_evidence_at_actual_later_emission(self):
        s = self.make(MemoryPolicy(1, 0)); k = np.eye(7)[1]
        p = np.zeros((49, 3)); p[12, :2] = [.8, .3]
        self.assertEqual(s.step(p, k, 1024), [])
        p[12, :2] = [.1, .8]
        events = s.step(p, k, 2048)
        self.assertEqual([(e['type'], e['pitch'], e['available_sample']) for e in events],
                         [('note_on', 52, 2048)])

    def test_current_floor_blocks_stale_presence(self):
        s = self.make(MemoryPolicy(1, 1)); k = np.eye(7)[1]
        p = np.zeros((49, 3)); p[12, :2] = [.1, .8]
        self.assertEqual(s.step(p, k, 1024), [])
        p[12, :2] = [.9, .1]
        self.assertEqual(s.step(p, k, 2048), [])

    def test_release_uses_current_heads(self):
        s = self.make(MemoryPolicy(3, 2)); k = np.eye(7)[1]
        p = np.zeros((49, 3)); p[12, :2] = [.8, .8]
        self.assertEqual(s.step(p, k, 1024)[0]['type'], 'note_on')
        p[12] = [.1, .8, .0]; s.step(p, k, 2048)
        p[12] = [.1, .1, .95]
        events = s.step(p, k, 3072)
        self.assertEqual([(e['type'], e['available_sample']) for e in events], [('note_off', 3072)])

    def test_future_outputs_cannot_change_prefix(self):
        rng = np.random.default_rng(712)
        p = rng.random((15, 49, 3)); changed = p.copy(); changed[9:] = 0
        k = np.tile(np.eye(7)[3], (15, 1)); ends = np.arange(1, 16) * 1024
        def run(values):
            selector = JointSelector(JointPolicy())
            return decode_memory(ends, values, k, selector, MemoryPolicy(3, 2), int(ends[-1]))
        before = lambda rows: [e for e in rows if e['available_sample'] <= ends[8]]
        self.assertEqual(before(run(p)), before(run(changed)))


if __name__ == '__main__':
    unittest.main()
