"""S622 avoid losing original good S613 NOTE-ons to an early fast-pitch lease."""
import unittest

from causal_note.dual_hop import DualSourceEventMerger, DualHopConfig


def event(kind, sample, pitch=52):
    return dict(type=kind,pitch=pitch,
                decision_sample=sample,available_sample=sample,
                time=sample/44100.,velocity=96 if kind=="note_on" else 0,
                reason="test")


class ReferencePriorityTests(unittest.TestCase):
    def test_genuine_later_baseline_note_is_not_silently_hidden(self):
        m=DualSourceEventMerger(reference_priority=True,grace_samples=512)
        self.assertEqual(len(m.step("fast",event("note_on",128))),1)
        takeover=m.step("reference",event("note_on",1536))
        self.assertEqual([x["type"] for x in takeover],["note_off","note_on"])
        self.assertTrue(m.current[52-40])
        self.assertEqual(m.reference_retriggers,1)
        # Original fast head ends: original bass note is still sustaining.
        self.assertEqual(m.step("fast",event("note_off",1600)),[])
        # Frozen reference release always owns the final audible duration.
        self.assertEqual(len(m.step("reference",event("note_off",2304))),1)
        self.assertFalse(m.current[52-40])
        self.assertEqual(m.finish(2304),[])

    def test_fast_and_reference_nearly_simultaneous_do_not_double_note(self):
        m=DualSourceEventMerger(reference_priority=True,grace_samples=1024)
        self.assertEqual(len(m.step("fast",event("note_on",256))),1)
        self.assertEqual(m.step("reference",event("note_on",512)),[])
        self.assertEqual(m.reference_retriggers,0)
        self.assertEqual(m.step("fast",event("note_off",768)),[])
        self.assertEqual(len(m.step("reference",event("note_off",1024))),1)

    def test_reference_release_retires_stale_fast_note(self):
        m=DualSourceEventMerger(reference_priority=True,grace_samples=256)
        m.step("fast",event("note_on",128))
        m.step("reference",event("note_on",256))
        out=m.step("reference",event("note_off",640))
        self.assertEqual(len(out),1)
        self.assertEqual(out[0]["type"],"note_off")
        self.assertEqual(m.fast_leases_cut_by_reference,1)
        self.assertFalse(m.current[12])
        # A late fast noteoff after reference taking over is harmless.
        self.assertEqual(m.step("fast",event("note_off",768)),[])

    def test_legacy_merger_behavior_unchanged_by_default(self):
        m=DualSourceEventMerger()
        m.step("fast",event("note_on",128))
        self.assertEqual(m.step("reference",event("note_on",1536)),[])
        self.assertEqual(m.reference_retriggers,0)
        self.assertEqual(m.step("fast",event("note_off",1600)),[])
        off=m.step("reference",event("note_off",2304))
        self.assertEqual(len(off),1)


if __name__=="__main__":
    unittest.main()
