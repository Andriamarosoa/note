"""S636 host tests do not pretend that CI contains a Windows sound card."""
from __future__ import annotations
import unittest

from scripts.s636_note_live_desktop import MidiOutput,DiagnosticSink,engine_factory


class FakePort:
    def __init__(self):
        self.records=[]
    def send(self,msg):
        self.records.append(msg)


def message_factory(kind,**kwargs):
    return dict(type=kind,**kwargs)


class HostTests(unittest.TestCase):
    def test_hardware_sink_note_pairs_and_all_notes_off(self):
        port=FakePort()
        midi=MidiOutput(port,channel=2,message_factory=message_factory)
        midi(dict(type="note_on",pitch=52,velocity=96))
        midi(dict(type="note_off",pitch=52,velocity=0))
        midi(dict(type="note_on",pitch=64,velocity=96))
        midi.panic()
        self.assertEqual([m["type"] for m in port.records],
                         ["note_on","note_off","note_on","note_off","control_change"])
        self.assertEqual(port.records[-1]["control"],123)
        self.assertEqual(midi.active,set())

    def test_duplicate_midi_on_is_rejected_and_closed_safely(self):
        port=FakePort()
        midi=MidiOutput(port,message_factory=message_factory)
        midi(dict(type="note_on",pitch=52,velocity=96))
        with self.assertRaises(RuntimeError):
            midi(dict(type="note_on",pitch=52,velocity=96))
        midi.panic()
        self.assertEqual(midi.active,set())

    def test_no_midi_mode_tracks_note_lifecycle(self):
        sink=DiagnosticSink()
        sink(dict(type="note_on",pitch=52))
        sink(dict(type="note_off",pitch=52))
        self.assertEqual(sink.counts["note_on"],1)
        self.assertEqual(sink.counts["note_off"],1)
        sink.panic()
        self.assertEqual(sink.active,set())

    def test_original_checkpoint_mode_is_real_and_protected(self):
        factory,label=engine_factory(None)
        self.assertEqual(label,"S614")
        stream=factory()
        self.assertTrue(hasattr(stream,"push"))
        self.assertTrue(hasattr(stream,"finish"))
        self.assertEqual(stream.heads.metadata["selection"]["max_notes"],6)


if __name__=="__main__":
    unittest.main()
