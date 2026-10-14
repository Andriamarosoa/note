"""Compatibility guards: live MIDI option MUST NOT modify existing Live or S80 K."""
from __future__ import annotations

from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
import struct
import tempfile
import unittest

from causal_note.live_k_midi_bridge import (
    LockedKLiveMidi, load_frozen_profile, parse_live_decision, save_midi_file,
)
from scripts.stream_k_midi_live import DEFAULT_PROFILE, process_jsonl


class KLockedLiveMidiContractTests(unittest.TestCase):
    def test_K0_harmonic_sustain_and_offset_cannot_invent_new_attacks(self):
        machine = LockedKLiveMidi()
        first = machine.process(sample=100, s80_K=2, midi_pitches=[52, 64])
        self.assertEqual([(x.kind, x.pitch) for x in first],
                         [("note_on", 52), ("note_on", 64)])
        self.assertEqual(machine.total_note_ons, machine.total_input_K)
        self.assertEqual(machine.process(sample=200, s80_K=0, midi_pitches=[]), ())
        self.assertEqual(len(machine.active_notes), 2,
                         "K0 must NOT terminate ringing notes")
        close = machine.process(sample=500, s80_K=0, midi_pitches=[],
                                note_off_ids=[first[0].note_id])
        self.assertEqual([(x.kind, x.pitch) for x in close], [("note_off", 52)])
        self.assertEqual([x.pitch for x in machine.active_notes], [64])
        done = machine.finalize(end_sample=1000)
        self.assertEqual([(x.kind, x.pitch) for x in done], [("note_off", 64)])
        self.assertEqual(machine.total_input_K, machine.total_note_ons)
        self.assertEqual(machine.total_note_ons, 2)

    def test_reattack_on_same_pitch_closes_prior_note_before_new_noteon(self):
        m = LockedKLiveMidi()
        old = m.process(sample=100, s80_K=1, midi_pitches=[60])[0]
        seq = m.process(sample=200, s80_K=1, midi_pitches=[60])
        self.assertEqual([t.kind for t in seq], ["note_off", "note_on"])
        self.assertEqual(seq[0].note_id, old.note_id)
        self.assertNotEqual(seq[1].note_id, old.note_id)
        self.assertEqual(m.total_note_ons, 2)
        self.assertEqual(len(m.active_notes), 1)
        self.assertEqual(len(m.finalize(end_sample=300)), 1)

    def test_failing_input_does_not_mutate_K_or_note_activity(self):
        m = LockedKLiveMidi()
        old = m.process(sample=100, s80_K=1, midi_pitches=[60])[0]
        failures = [
            dict(sample=110, s80_K=2, midi_pitches=[61]),
            dict(sample=110, s80_K=0, midi_pitches=[62]),
            dict(sample=110, s80_K=1, midi_pitches=[-1]),
            dict(sample=110, s80_K=1, midi_pitches=[89]),
            dict(sample=110, s80_K=7, midi_pitches=list(range(40, 47))),
            dict(sample=110, s80_K=2, midi_pitches=[61,61]),
            dict(sample=110, s80_K=True, midi_pitches=[]),
            dict(sample=90, s80_K=1, midi_pitches=[62]),
            dict(sample=110, s80_K=0, midi_pitches=[],
                 note_off_ids=["unknown"]),
            dict(sample=100, s80_K=1, midi_pitches=[60]),
        ]
        for kw in failures:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                m.process(**kw)
            self.assertEqual(m.total_input_K, 1)
            self.assertEqual(m.total_note_ons, 1)
            self.assertEqual([n.note_id for n in m.active_notes], [old.note_id])
        m.finalize(end_sample=111)
        with self.assertRaises(RuntimeError):
            m.process(sample=112, s80_K=0)
        with self.assertRaises(RuntimeError):
            m.finalize(end_sample=113)

    def test_K6_creates_six_distinct_real_note_on_outputs(self):
        m = LockedKLiveMidi()
        notes = m.process(sample=256, s80_K=6,
                          midi_pitches=[40, 45, 52, 60, 76, 88])
        self.assertEqual(len(notes), 6)
        self.assertEqual({n.pitch for n in notes}, {40,45,52,60,76,88})
        self.assertEqual(len(m.finalize(end_sample=512)), 6)
        self.assertEqual(m.total_input_K, m.total_note_ons)

    def test_terminal_eof_requires_true_end_sample_after_open_notes(self):
        m = LockedKLiveMidi()
        m.process(sample=50, s80_K=1, midi_pitches=[60])
        with self.assertRaises(ValueError):
            m.finalize(end_sample=50)
        self.assertEqual(len(m.active_notes), 1)
        self.assertEqual(len(m.finalize(end_sample=51)), 1)

    def test_stream_emits_without_waiting_until_end_of_stream_and_writes_real_midi(self):
        inp = StringIO('\n'.join([
            json.dumps(dict(type="decision",sample=100,s80_K=2,midi_pitches=[55,67])),
            json.dumps(dict(type="decision",sample=200,s80_K=0,midi_pitches=[])),
            json.dumps(dict(type="decision",sample=300,s80_K=0,midi_pitches=[],
                            note_off_ids=["note-000000001"])),
            json.dumps(dict(type="decision",sample=400,s80_K=1,midi_pitches=[67])),
            json.dumps(dict(type="eof",end_sample=500))
        ])+'\n')
        out=StringIO()
        with tempfile.TemporaryDirectory() as dirname:
            midi=Path(dirname)/"demo.mid"
            report=process_jsonl(inp,output=out,profile=DEFAULT_PROFILE,midi_file=midi)
            wire=[json.loads(line) for line in out.getvalue().splitlines()]
            self.assertEqual([w["type"] for w in wire],
                ["note_on","note_on","note_off","note_off","note_on","note_off"])
            self.assertEqual([w["sample"] for w in wire],
                [100,100,300,400,400,500])
            self.assertEqual(report["note_on_events"],3)
            self.assertEqual(report["input_S80_K_sum"],3)
            self.assertFalse(report["live_MIDI_accuracy_measured"])
            raw=midi.read_bytes()
            self.assertEqual(raw[:4],b"MThd")
            self.assertEqual(struct.unpack(">IHHH",raw[4:14]),(6,0,1,480))
            self.assertEqual(raw[14:18],b"MTrk")
            self.assertIn(bytes([0x90,55,80]),raw)
            self.assertIn(bytes([0x80,55,0]),raw)

    def test_eof_without_measured_time_rejected(self):
        source=StringIO(json.dumps(
            dict(type="decision",sample=100,s80_K=0,midi_pitches=[]))+'\n')
        with self.assertRaisesRegex(ValueError,"missing terminal EOF"):
            process_jsonl(source,output=StringIO(),profile=DEFAULT_PROFILE)

    def test_bad_stream_record_fails_closed(self):
        with self.assertRaisesRegex(ValueError,"requires sample"):
            parse_live_decision(dict(type="decision",s80_K=1,midi_pitches=[60]))
        with self.assertRaisesRegex(ValueError,"unknown live"):
            parse_live_decision(dict(type="decision",sample=1,s80_K=1,
                                     midi_pitches=[60],true_K=1))
        with self.assertRaisesRegex(ValueError,"array"):
            parse_live_decision(dict(type="decision",sample=1,s80_K=1,
                                     midi_pitches="60"))

    def test_exact_active_reference_manifest_is_pinned_and_noncausal_admitted(self):
        p=load_frozen_profile(DEFAULT_PROFILE)
        self.assertEqual(p["original_S80_ExactK_reference"],"S80_S397_K5_GUARDED")
        self.assertEqual(p["best_offline_native_challenger"],
                         "S422_S80_K_LOCKED_5S_CONTINUOUS_MIC_PITCH_SURVIVAL_OFFSET")
        self.assertFalse(p["S422_is_low_latency_causal"])
        self.assertFalse(p["live_quality_certified"])
        self.assertTrue(p["legacy_live_untouched"])
        with tempfile.TemporaryDirectory() as directory:
            bad=p.copy()
            bad["original_S80_ExactK_reference"]="some-unverified-ref"
            f=Path(directory)/"changed.json"
            f.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                load_frozen_profile(f)


if __name__ == "__main__":
    unittest.main()
