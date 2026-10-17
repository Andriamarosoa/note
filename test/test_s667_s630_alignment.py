"""Protect S630 batch behavior while opt-in S667 NoteOff thresholds are enabled."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import io
import json
import unittest

import numpy as np

from causal_note.strict_live import HOP, Selection, StrictLiveStream
from causal_note.batched_live import BatchedLiveStream
from scripts.live_strict_notes import main, s667_live_selection


class FakeFrozenHeads:
    """One scripted causal NOTE posterior per native 256-sample hop.

    Same neural/ONNX API on singleton and batched calls. Tests preserve every
    source-hop's original posterior; only the selector config may differ.
    """
    def __init__(self, selection):
        self.metadata = {"selection": asdict(selection)}
        self.next_frame = 0

    def predict(self, x):
        input_array = np.asarray(x)
        batched = input_array.ndim == 3
        batch_count = len(input_array) if batched else 1
        p = np.zeros((batch_count, 49, 3), np.float32)
        p[:, :, 0] = .01
        p[:, :, 1] = .1
        p[:, :, 2] = .1
        k = np.zeros((batch_count, 7), np.float32)
        k[:, 0] = 1.
        for j in range(batch_count):
            frame = self.next_frame + j
            if frame < 4:
                p[j, 0, 0] = .95
                p[j, 0, 1] = .95
            elif frame < 9:
                p[j, 0, 1] = .2
            else:
                p[j, 0, 1] = .1
        self.next_frame += batch_count
        return (p, k) if batched else (p[0], k[0])


def playback(stream):
    events = []
    for j in range(18):
        # 256-frame capture callbacks, same PCM and native head posteriors.
        events.extend(stream.push(np.zeros(HOP, np.float32), start_sample=j * HOP))
    events.extend(stream.finish())
    return [(e["type"],e["pitch"],e["decision_sample"]) for e in events], events


class S667S630AlignmentTest(unittest.TestCase):
    def setUp(self):
        self.original = Selection(onset=.60, active_on=.5, active_off=.25,
                                  onset_reset=.2, release_frames=3)
        self.metadata = {"selection": asdict(self.original)}

    def test_original_batch_and_no_optin_remain_identical(self):
        baseline = BatchedLiveStream(FakeFrozenHeads(self.original),batch_hops=4)
        explicit_none = BatchedLiveStream(FakeFrozenHeads(self.original),
                                          batch_hops=4,selection=None)
        self.assertEqual(playback(baseline)[0], playback(explicit_none)[0])
        self.assertIsNone(s667_live_selection(self.metadata))
        self.assertEqual(self.original.active_off,.25)

    def test_s667_offset_is_replayed_without_changing_pitch_or_onsets(self):
        chosen = s667_live_selection(self.metadata,3,.15)
        native, nev = playback(StrictLiveStream(FakeFrozenHeads(self.original),
                                                selection=chosen))
        batch, bev = playback(BatchedLiveStream(FakeFrozenHeads(self.original),
                                                batch_hops=4, selection=chosen))
        self.assertEqual(native,batch)
        self.assertEqual([e for e in native if e[0]=="note_on"],
                         [e for e in batch if e[0]=="note_on"])
        self.assertTrue(all(e["available_sample"]>=e["decision_sample"]
                            for e in bev))
        self.assertEqual(chosen.onset,self.original.onset)
        self.assertEqual(chosen.active_on,self.original.active_on)

    def test_override_delays_release_without_changing_original_head(self):
        original, _ = playback(BatchedLiveStream(FakeFrozenHeads(self.original),batch_hops=4))
        chosen = s667_live_selection(self.metadata,3,.15)
        modified, _ = playback(BatchedLiveStream(FakeFrozenHeads(self.original),
                                                 batch_hops=4,selection=chosen))
        orig_on = [e for e in original if e[0]=="note_on"]
        mod_on = [e for e in modified if e[0]=="note_on"]
        orig_off = [e for e in original if e[0]=="note_off"]
        mod_off = [e for e in modified if e[0]=="note_off"]
        self.assertEqual(orig_on,mod_on)
        self.assertEqual(len(orig_off),1)
        self.assertEqual(len(mod_off),1)
        self.assertGreater(mod_off[0][2],orig_off[0][2])
        self.assertEqual(self.original.active_off,.25)

    def test_cli_accepts_s667_with_s630_batch4(self):
        with redirect_stdout(io.StringIO()) as stdout:
            main(["--status","--inference-batch-hops","4",
                  "--s667-release-frames","5","--s667-active-off","0.15"])
        self.assertIn("EXPERIMENTAL_AUTONOMOUS_CAUSAL_NOTE_LIVE",stdout.getvalue())

    def test_invalid_s667_combinations_rejected(self):
        for args in (
            ["--s667-active-off","1.2"],
            ["--s667-release-frames","0"],
            ["--s667-active-off","0.15","--lookahead-ms","5"],
            ["--s667-active-off","0.15","--hop-size","128"],
            ["--s667-active-off","0.15","--harmonic-mode","shadow"],
        ):
            with self.subTest(args=args):
                with redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as err:
                        main(["--status",*args])
                self.assertEqual(err.exception.code,2)


if __name__=="__main__":
    unittest.main()
