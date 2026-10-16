"""S633 integration regression: frozen NOTE LIVE, including real CLI batch4.

The original model remains byte-identical in shadow. A dummy head with a
known subthreshold note tests guard through the REAL causal PCM frontend,
both native and batched. No microphone hardware is claimed.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
import wave
import numpy as np

from causal_note.strict_live import StrictLiveStream,SR,HOP,load_heads
from causal_note.batched_live import BatchedLiveStream
from causal_note.lookahead import LookaheadConfig
from causal_note.register_rescue import LIVE_PRESETS
from scripts.live_strict_notes import DEFAULT_MODEL

REPO=pathlib.Path(__file__).resolve().parents[1]


class WeakHighHead:
    metadata={"selection":dict(onset=.60,active_on=.50)}
    def predict(self,features):
        x=np.asarray(features)
        batched=x.ndim==3
        if not batched:x=x[None]
        p=np.zeros((len(x),49,3),np.float32)
        p[:,80-40,0]=.45
        p[:,80-40,1]=.85
        k=np.zeros((len(x),7),np.float32)
        k[:,1]=1.
        return (p,k) if batched else (p[0],k[0])


def audio_fixture():
    audio=np.zeros(round(.35*SR),np.float32)
    start=round(.095*SR)
    end=round(.255*SR)
    phase=np.arange(end-start)/SR
    audio[start:end]=.30*np.sin(2*np.pi*830.61*phase).astype(np.float32)
    return audio


def replay(stream,audio,block=256):
    events=[]
    for i in range(0,len(audio),block):
        events.extend(stream.push(audio[i:i+block],start_sample=i))
    events.extend(stream.finish())
    return events


def signatures(events):
    return [(e["type"],e["pitch"],e["decision_sample"]) for e in events]


class S633LiveIntegrationTests(unittest.TestCase):
    def test_strict_live_shadow_equals_off_guard_can_emit_new_high_note(self):
        x=audio_fixture()
        heads=WeakHighHead()
        reference=StrictLiveStream(heads)
        shadow=StrictLiveStream(heads,register_rescue_mode="shadow",
                    register_rescue_policy=LIVE_PRESETS["high_moderate"])
        guard=StrictLiveStream(heads,register_rescue_mode="guard",
                    register_rescue_policy=LIVE_PRESETS["high_moderate"])
        original=replay(reference,x)
        unchanged=replay(shadow,x)
        attacked=replay(guard,x)
        self.assertEqual(signatures(original),signatures(unchanged))
        self.assertEqual(len([e for e in original if e["type"]=="note_on"]),0)
        self.assertGreater(len([e for e in attacked if e["type"]=="note_on" and e["pitch"]==80]),0)
        self.assertGreater(shadow.register_rescue_stats["rescue_high_candidates"],0)
        for e in attacked:
            self.assertGreaterEqual(e["available_sample"],e["decision_sample"])

    def test_batched_shadow_equals_off_guard_can_emit_new_high_note(self):
        x=audio_fixture()
        heads=WeakHighHead()
        reference=BatchedLiveStream(heads,batch_hops=4)
        shadow=BatchedLiveStream(heads,batch_hops=4,
                    register_rescue_mode="shadow",
                    register_rescue_policy=LIVE_PRESETS["high_moderate"])
        guard=BatchedLiveStream(heads,batch_hops=4,
                    register_rescue_mode="guard",
                    register_rescue_policy=LIVE_PRESETS["high_moderate"])
        original=replay(reference,x)
        unchanged=replay(shadow,x)
        attacked=replay(guard,x)
        self.assertEqual(signatures(original),signatures(unchanged))
        self.assertEqual(len([e for e in original if e["type"]=="note_on"]),0)
        self.assertGreater(len([e for e in attacked if e["type"]=="note_on" and e["pitch"]==80]),0)
        self.assertGreater(shadow.stats["rescue_high_candidates"],0)
        self.assertEqual(guard.stats["frames_selected"],len(x)//HOP)

    def test_frozen_real_native_shadow_exact_parity(self):
        heads=load_heads(DEFAULT_MODEL)
        x=audio_fixture()
        for factory in (
            lambda kw: StrictLiveStream(heads,**kw),
            lambda kw: BatchedLiveStream(heads,batch_hops=4,**kw),
        ):
            original=replay(factory({}),x)
            shadow=replay(factory(dict(register_rescue_mode="shadow",
                  register_rescue_policy=LIVE_PRESETS["bass_k50_high_moderate"])),x)
            self.assertEqual(signatures(original),signatures(shadow))

    def test_reject_lookahead_guard_without_validation(self):
        with self.assertRaisesRegex(ValueError,"zero acoustic lookahead"):
            BatchedLiveStream(WeakHighHead(),batch_hops=4,
                lookahead=LookaheadConfig(milliseconds=5),
                register_rescue_mode="guard",
                register_rescue_policy=LIVE_PRESETS["high_moderate"])

    def test_real_cli_wav_batch4_register_shadow_guard_and_invalid_future(self):
        x=audio_fixture()
        with tempfile.TemporaryDirectory() as td:
            td=pathlib.Path(td)
            source=td/"test.wav"
            with wave.open(str(source),"wb") as f:
                f.setnchannels(1);f.setframerate(SR);f.setsampwidth(2)
                f.writeframes((x*32767).astype("<i2").tobytes())
            outputs=[]
            for mode in ("off","shadow","guard"):
                out=td/(mode+".jsonl")
                command=[sys.executable,"-m","scripts.live_strict_notes",
                    "--wav",str(source),"--out",str(out),
                    "--hop-size","256","--block-size","256",
                    "--lookahead-ms","0","--inference-batch-hops","4",
                    "--register-rescue-mode",mode,
                    "--register-rescue-policy","high_moderate"]
                result=subprocess.run(command,cwd=REPO,capture_output=True,
                                      text=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stderr[-2000:])
                outputs.append([json.loads(row) for row in out.read_text().splitlines()])
            self.assertEqual(signatures(outputs[0]),signatures(outputs[1]))
            result=subprocess.run([sys.executable,"-m","scripts.live_strict_notes",
                 "--status","--hop-size","256","--lookahead-ms","5",
                 "--inference-batch-hops","4","--register-rescue-mode","guard"],
                 cwd=REPO,capture_output=True,text=True,timeout=30)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("requires lookahead-ms 0",result.stderr)


if __name__ == "__main__":
    unittest.main()
