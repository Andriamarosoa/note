"""S634 tests: queue safety, identical recognition without gaps, panic recovery.

No dependence on Windows or a physical microphone; hardware loopback is
explicitly outside this suite. The original NOTE S614 selector is unchanged.
"""
from __future__ import annotations

import unittest
import numpy as np

from causal_note.strict_live import HOP, SR, StrictLiveStream
from causal_note.s634_safe_live_transport import SafeLiveTransport, TransportConfig


class FixedHeads:
    metadata={"selection":{"refractory_frames":2,"release_frames":3,
                           "onset":.6,"active_on":.5}}
    def predict(self,features):
        assert features.shape==(49,126)
        probability=np.zeros((49,3),np.float32)
        probability[10,0]=.89
        probability[10,1]=.91
        k=np.eye(7,dtype=np.float32)[1]
        return probability,k


def replay(signal, config=TransportConfig(), max_blocks=1):
    out=[]
    bridge=SafeLiveTransport(
        lambda:StrictLiveStream(FixedHeads()),
        out.append,config=config)
    for start in range(0,len(signal),HOP):
        bridge.capture(signal[start:start+HOP].copy(),start_sample=start)
        bridge.poll(max_blocks=max_blocks)
    bridge.close()
    return out,bridge.report()


class TransportTests(unittest.TestCase):
    def test_empty_close_is_idempotent(self):
        received=[]
        host=SafeLiveTransport(lambda:StrictLiveStream(FixedHeads()),
                               received.append)
        host.close();host.close()
        self.assertEqual(received,[])
        with self.assertRaises(RuntimeError):
            host.capture(np.zeros(HOP,np.float32),start_sample=0)

    def test_lossless_emissions_identical_to_original(self):
        samples=np.zeros(45*HOP,np.float32)
        reference=StrictLiveStream(FixedHeads())
        expected=[]
        for start in range(0,len(samples),HOP):
            expected.extend(reference.push(samples[start:start+HOP],start_sample=start))
        expected.extend(reference.finish())
        actual,summary=replay(samples)
        self.assertEqual(actual,expected)
        self.assertEqual(summary["recoveries"],0)
        self.assertEqual(summary["queue_drops"],0)
        self.assertEqual(summary["gap_samples"],0)
        self.assertLessEqual(len(actual),2)

    def test_hard_gap_closes_notes_and_starts_new_epoch(self):
        samples=np.zeros(HOP,np.float32)
        out=[]
        host=SafeLiveTransport(lambda:StrictLiveStream(FixedHeads()),out.append)
        host.capture(samples,start_sample=0)
        host.poll()
        host.capture(samples,start_sample=4*HOP)  # three lost capture blocks
        host.poll()
        host.close()
        types=[e["type"] for e in out]
        self.assertEqual(types,["note_on","note_off","note_on","note_off"])
        self.assertEqual(out[1]["reason"],"transport_capture_gap")
        self.assertGreaterEqual(out[1]["available_sample"],4*HOP)
        self.assertGreaterEqual(out[2]["decision_sample"],4*HOP)
        self.assertTrue(all(e["decision_sample"]<=e["available_sample"] for e in out))
        self.assertEqual(host.report()["recoveries"],1)

    def test_invalid_capture_panics_before_next_valid_audio(self):
        events=[]
        host=SafeLiveTransport(lambda:StrictLiveStream(FixedHeads()),events.append)
        block=np.zeros(HOP,np.float32)
        host.capture(block,start_sample=0);host.poll()
        self.assertFalse(host.capture(np.full(HOP,np.nan,dtype=np.float32),
                                      start_sample=HOP))
        host.capture(block,start_sample=2*HOP)
        host.poll();host.close()
        self.assertEqual([v["type"] for v in events],
                         ["note_on","note_off","note_on","note_off"])
        self.assertEqual(events[1]["reason"],"transport_invalid_capture")
        self.assertEqual(host.report()["invalid_blocks"],1)

    def test_bounded_queue_discards_old_audio_never_infers_stale_block(self):
        events=[]
        host=SafeLiveTransport(lambda:StrictLiveStream(FixedHeads()),events.append,
                              TransportConfig(queue_capacity=6,
                                              max_backlog_blocks=2))
        block=np.zeros(HOP,np.float32)
        host.capture(block,start_sample=0);host.poll()
        for frame in range(1,9):
            host.capture(block,start_sample=frame*HOP)
        # Old producer blocks should be skipped and the old note closed.
        host.poll()
        host.close()
        self.assertEqual([v["type"] for v in events],
                         ["note_on","note_off","note_on","note_off"])
        self.assertGreater(host.report()["queue_drops"],0)
        self.assertGreater(host.report()["backlog_drops"],0)
        self.assertGreaterEqual(host.report()["recoveries"],1)

    def test_microphone_stall_force_notes_off(self):
        times=[0.]
        event=[]
        host=SafeLiveTransport(lambda:StrictLiveStream(FixedHeads()),event.append,
                  config=TransportConfig(stall_seconds=.5),
                  clock=lambda:times[0])
        block=np.zeros(HOP,np.float32)
        host.capture(block,start_sample=0);host.poll()
        times[0]=.49
        self.assertFalse(host.check_stall())
        times[0]=.51
        self.assertTrue(host.check_stall())
        self.assertEqual(event[-1]["type"],"note_off")
        self.assertEqual(event[-1]["reason"],"transport_microphone_stall")
        self.assertEqual(host.report()["recoveries"],1)
        host.close()

    def test_missing_audio_cannot_be_added_as_silence(self):
        rng=np.random.default_rng(634)
        block=rng.normal(size=HOP).astype(np.float32)*.04
        event=[]
        host=SafeLiveTransport(lambda:StrictLiveStream(FixedHeads()),event.append)
        host.capture(block,start_sample=0);host.poll()
        host.capture(block,start_sample=HOP*5)
        host.poll()
        host.close()
        self.assertTrue(any(e.get("reason")=="transport_capture_gap" for e in event))
        self.assertTrue(all(e["available_sample"]>=e["decision_sample"] for e in event))

    def test_real_checkpoint_matches_s614_reference(self):
        from scripts.live_strict_notes import DEFAULT_MODEL
        from causal_note.strict_live import load_heads
        head=load_heads(DEFAULT_MODEL)
        rng=np.random.default_rng(634)
        audio=rng.normal(0,.02,70*HOP).astype(np.float32)
        original=StrictLiveStream(head)
        baseline=[]
        for start in range(0,len(audio),HOP):
            baseline.extend(original.push(audio[start:start+HOP],start_sample=start))
        baseline.extend(original.finish())
        protected,metric=replay_with_factory(audio,lambda:StrictLiveStream(head))
        self.assertEqual(protected,baseline)
        self.assertEqual(metric["recoveries"],0)


def replay_with_factory(audio,factory):
    output=[]
    s=SafeLiveTransport(factory,output.append)
    for start in range(0,len(audio),HOP):
        s.capture(audio[start:start+HOP],start_sample=start)
        s.poll()
    s.close()
    return output,s.report()


if __name__=="__main__":
    unittest.main()
