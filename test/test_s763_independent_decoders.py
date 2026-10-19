"""S763 contract: six separately stateful decoders cannot regress one another."""
import unittest
import numpy as np

from causal_note.s763_independent_decoders import (
    AXES, DECISION_SAMPLES, FIRST, VOICES, Fragment, Intent, AxisDecision,
    ExactKDecoder, ActiveKDecoder, OnsetDecoder, OffsetDecoder, FullNoteDecoder,
    IsolatedDecoderBank, ShadowArbiter, default_decoders)


def fragment(t, attack=(), release=(), active=(), k=2):
    post=np.zeros((49,3),np.float32)
    for pitch in active:
        post[pitch-FIRST,1]=.88
    for pitch in attack:
        post[pitch-FIRST,0]=.99
        post[pitch-FIRST,1]=.91
    for pitch in release:
        post[pitch-FIRST,2]=.95
        post[pitch-FIRST,1]=.10
    probs=np.zeros(7,np.float32);probs[k]=1.
    return Fragment(t*DECISION_SAMPLES,post,probs)


def demo():
    return [
        fragment(1,attack=(40,52),active=(40,52),k=2),
        fragment(2,active=(40,52),k=2),
        fragment(3,release=(40,),active=(52,),k=1),
        fragment(4,release=(52,),k=0),
        fragment(5,attack=(45,),active=(45,),k=1),
        fragment(6,active=(45,),k=1),
    ]


class MaliciousK(ExactKDecoder):
    def step(self, f):
        # Simulates a K decoder that mutates every input channel.
        # Other decoders cannot see this mutation.
        f.post.setflags(write=True)
        f.k.setflags(write=True)
        f.post[:]=1.
        f.k[:]=0.
        f.k[0]=1.
        return AxisDecision("exact_k",f.sample,count=0,confidence=1.)


class InvalidFutureOnset(OnsetDecoder):
    def step(self, f):
        return AxisDecision("onset",f.sample,intents=(
            Intent("note_on",40,f.sample+DECISION_SAMPLES,.9,"illegal"),))


class IndependentDecoderTests(unittest.TestCase):
    def test_all_six_axes_are_real_instances(self):
        instances=default_decoders()
        self.assertEqual(set(instances),set(AXES))
        self.assertEqual(len({id(v) for v in instances.values()}),6)

    def test_input_immutable_and_forked(self):
        f=demo()[0];g=f.fork()
        self.assertFalse(f.post.flags.writeable)
        self.assertFalse(f.k.flags.writeable)
        self.assertFalse(np.shares_memory(f.post,g.post))
        self.assertFalse(np.shares_memory(f.k,g.k))
        with self.assertRaises(ValueError):
            f.post[0,0]=.2
        a=np.zeros((49,3),np.float32)
        k=np.array([1.,0,0,0,0,0,0],np.float32)
        snapshot=Fragment(DECISION_SAMPLES,a,k)
        a[:]=1.;k[:]=0
        self.assertEqual(float(snapshot.post.sum()),0.)
        self.assertEqual(float(snapshot.k[0]),1.)

    def test_six_decoders_emit_real_distinct_outputs(self):
        bank=IsolatedDecoderBank()
        x=bank.step(demo()[0])
        self.assertEqual(set(x),set(AXES))
        self.assertEqual(x["exact_k"].count,2)
        self.assertEqual(x["active_exact_k"].count,2)
        self.assertEqual(x["frame_pitch"].pitches,(40,52))
        self.assertEqual({v.pitch for v in x["onset"].intents},{40,52})
        self.assertFalse(x["offset"].intents)
        self.assertEqual({v.pitch for v in x["full_note"].intents},{40,52})

    def test_changing_k_decoder_or_poisoning_input_cannot_change_other_axes(self):
        baseline=IsolatedDecoderBank()
        mutated=default_decoders();mutated["exact_k"]=MaliciousK()
        bank=IsolatedDecoderBank(mutated)
        for f in demo():
            before=baseline.step(f)
            after=bank.step(f)
            self.assertEqual(after["exact_k"].count,0)
            for name in AXES:
                if name!="exact_k":
                    self.assertEqual(after[name],before[name],name)

    def test_changing_onset_threshold_does_not_change_offset_or_full_note(self):
        basic=IsolatedDecoderBank()
        modified=default_decoders()
        modified["onset"]=OnsetDecoder(threshold=1.,sustain=1.)
        bank=IsolatedDecoderBank(modified)
        for f in demo():
            a,b=basic.step(f),bank.step(f)
            for name in AXES:
                if name!="onset":
                    self.assertEqual(a[name],b[name],name)

    def test_changing_offset_policy_does_not_change_onset_or_full_note(self):
        basic=IsolatedDecoderBank()
        modified=default_decoders()
        modified["offset"]=OffsetDecoder(release=1.,low_sustain=0.)
        bank=IsolatedDecoderBank(modified)
        for f in demo():
            a,b=basic.step(f),bank.step(f)
            for name in AXES:
                if name!="offset":
                    self.assertEqual(a[name],b[name],name)

    def test_execution_order_cannot_change_any_decoder(self):
        a=IsolatedDecoderBank(order=AXES)
        b=IsolatedDecoderBank(order=tuple(reversed(AXES)))
        for f in demo():
            self.assertEqual(a.step(f),b.step(f))

    def test_future_frames_cannot_change_emitted_prefix(self):
        original=demo()
        changed=demo()
        altered=np.zeros((49,3),np.float32)
        altered[:,0]=1.;altered[:,1]=.8
        replacement=Fragment(5*DECISION_SAMPLES,altered,
                             np.array([0,0,0,0,0,1,0],np.float32))
        changed[4]=replacement
        a=IsolatedDecoderBank();b=IsolatedDecoderBank()
        for i in range(4):
            self.assertEqual(a.step(original[i]),b.step(changed[i]))

    def test_full_note_manages_own_open_close_identity(self):
        bank=IsolatedDecoderBank()
        emitted=[]
        for f in demo():
            emitted.extend(bank.step(f)["full_note"].intents)
        opened={};pairs=[]
        for event in emitted:
            if event.kind=="note_on":
                self.assertNotIn(event.pitch,opened)
                opened[event.pitch]=event.sample
            else:
                self.assertIn(event.pitch,opened)
                self.assertGreater(event.sample,opened.pop(event.pitch))
                pairs.append(event.pitch)
        self.assertEqual(set(pairs),{40,52})
        self.assertEqual(set(opened),{45})
        final=bank.decoders["full_note"].finish(7*DECISION_SAMPLES)
        self.assertEqual(len(final),1)
        self.assertEqual(final[0].pitch,45)
        self.assertEqual(final[0].reason,"stream_end")

    def test_noncausal_clock_and_future_events_fail(self):
        with self.assertRaises(ValueError):
            Fragment(DECISION_SAMPLES+1,np.zeros((49,3)),np.eye(7)[0])
        with self.assertRaises(ValueError):
            Fragment(DECISION_SAMPLES,np.zeros((49,3)),np.ones(7))
        with self.assertRaises(ValueError):
            IsolatedDecoderBank().step(demo()[1])
        bad=default_decoders();bad["onset"]=InvalidFutureOnset()
        with self.assertRaises(AssertionError):
            IsolatedDecoderBank(bad).step(demo()[0])

    def test_arbiter_reports_conflicts_without_editing_source_midi(self):
        bank=IsolatedDecoderBank()
        arbiter=ShadowArbiter()
        x=bank.step(demo()[0])
        out=arbiter.observe(x)
        self.assertEqual(out.exact_k,2)
        self.assertEqual(out.active_k,2)
        self.assertEqual(out.active_pitch_count,2)
        self.assertEqual(out.onset_proposals,2)
        signed=[
            dict(type="note_on",pitch=40,available_sample=DECISION_SAMPLES,
                 decision_sample=DECISION_SAMPLES,time=DECISION_SAMPLES/44100.,
                 reason="signed_S708"),
            dict(type="note_off",pitch=40,available_sample=2*DECISION_SAMPLES,
                 decision_sample=2*DECISION_SAMPLES,time=2*DECISION_SAMPLES/44100.,
                 reason="signed_S707")]
        kept=arbiter.preserve_signed_stream(signed)
        self.assertEqual(kept,signed)
        self.assertIsNot(kept[0],signed[0])
        kept[0]["pitch"]=42
        self.assertEqual(signed[0]["pitch"],40)

    def test_decoder_registration_rejects_shared_state(self):
        a=default_decoders()
        a["offset"]=a["onset"]
        with self.assertRaises(ValueError):
            IsolatedDecoderBank(a)

if __name__=="__main__":
    unittest.main()
