"""S765 FIT-only arbiter tests: strict causality, grouping, source priority."""
import unittest
import numpy as np

from test.test_s764_six_decoder_shadow_cal32 import demo
from scripts.s764_six_decoder_shadow_cal32 import replay_six
from scripts.s765_fit_independent_arbiter import (
    DIM,RISK,candidate_rows,grouped_quality,positive_probability,
    emit_trial,train_one,acoustic_birth_pairs)

class ArbiterTests(unittest.TestCase):
    def fixture(self):
        ends,post,k,source,truth,native=demo()
        item=dict(decision_ends=ends,samples=int(native[-1]),x=np.zeros((len(ends),49,288),
                                                 np.float32),
                  native_post=post.copy())
        ctx=np.zeros((len(ends),49,16),np.float32)
        shadow=replay_six(ends,post,k,source,int(native[-1]))
        return item,post,k,source,shadow,ctx

    def test_all_six_experts_make_a_full_note_40d_candidate(self):
        item,post,k,source,shadow,ctx=self.fixture()
        pairs,X,indices,stats=candidate_rows(item,post,k,source,shadow,ctx)
        self.assertEqual(len(pairs),1)
        self.assertEqual(pairs[0][0]["pitch"],52)
        self.assertEqual(indices,[(3,12)])
        self.assertEqual(X.shape,(1,DIM))
        self.assertGreater(stats["source_active"],0)
        self.assertGreater(stats["eligible"],0)
        self.assertTrue(np.isfinite(X).all())

    def test_acoustic_independent_birth_when_fullnote_proposed_none(self):
        item,post,k,source,shadow,ctx=self.fixture()
        # An acoustically coherent low-onset/high-novelty pitch that the
        # FullNote attack threshold deliberately missed.
        pitch=56
        ix=pitch-40
        post=post.copy();ctx=ctx.copy()
        candidate=np.zeros((len(post),49,5),np.float32)
        post[2:5,ix,0]=.11
        post[2:5,ix,1]=.78
        post[5,ix,2]=.96
        ctx[2,ix,4]=.15
        ctx[2,ix,7]=.375
        ctx[2,ix,8]=.06
        candidate[2,ix,4]=.82
        pairs,stats=acoustic_birth_pairs(item,post,candidate,ctx)
        self.assertEqual(stats["acoustic_opened"],1)
        self.assertEqual(stats["acoustic_closed"],1)
        self.assertEqual(len(pairs),1)
        self.assertEqual(pairs[0][0]["pitch"],pitch)
        self.assertEqual(pairs[0][0]["available_sample"],
                         int(item["decision_ends"][2]))
        self.assertEqual(pairs[0][1]["available_sample"],
                         int(item["decision_ends"][5]))
        selected,X,indices,counts=candidate_rows(
            item,post,k,source,shadow,ctx,proposals=candidate)
        self.assertTrue(any(on["pitch"]==pitch for on,off in selected))
        self.assertGreaterEqual(counts["acoustic_opened"],1)
        self.assertEqual(X.shape[1],DIM)
        self.assertTrue(np.isfinite(X).all())

    def test_future_changes_do_not_affect_acoustic_birth_decision(self):
        item,post,k,source,shadow,ctx=self.fixture()
        post=post.copy();ctx=ctx.copy()
        q=np.zeros((len(post),49,5),np.float32)
        ix=16
        post[1:5,ix,0]=.10
        post[1:5,ix,1]=.75
        ctx[1,ix,4]=.14
        ctx[1,ix,7]=.25
        q[1,ix,4]=.82
        a,_=acoustic_birth_pairs(item,post,q,ctx)
        modified=post.copy()
        modified[4:,ix,2]=.99
        modified[4:,ix,1]=0.
        b,_=acoustic_birth_pairs(item,modified,q,ctx)
        self.assertEqual([(on["pitch"],on["available_sample"])
                          for on,off in a],
                         [(on["pitch"],on["available_sample"])
                          for on,off in b])
        self.assertNotEqual([off["available_sample"] for on,off in a],
                            [off["available_sample"] for on,off in b])

    def test_future_signed_source_event_cannot_change_present_decision(self):
        item,post,k,source,shadow,ctx=self.fixture()
        a=candidate_rows(item,post,k,source,shadow,ctx)
        future=dict(type="note_on",pitch=52,
                    decision_sample=int(item["decision_ends"][4]),
                    available_sample=int(item["decision_ends"][4]),
                    time=float(item["decision_ends"][4]/44100.),
                    reason="signed_future_on",velocity=96)
        future_off=dict(type="note_off",pitch=52,
                    decision_sample=int(item["decision_ends"][5]),
                    available_sample=int(item["decision_ends"][5]),
                    time=float(item["decision_ends"][5]/44100.),
                    reason="signed_future_off",velocity=0)
        b=candidate_rows(item,post,k,source+[future,future_off],shadow,ctx)
        self.assertEqual(a[2],b[2])
        np.testing.assert_array_equal(a[1],b[1])
        self.assertEqual([x[0] for x in a[0]],[x[0] for x in b[0]])

    def test_future_candidate_release_does_not_affect_birth_features(self):
        item,post,k,source,shadow,ctx=self.fixture()
        a=candidate_rows(item,post,k,source,shadow,ctx)
        future=dict(shadow)
        newer=[]
        for x in future["full"]:
            event=dict(x)
            if event["type"]=="note_off":
                event["available_sample"]+=4096
                event["decision_sample"]+=4096
                event["time"]=event["available_sample"]/44100.
            newer.append(event)
        future["full"]=newer
        b=candidate_rows(item,post,k,source,future,ctx)
        self.assertEqual(a[2],b[2])
        np.testing.assert_array_equal(a[1],b[1])

    def test_grouped_oof_on_training_only(self):
        rng=np.random.default_rng(765)
        X=rng.normal(size=(100,DIM)).astype(np.float32)
        labels=(X[:,0]+X[:,1]>0).astype(np.int32)
        groups=np.repeat(["BN1","Jazz1","Funk2","Rock3","SS1"],20)
        risk,cv=grouped_quality(X,labels,labels,groups)
        self.assertIn(risk,RISK)
        self.assertEqual(cv["folds"],5)
        self.assertEqual(cv["coverage"],len(X))
        self.assertEqual(cv["positive_complete"],int(labels.sum()))
        self.assertIsNotNone(train_one(X,labels))
        self.assertIsNone(train_one(X,np.zeros(len(X),np.int32)))

    def test_positive_probability_fails_closed(self):
        X=np.ones((3,DIM),np.float32)
        np.testing.assert_array_equal(positive_probability(None,X),
                                      np.zeros(3,np.float32))

    def test_s708_source_preserved_under_additive_fusion(self):
        item,post,k,source,shadow,ctx=self.fixture()
        pairs,X,indices,stats=candidate_rows(item,post,k,source,shadow,ctx)
        # A zero-risk gate tests the mixer mechanics, not learned promotion.
        merged,counts=emit_trial(source,pairs,X,indices,k,dict(
            onset=None,complete=None),risk=0.)
        original=[e for e in merged if e["reason"].startswith("signed_")]
        self.assertEqual(original,source)
        self.assertEqual(counts["accepted"],1)

if __name__=="__main__":
    unittest.main()
