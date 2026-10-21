"""S828 fail-closed private Active-K training contracts (synthetic smoke only)."""
import unittest
import numpy as np
from causal_note.s828_activek_s708_private_readout import (
    DIM_SELECT,S828Stream,summarize_one,fit_private,proposed,SCHEMA
)
from causal_note.s827_s708_activek_only import S827S708ActiveKOnly
from test.test_s827_s708_activek_only import model,policy,StubNative,wave


def row(name,comp,seed,n=147):
    rng=np.random.default_rng(seed)
    y=np.arange(n,dtype=np.uint8)%7
    rng.shuffle(y)
    x=rng.normal(0,1,(n,3*len(DIM_SELECT)+6)).astype(np.float32)
    x[:,0]=y+.4*rng.normal(size=n)
    b=np.clip(y+rng.integers(-1,2,size=n),0,6).astype(np.uint8)
    return dict(name=name,composition=comp,X=x,k=y,baseline=b)


class TestPrivateS828(unittest.TestCase):
    def test_source_abi_current_frame_and_history(self):
        x=np.ones((49,288),np.float32)
        a=summarize_one(x,[0,1,2,3],3,2,.8)
        self.assertEqual(len(a),3*len(DIM_SELECT)+6)
        self.assertTrue(np.isfinite(a).all())
        with self.assertRaisesRegex(ValueError,"288"):
            summarize_one(x[:,:-1],[0,1,2,3],3,2,.8)

    def test_one_native256_neural_and_private_vector_per_hop(self):
        nn=model()
        audio=wave(8)
        a=S828Stream(StubNative(),nn,policy())
        b=S827S708ActiveKOnly(StubNative(),nn,policy())
        for i in range(0,len(audio),128):
            a.push(audio[i:i+128],start_sample=i)
            b.push(audio[i:i+128],start_sample=i)
        x,old=a.collected()
        self.assertEqual(x.shape,(8,3*len(DIM_SELECT)+6))
        self.assertEqual(a.model_calls,8)
        self.assertEqual(b.model_calls,8)
        self.assertEqual(len(a.feature_debug),0)
        self.assertEqual(old.tolist(),a.s708_baselines)
        self.assertEqual(a.decoder.active.tolist(),b.decoder.active.tolist())

    def test_independent_train_calibration_and_only_private_counts(self):
        train=[row("a","train_a",1),row("b","train_b",2),
               row("c","train_c",3)]
        cal=[row("d","cal",4)]
        val=row("e","heldout",5)
        fit=fit_private(train,cal)
        self.assertEqual(fit.metadata["train"],["train_a","train_b","train_c"])
        self.assertEqual(fit.metadata["calibration"],["cal"])
        self.assertTrue(fit.metadata["no_other_axis_model_trained"])
        self.assertTrue(fit.metadata["frozen_s708_original_weights"])
        self.assertEqual(fit.metadata["training_hop_samples"],256)
        p=fit.predict(val["X"],val["baseline"])
        self.assertEqual(p.shape,val["k"].shape)
        self.assertFalse(p.flags.writeable)
        with self.assertRaisesRegex(ValueError,"composition-disjoint"):
            fit_private([train[0],cal[0]],cal)
        for name in ("s708_private","learned_private","confidence_router"):
            probs=np.zeros((3,7),np.float32);probs[:,2]=1
            p=proposed(probs,np.array([0,1,2]),name,0)
            self.assertTrue(np.all((p>=0)&(p<=6)))

    def test_no_future_readout_synthetic_stream(self):
        audio=wave(9)
        altered=audio.copy()
        altered[4*256:]=0
        nn=model()
        xs=[]
        for samples in (audio,altered):
            t=S828Stream(StubNative(),nn,policy())
            for j in range(0,len(samples),256):
                t.push(samples[j:j+256],start_sample=j)
            xs.append(t.collected()[0])
        np.testing.assert_array_equal(xs[0][:4],xs[1][:4])


if __name__=="__main__":
    unittest.main()
