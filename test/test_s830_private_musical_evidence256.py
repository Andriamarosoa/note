"""S830 independent musical evidence tests (synthetic, not musical validation)."""
import unittest
import numpy as np
from causal_note.s829_exactk_s809_native256 import (
    HOP, fit_exact, summarize_one
)
from causal_note.s830_private_musical_evidence256 import (
    SCHEMA,S830PCMAcoustics,S830FittedStream,LearnedEvidence,
    pitch_features,truth_pitches_to_three,train_evidence,
    counts_from_evidence,selected,choose_selector,apply
)


def synthetic(name,comp,seed,n=168):
    rng=np.random.default_rng(seed)
    audio=rng.normal(0,.1,(n,49,72)).astype(np.float16)
    labels=np.zeros((n,49),bool)
    for pitch in range(49):
        for start in range((pitch%11),n,27+(pitch%7)):
            labels[start:start+4+(pitch%9),pitch]=True
    k=np.minimum(labels.sum(axis=1),6).astype(np.uint8)
    acoustic_k=np.minimum(np.count_nonzero(audio[:,:,0]>.2,axis=1),6).astype(np.uint8)
    x=np.stack([summarize_one(audio[t].astype(np.float32),int(acoustic_k[t])) for t in range(n)])
    return dict(name=name,composition=comp,X=x,baseline=acoustic_k,
                pitch_features=audio,pitch_target=labels,k=k)


class TestS830(unittest.TestCase):
    def test_3_truth_labels_are_distinct(self):
        occ=np.zeros((4,49),bool)
        occ[1:3,0]=True
        y=truth_pitches_to_three(occ)
        self.assertEqual(y["onset"][:,0].astype(int).tolist(),[0,1,0,0])
        self.assertEqual(y["sustain"][:,0].astype(int).tolist(),[0,1,1,0])
        self.assertEqual(y["release"][:,0].astype(int).tolist(),[0,0,0,1])
        self.assertEqual(pitch_features(np.zeros((1,49,72),np.float32)).shape,(1,49,73))

    def test_independent_3_models_and_per_fragment_selector(self):
        train=[synthetic("a","trainA",1),synthetic("b","trainB",2)]
        cal=[synthetic("c","CAL",3)]
        evidence=train_evidence(train,seed=830)
        self.assertEqual(evidence.meta["schema"],SCHEMA)
        self.assertFalse(evidence.meta["shared_model_weights"])
        self.assertEqual(len({id(getattr(evidence,k)) for k in ("onset","sustain","release")}),3)
        p=evidence.posterior(cal[0]["pitch_features"][:9],batch_frames=4)
        self.assertEqual(p.shape,(9,49,3))
        self.assertTrue(np.all((p>=0)&(p<=1)))
        d,h=counts_from_evidence(p,.5)
        self.assertEqual((d.shape,h.shape),( (9,),(9,) ))
        base=np.zeros(9,np.uint8)
        confidences=np.full(9,.8)
        self.assertEqual(len(selected(base,d,h,confidences,"router")),9)
        best,_=choose_selector([dict(
            base=base,evidence=p,confidence=confidences,
            k=np.zeros(9,np.uint8))])
        self.assertEqual(len(best),3)

    def test_pcm256_true_clock_and_no_future_change(self):
        rng=np.random.default_rng(830)
        audio=rng.normal(0,.04,7*HOP).astype(np.float32)
        changed=audio.copy()
        changed[4*HOP:]*=-1
        kept=[]
        for x in (audio,changed):
            stream=S830PCMAcoustics()
            result=[]
            for i in range(0,len(x),128):
                result+=list(stream.push(x[i:i+128],start_sample=i))
            self.assertEqual(stream.calls,7)
            self.assertEqual([r[0] for r in result],list(range(HOP,8*HOP,HOP)))
            kept.append(np.stack([r[1] for r in result]))
        np.testing.assert_array_equal(kept[0][:4],kept[1][:4])

    def test_fitted_online_is_same_as_batch_without_future(self):
        tr=[synthetic("a","c1",21),synthetic("b","c2",22),
            synthetic("c","c3",23)]
        cal=[synthetic("d","cal",24)]
        kfit=fit_exact(tr,cal,seed=830)
        musical=train_evidence(tr,seed=831)
        rng=np.random.default_rng(26)
        sound=rng.normal(0,.12,5*HOP).astype(np.float32)
        ac=S830PCMAcoustics()
        features=[];raw=[];x=[]
        for idx in range(0,len(sound),HOP):
            e,p,guess=ac.push(sound[idx:idx+HOP],start_sample=idx)[0]
            features.append(p);raw.append(guess);x.append(summarize_one(p,guess))
        selector=("joint",.5,0)
        batch,baseline,_=apply(kfit,musical,np.stack(x),np.asarray(raw),
                               np.stack(features),selector)
        live=S830FittedStream(kfit,musical,selector)
        emitted=[]
        for idx in range(0,len(sound),HOP):
            decisions=live.push(sound[idx:idx+HOP],start_sample=idx)
            self.assertEqual(len(decisions),1)
            emitted.extend(decisions)
        self.assertEqual([d.count for d in emitted],batch.tolist())
        self.assertTrue(live.diagnostics()["independent_trained_heads"]==5)
        self.assertEqual(live.last_decision,5*HOP)


if __name__=="__main__":
    unittest.main()
