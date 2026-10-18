"""S732 independent sound-source discrimination contracts."""
import unittest
import numpy as np
from causal_note.s732_source_discriminator import (
    observation_matrix,sampled_fit_rows,fit_discriminator,
    apply_evidence,predict_track_causal,N_FEATURES)
from causal_note.s731_auditory_scene import AuditoryScene


class SourceDiscriminatorTests(unittest.TestCase):
    def base(self,n=9):
        x=np.zeros((n,49,288),np.float32)
        x[:,:,192:216]=2.
        p=np.zeros((n,49,3),np.float32)
        p[:,:,0]=.2;p[:,:,1]=.65
        k=np.zeros((n,7),np.float32);k[:,2]=.9;k[:,1]=.1
        return x,p,k

    def test_all_physical_evidence_available_from_past_only(self):
        x,p,k=self.base(13)
        e=[dict(type="note_on",pitch=45,
                decision_sample=3072,available_sample=3072),
           dict(type="note_off",pitch=45,
                decision_sample=5120,available_sample=8192)]
        ends=np.arange(1,14)*1024
        f=observation_matrix(x,p,k,e,ends)
        self.assertEqual(f.shape,(13,49,N_FEATURES))
        self.assertTrue(np.isfinite(f).all())
        alt=x.copy();alt[9:]*=4.
        new=observation_matrix(alt,p,k,e,ends)
        np.testing.assert_array_equal(f[:9],new[:9])
        self.assertEqual(f[6,17,3],1.)
        self.assertEqual(f[8,17,3],0.)
        self.assertEqual(f[5,17,3],1.)

    def test_independent_starts_not_dropped_from_FIT_supervision(self):
        rng=np.random.default_rng(732)
        f=rng.random((230,49,N_FEATURES),dtype=np.float32)
        y=np.zeros((230,49),np.uint8)
        y[20,3]=1;y[45,18]=1;y[95,28]=1
        x,t,w=sampled_fit_rows(f,y,seed=732,max_rows=3000)
        self.assertEqual(int(t.sum()),3)
        self.assertEqual(x.shape[1],N_FEATURES)
        self.assertEqual(len(x),len(w))
        self.assertTrue(np.all(w>=1))
        self.assertLessEqual(len(x),3000)

    def test_full_scene_and_control_fit_real_different_sources(self):
        rng=np.random.default_rng(732)
        f=rng.normal(size=(2200,N_FEATURES)).astype(np.float32)
        labels=(f[:,4]+f[:,6]-.8*f[:,5]>0.).astype(np.uint8)
        no=fit_discriminator(f,labels,np.ones(len(f)),with_scene=False)
        yes=fit_discriminator(f,labels,np.ones(len(f)),with_scene=True)
        p1=yes.predict_proba(f)[:,1]
        self.assertGreater(float(((p1>=.5)==labels).mean()),.80)
        self.assertEqual(len(no.predict_proba(f[:,(0,1,2,9,14,15,17,18,19,20)])),len(f))

    def test_K_rejects_uncorroborated_hypothesis_and_release_untouched(self):
        old=np.zeros((49,3),np.float32)
        old[:,0]=.15;old[:,1]=.7;old[:,2]=.1
        x=np.zeros((49,288),np.float32)
        state=np.zeros(49,bool);state[0]=True
        one=np.zeros(7,np.float32);one[1]=1.
        two=np.zeros(7,np.float32);two[2]=1.
        m=AuditoryScene();m.step(x,state,one)
        e1=m.step(x,state,one)
        e2=m.step(x,state,two)
        q=np.ones(49,np.float32)
        out=apply_evidence(old,one,e1,q,gain=.6)
        np.testing.assert_array_equal(out,old)
        boosted=apply_evidence(old,two,e2,q,gain=.6)
        self.assertGreater(float(boosted[12,0]),float(old[12,0]))
        np.testing.assert_array_equal(boosted[:,1:],old[:,1:])

    def test_no_external_labels_or_future_audio_in_inference_code(self):
        x,p,k=self.base()
        f=observation_matrix(x,p,k,[],np.arange(1,10)*1024)
        r=np.random.default_rng(732)
        f[:,:,-1]=r.random((len(f),49))
        xx,yy,ww=sampled_fit_rows(f,np.zeros((len(f),49),np.uint8),seed=3)
        # API must not accept labels at prediction; only FIT receives y.
        self.assertEqual(f.shape,(9,49,N_FEATURES))
        with self.assertRaisesRegex(ValueError,"ABI"):
            apply_evidence(p[0],k[0],AuditoryScene().step(x[0],np.zeros(49,bool),k[0]),
                           np.zeros(48))


if __name__=="__main__":
    unittest.main()
