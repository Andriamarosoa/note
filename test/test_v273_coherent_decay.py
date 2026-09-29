import unittest
import numpy as np
from scripts.probe_v273_coherent_decay import forecast,evidence,make_wave,LONG_HISTORY,POST_SAMPLES


class CoherentDecayTest(unittest.TestCase):
    def test_two_sustained_close_tones_are_predicted_from_past_only(self):
        wave=make_wave([110,116.54094],2.45,noise_db=None,event='none',seed=1)
        p,info=forecast(wave[:LONG_HISTORY],POST_SAMPLES,order=32,lag=64)
        np.testing.assert_allclose(p,wave[LONG_HISTORY:],atol=2e-6)
        self.assertGreater(info['rank'],0)

    def test_damped_partial_mixture_is_predictable(self):
        wave=make_wave([137*h for h in range(1,9)],1.8,noise_db=None,event='none',seed=1,envelope='decaying')
        p,_=forecast(wave[:LONG_HISTORY],POST_SAMPLES,order=32,lag=64)
        np.testing.assert_allclose(p,wave[LONG_HISTORY:],atol=3e-6)

    def test_a_new_attack_does_not_enter_the_fitted_history(self):
        a=make_wave([220,233.08188],1.767,noise_db=40,event='none',seed=1)
        b=make_wave([220,233.08188],1.767,noise_db=40,event='same_pitch',seed=1)
        np.testing.assert_array_equal(a[:LONG_HISTORY],b[:LONG_HISTORY])
        pa,da=forecast(a[:LONG_HISTORY],POST_SAMPLES,order=32,lag=64)
        pb,db=forecast(b[:LONG_HISTORY],POST_SAMPLES,order=32,lag=64)
        np.testing.assert_array_equal(pa,pb);self.assertEqual(da,db)
        ea,_=evidence(a);eb,_=evidence(b)
        self.assertGreater(eb['wave_long_wider_recurrence'],ea['wave_long_wider_recurrence']+.3)

    def test_silence_and_invalid_inputs(self):
        p,_=forecast(np.zeros(LONG_HISTORY),POST_SAMPLES,order=32,lag=64)
        np.testing.assert_array_equal(p,0)
        for x in (np.ones(20),np.full(LONG_HISTORY,np.nan)):
            with self.assertRaises(ValueError):forecast(x,POST_SAMPLES,order=32,lag=64)

    def test_nonmusical_noise_is_also_unpredictable(self):
        a=make_wave([220],0.,noise_db=None,event='none',seed=7)
        b=make_wave([220],0.,noise_db=None,event='noise_burst',seed=7)
        ea,_=evidence(a);eb,_=evidence(b)
        self.assertLess(ea['wave_long_wider_recurrence'],1e-5)
        self.assertGreater(eb['wave_long_wider_recurrence'],.3)
        # Innovation must not be silently equated with a musical note or K.


if __name__=='__main__':unittest.main()
