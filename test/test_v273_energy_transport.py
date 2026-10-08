"""Deterministic and label-free checks of the spectral continuity proxy."""
from __future__ import annotations
import unittest
import numpy as np
from scripts.extract_v273_energy_transport import constrained_transport,observable_from_energy

class EnergyTransportTests(unittest.TestCase):
    def test_perfect_neighbor_transfer(self):
        R=np.zeros((3,8),float)
        R[0,2]=-4.;R[0,3]=4.
        R[1,4]=2.;R[1,5]=-2.
        J,B,D,err=constrained_transport(R)
        self.assertLess(err,1e-12)
        self.assertAlmostEqual(float(np.abs(J).sum()),6.)
        self.assertAlmostEqual(float(B.sum()+D.sum()),0.)
        self.assertAlmostEqual(float(J[0,2]),4.)
        self.assertAlmostEqual(float(J[1,4]),-2.)
    def test_birth_and_sink_are_not_hidden(self):
        R=np.zeros((2,8),float)
        R[0,4]=3.;R[1,4]=-2.
        J,B,D,err=constrained_transport(R)
        self.assertLess(err,1e-12)
        self.assertAlmostEqual(float(J.sum()),0.)
        self.assertAlmostEqual(float(B.sum()),3.)
        self.assertAlmostEqual(float(D.sum()),2.)
    def test_isolated_signals_retain_temporal_information(self):
        t=np.arange(-76,141,3.0)
        E=np.full((len(t),24),1e-5)
        E[:,6]=1.
        idle=observable_from_energy(E,t)
        born=E.copy()
        born[t>=0,13]=2*np.exp(-np.maximum(t[t>=0],0)/160.)
        fb=observable_from_energy(born,t)
        self.assertGreater(fb["birth__total_mass"],idle["birth__total_mass"]+0.5)
        cut=E.copy()
        cut[t>=20,6]=.001
        fd=observable_from_energy(cut,t)
        self.assertGreater(fd["extinction__total_mass"],idle["extinction__total_mass"]+0.5)
        self.assertTrue(np.isfinite(np.array(list(fb.values()),float)).all())
    def test_nonlinear_time_order(self):
        t=np.arange(-76,141,3.0)
        E=np.full((len(t),24),1e-5)
        E[:,6]=1.
        E[t>=0,13]=2*np.exp(-t[t>=0]/160)
        original=observable_from_energy(E,t)
        idx=np.flatnonzero((t>=-8)&(t<=110))
        shuffled=E.copy()
        shuffled[idx]=E[idx[::-1]]
        fake=observable_from_energy(shuffled,t)
        self.assertNotAlmostEqual(
            original["birth__early_fraction"],fake["birth__early_fraction"],places=6
        )

if __name__=="__main__":unittest.main()
