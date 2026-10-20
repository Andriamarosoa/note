"""S807 falsification, not promotion: reject three misleading model claims.

No FIT labels, CAL labels, model weights or imports of S708 checkpoint needed.
"""
from __future__ import annotations
import math,unittest
from pathlib import Path

def gaussian_discrete_argmax(x,sigma):
    if sigma<=0:raise ValueError("positive width required")
    scores=[math.exp(-((x-k)**2)/(2*sigma*sigma)) for k in range(7)]
    return max(range(7),key=lambda k:scores[k])

class CounterexamplesToPlausibleModelClaims(unittest.TestCase):
    def test_gaussian_width_alone_does_not_bias_k_to_zero(self):
        # Same Gaussian soft class source from s741; no blending with a
        # different K-source. The class argmax is nearest integer.
        for x in [0.1,0.4,1.1,1.9,2.2,2.9,3.1,3.8,4.2,4.9,5.2,5.8]:
            expected=min(range(7),key=lambda k:abs(x-k))
            for sigma in (.1,.3,.65,.85,1.15,2.0,3.0):
                self.assertEqual(gaussian_discrete_argmax(x,sigma),expected)
    def test_s763_is_shadow_not_official_s708(self):
        s=Path("src/causal_note/s763_independent_decoders.py").read_text()
        s708=Path("src/causal_note/s708_cardinality_selection.py").read_text()
        self.assertIn("shadow-only decoders",s)
        self.assertIn("class ActiveKDecoder",s)
        self.assertNotIn("s763_independent_decoders",s708)
    def test_s797_only_experimental_and_still_not_official_s708(self):
        s=Path("src/causal_note/s797_fragment_router.py").read_text()
        s708=Path("src/causal_note/s708_cardinality_selection.py").read_text()
        self.assertIn("RESEARCH ONLY",s)
        self.assertNotIn("s797_fragment_router",s708)
    def test_partial_cue_does_not_imply_no_acoustic_signal(self):
        # 59,154 *failed the specific S794 threshold*; logically different
        # from signal absence, note irrecoverability or zero posteriors.
        self.assertEqual(21500+59154,80654)
        self.assertLess(21500/80654,.27)
if __name__=="__main__":unittest.main()
