import unittest
import numpy as np
from scripts.build_v273_variant_memory import preserve_variants


class VariantMemoryTests(unittest.TestCase):
    def test_negative_variant_keeps_its_unique_correction_and_does_not_replace_parent(self):
        ids=np.arange(5); y=np.array([2,3,2,4,3]); base=np.array([2,2,2,4,3])
        previous=base.copy(); candidate=np.array([3,3,3,3,3])
        entries,pred,outcomes,union=preserve_variants(ids,y,base,[
            dict(variant_id='old',parent_id=None,prediction=previous),
            dict(variant_id='new',parent_id='old',prediction=candidate)])
        self.assertEqual(entries[1]['versus_freeze']['global']['net'],-2)
        self.assertTrue(entries[1]['retained'])
        np.testing.assert_array_equal(pred[:,0],previous)
        np.testing.assert_array_equal(pred[:,1],candidate)
        self.assertEqual(outcomes[1,1],1)
        self.assertTrue(union[1])
        self.assertFalse(entries[1]['downstream_ready'])

    def test_duplicate_identity_is_rejected_instead_of_overwriting_history(self):
        a=np.array([2,3]);v=dict(variant_id='same',parent_id=None,prediction=a)
        with self.assertRaises(ValueError):
            preserve_variants(np.arange(2),a,a,[v,v])


if __name__ == '__main__':
    unittest.main()
