"""S791 robust tests: heterogeneous experts, dynamic masks, no lookahead."""
import unittest
import numpy as np
from causal_note.s791_framepitch_expert_bank import (
    EXPERT_NAMES,ALL_MASKS,eligibility,sample_expert,candidate_offers,
    frame_descriptors,oracle_mask_labels,selected_offer,contribution_table,
    retained_from_fit)

class S791ComplementarityTests(unittest.TestCase):
    def fixture(self,n=5):
        x=np.zeros((n,49,40),np.float32)
        post=np.zeros((n,49,3),np.float32)
        base=np.zeros((n,49),bool)
        x[:,12,30]=.9
        post[:,12,1]=.46
        post[:,0,1]=.5
        post[2,12,1]=.9
        probs={k:np.ones((n,49),np.float32)*.5 for k in EXPERT_NAMES}
        return x,post,base,probs

    def test_head_count_not_mandated_six(self):
        self.assertEqual(len(EXPERT_NAMES),4)
        self.assertEqual(len(ALL_MASKS),16)
        self.assertEqual(ALL_MASKS[0],0)
        self.assertEqual(ALL_MASKS[-1],15)

    def test_specialization_regions_differ(self):
        x,p,base,probs=self.fixture()
        domain=eligibility(x,p)
        self.assertTrue(domain["bass"][0,0])
        self.assertFalse(domain["bass"][0,12])
        self.assertTrue(domain["harmonic"][0,12])
        self.assertFalse(domain["harmonic"][0,0])
        self.assertTrue(domain["transition"][2,12])
        self.assertFalse(domain["transition"][0,40])

    def test_per_fragment_mask_and_head_effects(self):
        x,p,base,probs=self.fixture(n=4)
        probs["bass"][0,0]=.95
        probs["harmonic"][1,12]=.95
        probs["general"][2,18]=.95
        probs["transition"][3,12]=.95
        offers,domain=candidate_offers(x,p,base,probs)
        chosen=np.array([2,4,1,8])
        selected=selected_offer(offers,chosen)
        self.assertTrue(selected[0,0])
        self.assertTrue(selected[1,12])
        self.assertTrue(selected[2,18])
        self.assertTrue(selected[3,12])
        self.assertFalse(selected[0,12])
        ctx=frame_descriptors(x,p,base,probs,domain)
        self.assertEqual(ctx.shape,(4,19))

    def test_prefix_invariance_under_future_mutation(self):
        x,p,base,probs=self.fixture(n=5)
        oa,da=candidate_offers(x,p,base,probs)
        ca=frame_descriptors(x,p,base,probs,da)
        x2=x.copy();p2=p.copy();prob2={k:v.copy() for k,v in probs.items()}
        x2[3:]=9;p2[3:]=.99
        for k in prob2:prob2[k][3:]=.99
        ob,db=candidate_offers(x2,p2,base,prob2)
        cb=frame_descriptors(x2,p2,base,prob2,db)
        np.testing.assert_array_equal(oa[:3],ob[:3])
        np.testing.assert_array_equal(ca[:3],cb[:3])

    def test_oracle_targets_are_fit_only_and_no_label_in_inference(self):
        x,p,base,probs=self.fixture(n=4)
        probs["bass"][:,0]=.99
        offers,_=candidate_offers(x,p,base,probs)
        y=np.zeros((4,49),bool);y[:,0]=True
        wanted=oracle_mask_labels(offers,y)
        self.assertTrue(all(m&2 for m in wanted))
        np.testing.assert_array_equal(selected_offer(offers,np.zeros(4,int)),base)
        with self.assertRaises(ValueError):oracle_mask_labels(offers,y[:3])

    def test_prune_irrelevant_heads_based_on_fit(self):
        comps=["A","B","C","D","E"]
        source=dict(f1=.75,by_composition={c:.75 for c in comps})
        table={"source":source}
        for name in EXPERT_NAMES:
            good=name=="bass"
            table[name]=dict(f1=.77 if good else .749,
                by_composition={c:.76 if good else .749 for c in comps})
        kept,masks=retained_from_fit(table)
        self.assertEqual(kept,["bass"])
        self.assertEqual(masks,(0,2))

    def test_training_samples_are_region_specific(self):
        x,p,base,probs=self.fixture(n=100)
        y=np.zeros((100,49),bool);y[:50,:8]=True
        xx,yy=sample_expert(x,y,p,"bass",23)
        self.assertEqual(xx.shape[1],40)
        self.assertTrue(0<int(yy.sum())<len(yy))
        with self.assertRaises(KeyError):sample_expert(x,y,p,"not_a_domain",23)

if __name__=="__main__":
    unittest.main()
