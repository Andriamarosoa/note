"""S762 proves per-fragment four-expert routing and causal invariants."""
import unittest
import numpy as np
from causal_note.strict_live import SR,HOP
from causal_note.s746_instance_guard import merge_events,InstanceGate
from scripts.s762_fragment_ensemble_cal32 import (
    QUALITY,GRID,quality_score,route_fragment_proposals)

BLOCK=4*HOP

def ev(kind,pitch,at,reason="synthetic"):
    return dict(type=kind,pitch=pitch,decision_sample=int(at),
                available_sample=int(at),time=float(at/SR),
                reason=reason,velocity=96 if kind=="note_on" else 0)

def pair(pitch,on,off):
    return (ev("note_on",pitch,on),ev("note_off",pitch,off))

class FragmentEnsembleTests(unittest.TestCase):
    def make(self):
        k=np.zeros((12,7),np.float32)
        k[:,3]=1.
        # First fragment: same event proposed by two experts, score decides.
        # Second fragment: different expert discovers a different note.
        e0=([(pair(41,3*BLOCK,8*BLOCK))],[(2,1)],
            np.array([.8]),np.array([.75]),k)
        e1=([(pair(41,3*BLOCK,9*BLOCK)),
             (pair(42,5*BLOCK,10*BLOCK))],[(2,1),(4,2)],
            np.array([.5,.9]),np.array([.4,.88]),k)
        return k,[e0,e1]

    def test_declares_four_expert_dynamic_grid(self):
        self.assertEqual(len(GRID),12)
        self.assertEqual(QUALITY,("min","geometric","complete"))

    def test_selects_different_experts_at_different_fragments(self):
        k,e=self.make()
        pairs,probs,indices,fused,stats=route_fragment_proposals(e,k,"min")
        self.assertEqual(stats["proposals"],3)
        self.assertEqual(stats["unique"],2)
        self.assertEqual(stats["deduplicated"],1)
        self.assertEqual([p[0]["pitch"] for p in pairs],[41,42])
        self.assertEqual(indices,[(2,1),(4,2)])
        np.testing.assert_allclose(probs,[.75,.88],atol=1e-6)

    def test_current_decision_does_not_depend_on_future_closures(self):
        k,e=self.make()
        a=route_fragment_proposals(e,k,"min")
        e2=[]
        for pairs,indices,on,full,posterior in e:
            changed=[]
            for source_on,source_off in pairs:
                changed.append((source_on,dict(source_off,
                      available_sample=source_off["available_sample"]+3*BLOCK,
                      decision_sample=source_off["decision_sample"]+3*BLOCK,
                      time=source_off["time"]+3*BLOCK/SR,
                      reason="mutated_future_release")))
            e2.append((changed,indices,on,full,posterior))
        b=route_fragment_proposals(e2,k,"min")
        self.assertEqual([p[0] for p in a[0]],[p[0] for p in b[0]])
        np.testing.assert_array_equal(a[1],b[1])
        self.assertEqual(a[2],b[2])
        np.testing.assert_array_equal(a[3],b[3])

    def test_scores_are_exploratory_and_bounded(self):
        a=np.array([.2,.8],np.float32); b=np.array([.9,.4],np.float32)
        np.testing.assert_allclose(quality_score(a,b,"min"),[.2,.4])
        np.testing.assert_allclose(quality_score(a,b,"complete"),b)
        np.testing.assert_allclose(quality_score(a,b,"geometric"),
                                   np.sqrt(a*b))
        with self.assertRaises(ValueError):
            quality_score(a,b,"bad")

    def test_source_notes_preserved_and_extra_closed(self):
        k,e=self.make()
        pairs,probs,indices,fused,stats=route_fragment_proposals(e,k,"min")
        source=[ev("note_on",40,BLOCK,"signed_on"),
                ev("note_off",40,11*BLOCK,"signed_off")]
        out,detail=merge_events(source,pairs,probs,indices,fused,
                                InstanceGate(risk=.70,k_tail=.10))
        kept=[x for x in out if x["reason"] in ("signed_on","signed_off")]
        self.assertEqual(kept,source)
        self.assertEqual(detail["accepted"],2)
        self.assertEqual([x["type"] for x in out if x.get("_s746_extra")],
                         ["note_on","note_on","note_off","note_off"])

    def test_signed_future_on_preempts_extra_not_original(self):
        k,e=self.make()
        pairs,probs,indices,fused,_=route_fragment_proposals(e,k,"min")
        source=[ev("note_on",40,BLOCK,"signed_on"),
                ev("note_on",41,6*BLOCK,"signed_new"),
                ev("note_off",41,9*BLOCK,"signed_new_off"),
                ev("note_off",40,11*BLOCK,"signed_off")]
        out,detail=merge_events(source,pairs,probs,indices,fused,
                                InstanceGate(risk=.7,k_tail=.10))
        kept=[x for x in out if x["reason"] in (
            "signed_on","signed_new","signed_new_off","signed_off")]
        self.assertEqual(kept,source)
        self.assertEqual(detail["preempted"],1)
        self.assertTrue(any(x["reason"]=="s746_source_priority" for x in out))

    def test_rejects_misaligned_fragment(self):
        k,e=self.make()
        invalid=list(e)
        wrong=list(invalid[0])
        bad_on=dict(wrong[0][0][0],decision_sample=2*BLOCK)
        wrong[0]=[(bad_on,wrong[0][0][1])]
        invalid[0]=tuple(wrong)
        with self.assertRaises(AssertionError):
            route_fragment_proposals(invalid,k,"min")

if __name__=="__main__":
    unittest.main()
