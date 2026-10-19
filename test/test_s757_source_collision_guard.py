"""S757 causal source-onset collision-guard regression tests."""
import unittest
import numpy as np
from causal_note.strict_live import HOP,SR
from scripts.s757_source_onset_collision_guard import (
    GUARDS,GRID,ADMISSION,select_novel_births)

def e(kind,pitch,now,reason="synthetic"):
    return dict(type=kind,pitch=pitch,
        decision_sample=int(now),available_sample=int(now),
        time=float(now/SR),reason=reason,velocity=96 if kind=="note_on" else 0)


class S757OnsetCollisionTests(unittest.TestCase):
    def setup_case(self,candidate_index=7):
        ends=np.arange(1,15,dtype=np.int64)*4*HOP
        # Native pitch 41 ends before supplemental birth; the original
        # matching 80ms window can still conflict near two onsets.
        baseline=[
            e("note_on",41,ends[1],"signed_s708_on"),
            e("note_off",41,ends[2],"signed_s707_off")]
        on=ends[candidate_index]
        proposed=[e("note_on",41,on,"supplemental_birth"),
                  e("note_off",41,ends[candidate_index+1],"supplemental_release")]
        p=np.zeros((len(ends),49,3),np.float32)
        p[candidate_index,1]=[.9,.9,.1]
        source=np.zeros((len(ends),7),np.float32)
        source[:,1]=1.
        k=source.copy()
        k[candidate_index,:]=[0,.1,.9,0,0,0,0]
        return baseline,proposed,ends,p,source,k

    def check(self,guard,*,case=7,reference=None):
        baseline,trial,ends,post,native,poly=self.setup_case(case)
        return select_novel_births(
            baseline if reference is None else reference,
            trial,ends,post,native,poly,
            source_guard_seconds=guard)

    def test_four_sources_three_causal_guard_windows(self):
        self.assertEqual(len(GRID),12)
        self.assertEqual(GUARDS,(.085,.16,.24))
        self.assertEqual(ADMISSION,.55)

    def test_160ms_guard_rejects_139ms_native_collision(self):
        short=self.check(.085)
        protected=self.check(.16)
        self.assertEqual(len(short[0]),1)
        self.assertEqual(len(protected[0]),0)
        self.assertEqual(protected[3]["blocked_source_recent"],1)

    def test_240ms_guard_rejects_event_allowed_at_160ms(self):
        # Ten 1024-sample four-hop gaps = 232.2ms.
        short=self.check(.16,case=11)
        long=self.check(.24,case=11)
        self.assertEqual(len(short[0]),1)
        self.assertEqual(len(long[0]),0)

    def test_future_native_onset_cannot_change_past_acceptance(self):
        b,trial,ends,p,n,k=self.setup_case()
        future=b+[e("note_on",41,ends[10],"future_signed_on"),
                  e("note_off",41,ends[11],"future_signed_off")]
        a=select_novel_births(b,trial,ends,p,n,k,source_guard_seconds=.085)
        c=select_novel_births(future,trial,ends,p,n,k,source_guard_seconds=.085)
        self.assertEqual(len(a[0]),len(c[0]))
        np.testing.assert_array_equal(a[1],c[1])

    def test_guard_cannot_consume_a_future_offset(self):
        b,trial,ends,p,n,k=self.setup_case()
        a=select_novel_births(b,trial,ends,p,n,k,source_guard_seconds=.085)
        altered=list(trial)
        altered[-1]=e("note_off",41,ends[12],"delayed_off")
        c=select_novel_births(b,altered,ends,p,n,k,
                             source_guard_seconds=.085)
        np.testing.assert_array_equal(a[1],c[1])

    def test_unreasonable_guard_is_rejected(self):
        for bad in (-1.,0.,.45):
            with self.assertRaises(ValueError):
                self.check(bad)

if __name__=="__main__":
    unittest.main()
