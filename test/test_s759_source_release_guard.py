"""S759 prove strictly causal native release quarantine on supplemental note births."""
import unittest
import numpy as np

from causal_note.strict_live import HOP,SR
from scripts.s759_source_release_guard import (
    GRID,RELEASE_GUARDS,SOURCE_ON_GUARD,select_novel_births)

def midi(kind,pitch,when,reason="signed"):
    return dict(type=kind,pitch=pitch,
        decision_sample=int(when),available_sample=int(when),
        time=float(when/SR),reason=reason,
        velocity=96 if kind=="note_on" else 0)

class SourceReleaseGuardTests(unittest.TestCase):
    def make(self,index=6):
        ends=np.arange(1,21,dtype=np.int64)*4*HOP
        native=[midi("note_on",41,ends[1],"S708_native"),
                midi("note_off",41,ends[3],"S707_hazard")]
        extra=[midi("note_on",41,ends[index],"supplemental"),
               midi("note_off",41,ends[index+1],"S707_extra")]
        p=np.zeros((len(ends),49,3),np.float32)
        p[index,1,:]=[.95,.95,0]
        nk=np.zeros((len(ends),7),np.float32)
        nk[:,0]=1.
        fk=nk.copy()
        fk[index,:]=[0,.15,.85,0,0,0,0]
        return native,extra,ends,p,nk,fk

    def select(self,guard,index=6,native_override=None):
        native,extra,ends,p,nk,fk=self.make(index)
        return select_novel_births(
            native if native_override is None else native_override,
            extra,ends,p,nk,fk,
            source_guard_seconds=SOURCE_ON_GUARD,
            release_guard_seconds=guard)

    def test_four_models_times_three_guards(self):
        self.assertEqual(len(GRID),12)
        self.assertEqual(RELEASE_GUARDS,(.025,.085,.175))

    def test_69ms_since_native_off_passes_25_but_fails_85(self):
        short=self.select(.025,index=6)
        mid=self.select(.085,index=6)
        self.assertEqual(len(short[0]),1)
        self.assertEqual(len(mid[0]),0)
        self.assertEqual(mid[3]["blocked_source_recent_off"],1)

    def test_139ms_since_native_off_passes_85_but_fails_175(self):
        mid=self.select(.085,index=9)
        long=self.select(.175,index=9)
        self.assertEqual(len(mid[0]),1)
        self.assertEqual(len(long[0]),0)
        self.assertEqual(long[3]["blocked_source_recent_off"],1)

    def test_future_native_restrike_cannot_affect_old_birth(self):
        ref,_,ends,_,_,_=self.make(6)
        future=ref+[midi("note_on",41,ends[12],"future_signed"),
                    midi("note_off",41,ends[13],"future_release")]
        a=self.select(.025,index=6)
        b=self.select(.025,index=6,native_override=future)
        self.assertEqual(len(a[0]),len(b[0]))
        np.testing.assert_array_equal(a[1],b[1])

    def test_out_of_range_quarantine_refused(self):
        for seconds in (-.01,.31):
            with self.assertRaises(ValueError):
                self.select(seconds)

if __name__=="__main__":
    unittest.main()
