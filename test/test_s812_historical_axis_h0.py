"""S812: historical S708 events are H0, not simplified S810 proxies."""
import unittest
from causal_note.s812_historical_axis_h0 import (
    AXES,SR,HistoricalH0Axis,SixHistoricalH0,checked_h1_view,signature)

def event(kind,pitch,t):
    return dict(type=kind,pitch=pitch,decision_sample=t,available_sample=t,
      time=t/SR,reason="historical_signed")

class Poison(HistoricalH0Axis):
    def step(self,sample,events):
        for e in events:e["pitch"]=88
        return ("poisoned_private_lane",self.axis,sample)

class Tests(unittest.TestCase):
    def test_six_separate_historical_stream_projections(self):
        src=[event("note_on",40,256),event("note_on",52,512),
             event("note_off",40,768),event("note_off",52,1024)]
        bank=SixHistoricalH0();hist=[]
        for t in (256,512,768,1024):
            hist.append(bank.step(t,[x for x in src if x["available_sample"]==t]))
        self.assertEqual([x["exact_k"].count for x in hist],[1,2,1,0])
        self.assertEqual([x["active_exact_k"].count for x in hist],[1,2,1,0])
        self.assertEqual(hist[1]["frame_pitch"].pitches,(40,52))
        self.assertEqual(sum(len(x["onset"].emitted) for x in hist),2)
        self.assertEqual(sum(len(x["offset"].emitted) for x in hist),2)
        for a in AXES:self.assertEqual(signature(bank.lanes[a].events),signature(src))
    def test_malicious_private_lane_does_not_modify_other_five(self):
        for axis in AXES:
            inp=[event("note_on",40,256)]
            before=SixHistoricalH0().step(256,inp)
            lanes={x:HistoricalH0Axis(x) for x in AXES}
            lanes[axis]=Poison(axis)
            after=SixHistoricalH0(lanes).step(256,inp)
            for other in AXES:
                if other!=axis:self.assertEqual(after[other],before[other])
            self.assertEqual(inp[0]["pitch"],40)
    def test_overlays_remain_private_and_preserve_H0(self):
        orig=SixHistoricalH0().step(256,[event("note_on",40,256)])
        for axis in AXES:
            alt=checked_h1_view(orig,("H1_rejected",axis),axis)
            self.assertEqual(alt[axis],("H1_rejected",axis))
            for other in AXES:
                if axis!=other:self.assertEqual(alt[other],orig[other])
        self.assertEqual(orig["onset"].axis,"onset")
    def test_event_future_unavailable_and_clock_guard(self):
        with self.assertRaises(ValueError):
            SixHistoricalH0().step(256,[event("note_on",40,512)])
        x=event("note_on",40,256);x["decision_sample"]=512
        with self.assertRaises(ValueError):SixHistoricalH0().step(512,[x])
        bank=SixHistoricalH0();bank.step(256,[])
        with self.assertRaises(ValueError):bank.step(256,[])
    def test_reordered_axes_are_exactly_equivalent(self):
        a,b=SixHistoricalH0(),SixHistoricalH0(order=tuple(reversed(AXES)))
        for t,events in ((256,[event("note_on",40,256)]),
                         (512,[event("note_off",40,512)])):
            self.assertEqual(a.step(t,events),b.step(t,events))
    def test_future_suffix_does_not_alter_past_decisions(self):
        before=event("note_on",40,256)
        self.assertEqual(SixHistoricalH0().step(256,[before]),
                         SixHistoricalH0().step(256,[before]))
    def test_unbalanced_midi_rejected(self):
        with self.assertRaises(ValueError):
            SixHistoricalH0().step(256,[event("note_off",40,256)])
        with self.assertRaises(ValueError):
            SixHistoricalH0().step(256,[event("note_on",40,256),
                                        event("note_on",40,256)])
if __name__=="__main__":unittest.main()
