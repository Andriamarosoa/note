"""Exact sparse evaluation of the original exhaustive research policy grid.

Only possible emission frames need traversal. A threshold crossing can be
armed iff a low score occurred strictly after the last emitted frame (or no
frame has been emitted). This is equivalent to the original state machine.
No change to model inference, policies, matching tolerance or Live runtime.
"""
from dataclasses import asdict
import numpy as np
from numba import njit
from .acoustic_calibration import count_cache


@njit(cache=True)
def _count_sparse(frames, offsets, reference_allowed, last_low, gap, times, truth_offsets):
    tp=0
    predicted=0
    for pitch in range(49):
        last=-10**12
        cursor=truth_offsets[pitch]
        stop=truth_offsets[pitch+1]
        for i in range(offsets[pitch],offsets[pitch+1]):
            frame=frames[i]
            if not (reference_allowed[i] or last_low[frame,pitch]>last):continue
            if frame-last<gap:continue
            last=frame
            predicted+=1
            t=(frame+1)*256/44100.
            while cursor<stop and t>times[cursor] and abs(t-times[cursor])>.080:
                cursor+=1
            if cursor<stop and abs(t-times[cursor])<=.080:
                tp+=1;cursor+=1
    return tp,predicted-tp,len(times)-tp


def score_policies(probabilities, cache, policies):
    totals=np.zeros((len(policies),3),np.int64)
    cuts=sorted({p.cut for p in policies if p.enabled})
    for name,q in probabilities.items():
        q=np.asarray(q,np.float64)
        base=cache[name]['baseline'];truth=cache[name]['truth']
        frame_indices=np.arange(len(q),dtype=np.int64)[:,None]
        for cut in cuts:
            selected=[(i,p) for i,p in enumerate(policies) if p.enabled and p.cut==cut]
            lows={ratio:np.maximum.accumulate(np.where(q<cut*ratio,frame_indices,-1),axis=0)
                  for ratio in {p.reset_ratio for _,p in selected}}
            hits={}
            for i,p in selected:
                key=(p.use_baseline,p.baseline_cut)
                if key not in hits:
                    allowed=base & (q>=p.baseline_cut) if p.use_baseline else np.zeros(base.shape,bool)
                    pitch,frames=np.nonzero(((q>=cut)|allowed).T)
                    offsets=np.r_[0,np.cumsum(np.bincount(pitch,minlength=49))].astype(np.int64)
                    hits[key]=(frames.astype(np.int64),offsets,allowed[frames,pitch])
                frames,offsets,allowed=hits[key]
                totals[i]+=_count_sparse(frames,offsets,allowed,lows[p.reset_ratio],
                    p.gap_hops,*truth)
        for i,p in enumerate(policies):
            if not p.enabled:totals[i]+=count_cache(base,*truth)
    candidates=[]
    for p,(tp,fp,fn) in zip(policies,totals):
        tp,fp,fn=int(tp),int(fp),int(fn)
        score=dict(tp=tp,fp=fp,fn=fn,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.)
        candidates.append(dict(policy=asdict(p),score=score))
    return candidates
