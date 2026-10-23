"""Fast causal replay for calibration; verified against the live selector."""
import numpy as np
from numba import njit


@njit(cache=True)
def _decode(q, baseline, cut, ratio, gap, use_baseline, baseline_cut, enabled):
    result=np.zeros(q.shape,np.bool_)
    if not enabled:
        return baseline.copy()
    armed=np.ones(49,np.bool_)
    last=np.full(49,-10**12,np.int64)
    for frame in range(len(q)):
        for pitch in range(49):
            value=np.float64(q[frame,pitch])
            if value < cut*ratio:armed[pitch]=True
            ready=armed[pitch] and value>=cut
            if use_baseline and baseline[frame,pitch] and value>=baseline_cut:
                ready=True
            if ready and frame-last[pitch]>=gap:
                result[frame,pitch]=True
                last[pitch]=frame
                armed[pitch]=False
    return result


def decode_cache(q, baseline, policy):
    if (q.shape!=baseline.shape or q.ndim!=2 or q.shape[1]!=49 or
            not np.isfinite(q).all() or np.any((q<0)|(q>1))):
        raise ValueError('invalid complete probability cache')
    return _decode(np.asarray(q,np.float32),np.asarray(baseline,bool),
        policy.cut,policy.reset_ratio,policy.gap_hops,policy.use_baseline,
        policy.baseline_cut,policy.enabled)


def truth_arrays(truth):
    groups=[sorted(float(n['time']) for n in truth if n['pitch']==p)
            for p in range(40,89)]
    return (np.asarray([t for g in groups for t in g],np.float64),
            np.r_[0,np.cumsum([len(g) for g in groups])].astype(np.int64))


@njit(cache=True)
def count_cache(mask, times, offsets):
    """Maximum-cardinality interval matching, same TP as the original scorer.

    All events of a pitch have an equal +/-80ms tolerance. Earliest feasible
    matching is maximum-cardinality on these ordered intervals. Tests also
    compare adversarial and random cases to the original Hungarian matcher.
    """
    tp=0;predicted=0
    for pitch in range(49):
        cursor=offsets[pitch]
        stop=offsets[pitch+1]
        for frame in range(len(mask)):
            if not mask[frame,pitch]:continue
            predicted+=1
            time=(frame+1)*256/44100.
            while cursor<stop and time>times[cursor] and abs(time-times[cursor])>.080:
                cursor+=1
            if cursor<stop and abs(time-times[cursor])<=.080:
                tp+=1;cursor+=1
    return tp,predicted-tp,len(times)-tp
