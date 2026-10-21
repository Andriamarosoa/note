"""Only Active Exact-K benchmark count and historical reference score."""
import numpy as np
def active_score(pred,truth):
    p,y=np.asarray(pred),np.asarray(truth)
    if p.ndim!=1 or p.shape!=y.shape:raise ValueError("invalid active K score arrays")
    mask=y>0
    return dict(correct=int(((p==y)&mask).sum()),frames=int(mask.sum()),
                undercount=int(((p<y)&mask).sum()),overcount=int(((p>y)&mask).sum()))
def original_count(events,ends):
    occupancy=np.zeros((len(ends),49),np.uint8)
    held={}
    for e in events:
        i=int(e["pitch"])-40
        pos=int(np.searchsorted(ends,int(e["available_sample"])))
        if not 0<=i<49:raise ValueError("bad GT pitch")
        if e["type"]=="note_on":
            if i in held:raise AssertionError("duplicate")
            held[i]=pos
        elif e["type"]=="note_off":
            if i not in held:raise AssertionError("orphan")
            occupancy[held.pop(i):pos,i]=1
        else:raise ValueError("bad MIDI")
    for i,start in held.items():occupancy[start:,i]=1
    return np.minimum(occupancy.sum(1),6).astype(np.uint8)
def targets_at_ends(notes,ends):
    count=np.zeros(len(ends),np.int16)
    t=np.asarray(ends)/44100
    for n in notes:
        p=int(n["pitch"])
        if not 40<=p<=88:raise ValueError("pitch out of bound")
        count+=(t>=float(n["time"]))&(t<float(n["time"])+float(n["duration"]))
    return np.minimum(count,6).astype(np.uint8)
def totals(rows):
    t={k:sum(int(x[k]) for x in rows) for k in ("correct","frames","undercount","overcount")}
    if sum(t[k] for k in ("correct","undercount","overcount"))!=t["frames"]:raise AssertionError("active score mismatch")
    t["accuracy"]=t["correct"]/max(1,t["frames"])
    return t
