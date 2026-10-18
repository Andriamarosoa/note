"""S710 joint pitch-state/cardinality filtering from received model outputs."""
from dataclasses import dataclass
import numpy as np
from causal_note.strict_live import SR,HOP

@dataclass(frozen=True)
class StatePolicy:
    opening_penalty: float = 1.
    stay_bonus: float = 1.
    count_weight: float = 1.
    release_weight: float = 1.
    close_steps: int = 1
    acoustic_floor: float = .20
    restrike_onset: float = .60

class StateSelector:
    def __init__(self,policy):
        self.policy=policy
        self.active=np.zeros(49,bool)
        self.armed=np.ones(49,bool)
        self.low=np.zeros(49,np.int32)
        self.last_on=np.full(49,-10**12,np.int64)
        self.last_end=0

    def step(self,post,count,end):
        if int(end)!=self.last_end+4*HOP:raise ValueError("Non-contiguous received blocks")
        self.last_end=int(end)
        c=self.policy
        on,present,release=np.asarray(post).T
        if post.shape!=(49,3) or np.asarray(count).shape!=(7,):raise ValueError("Head shape")
        self.armed |= on < c.restrike_onset*.4
        age=end-self.last_on
        mandatory=self.active & (age<.035*SR)
        eligible=self.active | ((present>=c.acoustic_floor)&(age>=.065*SR))
        value=np.log(np.clip(present,1e-5,1-1e-5)/np.clip(1-present,1e-5,1))
        value+=np.where(self.active,c.stay_bonus-c.release_weight*release,
                         on-c.opening_penalty)
        free=np.flatnonzero(eligible & ~mandatory)
        free=free[np.argsort(-value[free],kind="stable")][:6-int(mandatory.sum())]
        n=int(mandatory.sum())
        totals=np.r_[0.,np.cumsum(value[free])]
        totals+=c.count_weight*np.log(np.clip(np.asarray(count)[n:n+len(totals)],1e-8,1))
        take=int(np.argmax(totals))
        target=mandatory.copy()
        target[free[:take]]=True
        self.low=np.where(self.active & ~target,self.low+1,0)
        events=[]
        def emit(kind,p,reason):
            events.append(dict(type=kind,pitch=int(p+40),decision_sample=int(end),
                available_sample=int(end),time=float(end/SR),reason=reason,
                velocity=96 if kind=="note_on" else 0))
        for p in np.flatnonzero(self.active & (self.low>=c.close_steps)):
            emit("note_off",p,"joint_state_release")
            self.active[p]=False
        for p in np.flatnonzero(self.active & target & self.armed &
                                (on>=c.restrike_onset) & (age>=.065*SR)):
            emit("note_off",p,"joint_state_restrike")
            emit("note_on",p,"joint_state_restrike")
            self.armed[p]=False;self.last_on[p]=end;self.low[p]=0
        for p in free[:take]:
            if self.active[p]:continue
            if self.active.sum()>=6:break
            emit("note_on",p,"joint_state_acoustic_open")
            self.active[p]=True;self.armed[p]=False;self.low[p]=0;self.last_on[p]=end
        return events

    def finish(self,sample):
        if sample<self.last_end:raise ValueError("EOF precedes last received block")
        events=[dict(type="note_off",pitch=int(p+40),decision_sample=int(sample),
            available_sample=int(sample),time=float(sample/SR),reason="stream_end",velocity=0)
            for p in np.flatnonzero(self.active)]
        self.active[:]=False
        return events

def decode_state(ends,post,count,policy,samples):
    decoder=StateSelector(policy);events=[]
    for end,p,k in zip(ends,post,count):events.extend(decoder.step(p,k,int(end)))
    events.extend(decoder.finish(samples))
    return events
