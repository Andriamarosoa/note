"""S731: causal auditory scene analysis by *counterfactual source explaining-away*.

NOT another append-only 32D neural input. This is an explicit online
old-plus-new source-competition mechanism, for every MIDI40..88 pitch and
every partial (up to 8 measured harmonics).

Given the ACTIVE NOTES ALREADY EMITTED by signed S708, compare:
    H0: a candidate peak is old-source harmonics continuing/being re-excited
    H1: unexplained new acoustic excitation supporting another pitch.

Both hypotheses observe only received spectrograms. The H0 projection uses
a physically grounded harmonic ownership graph up to parent partial 32,
including parent harmonics beyond the 8 actually sampled (never confuse
"not observed at a parent bin" with "not a possible parent harmonic").
An existing source's spectral level predicts its own next step by causal
decay extrapolation; common broadband change at a sustained parent's
non-child harmonics is explained away rather than mistaken for new notes.

This is an experimental decision mechanism, NOT an exact inverse of PCM,
a clone of human hearing, a note separator with oracle source identities,
or an alteration of the frozen S708 model / LIVE pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

FIRST_MIDI=40
N_PITCHES=49
PARTIALS=8
CHANNELS=288
MAX_PARENT_HARMONIC=32


def harmonic_ownership():
    """[candidate 49, source 49, partial 8] physical overlap graph.

    Parents may have any pitch 40..88, including above the candidate;
    equality requires a source's INTEGER partial 1..32 (tolerance 18 cents).
    Graph depends only on semitone frequencies, never label annotations.
    """
    hz=440.*2.**((np.arange(FIRST_MIDI,FIRST_MIDI+N_PITCHES)-69)/12.)
    frequencies=hz[:,None]*np.arange(1,PARTIALS+1)[None,:]
    ratio=frequencies[:,None,:]/hz[None,:,None]
    nearest=np.rint(ratio)
    cents=np.abs(1200*np.log2(np.maximum(ratio,1e-12)/np.maximum(nearest,1.)))
    return ((nearest>=1)&(nearest<=MAX_PARENT_HARMONIC)&(cents<=18.)&
            (np.arange(N_PITCHES)[:,None,None]!=np.arange(N_PITCHES)[None,:,None]))


OWNERSHIP=harmonic_ownership()


@dataclass(frozen=True)
class SceneEvidence:
    novelty: np.ndarray
    explained: np.ndarray
    novelty_shared: np.ndarray
    novelty_unshared: np.ndarray
    old_ownership: np.ndarray
    supports: np.ndarray
    count_room: float


class AuditoryScene:
    """49-source causal source-scene tracker (pitch family invariant)."""
    def __init__(self,trend_decay=.75):
        if not 0<=trend_decay<1:raise ValueError("S731 invalid decay factor")
        self.trend_decay=trend_decay
        self.last=None
        self.velocity=np.zeros((N_PITCHES,3,PARTIALS),np.float32)
        self.frames=0

    def step(self,x,active,k):
        xx=np.asarray(x,np.float32)
        active=np.asarray(active,bool)
        k=np.asarray(k,np.float32)
        if (xx.shape!=(N_PITCHES,CHANNELS) or active.shape!=(N_PITCHES,)
            or k.shape!=(7,) or not np.isfinite(xx).all() or
            not np.isfinite(k).all()):
            raise ValueError("S731 finite S708 288D/49 active/Exact-K K0..6 required")
        spectral=xx[:,192:216].reshape(N_PITCHES,3,PARTIALS)
        if self.last is None:
            delta=np.zeros((N_PITCHES,PARTIALS),np.float32)
            actual=np.zeros_like(spectral)
        else:
            predicted=self.last+np.clip(self.velocity,-.20,0.)
            actual=spectral-self.last
            delta=np.maximum((spectral-predicted).mean(axis=1),0.)
        # Parent broadband gain/repick is estimated from its OWN 8 partials
        # and must not be inferred from the candidate's overlapping bins.
        parent_common=np.percentile(np.maximum(actual,0.).mean(axis=1),35,axis=1)
        owner_mask=np.any(OWNERSHIP[:,active,:],axis=1) if np.any(active) else np.zeros((N_PITCHES,PARTIALS),bool)
        correction=np.zeros_like(delta)
        if np.any(active):
            for q in range(N_PITCHES):
                # Source p's partial h may explain the candidate frequency,
                # including p harmonics 9..32 unobserved in source features.
                for p in np.flatnonzero(active):
                    match=OWNERSHIP[q,p]
                    correction[q,match]=np.maximum(
                        correction[q,match],parent_common[p])
        residual=np.maximum(delta-correction,0.)
        weighted=np.maximum(spectral.mean(axis=1),0.)
        reliability=np.clip(weighted/(.50+weighted),0.,1.)
        residual*=reliability
        explained=np.maximum(delta*reliability-residual,0.)
        count_shared=owner_mask.sum(axis=1)
        count_unshared=(~owner_mask).sum(axis=1)
        share=(residual*owner_mask).sum(axis=1)/np.maximum(count_shared,1)
        unique=(residual*(~owner_mask)).sum(axis=1)/np.maximum(count_unshared,1)
        support=(residual>.13).sum(axis=1)
        # Reward evidence across at least two harmonics. Shared harmonics
        # are not rejected: they can contain an independently plucked note.
        coherence=np.minimum(1.,support/2.)
        novelty=(.65*share+.35*unique)*coherence
        novelty=np.where(count_shared==0,unique*coherence,novelty)
        ownership=count_shared/PARTIALS
        count=int(active.sum())
        count_room=float(np.clip(k[count+1:].sum(),0,1)) if count<6 else 0.
        # Maintain memory AFTER decisions, so no future spectrum is read.
        if self.last is not None:
            self.velocity=(self.trend_decay*self.velocity+
                  (1-self.trend_decay)*np.clip(actual,-.20,.20))
        self.last=spectral.copy()
        self.frames+=1
        return SceneEvidence(novelty.astype(np.float32),explained.mean(-1).astype(np.float32),
            share.astype(np.float32),unique.astype(np.float32),
            ownership.astype(np.float32),support.astype(np.int16),count_room)


def scene_scores(track,baseline_on,baseline_k,baseline_events,ends):
    """Compute physical scores using only SOURCE states from past S708 emissions.

    Calls are ordered per received decision; S708 events at current decision
    become active only AFTER its acoustic evidence is computed.
    """
    x=np.asarray(track,np.float32)
    ends=np.asarray(ends,np.int64)
    k=np.asarray(baseline_k,np.float32)
    if (x.ndim!=3 or x.shape[1:]!=(49,288) or
        ends.shape!=(len(x),) or k.shape!=(len(x),7) or
        np.any(np.diff(ends)<=0)):
        raise ValueError("S731 track/decision ABI mismatch")
    by_time={}
    for e in baseline_events:
        clock=int(e["decision_sample"])
        by_time.setdefault(clock,[]).append(e)
    state=np.zeros(49,bool)
    tracker=AuditoryScene()
    result=[]
    for time,row,kk in zip(ends,x,k):
        result.append(tracker.step(row,state,kk))
        for e in by_time.get(int(time),()):
            if 40<=int(e["pitch"])<=88:
                state[int(e["pitch"])-40]=(e["type"]=="note_on")
    return result


def reweight_onsets(post,evidence,*,gain=.20,explain=.00,require_k=True):
    """Causal auditory H1-H0 evidence applied to *onset channel only*.

    Original Sustain/Release posteriors stay intact; K0..K6 is not modified.
    Do not replace the original release or retroactively modify NoteOns.
    """
    p=np.asarray(post,np.float32)
    if p.shape!=(49,3):raise ValueError("S731 49 pitch, 3 head posteriors required")
    if not 0<=gain<=1 or not 0<=explain<=1:
        raise ValueError("S731 gain and explain weights are not probabilities")
    trial=p.copy()
    kweight=evidence.count_room if require_k else 1.
    boost=gain*kweight*np.clip(evidence.novelty,0,1)
    suppress=explain*np.clip(evidence.explained,0,1)
    trial[:,0]=np.clip(trial[:,0]+boost-suppress,0.,1.)
    return trial
