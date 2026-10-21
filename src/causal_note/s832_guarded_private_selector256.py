"""S832 Exact-K-only causal local risk gates.

The three S831 musical heads and two S829 count heads remain *private*.
A policy can only accept changes predicted by these heads. Calibration
truth is consumed by fit_rules() on inner-CAL, never by choose() at
inference and never by the music heads at runtime.

No other characteristic, pretrained checkpoint, future frame or hold state.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter,defaultdict
import math
import numpy as np
from causal_note.s830_private_musical_evidence256 import (
    counts_from_evidence,selected,positive_note_confidence
)

SCHEMA="S832_PRIVATE_CONSERVATIVE_LOCAL_VETO_PCM256_V1"
MODES=("rescue","blend","sustain","router")
THRESHOLDS=(.30,.40,.50,.60,.70)
PARAMETERS=(
    (64,1.35,.55,10),
    (96,1.4,.57,12),
    (128,1.5,.58,15),
    (160,1.7,.59,18)
)

def _bin(conf):
    """A 3-band confidence quantizer, not a certainty about correctness."""
    conf=np.asarray(conf,np.float32)
    if not np.isfinite(conf).all():
        raise ValueError("nonfinite positive-note confidence")
    return np.where(conf>=.80,2,np.where(conf>=.60,1,0)).astype(np.uint8)


def make_candidates(base,evidence):
    """Calculate each causal proposed cardinality once, no GT whatsoever."""
    b=np.asarray(base,np.uint8)
    e=np.asarray(evidence,np.float32)
    if e.ndim!=3 or e.shape[1:]!=(49,3) or b.shape!=(len(e),):
        raise ValueError("S832 private base and evidence shape")
    confidence=positive_note_confidence(e)
    candidate={}
    for threshold in THRESHOLDS:
        direct,state=counts_from_evidence(e,threshold)
        for mode in MODES:
            candidate[(mode,threshold)]=selected(
                b,direct,state,confidence,mode,0)
    return confidence,candidate


def signed_margin(corrected,regressed):
    """Wilson-style lower 95% success bound among changed-label outcomes."""
    n=corrected+regressed
    if n==0:return 0.
    z=1.96
    p=corrected/n
    return (p+z*z/(2*n)-z*np.sqrt(
        (p*(1-p)+z*z/(4*n))/n))/(1+z*z/n)


@dataclass
class Rule:
    key:tuple
    observed:int
    corrected:int
    regressed:int
    lower_bound:float
    score:float
    positive_tracks:int
    negative_tracks:int

    def record(self):
        return dict(key=[str(v) for v in self.key],
          observed=self.observed,corrected=self.corrected,
          regressed=self.regressed,lower_bound=self.lower_bound,
          score=self.score,positive_tracks=self.positive_tracks,
          negative_tracks=self.negative_tracks)


def _rule_key(mode,thr,base,proposed,confbin):
    return (str(mode),float(thr),int(base),int(proposed),int(confbin))


def learn_rules(cal_rows,*,params):
    """Build sparse, traceable, conservative local gates from CAL labels.

    Multiple candidate heads are assessed independently. A selected rule
    must correct more frames than it spoils and pass a Wilson lower bound.
    At inference only its non-GT key is used, with one outgoing decision
    for every incoming PCM256 boundary.
    """
    min_obs,ratio,min_lower,min_net=params
    stats=defaultdict(lambda:[0,0,0,Counter()])
    for row in cal_rows:
        b=np.asarray(row["base"],np.uint8)
        y=np.asarray(row["k"],np.uint8)
        confidence=np.asarray(row["confidence"],np.float32)
        labels=_bin(confidence)
        track=str(row["name"])
        if y.shape!=b.shape or labels.shape!=b.shape:
            raise ValueError("CAL candidate/GT alignment error")
        for (mode,thr),pred in row["candidates"].items():
            changed=np.flatnonzero(pred!=b)
            if not len(changed):continue
            key_components=np.column_stack((
                b[changed],pred[changed],labels[changed]))
            unique,inv=np.unique(key_components,axis=0,return_inverse=True)
            for j,group in enumerate(unique):
                ids=changed[inv==j]
                corrected=int(np.count_nonzero(
                    (pred[ids]==y[ids])&(b[ids]!=y[ids])))
                regressed=int(np.count_nonzero(
                    (pred[ids]!=y[ids])&(b[ids]==y[ids])))
                key=_rule_key(mode,thr,*group)
                s=stats[key];s[0]+=len(ids);s[1]+=corrected;s[2]+=regressed
                s[3][track]+=corrected-regressed
    passed={}
    candidates=[]
    for key,(observed,corrected,regressed,by_track) in stats.items():
        lb=signed_margin(corrected,regressed)
        positive_tracks=sum(v>0 for v in by_track.values())
        negative_tracks=sum(v<0 for v in by_track.values())
        # Conservative: risk smaller than a Wilson estimate and no single
        # track may dominate the calibration correction count.
        score=(corrected-regressed)/max(1,observed)
        record=Rule(key,observed,corrected,regressed,lb,score,
                    positive_tracks,negative_tracks)
        candidates.append(record)
        if (observed>=min_obs and corrected>=min_net and
             corrected-regressed>=min_net and
             corrected>=ratio*max(1,regressed) and
             lb>=min_lower and
             positive_tracks>=max(2,negative_tracks)):
            passed[key]=record
    return passed,candidates


def choose(base,confidence,candidates,rules):
    """One causal selection per current frame; never consult ground truth."""
    b=np.asarray(base,np.uint8)
    confidence=np.asarray(confidence,np.float32)
    if b.ndim!=1 or confidence.shape!=b.shape:
        raise ValueError("native256 shape mismatch")
    cbin=_bin(confidence)
    output=b.copy()
    strength=np.full(len(b),-np.inf,np.float64)
    for (mode,thr),proposal in sorted(candidates.items()):
        if proposal.shape!=b.shape:raise ValueError("candidate ABI drift")
        changed=np.flatnonzero(proposal!=b)
        for ix in changed:
            key=_rule_key(mode,thr,b[ix],proposal[ix],cbin[ix])
            rule=rules.get(key)
            if rule is None:continue
            # Ordering uses only pre-calibrated, non-label confidence.
            score=rule.lower_bound-.5+(rule.score*.25)
            if score>strength[ix]:
                strength[ix]=score;output[ix]=proposal[ix]
    return output


def calibration_score(rows,rules):
    correct=frames=under=over=0
    pertrack={}
    for row in rows:
        out=choose(row["base"],row["confidence"],row["candidates"],rules)
        truth=row["k"];base=row["base"]
        c=int(np.count_nonzero(out==truth))
        reference=int(np.count_nonzero(base==truth))
        pertrack[row["name"]]=dict(correct=c,reference=reference,
            improvement=c-reference,frames=len(truth))
        frames+=len(truth);correct+=c
        under+=int(np.count_nonzero(out<truth))
        over+=int(np.count_nonzero(out>truth))
    assert correct+under+over==frames
    return dict(correct=correct,frames=frames,undercount=under,
      overcount=over,accuracy=correct/max(1,frames),
      reference_correct=sum(t["reference"] for t in pertrack.values()),
      track_comparisons=pertrack)


def fit_guard(cal_rows):
    """Inner-CAL track-disjoint guard selection with fallback to S829.

    A training fragment never appears in the holdout scoring split.
    We split the 16 held-in CAL tracks deterministically (12 train/4
    validation), select between conservative fixed risks, then relearn
    rules on all 16 CAL tracks. The guard may be completely empty.
    """
    if not cal_rows:raise ValueError("empty CAL")
    names=sorted({str(x["name"]) for x in cal_rows})
    if len(names)<5:raise ValueError("at least five distinct CAL tracks")
    valset=set(names[3::4])
    train=[r for r in cal_rows if r["name"] not in valset]
    check=[r for r in cal_rows if r["name"] in valset]
    if not train or not check:raise AssertionError("no holdout tracks")
    baseline=sum(int(np.count_nonzero(r["base"]==r["k"])) for r in check)
    attempts=[]
    chosen=None
    for params in PARAMETERS:
        rules,summary=learn_rules(train,params=params)
        score=calibration_score(check,rules)
        change=score["correct"]-baseline
        changed=sum(x["corrected"]+x["regressed"] for x in rules.values())
        attempts.append(dict(params=list(params),rules=len(rules),
             changed_CAL_cases=changed,validation_delta=change,
             validation_frames=score["frames"]))
        # Reject if validation frame gain is negligible or regression at
        # the level of any held-out CAL track > 0.25 percentage points.
        worst=min((t["improvement"]/t["frames"] for t in
                  score["track_comparisons"].values()),default=0.)
        if (change>max(5,int(score["frames"]*.0005))
             and worst>=-.0025 and rules):
            if chosen is None or change>chosen[0]:
                chosen=(change,params)
    if chosen is None:
        return {},dict(reason="validation_veto",tried=attempts,
            guard_enabled=False,rule_count=0)
    params=chosen[1]
    learned,details=learn_rules(cal_rows,params=params)
    return learned,dict(reason="positive_CAL_holdout",tried=attempts,
        guard_enabled=bool(learned),rule_count=len(learned),
        params=list(params),rules=[r.record() for r in learned.values()])
