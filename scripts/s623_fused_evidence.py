"""S623: native NOTE 128-hop evidence assists the frozen 256-hop selector.

Unlike S620–S622, the fast 128 branch NEVER emits MIDI note_on/off.
The unchanged 256-hop frozen S613 neural+ONNX hybrid owns the sole note
lifecycle. Fast evidence can promote only a weak, already supported
256-hop candidate at the CURRENT 256 decision frame, before selection.

Every signal used at 256t is taken from 128-hop predictions whose end <=t.
No future audio / no post-attack hold. Frequency windows remain native
past-only 2048/4096/8192-sample spectrograms, with separate lag contexts.

Experiment only: the policies are DEV-calibrated, never promote on the
17 tunes used to select thresholds. No held-out data is opened.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict
import argparse
import json

import numpy as np

from causal_note.strict_live import load_heads, CausalSelection, Selection, SR
from scripts.s615_train_multihop import source_data
from scripts.s613_train_strict_live import event_score, real_notes, summary
from scripts.s615_exactk_audit import exact_k_onset_audit, merge_k_reports


@dataclass(frozen=True)
class EvidenceConfig:
    slow_weak_onset: float = .40
    fast_min_onset: float = .50
    fast_min_sustain: float = .50
    fast_novelty_ratio: float = .55
    fast_novelty_absolute: float = .04
    mode: str = "both"   # both fast frames, either fast frame, latest only
    require_slow_novelty: bool = False

    def __post_init__(self):
        if self.mode not in ("both","either","latest"):
            raise ValueError("Invalid S623 fast evidence mode")
        for x in (self.slow_weak_onset,self.fast_min_onset,
                  self.fast_min_sustain,self.fast_novelty_ratio):
            if not (0 < x <= 1):
                raise ValueError("S623 thresholds outside probability range")
        if self.fast_novelty_absolute < 0:
            raise ValueError("Invalid novelty floor")


def select_fast_evidence(two_fast_p, two_fast_novelty, config):
    """Two fast measurements in PRECEDING/current 256 interval.

    Shape [2,49,3] and [2,49]. Returns one boolean per candidate pitch.
    The novelty normalization is across musical pitches at each frame,
    not across future or the 17 DEV songs.
    """
    p=np.asarray(two_fast_p)
    n=np.asarray(two_fast_novelty)
    if p.shape!=(2,49,3) or n.shape!=(2,49):
        raise ValueError("Two timestamp-aligned hop128 frames required")
    rel=n/np.maximum(n.max(axis=1,keepdims=True),1e-5)
    ok=((p[:,:,0]>=config.fast_min_onset)&
        (p[:,:,1]>=config.fast_min_sustain)&
        (n>=config.fast_novelty_absolute)&
        (rel>=config.fast_novelty_ratio))
    if config.mode=="both":
        return ok[0]&ok[1]
    if config.mode=="either":
        return ok.any(axis=0)
    return ok[-1]


def run_one(record, config=None, *, selection):
    """Run S613 unchanged selector; never emit any event from hop128."""
    selector=CausalSelection(selection,hop=256)
    events=[]
    attempts=0
    selected=0
    if len(record["e128"])<2*len(record["e256"]):
        raise ValueError("Fast hop does not cover reference decisions")
    for idx,end in enumerate(record["e256"]):
        end=int(end)
        a=2*idx
        b=a+2
        if int(record["e128"][b-1])!=end:
            raise AssertionError("Fast and slow evidence not time-aligned")
        prob=record["p256"][idx]
        if config is not None:
            n=record["nov128"][a:b]
            qualified=select_fast_evidence(record["p128"][a:b],n,config)
            # Critically, the already correct S613 attacks are untouched.
            # Active voices and disarmed notes are also protected from
            # artificial fast restrikes.
            eligible=(qualified &
                      (~selector.active) & selector.armed &
                      (prob[:,0]>=config.slow_weak_onset)&
                      (prob[:,0]<selection.onset)&
                      (prob[:,1]>=selection.active_on))
            if config.require_slow_novelty:
                nv=record["nov256"][idx]
                eligible &= nv>=config.fast_novelty_absolute
            attempts+=int(qualified.sum())
            selected+=int(eligible.sum())
            if eligible.any():
                prob=prob.copy()
                prob[eligible,0]=min(1.0,float(selection.onset)+1e-5)
        # This is the ONLY note state machine. Original release, K0..6,
        # physical-hop refractory and max six NOTE voices remain intact.
        events+=selector.step(prob,record["k256"][idx],end,end)
    events+=selector.panic(record["frames"])
    return events,dict(fast_candidates=attempts,promoted_pitch_frames=selected)


def predict_source(root,source,slow_heads,fast_heads):
    e128,x128,_,_,notes,length=source_data(root,source,128)
    e256,x256,_,_,_,length2=source_data(root,source,256)
    if length!=length2:
        raise ValueError("Acoustic samples differ across hop schedules")
    p128,_=fast_heads.predict(x128)
    p256,k256=slow_heads.predict(x256)
    n128=np.maximum(x128[:,:,60:63],0).max(axis=2)
    n256=np.maximum(x256[:,:,60:63],0).max(axis=2)
    return dict(source=source,e128=e128,e256=e256,p128=p128,
                p256=p256,k256=k256,nov128=n128,nov256=n256,
                frames=length,truth=real_notes(notes,length/SR))


def configs():
    return [
        EvidenceConfig(slow_weak_onset=weak,fast_min_onset=fast,
                       fast_min_sustain=.50,fast_novelty_ratio=nov,
                       fast_novelty_absolute=.04,mode=mode,
                       require_slow_novelty=slow_nv)
        for weak in (.35,.45,.50)
        for fast in (.45,.60)
        for nov in (.45,.70)
        for mode in ("latest","either","both")
        for slow_nv in (False,True)
    ]


def evaluate(root,out,*,slow_model,fast_model):
    if out.exists():
        raise FileExistsError("Refusing to overwrite S623 report")
    slow=load_heads(slow_model)
    fast=load_heads(fast_model)
    if slow.metadata.get("architecture")!="native_causal_hybrid":
        raise ValueError("S623 needs the ORIGINAL frozen full S613 hybrid")
    if 128 not in fast.metadata.get("trained_hops",()):
        raise ValueError("Separate fast model MUST declare trained hop128")
    selection=Selection(**slow.metadata["selection"])
    protocol=json.loads((root/"protocol.json").read_text())
    sources=[s for s in protocol["sources"] if s["split"]=="dev"]
    if len(sources)!=17:
        raise ValueError("All 17 pinned original DEV songs required")
    recordings=[]
    for i,s in enumerate(sources,1):
        row=predict_source(root,s,slow,fast)
        old,_=run_one(row,selection=selection)
        score,pairs=event_score(old,row["truth"],.08)
        row["old_event_truth"]={j for _,j in pairs}
        row["baseline_score"]=score
        recordings.append(row)
        print(json.dumps(dict(loaded=i,source=s["name"],
                              reference_TP=score["TP"],
                              reference_FN=score["FN"])),flush=True)
    baseline=summary([s["baseline_score"] for s in recordings])
    baseline_k=merge_k_reports([
        exact_k_onset_audit(run_one(rec,selection=selection)[0],
                            rec["truth"]) for rec in recordings])
    reports=[]
    for cfg in configs():
        scores=[]; grouped=[]; gained=lost=0
        by_corpus=defaultdict(list)
        attempts=boosted=0
        for rec in recordings:
            events,n=run_one(rec,cfg,selection=selection)
            sc,pairs=event_score(events,rec["truth"],.08)
            new_matched={j for _,j in pairs}
            gained+=len(new_matched-rec["old_event_truth"])
            lost+=len(rec["old_event_truth"]-new_matched)
            scores.append(sc)
            by_corpus[rec["source"]["corpus"]].append(sc)
            grouped.append(exact_k_onset_audit(events,rec["truth"]))
            attempts+=n["fast_candidates"]
            boosted+=n["promoted_pitch_frames"]
        m=summary(scores)
        k=merge_k_reports(grouped)
        record=dict(config=asdict(cfg),
            TP=m["TP"],FP=m["FP"],FN=m["FN"],F1=m["F1"],
            saved_former_FN=gained,lost_old_TP=lost,
            delta_TP=m["TP"]-baseline["TP"],
            delta_FP=m["FP"]-baseline["FP"],
            candidate_pitch_frames=attempts,
            promoted_pitch_frames=boosted,
            K_by_true=k["by_true_k"],
            total_grouped_K_accuracy=k["onset_exact_k_rate"],
            pitch_set_accuracy=k["pitch_set_rate"],
            by_corpus={corpus:summary(scores) for corpus,scores in by_corpus.items()})
        assert record["delta_TP"]==gained-lost
        reports.append(record)
        print(json.dumps(dict(config=asdict(cfg),
              F1=round(m["F1"],5),TP=m["TP"],FP=m["FP"],FN=m["FN"],
              saved=gained,lost=lost,
              K3=k["by_true_k"].get("3"),K4=k["by_true_k"].get("4"))),flush=True)
    best_f1=max(reports,key=lambda r:r["F1"])
    best_recall=max(reports,key=lambda r:(r["TP"],-r["FP"]))
    result=dict(status="S623_DEV_EVIDENCE_ONLY_NO_PROMOTION",
        reference_model=str(slow_model),trained_fast_model=str(fast_model),
        sources=len(sources),hop_fast=128,hop_output=256,
        baseline=baseline,baseline_K=baseline_k,
        configurations=len(reports),tested=reports,
        best_f1=best_f1,best_recall=best_recall,
        no_future_audio=True,no_fast_independent_MIDI=True,
        warning="DEV-optimized, not independent. Only a single original MIDI event state machine; this is a two-context but not fast-emitted 128 MIDI mode.")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(
        status=result["status"],baseline=baseline,best_f1=best_f1["F1"],
        best_recall=best_recall["F1"],
        best_TP=best_f1["TP"],best_FP=best_f1["FP"],
        best_K3=best_f1["K_by_true"].get("3"),
        best_K4=best_f1["K_by_true"].get("4"))),flush=True)
    return result


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--root",type=Path,required=True)
    a.add_argument("--slow-model",type=Path,
                   default=Path("model/live/strict_note/NOTE_strict_live.npz"))
    a.add_argument("--fast-model",type=Path,required=True)
    a.add_argument("--out",type=Path,required=True)
    x=a.parse_args()
    evaluate(x.root,x.out,slow_model=x.slow_model,fast_model=x.fast_model)


if __name__=="__main__":
    main()
