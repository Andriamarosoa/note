"""S742 — Active Exact-K forensics on actual S708 MIDI note instances.

Research diagnostics only. The metric is *not* K-network argmax accuracy:
active_exact_k measures emitted MIDI active-note cardinality on true K>0
native-hop frames. Uses the exact S714 framing convention and checks the
official S704 metric bit-for-bit.

No oracle signal is used in prediction. Truth is consulted for forensics
only AFTER original streams and six acoustic K proposals are generated.
"""
from __future__ import annotations
from collections import Counter
import numpy as np

from causal_note.s708_cardinality_selection import GuidedSelector
from scripts.s714_audit_s708_cardinality import events_to_frames, annotation_frames
from scripts.s704_joint_metrics import metric
from scripts.s704_train_joint_loops import targets

K_CLASSES = tuple(range(7))


def _triple(arr, mask):
    return dict(frames=int(mask.sum()),correct=int(np.count_nonzero(arr[mask]==0)),
                under=int(np.count_nonzero(arr[mask]<0)),
                over=int(np.count_nonzero(arr[mask]>0)))


def exact_active_breakdown(events, truth, ends):
    """One exact source of cardinality truth: physically emitted notes.

    K6 groups K>=6 because S708 MIDI is limited to six simultaneous notes.
    K0 is counted in Exact-K but excluded from Active Exact-K.
    """
    pred=events_to_frames(events,ends)
    target,count=annotation_frames(truth,ends)
    delta=np.minimum(pred.sum(1),6)-np.minimum(count,6)
    by_k={}
    for k in K_CLASSES:
        hit=count==k if k<6 else count>=6
        by_k[str(k)]=_triple(delta,hit)
    active=count>0
    missing=target&~pred
    false=pred&~target
    impossible=(count>target.sum(1))&active
    return dict(
        frames=len(ends),
        correct=int(np.count_nonzero(delta==0)),
        active_frames=int(active.sum()),
        active_correct=int(np.count_nonzero(active&(delta==0))),
        under_active=int(np.count_nonzero(active&(delta<0))),
        over_active=int(np.count_nonzero(active&(delta>0))),
        active_exact_k=float(np.mean(delta[active]==0)) if active.any() else 0.,
        silent_over=int(np.count_nonzero((count==0)&(delta>0))),
        by_true_k=by_k,
        missing_pitch_frames=int(missing.sum()),
        spurious_pitch_frames=int(false.sum()),
        unison_or_same_pitch_impossible_frames=int(impossible.sum()),
        # This class-aware lower bound is a diagnostic, not a proposed policy.
        max_possible_active_exact_k_if_no_same_pitch_dup=(
            float(1-np.mean(impossible[active])) if active.any() else 0.),
    )


def k_source_diagnostics(frozen_k, k_heads, true_k):
    """Is the missing cardinality even represented by candidate K heads?"""
    k=np.asarray(frozen_k,np.float32)
    heads=np.asarray(k_heads,np.float32)
    truth=np.asarray(true_k,np.int64)
    n=len(k)
    if k.shape!=(n,7) or heads.shape!=(n,7,6) or truth.shape!=(n,):
        raise ValueError("Need [fragments,7,6] and real K0..K6 truth")
    if np.any(truth<0) or np.any(truth>6) or not np.allclose(
            heads.sum(1),1,atol=1e-4):
        raise ValueError("Unnormalized probabilities or K truth outside 0..6")
    hard=heads.argmax(axis=1)
    prior=hard[:,0]
    active=truth>0
    wrong=(prior!=truth)&active
    any_correct=np.any(hard[:,1:]==truth[:,None],axis=1)
    return dict(
        active_fragments=int(active.sum()),
        baseline_K_argmax_correct=int(np.count_nonzero((prior==truth)&active)),
        baseline_K_argmax_accuracy_active=float(
            np.mean(prior[active]==truth[active])) if active.any() else 0.,
        baseline_wrong_active=int(wrong.sum()),
        baseline_wrong_with_any_alternative_right=int(np.count_nonzero(wrong&any_correct)),
        baseline_wrong_without_any_alternative_right=int(np.count_nonzero(wrong&~any_correct)),
        by_head={f"H{h}":dict(
            correct_active=int(np.count_nonzero((hard[:,h]==truth)&active)),
            wrong_active=int(np.count_nonzero((hard[:,h]!=truth)&active)))
            for h in range(6)},
        by_true_k={str(c):dict(
            fragments=int(np.count_nonzero(truth==c)),
            wrong_H0=int(np.count_nonzero((truth==c)&(prior!=c))),
            H0_wrong_with_alternative=int(np.count_nonzero(
                (truth==c)&(prior!=c)&any_correct)))
            for c in K_CLASSES},
    )


def trace_weak_k_gate(post, frozen_k, decision_ends, policy, true_sustain):
    """Observe the frozen *source* decoder's actual eligible weak pitches.

    Trace instrumented before every causal decision; true_sustain is only
    compared to frozen states AFTER the full trace is produced.
    """
    t=len(post)
    k=np.asarray(frozen_k,np.float32)
    truth=np.asarray(true_sustain,bool)
    ends=np.asarray(decision_ends,np.int64)
    if post.shape!=(t,49,3) or k.shape!=(t,7) \
       or truth.shape!=(t,49) or ends.shape!=(t,):
        raise ValueError("Frozen decision block shapes changed")
    selector=GuidedSelector(policy)
    histories=[]
    for index,end in enumerate(ends):
        on,sustain,release=post[index].T
        old=selector.past[0] if selector.past else np.zeros_like(post[index])
        rising=(on-old[:,0]>=policy.rise) | \
               (sustain-old[:,1]>=2.5*policy.rise)
        armed=selector.armed | (on<policy.onset*.4)
        # Ignore pitches currently active before this decision; true
        # missing pitch can be due to a source NoteOff downstream.
        weak=((~selector.active)&armed&rising&
              (on>=policy.weak_onset)&(sustain>=policy.weak_sustain)&
              (release<.50)&
              (end-selector.last_on>=.12*44100))
        strong=((~selector.active)&armed&
                (on>=policy.onset)&(sustain>=policy.sustain_on)&
                (end-selector.last_on>=.065*44100))
        target=int(k[index].argmax())
        confident=float(k[index].max())>=policy.guide_confidence
        room=target>int(selector.active.sum())
        histories.append(dict(
            weak=weak.copy(),strong=strong.copy(),
            k_confident=confident,k_room=room,
            k_target=target,
            source_active=selector.active.copy(),
            source_active_count=int(selector.active.sum())))
        selector.step(post[index],k[index],int(end))
    selector.finish(int(ends[-1]))
    flags=Counter()
    for h,row in zip(histories,truth):
        missing=row & ~h["source_active"]
        if not missing.any():
            continue
        flags["missing_true_pitch_decision_fragments"]+=int(missing.sum())
        flags["missing_with_strong_eligible"]+=int((missing&h["strong"]).sum())
        flags["missing_with_weak_eligible"]+=int((missing&h["weak"]).sum())
        if not h["k_confident"]:
            flags["weak_eligible_but_k_low_confidence"]+=int(
                (missing&h["weak"]).sum())
        elif not h["k_room"]:
            flags["weak_eligible_but_k_no_room"]+=int(
                (missing&h["weak"]).sum())
        else:
            flags["weak_eligible_and_K_support"]+=int(
                (missing&h["weak"]).sum())
        flags["weak_eligible_any_nontruth_pitch"]+=int(
            (h["weak"]&~row).sum())
    return dict(flags=dict(flags))


def full_track_forensics(track):
    """Input is already generated by one_k_track, then scored with truth."""
    item=track["item"]
    events=track["baseline"]
    truth=track["truth"]
    native_ends=item["ends"]
    decision_ends=item["decision_ends"]
    original=metric(events,truth,native_ends)
    emitted=exact_active_breakdown(events,truth,native_ends)
    if (abs(original["active_exact_k"]-emitted["active_exact_k"])>1e-12 or
        original["active_k_correct"]!=emitted["active_correct"] or
        original["active_frames"]!=emitted["active_frames"]):
        raise AssertionError("S742 audit is not the exact original MIDI metric")
    _,true_k=targets(decision_ends,truth)
    return dict(
        name=item["spec"]["name"],
        composition=item["spec"]["composition"],
        texture=("comp" if str(item["spec"]["name"]).endswith("_comp") else
                 "solo" if str(item["spec"]["name"]).endswith("_solo") else
                 "unknown"),
        native_frames=emitted,
        k_proposals=k_source_diagnostics(track["k"],track["k_heads"],true_k),
        source_gate=trace_weak_k_gate(
            track["post"],track["k"],decision_ends,track["policy"],
            targets(decision_ends,truth)[0][:,:,1].astype(bool)),
    )


def aggregate_forensics(records):
    if not records:
        raise ValueError("No actual emitted-note diagnostics")
    active=sum(x["native_frames"]["active_frames"] for x in records)
    correct=sum(x["native_frames"]["active_correct"] for x in records)
    breakdown={}
    for k in K_CLASSES:
        breakdown[str(k)]={field:sum(x["native_frames"]["by_true_k"][str(k)][field]
                                 for x in records)
                           for field in ("frames","correct","under","over")}
        row=breakdown[str(k)]
        row["correct_pct"]=100*row["correct"]/max(1,row["frames"])
    gates=Counter()
    for row in records:
        gates.update(row["source_gate"]["flags"])
    return dict(
        tracks=len(records),
        active_exact_k=correct/max(1,active),
        active_frames=active,active_correct=correct,
        under_active=sum(x["native_frames"]["under_active"] for x in records),
        over_active=sum(x["native_frames"]["over_active"] for x in records),
        silent_over=sum(x["native_frames"]["silent_over"] for x in records),
        missing_pitch_frames=sum(x["native_frames"]["missing_pitch_frames"] for x in records),
        spurious_pitch_frames=sum(x["native_frames"]["spurious_pitch_frames"] for x in records),
        same_pitch_unison_unrepresentable_frames=sum(
            x["native_frames"]["unison_or_same_pitch_impossible_frames"]
            for x in records),
        K_heads_wrong_with_some_correct_alternative=sum(
            x["k_proposals"]["baseline_wrong_with_any_alternative_right"]
            for x in records),
        K_heads_wrong_without_correct_alternative=sum(
            x["k_proposals"]["baseline_wrong_without_any_alternative_right"]
            for x in records),
        by_true_k=breakdown,weak_onset_gate=dict(gates))
