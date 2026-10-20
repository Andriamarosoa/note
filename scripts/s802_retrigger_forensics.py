"""S802 FIT80 forensic analysis of synthetic attacks closed by native re-attacks.

Post-hoc truth is permitted ONLY here, to classify past predictions; no
truth/future-native-information enters the router or MIDI decoder.
Immutable S800 traces, actual S801 policy decoding, signed S801 metrics.
No model training, CAL selection, lookahead, NoteOff hold, or LIVE edits.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from causal_note.strict_live import SR
from causal_note.s795_activek_experts import HEADS,decode
from causal_note.s793_zero_added_latency import SOURCE_ID
from causal_note.s797_fragment_router import CUT
from scripts.s613_train_strict_live import event_score
from scripts.s704_joint_metrics import SCORES,metric,aggregate
from scripts.s795_train_activek_midi256 import COUNT_KEYS
from scripts.s800_oof_trace import SCHEMA as TRACE_SCHEMA,verify_snapshot
from scripts.s801_fit_fragment_ablations import route

POLICIES=("consensus","poly_gap","adaptive")
SCHEMA="S802_FIT80_NATIVE_SYNTHETIC_COLLISION_FORENSIC_V1"
BINS=(0,1024,2048,4096,8192,16384,32768)

def lag_bin(delta):
    if delta is None:return "other_release"
    for upper in BINS[1:]:
        if delta <= upper:return f"le_{upper}_samples"
    return "gt_32768_samples"

def classify_events(out,truth,decisions,X,P,composition,track,policy):
    """One-to-one onset truth matching *after* MIDI decisions are frozen.

    Future native re-attacks are used only as retrospective observations.
    They are never a model input, gating signal or emitted event adjustment.
    """
    _,pairs=event_score(out,truth)
    onsets=[e for e in out if e["type"]=="note_on"]
    matched={i:j for i,j in pairs}
    if len(matched)!=len(pairs):
        raise AssertionError("matching not one-to-one")
    marked={id(e):i for i,e in enumerate(onsets)}
    active={}
    decisions=np.asarray(decisions,np.int64)
    observations=[]
    for e in out:
        pitch=int(e["pitch"])
        if e["type"]=="note_on" and e.get("reason")=="s795_multihead_attack":
            if pitch in active:raise AssertionError("duplicate synthetic NoteOn")
            active[pitch]=e
        elif e["type"]=="note_off" and pitch in active:
            birth=active.pop(pitch)
            ix=marked[id(birth)]
            t=int(birth["available_sample"])
            f=int(np.searchsorted(decisions,t,side="left"))
            if f>=len(decisions) or decisions[f]!=t:
                raise AssertionError("birth without current decision frame")
            pi=pitch-40
            if not 0<=pi<49:raise AssertionError("pitch ABI")
            retrigger=e.get("reason")=="s795_source_retrigger"
            native_on=any(
                z["type"]=="note_on" and int(z["pitch"])==pitch
                and int(z["available_sample"])==int(e["available_sample"])
                and z.get(SOURCE_ID) is not None for z in out
            ) if retrigger else False
            if retrigger and not native_on:
                raise AssertionError("synthetic retrigger lacks matching source NoteOn")
            elapsed=int(e["available_sample"])-t
            if elapsed<0:raise AssertionError("negative synthetic note duration")
            target_id=matched.get(ix)
            observations.append(dict(
                composition=composition,track=track,policy=policy,
                pitch=pitch,first_eligible_sample=t,
                synthetic_matched_truth=target_id is not None,
                matched_truth_index=int(target_id) if target_id is not None else None,
                # Retrospective labels, NEVER candidate features.
                truth_onset_delta_ms=round(1000*(birth["time"]-truth[target_id]["time"]),4)
                    if target_id is not None else None,
                release_reason=e.get("reason"),
                source_retrigger=bool(retrigger),
                source_native_attacked_same_clock=bool(native_on),
                life_samples=elapsed,
                life_ms=round(1000*elapsed/SR,3),
                life_bin=lag_bin(elapsed if retrigger else None),
                on=float(X[f,pi,0]),sustain=float(X[f,pi,1]),
                off=float(X[f,pi,2]),on_rise=float(X[f,pi,5]),
                sustain_rise=float(X[f,pi,6]),
                harmonic_parent=float(X[f,pi,10]),
                general_p=float(P["general"][f,pi]),
                bass_p=float(P["bass"][f,pi]),
                polyphonic_p=float(P["polyphonic"][f,pi]),
                transient_p=float(P["transient"][f,pi]),
                harmonic_p=float(P["harmonic"][f,pi]),
            ))
    if active:raise AssertionError("unclosed artificial notes")
    if len(observations)!=sum(x.get("reason")=="s795_multihead_attack"
                             for x in out):
        raise AssertionError("some synthetic attacks lack paired NoteOff")
    return observations

def read_record(trace_dir,entry):
    info=verify_snapshot(entry,trace_dir)
    with np.load(Path(trace_dir)/f"fit_{entry['index']:03d}.npz",
                 allow_pickle=False) as z:
        X=z["features"].astype(np.float32)
        ends=z["metric_ends"]
        decisions=z["decision_ends"]
        P={h:z["head_"+h].astype(np.float32) for h in HEADS}
    return info,X,ends,decisions,P

def summarize(rows):
    n=len(rows)
    retriggers=[x for x in rows if x["source_retrigger"]]
    matched=[x for x in rows if x["synthetic_matched_truth"]]
    bad=[x for x in rows if not x["synthetic_matched_truth"]]
    bins=Counter(x["life_bin"] for x in retriggers)
    compositions={}
    for c in sorted({x["composition"] for x in rows}):
        subset=[x for x in rows if x["composition"]==c]
        compositions[c]=dict(total=len(subset),
            matched_truth=sum(x["synthetic_matched_truth"] for x in subset),
            unmatched_truth=sum(not x["synthetic_matched_truth"] for x in subset),
            native_retriggers=sum(x["source_retrigger"] for x in subset))
    return dict(total_synthetic_onsets=n,
        matched_ground_truth=len(matched),unmatched_ground_truth=len(bad),
        native_retrigger_closures=len(retriggers),
        retrigger_true_positives=sum(x["synthetic_matched_truth"] for x in retriggers),
        retrigger_false_positives=sum(not x["synthetic_matched_truth"] for x in retriggers),
        nonretrigger_true_positives=sum(x["synthetic_matched_truth"] for x in rows
                                        if not x["source_retrigger"]),
        closure_reasons=dict(Counter(x["release_reason"] for x in rows)),
        native_retrigger_life_bins=dict(sorted(bins.items())),
        by_composition=compositions)

def evaluate(manifest_path,traces,s801_path):
    m=json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    r=json.loads(Path(s801_path).read_text(encoding="utf-8"))
    if m.get("schema")!=TRACE_SCHEMA or not m.get("complete") or not m.get("FIT_source_parity_S795"):
        raise AssertionError("S800 signed real OOF manifest absent")
    if m.get("CAL_opened") or m.get("HELD_opened") or m.get("TEST_opened"):
        raise AssertionError("non-FIT split opened")
    if r.get("schema")!="S801_FIT80_CAUSAL_DYNAMIC_ABLATION_V1" or not r.get("completed"):
        raise AssertionError("S801 complete real FIT80 policy evidence required")
    if r["provisional_fit_candidate"]!="identity":
        raise AssertionError("source selection changed before forensics")
    if len(m["fit_files"])!=80 or len({q["index"] for q in m["fit_files"]})!=80:
        raise AssertionError("S800 80 tracks required")
    tracks=sorted(m["fit_files"],key=lambda x:x["index"])
    details={p:[] for p in POLICIES}
    metric_rows={p:[] for p in POLICIES}
    baseline_counts={k:0 for k in COUNT_KEYS}
    for entry in tracks:
        info,X,ends,decisions,P=read_record(traces,entry)
        src=metric(info["source_events"],info["fit_truth"],ends)
        if any(int(src[k])!=int(info["source_counts"][k]) for k in COUNT_KEYS):
            raise AssertionError("S800 native source score mismatch")
        for k in COUNT_KEYS:baseline_counts[k]+=int(src[k])
        for p in POLICIES:
            routed,mask,metadata=route(X,P,p)
            out,delta=decode(info["source_events"],decisions,X,routed,
                             31,CUT,int(info["samples"]))
            if any(delta[k] for k in ("max_added_native_delay_samples",
                                      "extra_lookahead_samples",
                                      "extra_hold_samples",
                                      "extra_confirmation_samples")):
                raise AssertionError("S802 replay changed event scheduling")
            metrics=metric(out,info["fit_truth"],ends)
            observations=classify_events(
                out,info["fit_truth"],decisions,X,P,
                info["composition"],info["name"],p)
            if len(observations)!=delta["new_onsets"] or (
                    delta["new_offsets"]!=delta["new_onsets"]):
                raise AssertionError("synthetic attacks not perfectly paired")
            metric_rows[p].append(metrics)
            details[p].extend(observations)
        print("S802_FORENSIC_TRACK",entry["index"],info["name"],flush=True)
    if baseline_counts!=m["S795_archived_fit_counts"] or (
            baseline_counts!=r["signed_source_counts"]):
        raise AssertionError("native baseline FIT counts drift")
    findings={}
    for p in POLICIES:
        scores=aggregate(metric_rows[p])
        expected=r["policy_results"][p]
        if any(int(scores[k])!=int(expected["counts"][k]) for k in COUNT_KEYS):
            raise AssertionError("S801 policy MIDI metric counts drift: "+p)
        if any(abs(scores[k]-expected["metrics"][k])>1e-12 for k in SCORES):
            raise AssertionError("S801 policy six ratios drift: "+p)
        info=summarize(details[p])
        if info["total_synthetic_onsets"]!=expected["aggregate_new_onsets"]:
            raise AssertionError("synthetic-onset count differs from S801: "+p)
        # Earlier S801 field contains all s795_ event reasons (incl new Onsets)
        source_retriggers=expected["added_offset_reason_totals"].get(
            "s795_source_retrigger",0)
        if info["native_retrigger_closures"]!=source_retriggers:
            raise AssertionError("S801 native retrigger count drift: "+p)
        findings[p]=info
    return dict(schema=SCHEMA,complete=True,
        dataset="GuitarSet FIT80 OOF real MIDI256",
        tracks=80,policy_list=list(POLICIES),
        signed_s800_native_fit_counts=baseline_counts,
        s801_policy_six_scores_and_counts_parity=True,
        human_oracle_in_runtime=False,
        truth_used_only_for_posthoc_forensics=True,
        future_native_used_only_for_posthoc_forensics=True,
        synthesis_policy_or_decoder_modified=False,
        production_LIVE_modified=False,
        latency_physical_measured=False,CAL_opened=False,
        HELD_opened=False,TEST_opened=False,promoted=False,
        by_policy=findings,observations=details)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--traces",type=Path,required=True)
    parser.add_argument("--s801",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError("S802 immutable output exists")
    report=evaluate(args.manifest,args.traces,args.s801)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",
                        encoding="utf-8")
    print("S802_POSTHOC_CAUSES",json.dumps({k:dict(
        n=x["total_synthetic_onsets"],
        closures=x["native_retrigger_closures"],
        real_tp=x["retrigger_true_positives"],
        real_fp=x["retrigger_false_positives"])
        for k,x in report["by_policy"].items()}),flush=True)

if __name__=="__main__":main()
