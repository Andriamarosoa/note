"""S767: audit hidden new-note recall under S765 causal source-priority filter.

Strictly OBSERVATIONAL on already-explored GuitarSet CAL32.
Does not train, alter reference notes, change LIVE or select/promo a policy.
Candidate MIDI proposals are fixed by causal S765/S763 generators before
CAL annotations are inspected. Annotations only label the audit afterward.

Question: Of the S765 proposals rejected because source pitch is active,
how many hit REAL baseline-missed notes / repeated same-pitch attacks?
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import torch

from causal_note.strict_live import SR
from causal_note.s708_cardinality_selection import GuidedPolicy
from scripts.s613_prepare_strict_live import download, labels, digest
from scripts.s613_train_strict_live import real_notes, event_score
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s704_joint_metrics import SCORES, metric, aggregate
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import FIT_COMPS, CAL_COMPS, load_model
from scripts.s733_train_onset_only import baseline_input
from scripts.s760_causal_confirmation_delay import pairs_from_events
from scripts.s764_six_decoder_shadow_cal32 import replay_six
from scripts.s765_fit_independent_arbiter import acoustic_birth_pairs, MIN_NATIVE_RECENT

CAUSAL_GROUPS=("eligible","source_active","source_recent","source_full")


def causal_candidates(item,post,source,acoustic,ctx):
    """Return source-filter category of each candidate, before any ground truth."""
    shadow=replay_six(item["decision_ends"],post,
                      item["_k"],source,item["samples"])
    native=pairs_from_events(shadow["full"])
    acoustic_pairs,acoustic_stats=acoustic_birth_pairs(item,post,acoustic,ctx)
    raw=[*native,*acoustic_pairs]
    dedup={}
    for pair in raw:
        on,off=pair
        key=(int(on["available_sample"]),int(on["pitch"]))
        if key not in dedup or on["reason"]=="s765_independent_acoustic_birth":
            dedup[key]=pair
    ordered=sorted(dedup.values(),key=lambda z:(
        int(z[0]["available_sample"]),int(z[0]["pitch"])))
    events=sorted(enumerate(source),key=lambda x:(
        int(x[1]["available_sample"]),x[0]))
    active=set()
    last=np.full(49,-10**12,dtype=np.int64)
    cursor=0
    proposals=[]
    stats=Counter(acoustic_stats)
    stats.update(raw=int(len(raw)),unique=int(len(ordered)))
    for on,off in ordered:
        t=int(on["available_sample"])
        pitch=int(on["pitch"]);idx=pitch-40
        if not 0<=idx<49:
            raise ValueError("S767 out-of-range pitch")
        while cursor<len(events) and int(events[cursor][1]["available_sample"])<=t:
            event=events[cursor][1]
            q=int(event["pitch"])
            if event["type"]=="note_on":
                active.add(q);last[q-40]=int(event["available_sample"])
            elif event["type"]=="note_off":
                active.discard(q)
            else:
                raise ValueError("S767 unknown native source MIDI event")
            cursor+=1
        if pitch in active:
            group="source_active"
        elif (t-last[idx])/SR<MIN_NATIVE_RECENT:
            group="source_recent"
        elif len(active)>=6:
            group="source_full"
        else:
            group="eligible"
        proposals.append(dict(on=on,off=off,group=group,
                              baseline_occupancy=len(active)))
        stats[group]+=1
    if sum(stats[g] for g in CAUSAL_GROUPS)!=len(proposals):
        raise AssertionError("S767 candidate partition incomplete")
    return proposals,dict(stats)


def audit_labels(proposals,truth,baseline_matches):
    """Offline audit ONLY. No truth can influence candidate selection."""
    predicted=[dict(type="note_on",pitch=p["on"]["pitch"],time=p["on"]["time"])
               for p in proposals]
    missing=[j for j in range(len(truth)) if j not in set(baseline_matches)]
    missed_truth=[truth[j] for j in missing]
    _,pairs=event_score(predicted,missed_truth,tolerance=.080)
    stats={g:Counter() for g in CAUSAL_GROUPS}
    for proposal in proposals:
        stats[proposal["group"]]["unique_proposals"]+=1
    coverage=defaultdict(set)
    for pi,mi in pairs:
        p=proposals[pi]
        truthidx=missing[mi]
        g=p["group"]
        stats[g]["missed_truth_onset_matched"]+=1
        coverage[g].add(truthidx)
        n=truth[truthidx]
        off=p["off"]
        release=(float(off["time"])-float(n["time"]+n["duration"]))
        if (off["reason"] not in ("stream_end","shutdown","stream_error")
                and abs(release)<=.050):
            stats[g]["missed_truth_complete_correct"]+=1
        stats[g]["missed_truth_pitch_low_E2_E3" if n["pitch"]<=52 else
                 "missed_truth_pitch_mid_F3_A4" if n["pitch"]<=69 else
                 "missed_truth_pitch_high"]+=1
    return dict(
        missing_truth=len(missing),
        group={k:dict(v) for k,v in stats.items()},
        matched_missing_truth_unique=len(set().union(*coverage.values())),
        matched_missing_truth_by_group={k:len(v) for k,v in coverage.items()})


def aggregate_audits(results):
    combined=Counter()
    groups={g:Counter() for g in CAUSAL_GROUPS}
    for detail in results:
        combined.update({k:int(detail[k]) for k in (
            "missing_truth","matched_missing_truth_unique")})
        for group,stats in detail["group"].items():
            groups[group].update(stats)
    return dict(
        missing_truth=int(combined["missing_truth"]),
        recoverable_by_all_existing_proposals=int(combined["matched_missing_truth_unique"]),
        candidates={g:dict(groups[g]) for g in CAUSAL_GROUPS})


def save(path,report):
    path.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")


def evaluate(args):
    if args.out.exists():
        raise FileExistsError("S767 immutable audit output already exists")
    args.out.mkdir(parents=True)
    signed=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    if sha!=signed["checkpoint_sha256"]:
        raise AssertionError("S767 signed source hash mismatch")
    choice=signed["frozen_candidate"]["choice"]
    if choice["release"]!=dict(gate=.8,persistence=1,grace=24):
        raise AssertionError("S767 source release changed")
    policy=GuidedPolicy(**choice["policy"])
    network=load_model(args.s708_dir/"s708_selected.pt")[0]
    torch.set_num_threads(2)
    report=dict(schema="S767_S765_FILTERED_MISSING_TRUTH_RECALL_CAL32",
        complete=False,promoted=False,source_sha256=sha,
        source_reference_parity=None,
        CAL_recordings_done=0,
        label_use="OFFLINE_ONLY_POST_CAUSAL_EMISSION",
        HIT_TARGET="BASELINE_MISSED_ONSET_WITHIN_80MS",
        no_training=True,TEST_opened=False,HELD_opened=False,
        LIVE_changed=False,reference_unchanged=True,
        CAL=[],summary={})
    save(args.out/"report.json",report)
    scored=[]
    audit=[]
    census=Counter()
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        proto,heads,selection,offset=setup(root,args.offset_checkpoint)
        for key,file in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            archive=proto["guitar_archives"][key]
            download(archive["url"],root/"data"/file,
                     archive["size"],archive["sha256"])
        train=[s for s in proto["sources"]
               if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as wav, \
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            expanded=expanded_train_specs(train,wav,ann)
            cal=[s for s in expanded if s["composition"] in CAL_COMPS]
            fit=[s for s in expanded if s["composition"] in FIT_COMPS]
            if len(cal)!=32 or len(fit)!=80 or set(FIT_COMPS)&set(CAL_COMPS):
                raise AssertionError("S767 dataset split drift")
            for i,spec in enumerate(cal):
                item,post,k,source,acoustic,ctx,_=baseline_input(
                    spec,wav,root,heads,selection,offset,network,
                    policy,choice["release"])
                item["_k"]=k
                proposals,stats=causal_candidates(item,post,source,acoustic,ctx)
                # Only after all proposals are finalized, inspect truth.
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                measured=metric(source,truth,item["ends"])
                results=audit_labels(proposals,truth,measured["matched_truth_ids"])
                audit.append(results)
                census.update(stats)
                scored.append(measured)
                report["CAL"].append(dict(
                    name=spec["name"],composition=spec["composition"],
                    source={m:float(measured[m]) for m in SCORES},
                    counts=stats,audit=results))
                report["CAL_recordings_done"]=i+1
                save(args.out/"report.json",report)
                print("S767 CAL",i+1,32,spec["name"],
                      "source_active",stats["source_active"],
                      "missed_hits",results["matched_missing_truth_unique"],flush=True)
    source=aggregate(scored)
    delta={k:float(source[k]-signed["frozen_candidate"]["CAL"][k])
           for k in SCORES}
    parity=all(abs(v)<1e-8 for v in delta.values())
    report.update(source_reference_parity=dict(valid=parity,delta=delta),
                  source={m:float(source[m]) for m in SCORES},
                  census=dict(census),
                  summary=aggregate_audits(audit),
                  complete=parity)
    save(args.out/"report.json",report)
    if not parity:
        raise AssertionError("S767 S708 baseline drift")
    print("S767_FILTERED_ONSET_RECALL",json.dumps({
        "source":report["source"],"census":report["census"],
        "summary":report["summary"],"promoted":False}),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s708-dir",type=Path,required=True)
    ap.add_argument("--offset-checkpoint",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    evaluate(ap.parse_args())
