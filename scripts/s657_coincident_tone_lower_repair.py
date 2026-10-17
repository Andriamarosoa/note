"""S657: causal selective Tone lower-octave rescue when HIGH onset is recent.

Controlled analysis of the S656 paired E3+E4 snippets only.
Native S614 high/other pitches, K and NoteOff selection stay original.
Only a lower-octave candidate is proposed when:
 - upper octave shows a credible, recent native onset,
 - the filtered low pitch shows an onset + sustain,
 - original low onset is below threshold, and
 - the lower pitch isn't already armed/active from earlier attack.
This targets S656 concurrent E3 losses while avoiding global Tone
filtering (which lost delayed E4 onsets).

No reconstruction of an artificial +2 pitch, no lookahead, no DEV
labels, no natural-song F1 claim. E2/E3 still needs real paired
bass-string sources; E3/E4 analogue is the measured target here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile
import zipfile
import numpy as np

from causal_note.strict_live import SR,HOP,Selection,CausalSelection,load_heads
from scripts.s637_guitarset_three_view_groundtruth import ROOT,MODEL
from scripts.s613_prepare_strict_live import download,digest
from scripts.s613_train_strict_live import event_score
from scripts.s642_real_octave_pair_comparison import (
    select_real_examples,conditions,CASES,LOW,HIGH
)
from scripts.s652_tone_filter_multiview import (
    causal_lowpass,predict_causal_view,FILTERS
)

POLICIES=("S614_original","gate_now_050","gate_last4_050",
          "gate_last8_050","gate_last4_070","gate_last8_070")
LOW_MIN=0
LOW_MAX=36


def condition(native,mild,dark,config,age,policy,armed,active):
    if policy=="S614_original":return native
    age_limit=int(policy.split("_")[1][4:]) if "last" in policy else 0
    floor=.7 if policy.endswith("_070") else .5
    p=native.copy()
    for i in range(LOW_MIN,LOW_MAX+1):
        upper=i+12
        recent=age[upper]<=age_limit
        if not recent or active[i] or not armed[i] or native[i,0]>=config.onset:
            continue
        if dark[i,0]<floor or dark[i,1]<config.active_on:
            continue
        # Independent physical lower-note support. A stronger low
        # fundamental than the high note in the filtered posterior
        # is not a proof of a new note; treat it as a conservative
        # requirement to suppress high-only octave hallucinations.
        if dark[i,1]<.65 and mild[i,1]<.65:
            continue
        p[i,0]=max(config.onset,float(dark[i,0]))
        p[i,1]=max(config.active_on,float(dark[i,1]))
    return p


def evaluate_pcm(wave,truth,heads,config):
    ends,native,K=predict_causal_view(wave,heads)
    variants=[]
    for fc in FILTERS.values():
        ef,p,k=predict_causal_view(causal_lowpass(wave,fc),heads)
        if not np.array_equal(ends,ef):raise AssertionError("Causal Tone clock")
        variants.append(p)
    mild,dark=variants
    states={name:CausalSelection(config) for name in POLICIES}
    events={name:[] for name in POLICIES}
    age=np.full(49,10**7,np.int32)
    for j,end in enumerate(ends):
        age=np.minimum(age+1,10**7)
        just=(native[j,:,0]>=config.onset)&(native[j,:,1]>=config.active_on)
        age[just]=0
        for name in POLICIES:
            policy_p=condition(native[j],mild[j],dark[j],
                               config,age,name,
                               states[name].armed,states[name].active)
            events[name].extend(states[name].step(policy_p,K[j],int(end),int(end)))
    out={}
    for name,dec in states.items():
        events[name].extend(dec.panic(len(wave)))
        score,pairs=event_score(events[name],truth)
        correct=Counter(truth[q]["pitch"] for _,q in pairs)
        emitted=[e for e in events[name] if e["type"]=="note_on"]
        out[name]=dict(TP=score["TP"],FP=score["FP"],FN=score["FN"],
                       E3_correct=correct[LOW],E4_correct=correct[HIGH],
                       spurious_E3_when_high_only=(sum(e["pitch"]==LOW for e in emitted)
                                if len(truth)==1 and truth[0]["pitch"]==HIGH else None),
                       other_emitted=sum(e["pitch"] not in (LOW,HIGH) for e in emitted))
    return out


def run(out):
    if out.exists():raise FileExistsError("S657 immutable evidence")
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    assert digest(MODEL.read_bytes())==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    proto=json.loads((ROOT/"protocol.json").read_text())
    heads=load_heads(MODEL,expected_sha256=frozen["model_sha256"])
    cfg=Selection(**heads.metadata["selection"])
    results=[]
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        for key,name in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            d=proto["guitar_archives"][key]
            download(d["url"],root/"data"/name,d["size"],d["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as arc:
            pairs,inventory=select_real_examples(proto,root,arc)
            for j,(low,high) in enumerate(pairs,1):
                waves,truth=conditions(low,high)
                cases={case:evaluate_pcm(waves[case],truth[case],heads,cfg)
                       for case in CASES}
                results.append(dict(low_source=low["name"],high_source=high["name"],
                                    cases=cases))
                print("S657 PAIR",j,"/8",{
                    p:cases["E3_plus_E4_simultaneous"][p]["E3_correct"]
                    for p in POLICIES},flush=True)
    totals={}
    for case in CASES:
        totals[case]={}
        for name in POLICIES:
            records=[row["cases"][case][name] for row in results]
            totals[case][name]={k:sum(r[k] for r in records)
                  for k in ("TP","FP","FN","E3_correct","E4_correct","other_emitted")}
            if case=="E4_alone":
                totals[case][name]["false_low_from_E4"]=sum(
                    r["spurious_E3_when_high_only"] or 0 for r in records)
        print("S657 CASE",case,json.dumps(totals[case]),flush=True)
    report=dict(status="S657_NATIVE_HIGH_TO_LOW_TONE_COINCIDENT_CAUSAL_RESUCE",
        sample_pairs=len(results),construct_GuitarSet_TRAIN_only=True,
        constructed_mix_NOT_real_concurrent_chord=True,
        E2_not_covered_by_isolated_pairs=True,
        native_K_and_other_notes_unchanged=True,
        no_added_future_audio=True,
        original_LIVE_not_modified=True,
        DEV_TEST_not_opened=True,
        results=totals,per_pair=results,
        source_inventory=inventory,promoted=False)
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S657 FINAL",json.dumps(totals),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        run(a.out)
