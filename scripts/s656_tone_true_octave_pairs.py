"""S656 — controlled causal Tone views on REAL E3 single / E3+E4 sources.

Only 8 digitally mixed pairs of isolated TRAIN GuitarSet guitar clips.
Use S642 identical E3 WAV for E3-alone and E3+E4; no synthetic sine
substitution. No measured Stratocaster, no inference Tone augmentation,
no original K changes and no DEV/TEST ground truth involved.

Ablate 8 native NOTE multi-Tone selection policies from S652, and
report E3/E4 NoteOn detections and other false attack events by case.
This IS NOT a natural-song or sealed generalization F1 score.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile
import zipfile
import numpy as np

from causal_note.strict_live import SR,Selection,CausalSelection,load_heads
from scripts.s637_guitarset_three_view_groundtruth import ROOT,MODEL
from scripts.s613_prepare_strict_live import download,digest
from scripts.s613_train_strict_live import event_score
from scripts.s642_real_octave_pair_comparison import (
    select_real_examples,conditions,CASES,LOW,HIGH,BASE_T,DELAY
)
from scripts.s652_tone_filter_multiview import (
    causal_lowpass,predict_causal_view,probabilities,FILTERS
)

POLICIES=("S614_original","tone_mild_2800","tone_dark_1100",
          "avg_original_mild","avg_three","vote_2of3",
          "original_OR_mild_085","original_AND_mild")
TOL=.080


def eval_condition(wave,truth,heads,config):
    e,native,k=predict_causal_view(wave,heads)
    views=[]
    for fc in FILTERS.values():
        ef,pi,ki=predict_causal_view(causal_lowpass(wave,fc),heads)
        if not np.array_equal(e,ef):
            raise AssertionError("No causal Tone time delay allowed")
        views.append(pi)
    mild,dark=views
    states={name:CausalSelection(config) for name in POLICIES}
    ev={name:[] for name in POLICIES}
    for j,end in enumerate(e):
        for name in POLICIES:
            p=probabilities(name,native[j],mild[j],dark[j],config)
            ev[name].extend(states[name].step(p,k[j],int(end),int(end)))
    result={}
    for name,state in states.items():
        ev[name].extend(state.panic(len(wave)))
        sc,match=event_score(ev[name],truth,tolerance=TOL)
        found=Counter(truth[q]["pitch"] for _,q in match)
        noteons=[x for x in ev[name] if x["type"]=="note_on"]
        result[name]=dict(
            TP=sc["TP"],FP=sc["FP"],FN=sc["FN"],
            E3_correct=int(found[LOW]),E4_correct=int(found[HIGH]),
            other_emitted=sum(x["pitch"] not in (LOW,HIGH) for x in noteons),
            emitted_E3=sum(x["pitch"]==LOW for x in noteons),
            emitted_E4=sum(x["pitch"]==HIGH for x in noteons))
    sample=int(np.searchsorted(e,round((BASE_T+.06)*SR),side="left"))
    sample=min(sample,len(e)-1)
    return result,dict(k_original_prediction=int(k[sample].argmax()),
                       true_initial_K=len({n["pitch"] for n in truth
                          if n["time"]<=BASE_T+.06 and
                          n["time"]+n["duration"]>BASE_T+.06}))


def run(out):
    if out.exists():raise FileExistsError("S656 immutable pair evidence")
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    assert digest(MODEL.read_bytes())==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    proto=json.loads((ROOT/"protocol.json").read_text())
    heads=load_heads(MODEL,expected_sha256=frozen["model_sha256"])
    cfg=Selection(**heads.metadata["selection"])
    data=[]
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        for key,path in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            source=proto["guitar_archives"][key]
            download(source["url"],root/"data"/path,source["size"],source["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as arc:
            pairs,inventory=select_real_examples(proto,root,arc)
            for index,(low,high) in enumerate(pairs,1):
                clips,targets=conditions(low,high)
                results={}
                K={}
                for case in CASES:
                    results[case],K[case]=eval_condition(clips[case],targets[case],heads,cfg)
                data.append(dict(low_name=low["name"],high_name=high["name"],
                                 low_sha256=low["sha256"],high_sha256=high["sha256"],
                                 cases=results,exact_k_control=K))
                print("S656 PAIR",index,"/8",
                      "E3_at_concurrent",results["E3_plus_E4_simultaneous"]["S614_original"]["E3_correct"],
                      "tone_mild",results["E3_plus_E4_simultaneous"]["tone_mild_2800"]["E3_correct"],
                      flush=True)
    totals={}
    for case in CASES:
        totals[case]={}
        for policy in POLICIES:
            r=[x["cases"][case][policy] for x in data]
            totals[case][policy]={k:sum(p[k] for p in r)
                                  for k in ("TP","FP","FN","E3_correct",
                                            "E4_correct","other_emitted",
                                            "emitted_E3","emitted_E4")}
        print("S656 CASE",case,json.dumps(totals[case]),flush=True)
    report=dict(status="S656_REAL_GUITARSET_CONTROLLED_OCTAVE_PAIR_TONE_FILTER",
        original_sources_isolated_GuitarSet_TRAIN_only=True,
        digitally_mixed_pairs_NOT_naturally_played_duets=True,
        uses_identical_E3_real_wave_in_single_and_overlay=True,
        example_E2_not_available=True,
        not_a_genuine_Stratocaster_Tone_measurement=True,
        unchanged_original_K_pitch_live=True,
        no_test_no_DEV_sources=True,
        sample_time_no_future_audio=True,
        pairs=len(data),
        totals=totals,
        pair_source=inventory,
        per_pair=data,
        automatic_promotion=False)
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S656 FINAL",json.dumps(totals),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        run(a.out)
