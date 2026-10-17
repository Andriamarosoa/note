"""S687: cross-composition robustness of S686's causal onset novelty rescue.

Preregister S686 thresholds, score ALL nine disjoint TRAIN compositions;
reject any candidate with aggregate F1/Exact-K regression, excessive false
onsets, or poor group consistency. Then check frozen S675 absolute offsets,
conditional accuracy, full-note F1, long-note success. DEV is never used to
select a candidate, and original LIVE remains untouched.
"""
from __future__ import annotations
import argparse,json,tempfile,zipfile,time
from collections import Counter
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from causal_note.strict_live import Selection,load_heads
from scripts.s613_prepare_strict_live import download,digest
from scripts.s672_causal_offset_timing_grid import ROOT,MODEL
from scripts.s682_onset_four_lock_guard import aggregate,full_aggregate,safe_full
from scripts.s686_causal_fundamental_novelty import POLICIES,record


def evidence(rows,policies):
    sums={p:aggregate([r["scores"] for r in rows],p) for p in policies}
    grouped={}
    for comp in sorted({r["composition"] for r in rows}):
        group=[r for r in rows if r["composition"]==comp]
        grouped[comp]={p:aggregate([r["scores"] for r in group],p)
                       for p in policies}
    return sums,grouped


def choose_train_only(sums,groups):
    baseline=sums["native"]
    audit={}
    accepted=[]
    for policy in POLICIES[1:]:
        cand=sums[policy]
        improvements=sum(groups[c][policy]["onset_F1"]>=
                         groups[c]["native"]["onset_F1"] for c in groups)
        decline=max(groups[c]["native"]["onset_F1"]-
                    groups[c][policy]["onset_F1"] for c in groups)
        tp_delta=cand["tp"]-baseline["tp"]
        fp_delta=cand["fp"]-baseline["fp"]
        approved=(tp_delta>=1 and fp_delta<=2*tp_delta and
                  cand["K_frames_correct"]>=baseline["K_frames_correct"] and
                  cand["onset_F1"]>baseline["onset_F1"] and
                  improvements>=6 and decline<.009)
        audit[policy]=dict(TP_gain=tp_delta,FP_gain=fp_delta,
                          compositions_not_worse=improvements,
                          worst_family_F1_loss=decline,
                          train_onset_F1=cand["onset_F1"],
                          train_exactK=cand["decoded_exact_k_accuracy"],
                          approved=bool(approved))
        if approved:accepted.append(policy)
    selected=max(accepted,key=lambda p:(sums[p]["onset_F1"],
                   sums[p]["K_frames_correct"])) if accepted else "native"
    return selected,audit,accepted


def run(out,checkpoint):
    if out.exists():raise FileExistsError("S687 immutable report exists")
    start=time.perf_counter()
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    proto=json.loads((ROOT/"protocol.json").read_text())
    assert digest(MODEL.read_bytes())==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    train=[s for s in proto["sources"]
           if s["corpus"]=="GuitarSet" and s["split"]=="train"]
    dev=[s for s in proto["sources"]
         if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    assert len(train)==36 and len(dev)==12 and len({x["composition"] for x in train})==9
    heads=load_heads(MODEL,expected_sha256=frozen["model_sha256"])
    sel=Selection(**heads.metadata["selection"])
    from causal_note.s672_live_offset import S672TrialNoteOff
    hazard=S672TrialNoteOff.load(checkpoint,heads)
    assert hazard.gate==.60 and hazard.persistence==1
    report=dict(schema="S687_CAUSAL_MULTI_COMPOSITION_ONSET_ROBUSTNESS",
        frozen_original_sha256=frozen["model_sha256"],
        S675_frozen_sha256=digest(checkpoint.read_bytes()),
        originally_S686_novelty_policies_preregistered=True,
        no_retraining_on_DEV=True,live_unmodified=True,
        K_head_pitch_heads_frozen=True,
        sustain_release_heads_frozen=True,
        K0_to_K6_native_logits_unchanged=True,
        causal_PCM_past_only=True,
        TEST_unopened=True,DEV_previously_exposed=True)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/"data").mkdir()
        for key,file in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            source=proto["guitar_archives"][key]
            download(source["url"],root/"data"/file,
                     source["size"],source["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as zipaudio:
            records=[]
            for i,spec in enumerate(train,1):
                records.append(record(spec,root,zipaudio,heads,sel,POLICIES))
                print("S687 TRAIN 9 FAMILIES",i,"/36",spec["name"],flush=True)
            totals,groups=evidence(records,POLICIES)
            selected,vetoes,accepted=choose_train_only(totals,groups)
            report["TRAIN_all_policy_scores"]=totals
            report["TRAIN_each_composition"]=groups
            report["TRAIN_policy_vetoes"]=vetoes
            report["TRAIN_selected"]=selected
            report["TRAIN_accepted"]=accepted
            print("S687 TRAIN FROZEN",selected,
                  json.dumps(dict(original=totals["native"],
                        candidate=totals[selected],passing=accepted)),flush=True)
            variants=("native",) if selected=="native" else ("native",selected)
            train_full=[]
            for i,spec in enumerate(train,1):
                row=record(spec,root,zipaudio,heads,sel,variants,hazard.model)
                train_full.append(row)
                print("S687 TRAIN S675",i,"/36",spec["name"],flush=True)
            full_train={p:full_aggregate([row["release"] for row in train_full],p)
                        for p in variants}
            report["TRAIN_S675_four_locks"]=full_train
            train_safe=bool(selected!="native" and safe_full(full_train["native"],
                                                               full_train[selected]))
            report["TRAIN_offsets_safe"]=train_safe
            print("S687 TRAIN FULL NOTE LOCK",train_safe,flush=True)
            devrows=[]
            for i,spec in enumerate(dev,1):
                devrows.append(record(spec,root,zipaudio,heads,sel,variants,hazard.model))
                print("S687 DEV",i,"/12",spec["name"],flush=True)
            dv={p:aggregate([r["scores"] for r in devrows],p) for p in variants}
            dvfull={p:full_aggregate([r["release"] for r in devrows],p)
                    for p in variants}
            report["DEV_onset_K"]=dv
            report["DEV_S675_full_note"]=dvfull
            report["DEV_by_recording"]=devrows
    safe=bool(selected!="native" and train_safe and
              dv[selected]["onset_F1"]>dv["native"]["onset_F1"] and
              dv[selected]["K_frames_correct"]>=dv["native"]["K_frames_correct"] and
              safe_full(dvfull["native"],dvfull[selected]))
    report["four_locks_passed"]=safe
    report["promoted"]=False
    report["elapsed_seconds"]=round(time.perf_counter()-start,2)
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S687 FINAL",json.dumps(dict(selected=selected,
          source_original=dv["native"],candidate=dv[selected],
          S675_original=dvfull["native"],S675_candidate=dvfull[selected],
          train_offsets_safe=train_safe,four_locks=safe)),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--checkpoint",required=True,type=Path)
    a=parser.parse_args()
    with threadpool_limits(limits=1):
        run(a.out,a.checkpoint)
