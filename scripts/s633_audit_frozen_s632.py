"""S633 independent replay *audit*, not model selection or training.

Frozen S632 explicit-cost Q models and exactly one preregistered
S632_assist_wait_p010 policy are replayed on the 17 original DEV songs.
Produce a fine-grained ledger of MIDI events added/removed, actual
newly matched reference NoteOn, lost matches, remaining FP and latency.
No TEST or source outside the original S614/S632 DEV split is read.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tempfile
import time
import zipfile
import joblib
import numpy as np

from causal_note.strict_live import Selection,load_heads
from causal_note.s632_risk_policy import RiskPolicy,causal_episodes,predicted_costs,apply_policy
from scripts.s626_train_sharp_native_onset import get_source
from scripts.s616_live_dev_ablation import ROOT,MODEL
from scripts.s613_prepare_strict_live import download,digest
from scripts.s613_train_strict_live import event_score,summary,offset_audit

SHA256S=(
    "207329be553ad179b461b725749e69244c080ac562b71364df6dc58066620218",
    "e57c818431e2544eba3d0b22d59d6df621cbac7637c7874df98424d8fb0a6a5b",
    "b54f631eaaeb0c31bc9f072bbfe2f6b76b9dcdd795f06401389c76a61824a4c6",
)
CHECK={"GuitarSet":dict(original=(998,585,582),new=(1001,585,579)),
       "URMP":dict(original=(243,149,256),new=(244,149,255))}


def normalize(pair_list):
    return {(int(i),int(j)) for i,j in pair_list}


def audit_source(row,heads,selection,costmodels):
    e=row["ends"];p=row["p"];k=row["k"];x=row["x"]
    candidate_episodes,frames=causal_episodes(row)
    cost_frames=predicted_costs(costmodels,frames)
    old,old_actions=apply_policy(row,cost_frames,selection,None)
    new,new_actions=apply_policy(row,cost_frames,selection,
        RiskPolicy(mode="assist",extra_wait_cost=.1,max_emit_cost=.65))
    original,matched_old=event_score(old,row["truth"])
    candidate,matched_new=event_score(new,row["truth"])
    old_truth={j for i,j in matched_old}
    new_truth={j for i,j in matched_new}
    gain=new_truth-old_truth
    losses=old_truth-new_truth
    old_on=[e for e in old if e["type"]=="note_on"]
    new_on=[e for e in new if e["type"]=="note_on"]
    as_key=lambda event:(int(event["pitch"]),int(event["decision_sample"]))
    native={as_key(e) for e in old_on}
    assisted={as_key(e) for e in new_on}
    old_idx={int(i):int(j) for i,j in matched_old}
    new_idx={int(i):int(j) for i,j in matched_new}
    lost_or_added=sorted(native.symmetric_difference(assisted))
    extra_onsets=sorted(assisted-native)
    deleted_onsets=sorted(native-assisted)
    corrections=[dict(pitch=int(row["truth"][i]["pitch"]),
                      reference_time_s=float(row["truth"][i]["time"]))
                 for i in sorted(gain)]
    regressions=[dict(pitch=int(row["truth"][i]["pitch"]),
                      reference_time_s=float(row["truth"][i]["time"]))
                 for i in sorted(losses)]
    off_old=offset_audit(old,row["truth"],matched_old,row["n"]/44100.)
    off_new=offset_audit(new,row["truth"],matched_new,row["n"]/44100.)
    return dict(name=row["name"],corpus=row["corpus"],
        original={k:original[k] for k in ("TP","FP","FN","F1")},
        revised={k:candidate[k] for k in ("TP","FP","FN","F1")},
        corrected_reference_onsets=corrections,
        regressed_reference_onsets=regressions,
        correction_count=len(gain),regression_count=len(losses),
        new_emission_count=len(extra_onsets),
        removed_emission_count=len(deleted_onsets),
        new_emissions=[dict(pitch=p,time_s=round(t/44100.,6)) for p,t in extra_onsets],
        removed_emissions=[dict(pitch=p,time_s=round(t/44100.,6)) for p,t in deleted_onsets],
        native_emission_count=len(old_on),assisted_emission_count=len(new_on),
        policy_risk_actions={key:new_actions[key] for key in ("waits","emits","rejects")},
        emitted_candidate_delay_ms=new_actions["delays_ms"],
        conditional_offset_before=off_old["conditional_offset_accuracy"],
        conditional_offset_after=off_new["conditional_offset_accuracy"])


def run(paths,out):
    if out.exists():raise FileExistsError("S633 audit output must be immutable")
    out.mkdir(parents=True)
    for path,sha in zip(paths,SHA256S):
        if digest(path.read_bytes())!=sha:
            raise ValueError("Wrong source checkpoint for S633 independent audit")
    models=[joblib.load(p) for p in paths]
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    original_sha=digest(MODEL.read_bytes())
    assert original_sha==frozen["model_sha256"]
    protocol=json.loads((ROOT/"protocol.json").read_text())
    wanted=[s for s in protocol["sources"]
            if s["split"]=="dev" and s["corpus"] in ("URMP","GuitarSet")]
    assert len(wanted)==17 and sum(s["corpus"]=="URMP" for s in wanted)==5
    original=load_heads(MODEL,expected_sha256=original_sha)
    sel=Selection(**original.metadata["selection"])
    started=time.perf_counter()
    records=[]
    with tempfile.TemporaryDirectory() as td:
        cache=Path(td);(cache/"data").mkdir()
        for spec in wanted:
            if spec["corpus"]=="URMP":
                row=get_source(spec,cache,original)
                records.append(audit_source(row,original,sel,models))
                print("S633 URMP",spec["name"],"new TP",
                    records[-1]["correction_count"],"lost TP",
                    records[-1]["regression_count"],flush=True)
        aud=protocol["guitar_archives"]["GuitarSet_audio"]
        ann=protocol["guitar_archives"]["GuitarSet_annotations"]
        download(aud["url"],cache/"data/GuitarSet_audio.zip",aud["size"],aud["sha256"])
        download(ann["url"],cache/"data/GuitarSet_annotations.zip",ann["size"],ann["sha256"])
        with zipfile.ZipFile(cache/"data/GuitarSet_audio.zip") as archive:
            for spec in wanted:
                if spec["corpus"]=="GuitarSet":
                    row=get_source(spec,cache,original,archive)
                    records.append(audit_source(row,original,sel,models))
                    print("S633 GUITAR",spec["name"],"new TP",
                        records[-1]["correction_count"],"lost TP",
                        records[-1]["regression_count"],flush=True)
    by={}
    for corpus in CHECK:
        selected=[r for r in records if r["corpus"]==corpus]
        current={key:sum(x["revised"][key] for x in selected) for key in ("TP","FP","FN")}
        old={key:sum(x["original"][key] for x in selected) for key in ("TP","FP","FN")}
        assert tuple(old.values())==CHECK[corpus]["original"],(corpus,old)
        assert tuple(current.values())==CHECK[corpus]["new"],(corpus,current)
        newly=sum(x["correction_count"] for x in selected)
        losses=sum(x["regression_count"] for x in selected)
        assert newly-losses == current["TP"]-old["TP"]
        by[corpus]=dict(original=old,revised=current,
                        new_truth_matches=newly,lost_truth_matches=losses,
                        new_emissions=sum(x["new_emission_count"] for x in selected),
                        removed_emissions=sum(x["removed_emission_count"] for x in selected),
                        regression_recordings=sum(x["revised"]["F1"]<x["original"]["F1"] for x in selected),
                        improved_recordings=sum(x["revised"]["F1"]>x["original"]["F1"] for x in selected))
    report=dict(status="S633_FIXED_S632_COST_POLICY_CORRECTIONS_REGRESSIONS_AUDITED",
                original_S80_S614_unchanged=True,
                original_S614_sha256=original_sha,
                unchanged_cost_model_hashes=list(SHA256S),
                exact_frozen_policy="S632_assist_wait_p010",
                no_training_in_this_run=True,
                original_test_not_opened=True,
                dev_preexposed_no_new_holdout_claim=True,
                by_corpus=by,sources=records,
                no_official_promotion=True,
                elapsed_seconds=round(time.perf_counter()-started,2))
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S633 INDEPENDENT REPLAY",json.dumps(by,indent=2),flush=True)


if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--models",nargs=3,type=Path,required=True)
    a.add_argument("--out",type=Path,required=True)
    opts=a.parse_args()
    run(opts.models,opts.out)
