"""H_FP_CAL_GRID_04: recalibrate (CAL only) 10 SHA-frozen trained MLP69 models.

Original two science jobs (#39900882849) failed because ALL 10 CAL
thresholds were zero. Original coarse minimum nonzero was .05, while
at .05 the TP-loss guard was already violated. This experiment changes
ONLY the granularity of the CAL cut grid, NOT neural weights, features,
data, composition splits, scoring definitions, scientific gates, latency,
or any official MIDI path. The original 70/74D comparator weights are
also SHA-identical. New CAL cuts are frozen BEFORE any OUTER replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import joblib
import numpy as np

from scripts.train_onset_fp_discriminator256 import (
    COMPS, load_signals, load_reference, approximate_score,
    MIN_CAL_FP_REMOVAL, MAX_CAL_TP_LOSS_FRACTION,
)

HYPOTHESIS_ISSUE=145
MODES=("track_uniform","track_class")
FINE_CUTS=(
    0., .00001, .00003, .00005, .0001, .0002, .0003, .0005,
    .0007, .001, .0015, .002, .003, .004, .005, .0075,
    .01, .015, .02, .025, .03, .035, .04, .045, .05,
    .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6,
    .65, .7, .75, .8, .85, .9, .95, .97, .99,
)
HISTORIC_69=dict(tp=8127,fp=4519,fn=3140)
HISTORIC_FOUR=dict(tp=8160,fp=4605,fn=3107)

def digest(file):
    h=hashlib.sha256()
    with Path(file).open("rb") as f:
        for block in iter(lambda:f.read(2*1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def run(signals, four_reference, old_models, mode, out):
    if mode not in MODES:
        raise ValueError("unregistered hypothesis #145 arm")
    out=Path(out)
    if out.exists():
        raise FileExistsError("scientific outputs must be immutable")
    rows=load_signals(signals)
    references=load_reference(four_reference)
    if ({r["track"] for r in rows}!=set(references)
        or any(references[r["track"]]["comp"]!=r["comp"] for r in rows)):
        raise AssertionError("CAL rows must match signed original 80 tracks")
    root=Path(old_models)
    original_report=json.loads((root/"report.json").read_text())
    if (original_report.get("mode")!=mode or
        original_report.get("models_really_trained")!=5 or
        original_report.get("training_hop_samples")!=256 or
        original_report.get("complete") is not True or
        len(original_report.get("folds",()))!=5):
        raise ValueError("wrong SHA-pinned original science arm")
    out.mkdir(parents=True)
    checkpoints=out/"checkpoints"
    checkpoints.mkdir()
    source_checkpoints=root/"checkpoints"
    results=[]
    for fold_i,outer in enumerate(COMPS):
        cal=COMPS[(fold_i+1)%len(COMPS)]
        source_fold=next(
            f for f in original_report["folds"] if f["outer"]==outer)
        if source_fold["cal"]!=cal or source_fold["selected_cut"]!=0.:
            raise AssertionError("source model must be from failed coarse-CAL case")
        stem=source_checkpoints/f"fp_control69_outer_{outer}"
        model_path=stem.with_suffix(".joblib")
        meta=json.loads(stem.with_suffix(".json").read_text())
        if (meta["outer_composition"]!=outer or
            meta["cal_composition"]!=cal or
            meta["sampling_mode"]!=mode or
            meta["trained_hop_samples"]!=256 or
            meta["calibrated_cut"]!=0. or
            outer in meta["train_compositions"] or
            cal in meta["train_compositions"]):
            raise AssertionError("uncertified TRAIN/CAL/OUTER or cutoff provenance")
        sha=digest(model_path)
        if (sha!=meta["checkpoint_sha256"] or
            sha!=source_fold["checkpoint_sha256"]):
            raise AssertionError("the genuine source MLP69 checkpoint changed")
        # A project-owned, SHA-checked sklearn model. No training or refitting.
        network=joblib.load(model_path)
        if network.n_features_in_!=69:
            raise AssertionError("wrong 69D native256 model checkpoint")
        cal_indices=[i for i,r in enumerate(rows) if r["comp"]==cal]
        if len({rows[i]["track"] for i in cal_indices})!=16:
            raise AssertionError("CAL composition must have 16 tracks")
        matrix=np.stack([rows[i]["X"][:69] for i in cal_indices])
        forecast=np.ones(len(rows),np.float32)
        forecast[cal_indices]=network.predict_proba(matrix)[:,1]
        original=approximate_score(rows,references,cal,forecast,0.)
        original_tp=sum(rows[i]["y"] for i in cal_indices)
        candidates=[]
        diagnostic=[]
        for cut in FINE_CUTS:
            result=approximate_score(rows,references,cal,forecast,cut)
            admissible=(
                cut==0. or
                (result["removed_fp"]>=MIN_CAL_FP_REMOVAL
                 and result["removed_tp"]<=
                     MAX_CAL_TP_LOSS_FRACTION*original_tp
                 and result["f1"]>original["f1"]))
            diagnostic.append(dict(
                cut=cut,removed_tp=result["removed_tp"],
                removed_fp=result["removed_fp"],
                f1_CAL_event_only_proxy=result["f1"],
                CAL_admissible=bool(admissible)))
            if admissible:
                candidates.append(dict(cut=cut,score=result))
        chosen=max(candidates,key=lambda p:(
            p["score"]["f1"],
            -p["score"]["removed_tp"],
            -p["cut"]))
        dst=checkpoints/model_path.name
        shutil.copy2(model_path,dst)
        if digest(dst)!=sha:
            raise AssertionError("do not train or edit model weights")
        metadata=dict(meta)
        metadata["calibrated_cut"]=float(chosen["cut"])
        metadata["source_coarse_cut"]=0.
        metadata["fine_CAL_only_preregistered_issue"]=HYPOTHESIS_ISSUE
        metadata["fine_grid"]=list(FINE_CUTS)
        metadata["only_threshold_changed_no_weights_trained"]=True
        metadata["same_science_CAL_rules"]=True
        metadata["source_trained_checkpoint_sha256"]=sha
        metadata["cal_only_proxy"]=chosen["score"]
        (checkpoints/(stem.name+".json")).write_text(
            json.dumps(metadata,indent=2)+"\n")
        for comparator in ("control70","temporal74"):
            prefix=source_checkpoints/f"fp_{comparator}_outer_{outer}"
            original_model=prefix.with_suffix(".joblib")
            original_meta=prefix.with_suffix(".json")
            info=json.loads(original_meta.read_text())
            if (info["outer_composition"]!=outer or
                info["variant"]!=comparator or
                digest(original_model)!=info["checkpoint_sha256"]):
                raise AssertionError("frozen control ablation SHA changed")
            shutil.copy2(original_model,checkpoints/original_model.name)
            shutil.copy2(original_meta,checkpoints/original_meta.name)
            if digest(checkpoints/original_model.name)!=info["checkpoint_sha256"]:
                raise AssertionError("copied control ablation SHA changed")
        record=dict(outer=outer,cal=cal,
            original_cut=0.,fine_CAL_cut=chosen["cut"],
            native256_model_sha256=sha,
            cal_baseline_proxy=original,
            fine_cal_selected_proxy=chosen["score"],
            all_CAL_cut_diagnostics=diagnostic,
            any_nonzero_cut_eligible=any(
                p["cut"]>0 for p in candidates),
            selected_cut_nonzero=bool(chosen["cut"]>0),
            CAL_truth_only_used_to_select_cut=True,
            OUTER_truth_not_used_in_this_script=True)
        results.append(record)
        print("ONSET_FP_FINE_CAL_GRID_ISSUE145_FREEZE",json.dumps({
            k:record[k] for k in ("outer","cal","original_cut",
                                 "fine_CAL_cut","any_nonzero_cut_eligible",
                                 "cal_baseline_proxy","fine_cal_selected_proxy")
        }),flush=True)
    report=dict(schema="ONSET_FP_FINE_CAL_ONLY_ISSUE145_V1",
        complete=True,mode=mode,
        scientific_hypothesis_issue=HYPOTHESIS_ISSUE,
        frozen_source_weights_sha256=True,
        model_weights_ever_retrained=False,
        unchanged_source_TRAIN_split=True,
        same_original_CAL_selection_rules=True,
        OUTER_unseen_by_this_recalibration_script=True,
        historical_OUTER_already_seen_in_prior_experiments=True,
        same_80_real_WAVs_to_be_replayed=True,
        existing_control69_baseline=HISTORIC_69,
        existing_four_head_baseline=HISTORIC_FOUR,
        n_folds_with_nonzero_cut=sum(x["selected_cut_nonzero"] for x in results),
        folds=results,
        real_audio_replay_completed=False,
        no_other_five_axes_changed=True,
        no_official_MIDI_changed=True,
        promoted=False)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("ONSET_FP_FINE_CAL_ONLY_NOT_REAL_WAV_F1",
          json.dumps({k:report[k] for k in (
              "mode","n_folds_with_nonzero_cut",
              "frozen_source_weights_sha256",
              "historical_OUTER_already_seen_in_prior_experiments")}),
          flush=True)
    return report

def main():
    p=argparse.ArgumentParser()
    for name in ("signals","four-reference","old-models","out"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--mode",choices=MODES,required=True)
    a=p.parse_args()
    return run(a.signals,a.four_reference,a.old_models,a.mode,a.out)
if __name__=="__main__":
    main()
