"""S824 same 80 signed original S795 OOF, ablate and retrain predefined feature families.

Unlike S822's genuine independent PCM256 DSP, this diagnostic uses S708
posteriors and previously emitted MIDI from S795; never eligible to promote.
"""
from __future__ import annotations
import argparse,hashlib,json
from dataclasses import asdict
from pathlib import Path
import joblib,numpy as np
from causal_note.s823_legacy23_asof_midi256 import asof_indices
from causal_note.s824_feature_family_ablation import (
    SCHEMA as TEACHER_SCHEMA, MODES, UNIQUE_MODES, MASKS, train_all, choose_on_cal
)
from causal_note.s814_activek_teacher import active_score
from causal_note.s813_teacher_contract import TeacherEvidence,REQUIRED_METRICS,unification_gate
from scripts.s813_prepare_native256 import targets_at_ends,save_json,digest_file
from scripts.s815_train_activek256 import original_count,totals
SOURCE_SCHEMA="S800_FIT80_OOF_PER_FRAGMENT_MIDI256_V1"
HEADS=("general","bass","polyphonic","transient","harmonic")
# Frozen S800 schema/HEADS constants; importing its writer would transitively
# require torch despite this being a read-only sklearn replay.

SCHEMA="S824_S795_PREDECLARED_FAMILY_CAUSAL_256_FIT80_V1"
EXPECTED_S823_FOLD_RAW={"BN1":38601,"Funk2":36567,"Jazz1":31449,"Rock3":38953,"SS1":45135}
COMPS=("BN1","Funk2","Jazz1","Rock3","SS1")

def run(manifest_dir,traces,frozen_dir,out):
    if out.exists():raise FileExistsError("immutable S824 ablation")
    manifest_path=manifest_dir/"report.json"
    source=json.loads(manifest_path.read_text())
    if (source.get("schema")!=SOURCE_SCHEMA or not source.get("complete")
        or source.get("decision_clock_samples")!=1024
        or source.get("metric_clock_samples")!=256
        or len(source.get("fit_files",[]))!=80
        or any(source.get(field) for field in ("CAL_opened","HELD_opened","TEST_opened",
                   "cal_used_for_training","cal_used_for_selection"))):
        raise ValueError("only completed original S795/S800 80 OOF snapshots permitted")
    original_s823=json.loads((frozen_dir/"report.json").read_text())
    if (original_s823.get("schema")!="S823_S795_LEGACY_TRUE23_ASOF1024_AT_256_FIT80_V1"
        or not original_s823.get("complete")
        or original_s823["by_mode"]["raw23"]["correct"]!=190705
        or original_s823["by_mode"]["raw23"]["frames"]!=249411):
        raise AssertionError("signed original S823 report required")
    records=[];source_scores=[]
    for spec in sorted(source["fit_files"],key=lambda x:x["index"]):
        stem=f"fit_{spec['index']:03d}"
        zpath=traces/(stem+".npz")
        jpath=traces/(stem+".json")
        if (digest_file(zpath)!=spec["npz_sha256"]
            or digest_file(jpath)!=spec["json_sha256"]):
            raise AssertionError("S800 recording signature failed")
        metadata=json.loads(jpath.read_text())
        if (metadata["name"]!=spec["name"] or
            metadata["composition"]!=spec["composition"] or
            metadata["source_counts"]!=spec["source_counts"] or
            not metadata["head_training_fold_excluded"]):
            raise AssertionError("S800 snapshot metadata/parity or held-out fold changed")
        with np.load(zpath,allow_pickle=False) as z:
            if set(z.files)!=set(("features","onset_targets_fit",
                  "decision_ends","metric_ends"))|{"head_"+x for x in HEADS}:
                raise AssertionError("original S795 signed snapshot ABI mismatch")
            X=z["features"];decision_ends=z["decision_ends"];ends=z["metric_ends"]
        if X.shape!=(spec["fragment_frames"],49,23) or len(ends)!=spec["frames"]:
            raise AssertionError("S800 sample/frame count changed")
        # This is a source MIXED posteriors/notes acoustic feature; never an
        # independent PCM-only observation, despite causal resampling.
        indices=asof_indices(decision_ends,ends)
        _,k=targets_at_ends(metadata["fit_truth"],ends)
        if len(k)!=len(ends):
            raise AssertionError("mismatched GT and MIDI frame clock")
        original=original_count(metadata["source_events"],ends)
        historical=active_score(original,k)
        if (historical["correct"]!=metadata["source_counts"]["active_k_correct"]
            or historical["frames"]!=metadata["source_counts"]["active_frames"]):
            raise AssertionError("H0 source event MIDI256 signed parity failed")
        records.append(dict(index=spec["index"],name=spec["name"],
             composition=spec["composition"],texture=metadata["texture"],
             X=X,k=k,asof=indices,source=historical,
             valid_asof_frames=int(np.count_nonzero(indices>=0)),
             initial_missing=int(np.count_nonzero(indices<0)),
             unique_source_snapshots_used=int(len(np.unique(indices[indices>=0]))),
             legacy_1024_snapshot_count=len(decision_ends),
             source_sha256=spec["npz_sha256"]))
        source_scores.append(historical)
    if (len(records)!=80 or len({r["name"] for r in records})!=80
        or {r["composition"] for r in records}!=set(COMPS)
        or {r["index"] for r in records}!=set(range(80))
        or {r["composition"]:len([q for q in records if q["composition"]==r["composition"]]) for r in records}
           != {c:16 for c in COMPS}):
        raise AssertionError("GuitarSet 80 folds identity")
    base=totals(source_scores)
    if base["frames"]!=249411 or base["correct"]!=142213 or sum(len(x["k"]) for x in records)!=306536:
        raise AssertionError("H0 MIDI256 data original signed counts differ")
    out.mkdir(parents=True)
    (out/"models").mkdir()
    result=dict(schema=SCHEMA,teacher_schema=TEACHER_SCHEMA,complete=False,
         promotion=False,series="S824",source_run=39352880035,
         predeclared_feature_masks={k:list(v) for k,v in MASKS.items()},
         original_S823_raw23_expected_correct=190705,
         ablation_s823_exact_parity_per_composition=False,
         no_retraining_of_S708=True,all_models_shadow_only=True,
         manifest_sha256=digest_file(manifest_path),
         observed_legacy_source_clock_samples=1024,
         decision_hop_samples=256,metric_hop_samples=256,
         native_PCM256_audio_reextracted=False,
         S795_23_genuine_original_channels=True,
         S795_depends_on_S708_posteriors_K_and_source_MIDI=True,
         six_axis_independence_verified=False,independent_teacher_eligible=False,
         no_future_audio_or_predictions=True,
         added_lookahead_samples=0,added_hold_samples=0,
         source_MIDI_modified=False,production_LIVE_modified=False,
         original_H0_activeK=base,CAL_opened=False,HELD_opened=False,TEST_opened=False,
         research_fit_previously_explored=True,
         physical_latency_measured=False,unified_MIDI_measured=False,
         observations_without_snapshot=sum(r["initial_missing"] for r in records),
         source_observations=sum(r["legacy_1024_snapshot_count"] for r in records),
         five_composition_OOF=True,train_tracks_per_fold=48,inner_cal_tracks=16,
         outer_tracks=16,folds={},tracks=[])
    save_json(out/"report.json",result)
    for fold,c in enumerate(COMPS):
        inner=COMPS[(fold+1)%5]
        train=[r for r in records if r["composition"] not in (c,inner)]
        cal=[r for r in records if r["composition"]==inner]
        test=[r for r in records if r["composition"]==c]
        if tuple(map(len,(train,cal,test)))!=(48,16,16):
            raise AssertionError("48/16/16 outer composition fold changed")
        models=train_all(train,seed=823,cap=840)
        signed=original_s823["folds"][c]["candidate_models"]["raw23"]
        checkpoint=frozen_dir/signed["path"]
        if digest_file(checkpoint)!=signed["sha256"]:
            raise AssertionError("immutable S823 original raw23 checkpoint hash changed")
        frozen=joblib.load(checkpoint)
        if (frozen.mode!="raw23"
            or frozen.metadata["train_compositions"]!=sorted({r["composition"] for r in train})
            or frozen.metadata["feature_dim"]!=23):
            raise AssertionError("original S823 FIT fold lineage differs")
        from causal_note.s824_feature_family_ablation import Teacher
        models["frozen_s823"]=Teacher(frozen.model,"full23",dict(
            frozen.metadata,original_frozen_s823=True,promotion=False))
        if models["full23"].model is models["acoustic_only"].model:
            raise AssertionError("full and removed-feature models share a fit")
        selected,cal_results=choose_on_cal(cal,models)
        paths={}
        for mode in UNIQUE_MODES:
            m=models[mode]
            p=out/"models"/f"{c}_{mode}.joblib"
            joblib.dump(m,p,compress=3)
            paths[mode]=dict(path=p.relative_to(out).as_posix(),
                sha256=digest_file(p),metadata=m.metadata)
        result["folds"][c]=dict(inner_calibration=inner,
             train_compositions=sorted({r["composition"] for r in train}),
             candidate_models=paths,selected_by_inner_cal=selected,
             calibration_scores=cal_results)
        for row in test:
            pred={mode:models[mode].predict(row) for mode in (*MODES,"frozen_s823")}
            by_mode={mode:active_score(p,row["k"]) for mode,p in pred.items()}
            truth=np.asarray(row["k"])
            by_k={mode:{str(k):dict(correct=int(np.count_nonzero((p==k)&(truth==k))),
                     frames=int(np.count_nonzero(truth==k)),
                     undercount=int(np.count_nonzero((p<k)&(truth==k))),
                     overcount=int(np.count_nonzero((p>k)&(truth==k))))
                     for k in range(1,7)} for mode,p in pred.items()}
            result["tracks"].append(dict(index=row["index"],name=row["name"],
                composition=c,texture=row["texture"],source=row["source"],
                by_mode=by_mode,by_true_k=by_k,selected_by_CAL=selected,
                s795_source_trace_sha256=row["source_sha256"],
                first_midi256_frames_no_1024_observation=row["initial_missing"],
                original_S795_snapshots=row["legacy_1024_snapshot_count"]))
        fullscore=totals([r["by_mode"]["full23"] for r in result["tracks"] if r["composition"]==c])
        frozenscore=totals([r["by_mode"]["frozen_s823"] for r in result["tracks"] if r["composition"]==c])
        if frozenscore["correct"]!=EXPECTED_S823_FOLD_RAW[c]:
            raise AssertionError("frozen S823 original predictions not reproduced on "+c)
        result["folds"][c]["frozen_s823_sha256"]=signed["sha256"]
        result["folds"][c]["retrain_full_minus_frozen_correct"]=fullscore["correct"]-frozenscore["correct"]
        save_json(out/"report.json",result)
        print("S824_FOLD",c,"CAL",selected,{mode:totals([
            r["by_mode"][mode] for r in result["tracks"] if r["composition"]==c])
            for mode in (*UNIQUE_MODES,"frozen_s823")},flush=True)
    result["by_mode"]={mode:totals([t["by_mode"][mode] for t in result["tracks"]])
                       for mode in (*MODES,"frozen_s823")}
    result["cal_selected"]=totals([t["by_mode"][t["selected_by_CAL"]] for t in result["tracks"]])
    result["by_composition"]={c:{mode:totals([
        t["by_mode"][mode] for t in result["tracks"] if t["composition"]==c])
        for mode in (*MODES,"frozen_s823")} for c in COMPS}
    result["by_true_k"]={mode:{str(k):totals([
        r["by_true_k"][mode][str(k)] for r in result["tracks"]])
        for k in range(1,7)} for mode in (*MODES,"frozen_s823")}
    if (result["by_mode"]["frozen_s823"]["correct"]!=190705
        or result["by_mode"]["frozen_s823"]["frames"]!=249411):
        raise AssertionError("S823 original signed shadow model not reproduced")
    result["ablation_s823_exact_parity_per_composition"]=True
    result["frozen_s823_checkpoint_sha256_verified"]=True
    result["retrained_full_minus_frozen_correct"]=(result["by_mode"]["full23"]["correct"]-
        result["by_mode"]["frozen_s823"]["correct"])
    result["delta_vs_full_pp"]={mode:100*(result["by_mode"][mode]["accuracy"]-
        result["by_mode"]["full23"]["accuracy"]) for mode in MODES}
    result["cal_selection_only_in_inner_composition"]=True
    result["complete"]=True
    evidence=TeacherEvidence(axis="active_exact_k",score=result["cal_selected"]["accuracy"],
        metric_name=REQUIRED_METRICS["active_exact_k"],
        source_id=result["manifest_sha256"],
        evaluation_id=hashlib.sha256(SCHEMA.encode()).hexdigest(),
        evaluation_role="fit_research",generalization_verified=False,
        private_parameters_verified=True,private_state_verified=False,
        causal_observations_verified=True,inference_uses_labels=False)
    result["teacher_evidence"]=asdict(evidence)
    result["unification_gate"]=unification_gate([evidence])
    if result["unification_gate"]["allow_unification"]:
        raise AssertionError("legacy S708-dependent features cannot unify")
    save_json(out/"report.json",result)
    print("S824_COMPLETE",json.dumps({"BY_MASK":result["by_mode"],
         "CAL_SELECTED":result["cal_selected"],"DELTA_VS_FULL_PP":result["delta_vs_full_pp"],
         "H0":base,"legacy_1024_snapshots":result["source_observations"],
         "missing_first_1024":result["observations_without_snapshot"],
         "promotion":False},sort_keys=True),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for n in ("manifest-dir","traces","frozen-dir","out"):
        parser.add_argument("--"+n,required=True,type=Path)
    a=parser.parse_args()
    run(a.manifest_dir,a.traces,a.frozen_dir,a.out)
