"""S825: 5×(3 internal pitch OOF folds) native PCM256 multi-loop cardinality learner.

All S825 outputs share 249411 GT-active MIDI256 frames and signed FIT80 PCM.
No S708 or emitted MIDI in predictions, no other axis checkpoint loaded.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from dataclasses import asdict
from pathlib import Path
import joblib,numpy as np
from causal_note.s825_crossfit_activek import fit_teacher,SCHEMA as HEAD_SCHEMA,MODES
from causal_note.s814_activek_teacher import active_score
from causal_note.s813_teacher_contract import TeacherEvidence,REQUIRED_METRICS,unification_gate
from scripts.s813_prepare_native256 import SCHEMA as DATA_SCHEMA,digest_file,save_json
from scripts.s815_train_activek256 import original_count,totals

SCHEMA="S825_OOF_PRIVATE_PITCH_3_LOOP_NATIVE_MIDI256_FIT80_V1"
COMPS=("BN1","Funk2","Jazz1","Rock3","SS1")
EXPECTED_H0_CORRECT=142213
EXPECTED_ACTIVE=249411
EXPECTED_TOTAL_FRAMES=306536

def run(cache,traces,out):
    if out.exists():raise FileExistsError("immutable S825 output")
    e=json.loads((cache/"report.json").read_text())
    if (e.get("schema")!=DATA_SCHEMA or not e.get("complete")
        or len(e.get("tracks",[]))!=80
        or e.get("total_frames")!=EXPECTED_TOTAL_FRAMES
        or any(e.get(v)!=256 for v in
           ("feature_hop_samples","decision_hop_samples","metric_hop_samples"))
        or e.get("H0_events_or_posteriors_used_as_features")
        or any(e.get(k) for k in ("CAL_opened","HELD_opened","TEST_opened"))):
        raise AssertionError("not exactly genuine FIT80 MIDI256, unsupervised DSP")
    records=[];source=[]
    for row in e["tracks"]:
        data={}
        for key,f in row["files"].items():
            path=cache/f["file"]
            if digest_file(path)!=f["sha256"]:
                raise AssertionError("source PCM/audio truth file SHA drift")
            data[key]=np.load(path,mmap_mode="r",allow_pickle=False)
        trace=traces/f"fit_{row['index']:03d}.json"
        if digest_file(trace)!=row["original_json_sha256"]:
            raise AssertionError("S800 truth or source events modified")
        meta=json.loads(trace.read_text())
        if meta["name"]!=row["name"] or meta["composition"]!=row["composition"]:
            raise AssertionError("FIT source metadata swapped")
        old=original_count(meta["source_events"],data["ends"])
        historical=active_score(old,data["count_target"])
        old_signed=meta["source_counts"]
        if (historical["correct"]!=old_signed["active_k_correct"]
            or historical["frames"]!=old_signed["active_frames"]):
            raise AssertionError("original signed MIDI256 source truth parity lost")
        records.append(dict(X=data["features"],y=data["pitch_target"],
            k=data["count_target"],name=row["name"],index=row["index"],
            composition=row["composition"],texture=row["texture"],source=historical))
        source.append(historical)
    if (len(records)!=80 or len({r["name"] for r in records})!=80
        or set(r["composition"] for r in records)!=set(COMPS)
        or len(set(r["index"] for r in records))!=80):
        raise AssertionError("original 80 FIT guitar recordings not recovered")
    baseline=totals(source)
    if baseline["correct"]!=EXPECTED_H0_CORRECT or baseline["frames"]!=EXPECTED_ACTIVE:
        raise AssertionError("H0 57.0195% signed original FIT80 not recovered")
    out.mkdir(parents=True)
    (out/"models").mkdir();(out/"predictions").mkdir()
    report=dict(schema=SCHEMA,head_schema=HEAD_SCHEMA,series="S825",
        complete=False,promotion=False,target_axis="active_exact_k",
        scope="shadow_private_pitch_cardinality_crossfit_not_unified_MIDI",
        original_fit80_active_k=baseline,source_cache_sha256=digest_file(cache/"report.json"),
        sample_rate_hz=44100,feature_hop_samples=256,decision_hop_samples=256,
        metric_hop_samples=256,resolution_ms=256/44100*1000,
        strict_training_oof_pitch=True,three_ablation_loops_declared=list(MODES),
        S708_posteriors_or_MIDI_features_used=False,
        source_feature_hop_is_256=True,
        original_S708_MIDI_modified=False,other_five_axes_modified=False,
        production_LIVE_modified=False,
        added_lookahead_samples=0,added_hold_samples=0,
        CAL_opened=False,HELD_opened=False,TEST_opened=False,
        fit_historically_explored=True,external_generalization_verified=False,
        physical_latency_measured=False,unified_midi_measured=False,
        train_tracks_per_fold=48,inner_calibration_tracks=16,
        held_out_tracks=16,folds={},tracks=[])
    save_json(out/"report.json",report)
    for i,outer in enumerate(COMPS):
        inner=COMPS[(i+1)%len(COMPS)]
        tr=[r for r in records if r["composition"] not in (outer,inner)]
        cal=[r for r in records if r["composition"]==inner]
        test=[r for r in records if r["composition"]==outer]
        if tuple(map(len,(tr,cal,test)))!=(48,16,16):
            raise AssertionError("48 TRAIN 16 CAL 16 outer evaluation contract")
        now=time.perf_counter()
        teacher=fit_teacher(tr,cal,seed=825,pitch_max=650,count_max=560)
        if (outer in teacher.metadata["train_compositions"]
            or outer in teacher.metadata["calibration_compositions"]
            or inner in teacher.metadata["train_compositions"]
            or teacher.metadata["S708_posteriors_or_MIDI_used"]):
            raise AssertionError("outer/CAL leak or imported foreign axis")
        selected=teacher.selector
        checkpoint=out/"models"/f"{outer}_private_pitch_3loop.joblib"
        joblib.dump(teacher,checkpoint,compress=3)
        report["folds"][outer]=dict(inner_calibration=inner,
            held_out=outer,checkpoint=checkpoint.relative_to(out).as_posix(),
            checkpoint_sha256=digest_file(checkpoint),selected_by_inner_cal=selected,
            calibration_counts=teacher.metadata["inner_calibration_scores"],
            training_seconds=time.perf_counter()-now,
            train_compositions=sorted({r["composition"] for r in tr}),
            crossfit=teacher.metadata["crossfit_oof_composition_proofs"],
            teacher_metadata=teacher.metadata)
        for row in test:
            outputs=teacher.predict_all(row["X"])
            truth=np.asarray(row["k"],np.uint8)
            scores={mode:active_score(z,truth) for mode,z in outputs.items()}
            by_k={mode:{str(k):dict(correct=int(np.count_nonzero((pred==truth)&(truth==k))),
                frames=int(np.count_nonzero(truth==k)),
                undercount=int(np.count_nonzero((pred<truth)&(truth==k))),
                overcount=int(np.count_nonzero((pred>truth)&(truth==k))))
                for k in range(1,7)} for mode,pred in outputs.items()}
            path=out/"predictions"/f"fit_{row['index']:03d}.npz"
            np.savez_compressed(path,**{f"{mode}_K":z for mode,z in outputs.items()})
            report["tracks"].append(dict(index=row["index"],name=row["name"],
                composition=outer,texture=row["texture"],source=row["source"],
                active_k=scores,by_true_k=by_k,cal_chosen=selected,
                predictions=path.relative_to(out).as_posix(),
                predictions_sha256=digest_file(path)))
        save_json(out/"report.json",report)
        print("S825_FOLD",outer,"cal_chosen",selected,
              {mode:totals([t["active_k"][mode] for t in report["tracks"] if t["composition"]==outer])
                  for mode in MODES},flush=True)
    report["by_mode"]={mode:totals([r["active_k"][mode] for r in report["tracks"]])
          for mode in MODES}
    report["cal_chosen_active_k"]=totals([r["active_k"][r["cal_chosen"]] for r in report["tracks"]])
    report["by_composition"]={comp:{
        mode:totals([r["active_k"][mode] for r in report["tracks"] if r["composition"]==comp])
        for mode in MODES} for comp in COMPS}
    report["by_true_k"]={mode:{str(k):totals([r["by_true_k"][mode][str(k)] for r in report["tracks"]])
                              for k in range(1,7)} for mode in MODES}
    # Historical S815 measured independently with the exact same source and denominator.
    report["historical_s815_comparator"]=dict(correct=141946,frames=EXPECTED_ACTIVE,
        accuracy=141946/EXPECTED_ACTIVE,undercount=65530,overcount=41935)
    report["delta_vs_H0_pp"]={mode:100*(report["by_mode"][mode]["accuracy"]-
        baseline["accuracy"]) for mode in MODES}
    report["delta_vs_S815_pp"]={mode:100*(report["by_mode"][mode]["accuracy"]-
        141946/249411) for mode in MODES}
    report["complete"]=True
    evidence=TeacherEvidence(axis="active_exact_k",
        score=report["cal_chosen_active_k"]["accuracy"],
        metric_name=REQUIRED_METRICS["active_exact_k"],
        source_id=report["source_cache_sha256"],
        evaluation_id=hashlib.sha256(json.dumps(
            {"series":SCHEMA,"folds":COMPS},sort_keys=True).encode()).hexdigest(),
        evaluation_role="fit_research",generalization_verified=False,
        private_parameters_verified=True,private_state_verified=True,
        causal_observations_verified=True,inference_uses_labels=False)
    report["teacher_evidence"]=asdict(evidence)
    report["unification_gate"]=unification_gate([evidence])
    if report["unification_gate"]["allow_unification"]:
        raise AssertionError("cannot promote FIT80 research")
    save_json(out/"report.json",report)
    print("S825_COMPLETE",json.dumps({"by_mode":report["by_mode"],
        "cal_selected":report["cal_chosen_active_k"],
        "H0":baseline,"delta_vs_H0_pp":report["delta_vs_H0_pp"]},
        sort_keys=True),flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser()
    for key in ("cache","traces","out"):
        a.add_argument("--"+key,required=True,type=Path)
    p=a.parse_args()
    run(p.cache,p.traces,p.out)
