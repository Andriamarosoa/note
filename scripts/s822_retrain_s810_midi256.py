"""S822 authenticated five-fold train/test of S810-style supervised K at MIDI256.

All three S822 outputs use SAME ground truth 249411 active frames; direct
RAW23 vs AGG61 versus an inner-CAL-only chosen one. No MIDI event synthesis,
no temporal retiming and no other axis touched.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from dataclasses import asdict
from pathlib import Path
import joblib,numpy as np
from causal_note.s822_s810_style_teacher import fit_teacher,choose,SCHEMA as HEAD_SCHEMA,MODES
from causal_note.s814_activek_teacher import active_score
from causal_note.s813_teacher_contract import TeacherEvidence,REQUIRED_METRICS,unification_gate
from scripts.s813_prepare_native256 import SCHEMA as DATA_SCHEMA,digest_file,save_json
from scripts.s815_train_activek256 import original_count,totals

SCHEMA="S822_S810_STYLE_AGG61_ABLATION_NATIVE_MIDI256_FIT80_V1"
COMPS=("BN1","Funk2","Jazz1","Rock3","SS1")
EXPECTED_H0_CORRECT=142213
EXPECTED_ACTIVE=249411
EXPECTED_TOTAL_FRAMES=306536

def run(cache,traces,out):
    if out.exists():raise FileExistsError("immutable S822 output")
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
        records.append(dict(X=data["features"],k=data["count_target"],
            name=row["name"],index=row["index"],composition=row["composition"],
            texture=row["texture"],source=historical))
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
    report=dict(schema=SCHEMA,head_schema=HEAD_SCHEMA,series="S822",
        complete=False,promotion=False,target_axis="active_exact_k",
        scope="shadow_independent_activeK_not_unified_MIDI",
        original_fit80_active_k=baseline,source_cache_sha256=digest_file(cache/"report.json"),
        sample_rate_hz=44100,feature_hop_samples=256,decision_hop_samples=256,
        metric_hop_samples=256,resolution_ms=256/44100*1000,
        s810_style_61_aggregation_equal=True,
        historical_s810_original_1024_channel_abi_equal=False,
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
        heads={mode:fit_teacher(tr,mode,seed=822,cap=1440) for mode in MODES}
        if heads["raw23"].model is heads["agg61"].model:
            raise AssertionError("heads must have separate trained weights")
        selected,cal_counts=choose(cal,heads)
        files={}
        for mode,model in heads.items():
            p=out/"models"/f"{outer}_{mode}_activek.joblib"
            joblib.dump(model,p,compress=3)
            files[mode]=dict(path=p.relative_to(out).as_posix(),
                sha256=digest_file(p),metadata=model.metadata)
        report["folds"][outer]=dict(inner_calibration=inner,
            held_out=outer,models=files,selected_by_inner_cal=selected,
            calibration_counts=cal_counts,training_seconds=time.perf_counter()-now,
            feature_references="S810 formula on NEW S813 acoustic 23 channels",
            train_compositions=sorted({r["composition"] for r in tr}))
        for row in test:
            outputs={mode:heads[mode].predict(row["X"]) for mode in MODES}
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
        print("S822_FOLD",outer,"cal_chosen",selected,
              {mode:totals([t["active_k"][mode] for t in report["tracks"] if t["composition"]==outer])
                  for mode in MODES},flush=True)
    report["raw23_active_k"]=totals([r["active_k"]["raw23"] for r in report["tracks"]])
    report["agg61_active_k"]=totals([r["active_k"]["agg61"] for r in report["tracks"]])
    report["cal_chosen_active_k"]=totals([r["active_k"][r["cal_chosen"]] for r in report["tracks"]])
    report["by_composition"]={comp:{
        mode:totals([r["active_k"][mode] for r in report["tracks"] if r["composition"]==comp])
        for mode in MODES} for comp in COMPS}
    report["by_true_k"]={mode:{str(k):totals([r["by_true_k"][mode][str(k)] for r in report["tracks"]])
                              for k in range(1,7)} for mode in MODES}
    # Historical S815 measured independently with the exact same source and denominator.
    report["historical_s815_comparator"]=dict(correct=141946,frames=EXPECTED_ACTIVE,
        accuracy=141946/EXPECTED_ACTIVE,undercount=65530,overcount=41935)
    report["delta_agg_minus_raw_pp"]=100*(report["agg61_active_k"]["accuracy"]-
        report["raw23_active_k"]["accuracy"])
    report["delta_agg_minus_H0_pp"]=100*(report["agg61_active_k"]["accuracy"]-
        baseline["accuracy"])
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
    print("S822_COMPLETE",json.dumps({"raw23":report["raw23_active_k"],
        "agg61":report["agg61_active_k"],
        "cal_selected":report["cal_chosen_active_k"],
        "H0":baseline,"delta_agg_raw_pp":report["delta_agg_minus_raw_pp"]},
        sort_keys=True),flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser()
    for key in ("cache","traces","out"):
        a.add_argument("--"+key,required=True,type=Path)
    p=a.parse_args()
    run(p.cache,p.traces,p.out)
