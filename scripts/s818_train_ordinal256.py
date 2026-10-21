"""S818: authenticated FIT80 six-ordinal-head acoustic experiment.

Freeze S815's three private Active-K models for comparison only; fit
six brand-new independently trained K>=j ordinal classifiers for Active-K. Same 48/16/16 five composition folds;
strict MIDI256, no S708 output changes, no promotion.
"""
from __future__ import annotations
import argparse,hashlib,json
from dataclasses import asdict
from pathlib import Path
import joblib
import numpy as np

from causal_note.s818_activek_ordinal import fit_teacher,SCHEMA as TEACHER_SCHEMA
from causal_note.s814_activek_teacher import active_score
from causal_note.s813_teacher_contract import TeacherEvidence,REQUIRED_METRICS,unification_gate
from scripts.s813_prepare_native256 import digest_file,save_json
from scripts.s814_train_activek256 import original_count,totals

SCHEMA="S818_REAL_FIT80_SIX_INDEPENDENT_ORDINAL_ACTIVEK_HEADS_MIDI256_V1"
COMPS=("BN1","Funk2","Jazz1","Rock3","SS1")


def class_scores(pred,target):
    pred=np.asarray(pred)
    target=np.asarray(target)
    return {str(k):dict(correct=int(np.count_nonzero((pred==k)&(target==k))),
        frames=int(np.count_nonzero(target==k)),
        undercount=int(np.count_nonzero((pred<k)&(target==k))),
        overcount=int(np.count_nonzero((pred>k)&(target==k))))
        for k in range(7)}


def run(cache,traces,previous_dir,out):
    if out.exists():
        raise FileExistsError("immutable S818 experiment already exists")
    fresh=json.loads((cache/"report.json").read_text())
    prior=json.loads((previous_dir/"report.json").read_text())
    if (fresh.get("schema")!="S813_FIT80_FRESH_UNSUPERVISED_DSP_MIDI256_V1"
        or not fresh.get("complete") or len(fresh.get("tracks",[]))!=80
        or prior.get("schema")!="S815_ACTIVE_EXACT_K_REAL_FRESH256_FIT80_V1"
        or not prior.get("complete") or len(prior.get("tracks",[]))!=80
        or any(prior.get(key) for key in ("CAL_opened","HELD_opened","TEST_opened","promotion"))):
        raise ValueError("original S815 research and S813 native256 source required")
    prior_cache_path=previous_dir.parent/"fresh256"/"report.json"
    if digest_file(prior_cache_path)!=prior["source_cache_sha256"]:
        raise AssertionError("old S815 training extraction identity mismatch")
    oldcache=json.loads(prior_cache_path.read_text())
    for key in ("source_manifest_sha256","audio_archive_sha256",
                "total_frames","sample_rate_hz","feature_dimension",
                "feature_hop_samples","decision_hop_samples","metric_hop_samples"):
        if fresh[key]!=oldcache[key]:
            raise AssertionError("immutable source evidence changed "+key)
    identity=lambda rows:sorted((int(r["index"]),r["name"],r["composition"],
        int(r["frames"]),r["original_json_sha256"],r["audio_member_sha256"]) for r in rows)
    if identity(fresh["tracks"])!=identity(oldcache["tracks"]):
        raise AssertionError("original audio/annotation metadata changed")
    if (prior["teacher_active_exact_k"]["correct"]!=141946
        or prior["original_fit80_active_k"]["correct"]!=142213):
        raise AssertionError("wrong published S815 baseline")
    saved={int(t["index"]):t for t in prior["tracks"]}
    records=[]
    for row in fresh["tracks"]:
        files=row["files"]
        for f in files.values():
            if digest_file(cache/f["file"])!=f["sha256"]:
                raise AssertionError("corrupted reproducible source cache")
        source=traces/f"fit_{row['index']:03d}.json"
        if digest_file(source)!=row["original_json_sha256"]:
            raise AssertionError("corrupted signed original metadata")
        d=json.loads(source.read_text())
        if d["name"]!=row["name"] or d["composition"]!=row["composition"]:
            raise AssertionError("wrong signed track")
        arrays={name:np.load(cache/info["file"],mmap_mode="r",allow_pickle=False)
                for name,info in files.items()}
        k=arrays["count_target"]
        src=active_score(original_count(d["source_events"],arrays["ends"]),k)
        if src!=saved[row["index"]]["source"] or src["correct"]!=d["source_counts"]["active_k_correct"]:
            raise AssertionError("S704 MIDI historical event identity failed")
        records.append(dict(index=row["index"],name=row["name"],composition=row["composition"],
             X=arrays["features"],k=k,ends=arrays["ends"],source=src))
    if tuple(sorted({r["composition"] for r in records}))!=COMPS:
        raise ValueError("expected five original disjoint compositions")
    out.mkdir(parents=True)
    (out/"models").mkdir()
    (out/"predictions").mkdir()
    report=dict(schema=SCHEMA,teacher_schema=TEACHER_SCHEMA,complete=False,
      original_S815_run=39536863317,original_s815_report_sha256=digest_file(previous_dir/"report.json"),
      original_s815_cache_sha256=digest_file(prior_cache_path),
      new_cache_sha256=digest_file(cache/"report.json"),
      sample_rate_hz=44100,feature_hop_samples=256,decision_hop_samples=256,
      metric_hop_samples=256,resolution_ms=256*1000/44100,
      six_independently_fitted_ordinal_heads=True, previous_three_heads_frozen_for_comparison=True,
      fresh_PCM_history_only=True,added_lookahead_samples=0,extra_hold_samples=0,
      source_MIDI_modified=False,other_characteristics_modified=False,
      CAL_opened=False,HELD_opened=False,TEST_opened=False,
      historically_explored_FIT80=True,external_generalization_verified=False,
      unified_MIDI_measured=False,live_physical_latency_measured=False,
      production_LIVE_modified=False,promotion=False,folds={},tracks=[])
    save_json(out/"report.json",report)
    for i,comp in enumerate(COMPS):
        cal_comp=COMPS[(i+1)%len(COMPS)]
        train=[r for r in records if r["composition"] not in (comp,cal_comp)]
        cal=[r for r in records if r["composition"]==cal_comp]
        evaluate=[r for r in records if r["composition"]==comp]
        if tuple(map(len,(train,cal,evaluate)))!=(48,16,16):
            raise AssertionError("48/16/16 composition split changed")
        ref=prior["folds"][comp]
        path=previous_dir/ref["checkpoint"]
        if digest_file(path)!=ref["checkpoint_sha256"]:
            raise AssertionError("frozen S815 weights differ from original run")
        base=joblib.load(path)
        if comp in base.metadata["train_compositions"]+base.metadata["calibration_compositions"]:
            raise AssertionError("frozen S815 checkpoint trained on outer composition")
        teacher=fit_teacher(train,cal,seed=818,cap=130,model_iter=55)
        if comp in teacher.metadata["train_compositions"]+teacher.metadata["calibration_compositions"]:
            raise AssertionError("ordinal model evaluation leakage")
        ck=out/"models"/f"{comp}_ordinal.joblib"
        joblib.dump(teacher,ck,compress=3)
        report["folds"][comp]=dict(checkpoint=str(ck.relative_to(out)),
            checkpoint_sha256=digest_file(ck),model_metadata=teacher.metadata,
            frozen_s815_checkpoint_sha256=digest_file(path),train_tracks=48,
            calibration_tracks=16,test_tracks=16)
        print("S818_TRAINED_PRIVATE",comp,teacher.metadata["selected"],flush=True)
        for r in evaluate:
            old_prediction=base.predict(r["X"])
            old=active_score(old_prediction,r["k"])
            if old!=saved[r["index"]]["active_k"]:
                raise AssertionError("frozen S815 output parity drift")
            revised=teacher.predict(r["X"])
            scored=active_score(revised,r["k"])
            output=out/"predictions"/f"fit_{r['index']:03d}.npz"
            np.savez_compressed(output,original_s815_k=old_prediction,
                new_active_k=revised,metric_ends=np.asarray(r["ends"]))
            report["tracks"].append(dict(index=r["index"],name=r["name"],
                composition=comp,source=r["source"],frozen_s815=old,new=scored,
                by_true_k=class_scores(revised,r["k"]),
                decisions_sha256=digest_file(output),
                decisions_file=str(output.relative_to(out))))
        save_json(out/"report.json",report)
        print("S818_FOLD",comp,totals([r["new"] for r in report["tracks"]
              if r["composition"]==comp]),flush=True)
    historical=totals([r["source"] for r in report["tracks"]])
    frozen=totals([r["frozen_s815"] for r in report["tracks"]])
    current=totals([r["new"] for r in report["tracks"]])
    if historical!=prior["original_fit80_active_k"] or frozen!=prior["teacher_active_exact_k"]:
        raise AssertionError("complete prior/historical parity not reproduced")
    by_k={str(k):totals([r["by_true_k"][str(k)] for r in report["tracks"]])
          for k in range(1,7)}
    report.update(complete=True,historical_fit_active_k=historical,
      original_s815_fit_active_k=frozen,new_s818_fit_active_k=current,
      by_true_active_k=by_k,
      delta_vs_s815_pp=100*(current["accuracy"]-frozen["accuracy"]),
      delta_vs_h0_pp=100*(current["accuracy"]-historical["accuracy"]))
    ev=TeacherEvidence(axis="active_exact_k",score=current["accuracy"],
      metric_name=REQUIRED_METRICS["active_exact_k"],
      source_id=fresh["source_manifest_sha256"]+"-"+fresh["audio_archive_sha256"],
      evaluation_id=hashlib.sha256(f"{SCHEMA}:818".encode()).hexdigest(),
      evaluation_role="fit_research",generalization_verified=False,
      private_parameters_verified=True,private_state_verified=True,
      causal_observations_verified=True,inference_uses_labels=False)
    report["teacher_evidence"]=asdict(ev)
    report["unification_gate"]=unification_gate([ev])
    if report["unification_gate"]["allow_unification"]:
        raise AssertionError("S818 researched FIT80 may not authorize distillation")
    save_json(out/"report.json",report)
    print("S818_COMPLETE",current,"S815",frozen,"H0",historical,
          "DIFF_PP",report["delta_vs_s815_pp"],report["delta_vs_h0_pp"],
          "PROMOTION",False,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("cache","traces","previous-dir","out"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    run(a.cache,a.traces,a.previous_dir,a.out)
