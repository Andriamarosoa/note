"""S819 forensic-only: can S817's two private high-polyphony heads ever change K?

The S817 5-fold models, S815 5-fold baseline and source PCM/annotations are
frozen and SHA checked. For all 80 FIT80 recordings calculate the real gating
rates and fixed-policy counterfactual outputs; NEVER choose a new policy using
outer GT and NEVER promote/modify MIDI. This evaluates hypotheses from actual
NVIDIA NIM advisers without accepting their claims as scientific proof.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib
import numpy as np
from causal_note.s817_highpoly_temporal import proposals,select,SELECTORS,THRESHOLDS
from causal_note.s814_activek_teacher import active_score
from scripts.s813_prepare_native256 import digest_file,save_json
from scripts.s814_train_activek256 import totals

SCHEMA="S819_S817_FROZEN_HIGH_POLYPHONY_GATE_FORENSIC_FIT80_V1"
COMPS=("BN1","Funk2","Jazz1","Rock3","SS1")
POLICIES=tuple((m,float(t)) for m in SELECTORS for t in THRESHOLDS)

def row_score(pred,labels):
    return active_score(np.asarray(pred),np.asarray(labels))

def accum(v):
    return totals(v)

def main():
    parser=argparse.ArgumentParser()
    for key in ("cache","s817-dir","s815-dir","out"):
        parser.add_argument("--"+key,required=True,type=Path)
    a=parser.parse_args()
    run(a.cache,a.s817_dir,a.s815_dir,a.out)

def run(cache,s817_dir,s815_dir,out):
    if out.exists():raise FileExistsError("immutable S819 diagnostic already exists")
    fresh=json.loads((cache/"report.json").read_text())
    old817=json.loads((s817_dir/"report.json").read_text())
    old815=json.loads((s815_dir/"report.json").read_text())
    if (fresh.get("complete") is not True or len(fresh.get("tracks",[]))!=80
        or fresh.get("schema")!="S813_FIT80_FRESH_UNSUPERVISED_DSP_MIDI256_V1"
        or old817.get("schema")!="S817_REAL_FIT80_FIVE_PRIVATE_ACTIVEK_HEADS_MIDI256_V1"
        or old815.get("schema")!="S815_ACTIVE_EXACT_K_REAL_FRESH256_FIT80_V1"
        or old817.get("complete") is not True or old815.get("complete") is not True):
        raise AssertionError("genuine completed S813/S815/S817 input required")
    if (old817["historical_fit_active_k"]["correct"]!=142213 or
        old817["original_s815_fit_active_k"]["correct"]!=141946 or
        old817["new_s817_fit_active_k"]["correct"]!=141946 or
        old815["teacher_active_exact_k"]["correct"]!=141946):
        raise AssertionError("original signed baseline/S817 mismatch")
    if (fresh["feature_hop_samples"],fresh["metric_hop_samples"])!=(256,256):
        raise AssertionError("MIDI256 source required")
    old817_cache=s817_dir.parent/"fresh256"/"report.json"
    if digest_file(old817_cache)!=old817["new_cache_sha256"]:
        raise AssertionError("original S817 cache provenance drift")
    cachemeta=json.loads(old817_cache.read_text())
    for key in ("source_manifest_sha256","audio_archive_sha256",
                "feature_hop_samples","decision_hop_samples",
                "metric_hop_samples","total_frames","feature_dimension"):
        if fresh[key]!=cachemeta[key]:
            raise AssertionError("S819 source PCM differs from original S817: "+key)
    identity=lambda rows:sorted((int(r["index"]),r["name"],r["composition"],
       r["frames"],r["original_json_sha256"],r["audio_member_sha256"]) for r in rows)
    if identity(fresh["tracks"])!=identity(cachemeta["tracks"]):
        raise AssertionError("80 annotated original recordings differ")
    oldtracks={int(r["index"]):r for r in old817["tracks"]}
    if len(oldtracks)!=80:raise AssertionError("S817 duplicate tracks")
    out.mkdir(parents=True)
    report=dict(schema=SCHEMA,complete=False,source_s817_run=39543749810,
        source_s815_run=39536863317,
        source_817_report_sha256=digest_file(s817_dir/"report.json"),
        source_815_report_sha256=digest_file(s815_dir/"report.json"),
        original_wav_source_verified=True,
        genuine_active_k_frames=249411,
        metric_hop_samples=256,decision_hop_samples=256,
        added_hold_samples=0,added_lookahead_samples=0,
        new_training=False,posthoc_outer_policies_only=True,
        outer_labels_used_for_calibration=False,
        FIT80_explored=True,external_generalization_measured=False,
        other_five_teacher_outputs_modified=False,production_live_modified=False,
        promotion=False,folds={},tracks=[],outer_policy_totals={})
    save_json(out/"report.json",report)
    for comp in COMPS:
        m817=old817["folds"][comp]
        ck=s817_dir/m817["checkpoint"]
        if digest_file(ck)!=m817["checkpoint_sha256"]:
            raise AssertionError("S817 private highpoly checkpoint digest mismatch")
        candidate=joblib.load(ck)
        m815=old815["folds"][comp]
        baseline_path=s815_dir/m815["checkpoint"]
        if digest_file(baseline_path)!=m815["checkpoint_sha256"]:
            raise AssertionError("S815 baseline checkpoint digest mismatch")
        base_model=joblib.load(baseline_path)
        if (comp in candidate.metadata["train_compositions"]
            or comp in candidate.metadata["calibration_compositions"]
            or comp in base_model.metadata["train_compositions"]
            or comp in base_model.metadata["calibration_compositions"]):
            raise AssertionError("training/cal/test composition leakage")
        report["folds"][comp]=dict(
            s817_original_selection=candidate.metadata["selected"],
            original_calibration_policy_results=candidate.metadata["calibration_results"],
            s817_checkpoint_sha256=m817["checkpoint_sha256"],
            s815_checkpoint_sha256=m815["checkpoint_sha256"],
            tested_records=0,
            actual_original_s817_changed_frames=0,
            gate_activation_rates={},
            potential_changed_frames_by_policy={},
            actual_scores_by_fixed_policy={},
            # Analysis of posthoc outer scores is never a policy choice.
            outer_scores_may_not_authorize_promotion=True)
        for row in sorted((r for r in fresh["tracks"] if r["composition"]==comp),
                          key=lambda r:int(r["index"])):
            r817=oldtracks[int(row["index"])]
            if row["name"]!=r817["name"] or r817["composition"]!=comp:
                raise AssertionError("original track index/name mismatch")
            files=row["files"]
            for f in files.values():
                if digest_file(cache/f["file"])!=f["sha256"]:
                    raise AssertionError("source acoustic cache tampered")
            X=np.load(cache/files["features"]["file"],mmap_mode="r",allow_pickle=False)
            target=np.load(cache/files["count_target"]["file"],mmap_mode="r",allow_pickle=False)
            original=base_model.predict(X)
            s815_score=row_score(original,target)
            if s815_score!=r817["frozen_s815"]:
                raise AssertionError("independently replayed original S815 unequal")
            s817_output=candidate.predict(X,original)
            s817_score=row_score(s817_output,target)
            if s817_score!=r817["new"]:
                raise AssertionError("independently replayed original S817 unequal")
            conf,high=proposals(X,candidate.gate,candidate.high)
            re_selected=select(original,conf,high,
              candidate.selector,candidate.threshold)
            if not np.array_equal(re_selected,s817_output):
                raise AssertionError("S817 high-polyphony published selector drift")
            original_changed=int(np.count_nonzero(original!=s817_output))
            # Register *all* pre-declared counterfactual policies. No outer
            # calibration, no test-picked winners, no changed production MIDI.
            policies={}
            for mode,threshold in POLICIES:
                output=select(original,conf,high,mode,threshold)
                delta=int(np.count_nonzero(output!=original))
                score=row_score(output,target)
                key=f"{mode}@{threshold:.2f}"
                policies[key]=dict(changed_frames=delta,
                    correct=score["correct"],frames=score["frames"],
                    undercount=score["undercount"],overcount=score["overcount"])
            report["tracks"].append(dict(index=int(row["index"]),name=row["name"],
              composition=comp,original_s815=s815_score,
              selected_s817=s817_score,selection_changed_frames=original_changed,
              n_audio_frames=len(X),by_policy=policies,
              gate_fires={f"{t:.2f}":int(np.count_nonzero(conf>=t)) for t in THRESHOLDS},
              gate_and_high_disagrees={f"{t:.2f}":int(np.count_nonzero(
                  (conf>=t)&(high!=original))) for t in THRESHOLDS}))
            report["folds"][comp]["tested_records"]+=1
            report["folds"][comp]["actual_original_s817_changed_frames"]+=original_changed
        tracks=[r for r in report["tracks"] if r["composition"]==comp]
        for t in THRESHOLDS:
            key=f"{t:.2f}"
            report["folds"][comp]["gate_activation_rates"][key]=sum(r["gate_fires"][key] for r in tracks)
        for mode,t in POLICIES:
            key=f"{mode}@{t:.2f}"
            report["folds"][comp]["potential_changed_frames_by_policy"][key]=sum(
                r["by_policy"][key]["changed_frames"] for r in tracks)
            s=accum([r["by_policy"][key] for r in tracks])
            report["folds"][comp]["actual_scores_by_fixed_policy"][key]=s
        print("S819_GATE_FORENSIC_FOLD",comp,
              "original_changed",report["folds"][comp]["actual_original_s817_changed_frames"],
              "gate_hits",report["folds"][comp]["gate_activation_rates"],
              flush=True)
        save_json(out/"report.json",report)
    baseline=accum([r["original_s815"] for r in report["tracks"]])
    teacher=accum([r["selected_s817"] for r in report["tracks"]])
    if baseline!=old817["original_s815_fit_active_k"] or teacher!=old817["new_s817_fit_active_k"]:
        raise AssertionError("five-fold S815/S817 source parity drift")
    report["original_baseline_s815"]=baseline
    report["original_teacher_s817"]=teacher
    report["n_original_decisions_changed"]=sum(r["selection_changed_frames"] for r in report["tracks"])
    for mode,t in POLICIES:
        key=f"{mode}@{t:.2f}"
        report["outer_policy_totals"][key]=dict(
           active_k=accum([r["by_policy"][key] for r in report["tracks"]]),
           changed_frames=sum(r["by_policy"][key]["changed_frames"] for r in report["tracks"]))
    report["complete"]=True
    save_json(out/"report.json",report)
    print("S819_GATE_FORENSIC_COMPLETE",json.dumps(
          {"n_original_decisions_changed":report["n_original_decisions_changed"],
           "original_s815":baseline,"s817_selected":teacher,
           "possible_changes":{k:v["changed_frames"] for k,v in
                                report["outer_policy_totals"].items()}},
           sort_keys=True),flush=True)

if __name__=="__main__":main()
