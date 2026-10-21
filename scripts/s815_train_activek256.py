"""S815 real FIT80 independent Active Exact-K experiment using S813 native PCM256.

Run only after S813 source manifest and full original PCM cache are verified.
No access to held-out project CAL/HELD/TEST; FIT80 is already explored research.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import joblib
import numpy as np

from causal_note.s815_activek_teacher import fit_teacher, SCHEMA as TEACHER_SCHEMA
from causal_note.s814_activek_teacher import active_score
from causal_note.s813_teacher_contract import TeacherEvidence, REQUIRED_METRICS, unification_gate
from scripts.s813_prepare_native256 import SCHEMA as DATA_SCHEMA, digest_file, save_json

SCHEMA = "S815_ACTIVE_EXACT_K_REAL_FRESH256_FIT80_V1"
COMPS = ["BN1", "Funk2", "Jazz1", "Rock3", "SS1"]


def original_count(events, ends):
    """Only for scoring original reference, never teacher inference."""
    occupancy = np.zeros((len(ends), 49), np.uint8)
    active = {}
    for e in events:
        p = int(e["pitch"]) - 40
        at = int(np.searchsorted(ends, int(e["available_sample"])))
        if not (0 <= p < 49):
            raise ValueError("out of reference pitch range")
        if e["type"] == "note_on":
            if p in active:
                raise AssertionError("unpaired duplicate historical NoteOn")
            active[p] = at
        elif e["type"] == "note_off":
            if p not in active:
                raise AssertionError("orphan historical NoteOff")
            occupancy[active.pop(p):at, p] = 1
        else:
            raise ValueError("invalid historic MIDI event")
    for p, begin in active.items():
        occupancy[begin:, p] = 1
    return np.minimum(occupancy.sum(1), 6).astype(np.uint8)


def totals(rows):
    result = {k: sum(int(r[k]) for r in rows) for k in
              ("correct", "frames", "undercount", "overcount")}
    if result["correct"] + result["undercount"] + result["overcount"] != result["frames"]:
        raise AssertionError("true active denominator classification drift")
    result["accuracy"] = result["correct"] / max(1, result["frames"])
    return result


def run(cache, traces, out):
    if out.exists():
        raise FileExistsError("immutable S815 result directory already exists")
    extraction = json.loads((cache / "report.json").read_text())
    if (extraction.get("schema") != DATA_SCHEMA or not extraction.get("complete")
            or len(extraction.get("tracks", [])) != 80
            or any(extraction.get(k) for k in ("CAL_opened", "HELD_opened", "TEST_opened"))
            or extraction.get("H0_events_or_posteriors_used_as_features")
            or any(extraction.get(k) != 256 for k in (
                "feature_hop_samples", "decision_hop_samples", "metric_hop_samples"))):
        raise ValueError("authentic untainted native MIDI256 source required")
    records, h0_rows = [], []
    for row in extraction["tracks"]:
        arrays = {}
        for name, info in row["files"].items():
            path = cache / info["file"]
            if digest_file(path) != info["sha256"]:
                raise AssertionError("modified source cache " + str(path))
            arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
        src_path = traces / f"fit_{row['index']:03d}.json"
        if digest_file(src_path) != row["original_json_sha256"]:
            raise AssertionError("altered S800 event or annotation provenance")
        source = json.loads(src_path.read_text())
        if source["name"] != row["name"] or source["composition"] != row["composition"]:
            raise AssertionError("source track mismatch")
        truth = arrays["count_target"]
        src_pred = original_count(source["source_events"], arrays["ends"])
        reference = active_score(src_pred, truth)
        signed = source["source_counts"]
        if (reference["frames"] != int(signed["active_frames"])
                or reference["correct"] != int(signed["active_k_correct"])):
            raise AssertionError("original S704 MIDI256 Active-K signed parity failed")
        records.append(dict(X=arrays["features"], y=arrays["pitch_target"], k=truth, name=row["name"],
                            composition=row["composition"], texture=row["texture"],
                            index=row["index"], source=reference,
                            ends=arrays["ends"]))
        h0_rows.append(reference)
    if (sorted({r["composition"] for r in records}) != COMPS
            or len({r["index"] for r in records}) != 80):
        raise ValueError("FIT80 five-composition data contract drift")
    source_report = totals(h0_rows)
    out.mkdir(parents=True)
    (out / "models").mkdir()
    (out / "predictions").mkdir()
    report = dict(schema=SCHEMA, teacher_schema=TEACHER_SCHEMA, complete=False,
        series="S815", target_axis="active_exact_k", score_scope="independent_teacher",
        sample_rate_hz=44100, feature_hop_samples=256, decision_hop_samples=256,
        metric_hop_samples=256, resolution_ms=256/44100*1000,
        source_cache_sha256=digest_file(cache/"report.json"),
        original_fit80_active_k=source_report,
        five_composition_oof=True, training_tracks_per_fold=48,
        inner_calibration_tracks_per_fold=16, test_tracks_per_fold=16,
        zero_future_lookahead_in_teacher=True, teacher_train_label_input_only=True,
        other_five_axes_untouched=True, original_emitted_midi_untouched=True,
        CAL_opened=False, HELD_opened=False, TEST_opened=False,
        fit_historically_explored=True, generalization_verified=False,
        physical_microphone_latency_measured=False, unified_midi_evaluated=False,
        production_LIVE_modified=False, promotion=False, folds={}, tracks=[])
    save_json(out/"report.json", report)
    for fold, outer in enumerate(COMPS):
        inner = COMPS[(fold+1) % 5]
        train = [r for r in records if r["composition"] not in (outer,inner)]
        cal = [r for r in records if r["composition"] == inner]
        evaluate = [r for r in records if r["composition"] == outer]
        if tuple(map(len,(train,cal,evaluate))) != (48,16,16):
            raise AssertionError("predeclared 48/16/16 split violated")
        started=time.perf_counter()
        teacher=fit_teacher(train,cal,cap=1400,pitch_cap=2500,seed=815)
        model_path=out/"models"/f"{outer}_active_k.joblib"
        joblib.dump(teacher,model_path,compress=3)
        if (outer in teacher.metadata["train_compositions"]
                or outer in teacher.metadata["calibration_compositions"]):
            raise AssertionError("outer split leakage")
        cpu=[]
        example=evaluate[0]["X"]
        teacher.predict(example[:1])
        for i in np.linspace(0,len(example)-1,64,dtype=int):
            t=time.perf_counter()
            teacher.predict(example[i:i+1])
            cpu.append(1000*(time.perf_counter()-t))
        report["folds"][outer]=dict(
            inner_calibration=inner, train_seconds=time.perf_counter()-started,
            checkpoint=model_path.relative_to(out).as_posix(),
            checkpoint_sha256=digest_file(model_path), heads=teacher.metadata,
            inference_only_ms_p50_p95_p99=np.percentile(cpu,[50,95,99]).tolist(),
            extraction_and_audio_latency_excluded=True)
        for row in evaluate:
            pred=teacher.predict(row["X"])
            truth=np.asarray(row["k"],np.uint8)
            metric=active_score(pred,truth)
            by_k={str(k):dict(correct=int(np.count_nonzero((pred==k)&(truth==k))),
                frames=int(np.count_nonzero(truth==k)),
                undercount=int(np.count_nonzero((pred<k)&(truth==k))),
                overcount=int(np.count_nonzero((pred>k)&(truth==k))))
                for k in range(7)}
            output=out/"predictions"/f"fit_{row['index']:03d}.npz"
            np.savez_compressed(output, predicted_count=pred,
                                metric_ends=row["ends"])
            report["tracks"].append(dict(index=row["index"],name=row["name"],
                composition=outer,texture=row["texture"],
                frames=len(truth),active_k=metric,source=row["source"],by_k=by_k,
                predictions=output.relative_to(out).as_posix(),
                predictions_sha256=digest_file(output)))
        save_json(out/"report.json",report)
        foldrows=[r["active_k"] for r in report["tracks"] if r["composition"]==outer]
        print("S815_FOLD",outer,totals(foldrows),flush=True)
    outcome=totals([r["active_k"] for r in report["tracks"]])
    by_k={str(k):totals([r["by_k"][str(k)] for r in report["tracks"]])
          for k in range(1,7)}
    by_comp={c:totals([r["active_k"] for r in report["tracks"]
                      if r["composition"]==c]) for c in COMPS}
    report.update(complete=True, teacher_active_exact_k=outcome,
        three_independent_heads_trained=True,
        delta_against_h0_fit80_pp=100*(outcome["accuracy"]-source_report["accuracy"]),
        by_true_active_k=by_k, by_composition=by_comp,
        teacher_exceeds_85_fit=outcome["accuracy"]>.85)
    evidence=TeacherEvidence(axis="active_exact_k",score=outcome["accuracy"],
        metric_name=REQUIRED_METRICS["active_exact_k"],
        source_id=report["source_cache_sha256"],
        evaluation_id=hashlib.sha256(json.dumps(
            {"folds":COMPS,"schema":SCHEMA},sort_keys=True).encode()).hexdigest(),
        evaluation_role="fit_research",generalization_verified=False,
        private_parameters_verified=True,private_state_verified=True,
        causal_observations_verified=True,inference_uses_labels=False)
    report["teacher_evidence"]=asdict(evidence)
    report["unification_gate"]=unification_gate([evidence])
    if report["unification_gate"]["allow_unification"]:
        raise AssertionError("historically explored FIT80 cannot certify >85%")
    save_json(out/"report.json",report)
    print("S815_COMPLETE",outcome,"BASE",source_report,
          "PROMOTION",False,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for arg in ("cache","traces","out"):
        p.add_argument("--"+arg,type=Path,required=True)
    q=p.parse_args()
    run(q.cache,q.traces,q.out)
