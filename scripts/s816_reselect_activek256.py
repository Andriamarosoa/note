"""S816 read-only re-selection of S815 private Active-K heads, real MIDI256.

Five S815 composition-excluded checkpoints are held fixed. Only the inner
composition's saved calibration counts may determine risk-penalized selector
and shift. No refit, no outer-label policy choice, no historical MIDI edits.
FIT80 has been explored in prior rounds; this is research, never an
independent out-of-sample certification or deployment candidate.
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

from causal_note.s815_activek_teacher import _count_outputs, _select, SELECTORS, SHIFTS
from causal_note.s814_activek_teacher import active_score
from causal_note.s813_teacher_contract import TeacherEvidence, REQUIRED_METRICS, unification_gate
from scripts.s813_prepare_native256 import digest_file, save_json
from scripts.s814_train_activek256 import original_count, totals

SCHEMA="S816_FROZEN_PRIVATE_S815_ACTIVEK_RISK_SELECTOR_MIDI256_V1"
PENALTY_OVERS=0.50     # Pre-declared, not tuned on outer composition
MAX_CAL_ACC_LOSS=0.02 # Risk-regularization cannot sacrifice >2pp on inner cal
COMPS=("BN1","Funk2","Jazz1","Rock3","SS1")


def choose_policy(calibration, *, penalty=PENALTY_OVERS, max_loss=MAX_CAL_ACC_LOSS):
    """Select a predeclared policy using inner calibration ONLY.

    The target function penalizes overcounts only after enforcing near-best
    inner-calibration accuracy. No outer/test data accepted in this API.
    """
    if not (np.isfinite(penalty) and 0 <= penalty <= 1
            and np.isfinite(max_loss) and 0 <= max_loss <= .1):
        raise ValueError("invalid fixed penalty or calibration guard")
    by={}
    for v in calibration:
        key=(v["selector"], int(v["shift"]))
        if key in by or key[0] not in SELECTORS or key[1] not in SHIFTS:
            raise ValueError("duplicate or undeclared policy")
        correct,frames,over=(int(v[k]) for k in ("correct","frames","overcount"))
        if not 0 <= correct <= frames or not 0 <= over <= frames-correct or frames<=0:
            raise ValueError("invalid inner calibration counts")
        by[key]=dict(correct=correct,frames=frames,overcount=over,
                     accuracy=correct/frames,over_rate=over/frames)
    if set(by)!={(s,shift) for s in SELECTORS for shift in SHIFTS}:
        raise ValueError("full fifteen-policy calibration table required")
    if len({v["frames"] for v in by.values()})!=1:
        raise ValueError("candidate comparison requires same inner calibration frames")
    best=max(x["accuracy"] for x in by.values())
    eligible=[k for k,v in by.items() if v["accuracy"]>=best-max_loss-1e-12]
    def rank(k):
        v=by[k]
        return (v["accuracy"]-penalty*v["over_rate"],
                v["accuracy"],-v["over_rate"],
                -SELECTORS.index(k[0]),-SHIFTS.index(k[1]))
    selected=max(eligible,key=rank)
    return dict(selector=selected[0],shift=selected[1],
        penalty_overs=penalty,max_cal_accuracy_loss=max_loss,
        calibration_best_accuracy=best,calibration_selected=by[selected],
        calibration_selected_utility=rank(selected)[0],
        n_eligible=len(eligible),choices=by)


def breakdown(pred,truth):
    p,y=np.asarray(pred),np.asarray(truth)
    return {str(k):dict(correct=int(np.count_nonzero((p==k)&(y==k))),
        frames=int(np.count_nonzero(y==k)),
        undercount=int(np.count_nonzero((p<k)&(y==k))),
        overcount=int(np.count_nonzero((p>k)&(y==k)))) for k in range(7)}


def run(cache,source_dir,previous_dir,out):
    if out.exists():
        raise FileExistsError("S816 immutable output already exists")
    fresh=json.loads((cache/"report.json").read_text())
    previous=json.loads((previous_dir/"report.json").read_text())
    if not fresh.get("complete") or fresh.get("schema")!="S813_FIT80_FRESH_UNSUPERVISED_DSP_MIDI256_V1":
        raise ValueError("S813 authentic PCM256 cache mandatory")
    if (previous.get("schema")!="S815_ACTIVE_EXACT_K_REAL_FRESH256_FIT80_V1"
       or not previous.get("complete") or len(previous.get("tracks",[]))!=80
       or any(previous.get(k) for k in ("CAL_opened","HELD_opened","TEST_opened",
                                        "production_LIVE_modified","promotion"))):
        raise ValueError("frozen authentic S815 result/cache identity mismatch")
    # An extraction report contains wall-clock extraction timings, so its raw
    # JSON SHA changes on a legitimate fresh run. Instead authenticate the
    # old report as saved, then compare immutable source IDs and source audio
    # checksums across the two independently extracted caches.
    prior_cache_path=previous_dir.parent/"fresh256"/"report.json"
    if not prior_cache_path.is_file():
        raise ValueError("S815 original acoustic source manifest must accompany checkpoints")
    if digest_file(prior_cache_path)!=previous["source_cache_sha256"]:
        raise AssertionError("S815 original saved cache provenance mismatch")
    prior_cache=json.loads(prior_cache_path.read_text())
    for key in ("schema","source_manifest_sha256","audio_archive_sha256",
                "total_frames","feature_dimension","sample_rate_hz",
                "feature_hop_samples","decision_hop_samples","metric_hop_samples"):
        if fresh[key]!=prior_cache[key]:
            raise AssertionError("immutable prior/new extraction source contract changed: "+key)
    source_identity=lambda rows: sorted((
        int(r["index"]),r["name"],r["composition"],int(r["frames"]),
        r["original_json_sha256"],r["audio_member_sha256"]) for r in rows)
    if source_identity(fresh["tracks"])!=source_identity(prior_cache["tracks"]):
        raise AssertionError("original signed PCM and annotations differ from S815")
    if (int(previous["teacher_active_exact_k"]["correct"])!=141946
        or int(previous["original_fit80_active_k"]["correct"])!=142213
        or int(previous["original_fit80_active_k"]["frames"])!=249411):
        raise ValueError("S815 known scientific result mismatch")
    if len(fresh["tracks"])!=80 or tuple(sorted(previous["folds"]))!=COMPS:
        raise ValueError("expected all FIT80 composition folds")
    report=dict(schema=SCHEMA,series="S816",complete=False,
        source_run=39536863317,source_run_commit="dafb9eba0e587ae48156c46859bcfad034bfbaad",
        source_report_sha256=digest_file(previous_dir/"report.json"),
        source_cache_sha256=digest_file(cache/"report.json"),
        sample_rate_hz=44100, feature_hop_samples=256,decision_hop_samples=256,
        metric_hop_samples=256,
        penalty_overcount=PENALTY_OVERS,
        max_calibration_accuracy_loss=MAX_CAL_ACC_LOSS,
        retrained_models=False,added_lookahead_samples=0,added_hold_samples=0,
        other_axes_modified=False,source_MIDI_modified=False,
        CAL_opened=False,HELD_opened=False,TEST_opened=False,
        FIT80_historically_explored=True,generalization_verified=False,
        LIVE_measured=False,promotion=False,folds={},tracks=[])
    out.mkdir(parents=True)
    (out/"decisions").mkdir()
    save_json(out/"report.json",report)
    baseline_rows=[]; original_s815_rows=[];candidate_rows=[]
    indexed={int(r["index"]):r for r in previous["tracks"]}
    if len(indexed)!=80:
        raise ValueError("duplicate S815 track index")
    for outer in COMPS:
        oldfold=previous["folds"][outer]
        checkpoint=previous_dir/oldfold["checkpoint"]
        if digest_file(checkpoint)!=oldfold["checkpoint_sha256"]:
            raise AssertionError("frozen S815 head checkpoint integrity failed")
        model=joblib.load(checkpoint)
        if (len(model.heads)!=3 or
                model.metadata["calibration_compositions"] !=
                [COMPS[(COMPS.index(outer)+1)%5]] or
                outer in model.metadata["train_compositions"] or
                outer in model.metadata["calibration_compositions"]):
            raise AssertionError("composition independence or private-head contract violated")
        selection=choose_policy(oldfold["heads"]["calibration_results"])
        report["folds"][outer]=dict(
            original_selector=model.selector,original_shift=model.shift,
            selected_selector=selection["selector"],selected_shift=selection["shift"],
            selection_changed=(selection["selector"],selection["shift"])!=(model.selector,model.shift),
            inner_calibration=model.metadata["calibration_compositions"],
            selected_calibration_details={
                k:selection[k] for k in ("calibration_best_accuracy",
                      "calibration_selected","calibration_selected_utility","n_eligible")},
            checkpoint_sha256=oldfold["checkpoint_sha256"],
            evaluated_recordings=0)
        for row in sorted([r for r in fresh["tracks"] if r["composition"]==outer],
                          key=lambda item:item["index"]):
            current=indexed[int(row["index"])]
            if current["name"]!=row["name"] or current["composition"]!=outer:
                raise AssertionError("S815/S813 recording identity mismatch")
            files=row["files"]
            for src in files.values():
                if digest_file(cache/src["file"])!=src["sha256"]:
                    raise AssertionError("tampered 256 acoustic/labels")
            track=source_dir/f"fit_{row['index']:03d}.json"
            if digest_file(track)!=row["original_json_sha256"]:
                raise AssertionError("tampered S800 note events")
            events=json.loads(track.read_text())
            if events["name"]!=row["name"] or events["composition"]!=outer:
                raise ValueError("mismatched MIDI source")
            x=np.load(cache/files["features"]["file"],mmap_mode="r")
            y=np.load(cache/files["count_target"]["file"],mmap_mode="r")
            ends=np.load(cache/files["ends"]["file"],mmap_mode="r")
            h0=active_score(original_count(events["source_events"],ends),y)
            if h0!=current["source"] or h0["correct"]!=events["source_counts"]["active_k_correct"]:
                raise AssertionError("signed H0 S704 parity drift")
            # Baseline reconstruction ensures no moved scorer or models.
            current_pred=model.predict(x)
            o=active_score(current_pred,y)
            if o!=current["active_k"]:
                raise AssertionError("S815 saved-source score not reconstructed exactly")
            p,r,q=_count_outputs(np.asarray(x),model.direct,model.regression,model.pitch_occupancy)
            revised=_select(p,r,q,selection["selector"],selection["shift"])
            evaluation=active_score(revised,y)
            if any(v < 0 for v in evaluation.values()):
                raise AssertionError("invalid new metric")
            save=np.asarray(revised,dtype=np.uint8)
            dest=out/"decisions"/f"fit_{row['index']:03d}.npz"
            np.savez_compressed(dest,predicted_count=save,
                                metric_ends=np.asarray(ends,dtype=np.int64))
            report["tracks"].append(dict(index=row["index"],name=row["name"],
                composition=outer,source=h0,frozen_s815=o,candidate=evaluation,
                by_true_k=breakdown(save,y),decision_sha256=digest_file(dest),
                decision_file=dest.relative_to(out).as_posix()))
            baseline_rows.append(h0);original_s815_rows.append(o);candidate_rows.append(evaluation)
            report["folds"][outer]["evaluated_recordings"]+=1
        save_json(out/"report.json",report)
        local=totals([r["candidate"] for r in report["tracks"] if r["composition"]==outer])
        print("S816_FOLD",outer,local,report["folds"][outer]["selected_selector"],
              report["folds"][outer]["selected_shift"],flush=True)
    source_score=totals(baseline_rows)
    old_score=totals(original_s815_rows)
    new_score=totals(candidate_rows)
    if (source_score!=previous["original_fit80_active_k"] or
         old_score!=previous["teacher_active_exact_k"]):
        raise AssertionError("full genuine S815/H0 historic score parity failed")
    by_k={str(k):totals([r["by_true_k"][str(k)] for r in report["tracks"]])
          for k in range(1,7)}
    report.update(complete=True,source_active_k=source_score,
        frozen_s815_active_k=old_score,candidate_active_k=new_score,
        by_true_active_k=by_k,
        changed_selector_folds=sum(r["selection_changed"] for r in report["folds"].values()),
        delta_vs_h0_pp=100*(new_score["accuracy"]-source_score["accuracy"]),
        delta_vs_s815_pp=100*(new_score["accuracy"]-old_score["accuracy"]),
        fit85_threshold_exceeded=new_score["accuracy"]>.85)
    ev=TeacherEvidence(axis="active_exact_k",score=new_score["accuracy"],
        metric_name=REQUIRED_METRICS["active_exact_k"],source_id=report["source_cache_sha256"],
        evaluation_id=hashlib.sha256(f"{SCHEMA}:risk05:guard02".encode()).hexdigest(),
        evaluation_role="fit_research",generalization_verified=False,
        inference_uses_labels=False,private_parameters_verified=True,
        private_state_verified=True,causal_observations_verified=True)
    report["teacher_evidence"]=asdict(ev)
    report["unification_gate"]=unification_gate([ev])
    if report["unification_gate"]["allow_unification"]:
        raise AssertionError("no retrospective FIT study may unlock model unification")
    save_json(out/"report.json",report)
    print("S816_COMPLETE",new_score,"H0",source_score,"S815",old_score,flush=True)
    print("S816_DELTAS_PP",report["delta_vs_h0_pp"],report["delta_vs_s815_pp"],
          "changed_folds",report["changed_selector_folds"],flush=True)


def main():
    p=argparse.ArgumentParser()
    for flag in ("cache","source-dir","previous-dir","out"):
        p.add_argument("--"+flag,required=True,type=Path)
    a=p.parse_args()
    run(a.cache,a.source_dir,a.previous_dir,a.out)

if __name__=="__main__":
    main()
