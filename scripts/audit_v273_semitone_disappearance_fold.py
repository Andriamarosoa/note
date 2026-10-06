"""Progressive semitone pitch-shift disappearance audit for frozen K2/K3 cases.

Each row is evaluated by the robust V27.3 model for its own held-out internal
fold. Only the spectral map is rebuilt from pitch-shifted audio; temporal
candidate inputs remain fixed. Fold 3 is forbidden.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from causal_note.guitarset import SAMPLE_RATE, index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.train_v100_spectral_string_slots import _pcm_window, _spectral_map_from_segment
from scripts.spectral_window import COVERED
from scripts.audit_v273_internal_b_like_boundary_corrector import (
    FOLDS, NEURONS, SEED, discover_reports, nested_base
)
from scripts.train_v273_group_gate_ab import build_model
from scripts.v273_window_experiment import load_bundle, batch_inputs, require
from scripts.v273_residual_audit import sha256_file, write_json

PROTOCOL = Path("analysis/v273-semitone-disappearance-protocol.md")
STEPS = tuple(range(25))
PAD = 4096


def robust_model(fold_root, val_fold):
    reports = discover_reports(fold_root)
    folder = reports[val_fold][0].parent
    uniform = build_model("learned_gate", SEED)
    uniform.load_weights(folder / "uniform.weights.h5")
    robust = build_model("learned_gate", SEED)
    robust.load_weights(folder / "freeze_local_combo.weights.h5")

    bu = nested_base(uniform)
    br = nested_base(robust)
    ku, bu_bias = [np.asarray(x).copy() for x in bu.get_layer("candidate_hidden1").get_weights()]
    kr, br_bias = [np.asarray(x).copy() for x in br.get_layer("candidate_hidden1").get_weights()]
    idx = np.asarray(NEURONS, np.int64)
    kr[:, idx] = ku[:, idx]
    br_bias[idx] = bu_bias[idx]
    br.get_layer("candidate_hidden1").set_weights([kr, br_bias])
    return robust


def model_probability(model, cache, ids, spectral):
    x = batch_inputs(cache, ids, COVERED.time_frames)
    x["spectral_map"] = np.asarray(spectral, np.float32)
    p = np.asarray(model.predict(x, batch_size=128, verbose=0), np.float64)
    require(p.shape == (len(ids), 7), "bad model output")
    require(np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1.0, atol=1e-5),
            "invalid probabilities")
    return p


def shifted_map(samples, start, step):
    if step == 0:
        segment = _pcm_window(
            samples,
            int(start) - COVERED.pre_samples,
            COVERED.segment_samples,
        )
    else:
        import librosa
        left = PAD + COVERED.pre_samples
        right = PAD + COVERED.post_samples
        local = _pcm_window(samples, int(start) - left, left + right)
        shifted = librosa.effects.pitch_shift(
            local.astype(np.float32),
            sr=SAMPLE_RATE,
            n_steps=float(step),
            bins_per_octave=12,
            res_type="soxr_hq",
            scale=False,
        ).astype(np.float32)
        if len(shifted) < len(local):
            shifted = np.pad(shifted, (0, len(local) - len(shifted)))
        elif len(shifted) > len(local):
            shifted = shifted[:len(local)]
        segment = shifted[PAD:PAD + COVERED.segment_samples]
        require(segment.shape == (COVERED.segment_samples,), "shifted crop drift")

    # Historical caches are float16. Quantize the rebuilt map the same way
    # before feeding it back into the model.
    return _spectral_map_from_segment(segment, window=COVERED).astype(np.float16).astype(np.float32)


def first_step(values, predicate, default=25):
    for step, value in enumerate(values):
        if predicate(int(value)):
            return step
    return default


def trajectory_summary(pred):
    pred = np.asarray(pred, np.int32)
    first_non3 = first_step(pred, lambda k: k != 3)
    first_below3 = first_step(pred, lambda k: k < 3)
    first_k2 = first_step(pred, lambda k: k == 2)

    reentered = False
    if first_non3 < len(pred) - 1:
        reentered = bool(np.any(pred[first_non3 + 1:] == 3))
    monotone = bool(np.all(np.diff(pred) <= 0))
    return {
        "first_non3_step": int(first_non3),
        "first_below3_step": int(first_below3),
        "first_k2_step": int(first_k2),
        "reentered_k3": reentered,
        "monotone_nonincreasing": monotone,
    }


def summarize_class(rows, true_k):
    selected = [r for r in rows if r["true_K"] == true_k]
    out = {"rows": len(selected)}
    for key in ("first_non3_step", "first_below3_step", "first_k2_step"):
        x = np.asarray([r[key] for r in selected], np.float64)
        out[key] = {
            "mean": float(x.mean()) if len(x) else None,
            "median": float(np.median(x)) if len(x) else None,
            "q25": float(np.quantile(x, .25)) if len(x) else None,
            "q75": float(np.quantile(x, .75)) if len(x) else None,
            "censored_at_25": int(np.sum(x == 25)) if len(x) else 0,
        }
    out["reentered_k3_rate"] = float(np.mean([r["reentered_k3"] for r in selected])) if selected else None
    out["monotone_nonincreasing_rate"] = float(np.mean([r["monotone_nonincreasing"] for r in selected])) if selected else None
    out["predicted_k2_by_step"] = {
        str(step): int(sum(r["trajectory"][step] == 2 for r in selected)) for step in STEPS
    }
    out["still_k3_by_step"] = {
        str(step): int(sum(r["trajectory"][step] == 3 for r in selected)) for step in STEPS
    }
    return out


def run(a):
    require(a.val_fold in FOLDS, "bad fold")
    require(a.val_fold != 3, "outer fold forbidden")
    require(not a.output.exists(), "refusing overwrite")
    require(PROTOCOL.exists(), "missing preregistered protocol")

    all_cases = [json.loads(line) for line in a.cases.read_text().splitlines()]
    cases = [r for r in all_cases if int(r["fold"]) == a.val_fold]
    require(cases and all(r["true_K"] in (2, 3) for r in cases), "bad frozen cohort")

    cache, _, _ = load_bundle(a.bundle, a.config)
    ids = np.asarray([int(r["row_id"]) for r in cases], np.int64)
    require(np.all(ids >= 0) and np.all(ids < len(cache["exact"])), "row ids out of bundle")
    require(np.array_equal(np.asarray(cache["members"][ids]).astype(str),
                           np.asarray([r["recording_id"] for r in cases]).astype(str)),
            "recording identity drift")
    require(np.array_equal(np.asarray(cache["cluster_start_samples"][ids], np.int64),
                           np.asarray([r["start_sample"] for r in cases], np.int64)),
            "start-sample identity drift")
    require(np.array_equal(np.minimum(np.asarray(cache["exact"][ids], np.int32), 6),
                           np.asarray([r["true_K"] for r in cases], np.int32)),
            "true-K drift")

    model = robust_model(a.fold_root, a.val_fold)

    cached_p = model_probability(
        model, cache, ids, np.asarray(cache["spectral"][ids, :COVERED.time_frames], np.float32)
    )
    cached_pred = cached_p.argmax(axis=1).astype(np.int32)
    require(np.all(cached_pred == 3), "frozen cohort no longer starts at predicted K3")

    wanted = set(np.asarray(cache["members"][ids]).astype(str))
    tracks = {t.annotation_member: t for t in index_guitarset(a.dataset) if t.annotation_member in wanted}
    require(set(tracks) == wanted, "missing GuitarSet tracks")
    audio = {}
    for member in sorted(wanted):
        wav = decode_pcm16_mono_wav(tracks[member].audio_zip, tracks[member].audio_member)
        audio[member] = np.asarray(wav.samples, np.float32) / 32768.0

    predictions = np.zeros((len(ids), len(STEPS)), np.int8)
    probabilities = np.zeros((len(ids), len(STEPS), 7), np.float32)
    map_error_max = 0.0
    started = time.monotonic()

    for step in STEPS:
        maps = np.empty((len(ids), COVERED.time_frames, 64, 3), np.float32)
        for j, row in enumerate(cases):
            member = row["recording_id"]
            maps[j] = shifted_map(audio[member], int(row["start_sample"]), step)
        if step == 0:
            cached_maps = np.asarray(cache["spectral"][ids, :COVERED.time_frames], np.float32)
            map_error_max = float(np.max(np.abs(maps - cached_maps)))
            require(map_error_max <= 1e-3, f"+0 spectral replay drift: {map_error_max}")
        p = model_probability(model, cache, ids, maps)
        predictions[:, step] = p.argmax(axis=1)
        probabilities[:, step] = p.astype(np.float32)
        if step == 0:
            require(np.array_equal(predictions[:, 0], cached_pred), "+0 prediction replay drift")
        print(json.dumps({
            "fold": a.val_fold,
            "step": step,
            "rows": len(ids),
            "still_k3": int(np.sum(predictions[:, step] == 3)),
            "k2": int(np.sum(predictions[:, step] == 2)),
            "seconds": round(time.monotonic() - started, 1),
        }), flush=True)

    rows = []
    transitions = Counter()
    for j, source in enumerate(cases):
        tr = predictions[j].astype(int).tolist()
        for step in range(1, len(STEPS)):
            transitions[(tr[step - 1], tr[step])] += 1
        summary = trajectory_summary(tr)
        rows.append({
            "row_id": int(source["row_id"]),
            "fold": a.val_fold,
            "group": source["group"],
            "true_K": int(source["true_K"]),
            "recording_id": source["recording_id"],
            "start_sample": int(source["start_sample"]),
            "trajectory": tr,
            "p_k2": probabilities[j, :, 2].astype(float).tolist(),
            "p_k3": probabilities[j, :, 3].astype(float).tolist(),
            **summary,
        })

    y = np.asarray([r["true_K"] for r in rows])
    feature = np.asarray([r["first_k2_step"] for r in rows], np.float64)
    raw_auc = float(roc_auc_score((y == 2).astype(int), feature)) if len(np.unique(y)) == 2 else None
    oriented_auc = None if raw_auc is None else max(raw_auc, 1.0 - raw_auc)
    direction = None if raw_auc is None else ("later_implies_K2" if raw_auc >= .5 else "earlier_implies_K2")

    report = {
        "status": "completed",
        "experiment": "v273_semitone_disappearance_fold",
        "validation_fold": a.val_fold,
        "steps": list(STEPS),
        "rows": len(rows),
        "K2_rows": int(np.sum(y == 2)),
        "K3_rows": int(np.sum(y == 3)),
        "outer_fold_3_used": False,
        "candidate_inputs_recomputed": False,
        "spectral_map_recomputed": True,
        "pitch_shift": "librosa duration-preserving local waveform, +1 semitone increments",
        "plus0_spectral_max_abs_error": map_error_max,
        "plus0_prediction_parity": True,
        "feature": "first_k2_step; 25 means no K2 through +24",
        "first_k2_raw_auc_K2": raw_auc,
        "first_k2_oriented_auc": oriented_auc,
        "first_k2_direction": direction,
        "by_true_k": {
            "2": summarize_class(rows, 2),
            "3": summarize_class(rows, 3),
        },
        "transition_counts": {
            f"{a0}->{b0}": int(n) for (a0, b0), n in sorted(transitions.items())
        },
        "source_sha256": {
            "protocol": sha256_file(PROTOCOL),
            "script": sha256_file(__file__),
            "cases": sha256_file(a.cases),
        },
        "runtime_seconds": float(time.monotonic() - started),
        "limitations": [
            "Temporal candidate features are frozen while only the spectral map is pitch-shifted.",
            "Internal folds were previously inspected; this is a development audit.",
            "Pitch shift may introduce phase-vocoder artifacts, especially at large positive shifts.",
        ],
    }

    a.output.mkdir(parents=True)
    np.savez_compressed(
        a.output / "trajectories.npz",
        row_id=ids,
        true_k=y.astype(np.int8),
        predicted=predictions,
        probability=probabilities,
        steps=np.asarray(STEPS, np.int8),
        first_k2_step=np.asarray([r["first_k2_step"] for r in rows], np.int8),
        first_non3_step=np.asarray([r["first_non3_step"] for r in rows], np.int8),
        first_below3_step=np.asarray([r["first_below3_step"] for r in rows], np.int8),
        reentered_k3=np.asarray([r["reentered_k3"] for r in rows], bool),
        monotone_nonincreasing=np.asarray([r["monotone_nonincreasing"] for r in rows], bool),
    )
    (a.output / "rows.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    report["rows_sha256"] = sha256_file(a.output / "rows.jsonl")
    report["trajectories_sha256"] = sha256_file(a.output / "trajectories.npz")
    write_json(a.output / "report.json", report)

    def fmt(v):
        return "n/a" if v is None else f"{v:.3f}"

    lines = [
        f"# Semitone disappearance — fold {a.val_fold}",
        "",
        f"Rows: **{len(rows)}** (K2={report['K2_rows']}, K3={report['K3_rows']}).",
        f"+0 spectral replay max error: **{map_error_max:.6g}**; prediction parity: **yes**.",
        "",
        "| true K | median first K2 | q25 | q75 | censored >+24 | reentered K3 | monotone K |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in (2, 3):
        s = report["by_true_k"][str(k)]
        f = s["first_k2_step"]
        lines.append(
            f"| {k} | {f['median']:.2f} | {f['q25']:.2f} | {f['q75']:.2f} | "
            f"{f['censored_at_25']} | {s['reentered_k3_rate']:.3f} | "
            f"{s['monotone_nonincreasing_rate']:.3f} |"
        )
    lines += [
        "",
        f"First-K2 oriented AUC: **{fmt(oriented_auc)}** "
        f"({direction or 'n/a'}).",
        "",
        "No correction rule is selected in this fold job; selection is done cross-fold on FIT only.",
    ]
    (a.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cases", "dataset", "bundle", "config", "fold-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--val-fold", type=int, required=True)
    run(p.parse_args())
