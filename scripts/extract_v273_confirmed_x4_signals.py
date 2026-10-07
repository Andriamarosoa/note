"""Extract inference-safe signals on x4 disappearances confirmed by both engines.

Cohort: rows where BOTH Librosa x4 and Rubber Band x4 change the frozen
base prediction K3 -> K2. This cohort is expected to contain 76 rows globally
(40 true K2 / 36 true K3).

Feature families are extracted without using truth labels:
- raw pick/transient waveform features;
- selected-F0 attack novelty + exclusive harmonic support;
- fine-STFT harmonic reconstruction features;
- explicit transition-spectrum morphology;
- progressive semitone trajectory shape/confidence;
- Librosa/RubberBand x4 confidence and margin-change features.

Truth K is stored only for later cross-fold evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, peak_widths

from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.audit_v273_raw_pcm_pick_transient import extract_segment, transient_features
from scripts.audit_v273_attack_novelty import raw_powers
from scripts.audit_v273_selected_f0_attack_exclusive import row_features as selected_f0_features
from scripts.audit_v273_internal_b_low_harmonic_strata import transition_spectrum, extract_one
from scripts.v273_residual_audit import FOLDS, require

EXPECTED_GLOBAL_COMMON = 76


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def spectral_entropy(p):
    p = np.asarray(p, np.float64)
    p = np.maximum(p, 0)
    p = p / (p.sum() + 1e-12)
    return float(-np.sum(p * np.log(p + 1e-12)) / max(1e-12, math.log(len(p))))


def spectrum_morphology(freq, x):
    """Explicit peak morphology on the positive post-minus-pre attack spectrum."""
    freq = np.asarray(freq, np.float64)
    x = np.asarray(x, np.float64)
    require(len(freq) == len(x) and len(x) > 8, "bad transition spectrum")
    p = np.maximum(x, 0.0)
    total = float(p.sum()) + 1e-12
    pn = p / total
    centroid = float(np.sum(freq * pn))
    spread = float(np.sqrt(np.sum(((freq - centroid) ** 2) * pn)))
    skew = float(np.sum(((freq - centroid) / (spread + 1e-12)) ** 3 * pn))
    kurt = float(np.sum(((freq - centroid) / (spread + 1e-12)) ** 4 * pn))

    # Robust relative peak detection.
    height = max(float(np.max(p)) * 0.05, float(np.quantile(p, 0.90)) * 0.25)
    prominence = max(float(np.max(p)) * 0.02, 1e-12)
    peaks, props = find_peaks(p, height=height, prominence=prominence, distance=2)
    if not len(peaks):
        peaks = np.asarray([int(np.argmax(p))], np.int64)
        prominences = np.asarray([float(np.max(p))])
    else:
        prominences = np.asarray(props["prominences"], np.float64)

    widths = peak_widths(p, peaks, rel_height=0.5)[0] if len(peaks) else np.asarray([0.0])
    df = float(np.median(np.diff(freq)))
    widths_hz = widths * df

    order = np.argsort(-p[peaks], kind="stable")
    top = peaks[order[: min(3, len(order))]]
    sym = []
    local_contrast = []
    for idx in top:
        radius = 5
        l = p[max(0, idx - radius):idx]
        r = p[idx + 1:min(len(p), idx + 1 + radius)]
        le = float(l.sum())
        re = float(r.sum())
        sym.append(1.0 - abs(le - re) / (le + re + 1e-12))
        shoulder = np.r_[l, r]
        local_contrast.append(float(p[idx] / (np.mean(shoulder) + 1e-12)) if len(shoulder) else 0.0)

    top_energy = np.sort(p[peaks])[-min(3, len(peaks)):].sum() / total
    return {
        "peak_count": float(len(peaks)),
        "peak_density_per_khz": float(len(peaks) / max((freq[-1] - freq[0]) / 1000.0, 1e-12)),
        "dominant_peak_hz": float(freq[int(peaks[np.argmax(p[peaks])])]),
        "dominant_peak_fraction": float(np.max(p[peaks]) / total),
        "top3_peak_fraction": float(top_energy),
        "peak_width_hz_mean": float(np.mean(widths_hz)),
        "peak_width_hz_max": float(np.max(widths_hz)),
        "peak_prominence_mean": float(np.mean(prominences)),
        "peak_prominence_max": float(np.max(prominences)),
        "peak_symmetry_mean": float(np.mean(sym)),
        "peak_symmetry_min": float(np.min(sym)),
        "local_peak_contrast_mean": float(np.mean(local_contrast)),
        "spectral_entropy": spectral_entropy(p),
        "spectral_centroid_hz": centroid,
        "spectral_spread_hz": spread,
        "spectral_skew": skew,
        "spectral_kurtosis": kurt,
    }


def trajectory_features(pred, prob):
    pred = np.asarray(pred, np.int32)
    prob = np.asarray(prob, np.float64)
    require(pred.shape == (25,) and prob.shape == (25, 7), "trajectory shape drift")
    margin = prob[:, 2] - prob[:, 3]
    steps = np.arange(25, dtype=np.float64)
    slope = float(np.polyfit(steps, margin, 1)[0])
    transitions = int(np.sum(pred[1:] != pred[:-1]))
    reentry = int(np.sum((pred[:-1] != 3) & (pred[1:] == 3)))
    first_k2 = next((i for i, k in enumerate(pred) if int(k) == 2), 25)
    first_non3 = next((i for i, k in enumerate(pred) if int(k) != 3), 25)
    first_below3 = next((i for i, k in enumerate(pred) if int(k) < 3), 25)
    return {
        "first_k2_step": float(first_k2),
        "first_non3_step": float(first_non3),
        "first_below3_step": float(first_below3),
        "k2_step_count": float(np.sum(pred == 2)),
        "k3_step_count": float(np.sum(pred == 3)),
        "transition_count": float(transitions),
        "reentry_k3_count": float(reentry),
        "total_variation_k": float(np.sum(np.abs(np.diff(pred)))),
        "unique_predicted_k": float(len(np.unique(pred))),
        "margin23_start": float(margin[0]),
        "margin23_end": float(margin[-1]),
        "margin23_delta": float(margin[-1] - margin[0]),
        "margin23_mean": float(np.mean(margin)),
        "margin23_std": float(np.std(margin)),
        "margin23_min": float(np.min(margin)),
        "margin23_max": float(np.max(margin)),
        "margin23_slope": slope,
    }


def confidence_features(base_prob, lib_prob, rub_prob):
    base_prob = np.asarray(base_prob, np.float64)
    lib_prob = np.asarray(lib_prob, np.float64)
    rub_prob = np.asarray(rub_prob, np.float64)
    for p in (base_prob, lib_prob, rub_prob):
        require(p.shape == (7,) and np.isfinite(p).all(), "probability shape drift")

    bm = float(base_prob[2] - base_prob[3])
    lm = float(lib_prob[2] - lib_prob[3])
    rm = float(rub_prob[2] - rub_prob[3])
    return {
        "base_p2": float(base_prob[2]),
        "base_p3": float(base_prob[3]),
        "base_margin23": bm,
        "lib_x4_p2": float(lib_prob[2]),
        "lib_x4_p3": float(lib_prob[3]),
        "lib_x4_margin23": lm,
        "rub_x4_p2": float(rub_prob[2]),
        "rub_x4_p3": float(rub_prob[3]),
        "rub_x4_margin23": rm,
        "mean_x4_p2": float((lib_prob[2] + rub_prob[2]) / 2.0),
        "min_x4_p2": float(min(lib_prob[2], rub_prob[2])),
        "mean_x4_margin23": float((lm + rm) / 2.0),
        "min_x4_margin23": float(min(lm, rm)),
        "engine_margin_gap": float(abs(lm - rm)),
        "lib_margin_gain": float(lm - bm),
        "rub_margin_gain": float(rm - bm),
        "mean_margin_gain": float((lm + rm) / 2.0 - bm),
        "lib_entropy": spectral_entropy(lib_prob),
        "rub_entropy": spectral_entropy(rub_prob),
        "base_entropy": spectral_entropy(base_prob),
    }


def prefix(d, p):
    return {p + k: float(v) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("cases", "dataset", "comparison", "trajectory", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--fold", type=int, required=True)
    a = ap.parse_args()

    require(a.fold in FOLDS and a.fold != 3, "bad fold")
    require(not a.output.exists(), "refusing overwrite")

    cases_all = [json.loads(line) for line in a.cases.read_text().splitlines()]
    case_map = {int(r["row_id"]): r for r in cases_all}
    require(len(case_map) == 488, "case cohort drift")

    comp = load_npz(a.comparison)
    traj = load_npz(a.trajectory)
    ids = np.asarray(comp["row_id"], np.int64)
    y = np.asarray(comp["true_k"], np.int32)
    lib_take = np.asarray(comp["librosa_k2_action"], bool)
    rub_take = np.asarray(comp["rubberband_k2_action"], bool)
    common = lib_take & rub_take

    tids = np.asarray(traj["row_id"], np.int64)
    require(set(tids.tolist()) == set(ids.tolist()), "trajectory/comparison row identity drift")
    tpos = {int(v): i for i, v in enumerate(tids)}
    order = np.asarray([tpos[int(v)] for v in ids], np.int64)
    tpred = np.asarray(traj["predicted"], np.int32)[order]
    tprob = np.asarray(traj["probability"], np.float64)[order]
    require(np.array_equal(np.asarray(traj["true_k"], np.int32)[order], y), "truth drift")

    selected_ids = ids[common]
    selected_y = y[common]
    selected_pred = tpred[common]
    selected_prob = tprob[common]
    selected_lib_prob = np.asarray(comp["librosa_x4_probability"], np.float64)[common]
    selected_rub_prob = np.asarray(comp["rubberband_x4_probability"], np.float64)[common]

    rows = [case_map[int(rid)] for rid in selected_ids]
    require(all(int(r["fold"]) == a.fold for r in rows), "fold identity drift")
    require(np.array_equal(np.asarray([r["true_K"] for r in rows], np.int32), selected_y), "case truth drift")

    wanted = {r["recording_id"] for r in rows}
    tracks = {t.annotation_member: t for t in index_guitarset(a.dataset) if t.annotation_member in wanted}
    require(set(tracks) == wanted, "audio coverage incomplete")
    audio = {}
    for member, track in tracks.items():
        wav = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        audio[member] = np.asarray(wav.samples, np.float64) / 32768.0

    measured = []
    for i, row in enumerate(rows):
        samples = audio[row["recording_id"]]
        start = int(row["start_sample"])

        trans = transient_features(extract_segment(samples, start))
        freq_raw, powers, _ = raw_powers(samples, start)
        f0feat = selected_f0_features(row, freq_raw, powers)
        freq, x = transition_spectrum(samples, start)
        harm = extract_one(freq, x)
        require(harm is not None, "harmonic extraction failed")
        morph = spectrum_morphology(freq, x)
        tfeat = trajectory_features(selected_pred[i], selected_prob[i])
        cfeat = confidence_features(
            selected_prob[i, 0],
            selected_lib_prob[i],
            selected_rub_prob[i],
        )

        feat = {}
        feat.update(prefix(trans, "transient__"))
        feat.update(prefix(f0feat, "attackf0__"))
        feat.update(prefix(harm, "harmonic__"))
        feat.update(prefix(morph, "morph__"))
        feat.update(prefix(tfeat, "trajectory__"))
        feat.update(prefix(cfeat, "confidence__"))
        require(np.isfinite(np.asarray(list(feat.values()), np.float64)).all(), "non-finite feature")

        measured.append({
            "row_id": int(row["row_id"]),
            "fold": int(a.fold),
            "true_k": int(row["true_K"]),
            "recording_id": row["recording_id"],
            "start_sample": start,
            "features": feat,
        })

    a.output.mkdir(parents=True)
    (a.output / "rows.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in measured)
    )
    report = {
        "status": "completed",
        "experiment": "v273_confirmed_x4_signal_extract",
        "fold": int(a.fold),
        "rows": len(measured),
        "K2": int(np.sum(selected_y == 2)),
        "K3": int(np.sum(selected_y == 3)),
        "feature_count": len(measured[0]["features"]) if measured else 0,
        "feature_families": {
            "transient": sum(k.startswith("transient__") for k in measured[0]["features"]),
            "attackf0": sum(k.startswith("attackf0__") for k in measured[0]["features"]),
            "harmonic": sum(k.startswith("harmonic__") for k in measured[0]["features"]),
            "morphology": sum(k.startswith("morph__") for k in measured[0]["features"]),
            "trajectory": sum(k.startswith("trajectory__") for k in measured[0]["features"]),
            "confidence": sum(k.startswith("confidence__") for k in measured[0]["features"]),
        },
        "outer_fold_3_used": False,
        "prediction_changes": False,
    }
    (a.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
