"""Energy-continuity audit: couple energy state and flow instead of concatenating them.

This is a second, stricter test of the user's "sound as flow" hypothesis.
The first audit showed that absolute signed flow carried signal, while simply
concatenating static energy and flow degraded it.  Here the coupling is put
inside the observable itself.

For each log-frequency band we form:
  relative flow:
      g = 2 (E[t+1] - E[t]) / (E[t+1] + E[t] + eps)

and a passive-continuation residual.  A per-band continuation ratio is learned
from the PRE window only (no labels):
      r_b = median(E[t+1]/E[t]) on pre-onset frames
      R_b(t) = E_b[t+1] - r_b E_b[t]

R > 0 is energy unexplained by passive continuation (source/injection proxy).
R < 0 is faster-than-expected loss (sink/mute proxy).

This is NOT Navier-Stokes reconstruction.  It is a pickup-domain conservation
proxy designed to distinguish source, continuation, and sink.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import label as cc_label

from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.extract_v273_energy_flow import (
    SAMPLE_RATE, PRE_MS, POST_MS, _safe_segment, _band_energy, _effective_rank
)
from scripts.v273_residual_audit import FOLDS, require


def _stat(d, name, x):
    x = np.asarray(x, np.float64)
    d[name + "_mean"] = float(np.mean(x))
    d[name + "_std"] = float(np.std(x))
    d[name + "_min"] = float(np.min(x))
    d[name + "_max"] = float(np.max(x))


def _island_stats(x, q=0.85):
    x = np.asarray(x, np.float64)
    pos = x[x > 0]
    if not len(pos):
        return 0.0, 0.0, 0.0
    thr = float(np.quantile(pos, q))
    labs, n = cc_label(x >= thr, structure=np.ones((3, 3), np.int8))
    if not n:
        return 0.0, 0.0, 0.0
    mass = np.asarray([x[labs == i].sum() for i in range(1, n + 1)], np.float64)
    total = float(np.sum(mass)) + 1e-12
    mass.sort()
    return float(n), float(mass[-1] / total), float(np.sum(mass[-min(3,n):]) / total)


def _temporal_persistence(x):
    """Weighted probability that an active band remains active next frame."""
    x = np.asarray(x, np.float64)
    if x.shape[0] < 2 or np.sum(x[:-1]) <= 1e-12:
        return 0.0
    a = x[:-1]
    b = x[1:]
    retained = np.minimum(a, b)
    return float(np.sum(retained) / (np.sum(a) + 1e-12))


def continuity_features(samples, start):
    pre = int(round(PRE_MS * SAMPLE_RATE / 1000.0))
    post = int(round(POST_MS * SAMPLE_RATE / 1000.0))
    segment = _safe_segment(samples, int(start), pre, post)
    E, centers, t = _band_energy(segment)
    E = np.asarray(E, np.float64)
    require(np.isfinite(E).all() and np.all(E >= 0), "bad energy")

    # Per-row scale only removes pickup/loudness scale. It does not mix labels.
    scale = float(np.median(np.sum(E, axis=1)) + 1e-12)
    En = E / scale

    ft = (t[:-1] + t[1:]) / 2.0
    pre_flow = (ft >= -60.0) & (ft < -8.0)
    event = (ft >= -8.0) & (ft <= 110.0)
    require(pre_flow.any() and event.any(), "empty temporal masks")

    E0, E1 = En[:-1], En[1:]
    dE = E1 - E0

    # Symmetric relative change, bounded and directly energy-conditioned.
    local_floor = np.maximum(
        np.median(En[t < -8.0], axis=0, keepdims=True) * 1e-3,
        1e-10,
    )
    denom = E1 + E0 + local_floor
    rel = 2.0 * dE / denom

    # Log-energy flow: relative growth/decay rate of each modal band.
    logflow = np.log(E1 + local_floor) - np.log(E0 + local_floor)

    # Estimate passive continuation from PRE only. Robust ratios are clipped
    # solely for numerical stability; no truth labels or held-out tuning.
    ratio = (E1[pre_flow] + local_floor) / (E0[pre_flow] + local_floor)
    passive_ratio = np.median(ratio, axis=0)
    passive_ratio = np.clip(passive_ratio, 0.50, 1.50)
    predicted = E0 * passive_ratio[None, :]
    residual = E1 - predicted
    residual_rel = 2.0 * residual / (E1 + predicted + local_floor)

    # Event slice.
    A = dE[event]
    G = rel[event]
    L = logflow[event]
    R = residual[event]
    Q = residual_rel[event]

    # Weight relative rates by an energy-presence measure. This suppresses
    # numerically huge changes in nearly empty bands without turning the
    # observable back into raw absolute flow.
    presence = np.sqrt(np.maximum(E0[event], 0.0) * np.maximum(E1[event], 0.0))
    pnorm = presence / (np.sum(presence, axis=1, keepdims=True) + 1e-12)
    WG = G * pnorm
    WQ = Q * pnorm

    feat = {}

    # Control: absolute flow from the first audit, recomputed on the same window.
    apos = np.maximum(A, 0.0)
    aneg = np.maximum(-A, 0.0)
    for name, x in (
        ("positive_frame", np.sum(apos, axis=1)),
        ("negative_frame", np.sum(aneg, axis=1)),
        ("gross_frame", np.sum(np.abs(A), axis=1)),
        ("net_frame", np.sum(A, axis=1)),
    ):
        _stat(feat, "absolute__" + name, x)

    # Relative energy flow: source/sink rate relative to existing state.
    for prefix, X in (("relative__", G), ("logflow__", L), ("weighted_relative__", WG)):
        pos = np.maximum(X, 0.0)
        neg = np.maximum(-X, 0.0)
        _stat(feat, prefix + "positive_frame", np.sum(pos, axis=1))
        _stat(feat, prefix + "negative_frame", np.sum(neg, axis=1))
        _stat(feat, prefix + "net_frame", np.sum(X, axis=1))
        _stat(feat, prefix + "gross_frame", np.sum(np.abs(X), axis=1))
        feat[prefix + "positive_mass"] = float(np.sum(pos))
        feat[prefix + "negative_mass"] = float(np.sum(neg))
        feat[prefix + "birth_fraction"] = float(np.sum(pos) / (np.sum(np.abs(X)) + 1e-12))
        feat[prefix + "coexistence_fraction"] = float(
            np.mean((np.sum(pos, axis=1) > 1e-8) & (np.sum(neg, axis=1) > 1e-8))
        )
        feat[prefix + "positive_persistence"] = _temporal_persistence(pos)
        feat[prefix + "negative_persistence"] = _temporal_persistence(neg)
        er, s1, s2 = _effective_rank(X)
        feat[prefix + "effective_rank"] = er
        feat[prefix + "sv1_fraction"] = s1
        feat[prefix + "sv2_fraction"] = s2

    # Continuity residual after subtracting expected passive continuation.
    for prefix, X in (("residual__", R), ("relative_residual__", Q), ("weighted_residual__", WQ)):
        src = np.maximum(X, 0.0)
        sink = np.maximum(-X, 0.0)
        _stat(feat, prefix + "source_frame", np.sum(src, axis=1))
        _stat(feat, prefix + "sink_frame", np.sum(sink, axis=1))
        _stat(feat, prefix + "net_frame", np.sum(X, axis=1))
        _stat(feat, prefix + "gross_frame", np.sum(np.abs(X), axis=1))
        feat[prefix + "source_mass"] = float(np.sum(src))
        feat[prefix + "sink_mass"] = float(np.sum(sink))
        feat[prefix + "source_fraction"] = float(np.sum(src) / (np.sum(np.abs(X)) + 1e-12))
        feat[prefix + "source_persistence"] = _temporal_persistence(src)
        feat[prefix + "sink_persistence"] = _temporal_persistence(sink)
        feat[prefix + "source_sink_coexistence"] = float(
            np.mean((np.sum(src, axis=1) > 1e-8) & (np.sum(sink, axis=1) > 1e-8))
        )
        n, top, top3 = _island_stats(src)
        feat[prefix + "source_islands"] = n
        feat[prefix + "source_top_fraction"] = top
        feat[prefix + "source_top3_fraction"] = top3
        n, top, top3 = _island_stats(sink)
        feat[prefix + "sink_islands"] = n
        feat[prefix + "sink_top_fraction"] = top
        feat[prefix + "sink_top3_fraction"] = top3
        er, s1, s2 = _effective_rank(X)
        feat[prefix + "effective_rank"] = er
        feat[prefix + "sv1_fraction"] = s1
        feat[prefix + "sv2_fraction"] = s2

    # Passive state itself is informative only as a dynamical parameter, not
    # static loudness: how much of the prior energy is expected to survive.
    feat["passive__ratio_mean"] = float(np.mean(passive_ratio))
    feat["passive__ratio_std"] = float(np.std(passive_ratio))
    feat["passive__ratio_min"] = float(np.min(passive_ratio))
    feat["passive__ratio_max"] = float(np.max(passive_ratio))
    feat["passive__decaying_band_fraction"] = float(np.mean(passive_ratio < 1.0))
    feat["passive__growing_band_fraction"] = float(np.mean(passive_ratio > 1.0))

    vals = np.asarray(list(feat.values()), np.float64)
    require(np.isfinite(vals).all(), "nonfinite continuity feature")
    return feat


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    require(a.fold in FOLDS and a.fold != 3, "bad fold")
    require(not a.output.exists(), "refusing overwrite")
    cases = [json.loads(x) for x in a.cases.read_text().splitlines() if x.strip()]
    require(len(cases) == 488, f"cohort drift {len(cases)}")
    rows = [r for r in cases if int(r["fold"]) == a.fold]
    require(rows and all(int(r["true_K"]) in (2,3) for r in rows), "bad cohort")

    wanted = {r["recording_id"] for r in rows}
    tracks = {
        t.annotation_member: t for t in index_guitarset(a.dataset)
        if t.annotation_member in wanted
    }
    require(set(tracks) == wanted, "audio coverage incomplete")
    audio = {}
    for member, track in tracks.items():
        wav = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        require(int(wav.sample_rate) == SAMPLE_RATE, "sample-rate drift")
        audio[member] = np.asarray(wav.samples, np.float64) / 32768.0

    out = []
    for row in rows:
        feat = continuity_features(audio[row["recording_id"]], int(row["start_sample"]))
        out.append({
            "row_id": int(row["row_id"]),
            "fold": int(a.fold),
            "true_k": int(row["true_K"]),
            "features": feat,
        })

    names = sorted(out[0]["features"])
    require(all(sorted(r["features"]) == names for r in out), "schema drift")
    a.output.mkdir(parents=True)
    (a.output / "rows.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in out)
    )
    report = {
        "status":"completed",
        "experiment":"v273_energy_continuity_extract",
        "fold":int(a.fold),
        "rows":len(out),
        "K2":int(sum(r["true_k"]==2 for r in out)),
        "K3":int(sum(r["true_k"]==3 for r in out)),
        "feature_count":len(names),
        "families":{
            p:int(sum(n.startswith(p+"__") for n in names))
            for p in ("absolute","relative","logflow","weighted_relative","residual",
                      "relative_residual","weighted_residual","passive")
        },
        "equation":"R_b(t)=E_b(t+1)-r_b*E_b(t), r_b estimated from PRE only",
        "labels_used_during_feature_extraction":False,
        "fold3_used":False,
        "automatic_promotion":False,
    }
    (a.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
