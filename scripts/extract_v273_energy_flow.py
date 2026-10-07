"""Energy-flow audit on the frozen 488-row V27.3 K2/K3 cohort.

This experiment tests a deliberately different physical view of Exact-K:
the pickup waveform is treated as a time-varying energy signal.  We do NOT
claim to reconstruct Navier-Stokes or acoustic intensity from a mono magnetic
pickup.  Instead we extract label-independent proxies for:

* local pickup energy;
* signed energy flow dE/dt (positive injection / negative dissipation);
* cross-band redistribution where positive and negative flows coexist;
* geometry/rank of the multi-band flow field;
* persistence of newly injected energy;
* connected "birth/death islands" in the band x time flow map.

Truth labels are stored only for later cross-fold evaluation.
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
from scripts.v273_residual_audit import FOLDS, require

SAMPLE_RATE = 44100
PRE_MS = 80.0
POST_MS = 160.0
FRAME = 512
HOP = 128
BANDS = 24
FMIN = 55.0
FMAX = 8000.0


def _safe_segment(x, center, pre, post):
    lo = int(center) - int(pre)
    hi = int(center) + int(post)
    out = np.zeros(int(pre + post), np.float64)
    src_lo = max(0, lo)
    src_hi = min(len(x), hi)
    if src_hi > src_lo:
        out[src_lo - lo:src_hi - lo] = x[src_lo:src_hi]
    return out


def _frames(x):
    require(len(x) >= FRAME, "segment shorter than frame")
    starts = np.arange(0, len(x) - FRAME + 1, HOP, dtype=np.int64)
    w = np.hanning(FRAME).astype(np.float64)
    return np.stack([x[s:s + FRAME] * w for s in starts]), starts


def _band_energy(segment):
    frames, starts = _frames(segment)
    spec = np.fft.rfft(frames, axis=1)
    power = np.square(np.abs(spec))
    freq = np.fft.rfftfreq(FRAME, 1.0 / SAMPLE_RATE)

    edges = np.geomspace(FMIN, FMAX, BANDS + 1)
    band = np.zeros((len(frames), BANDS), np.float64)
    centers = np.sqrt(edges[:-1] * edges[1:])
    for b, (a, z) in enumerate(zip(edges[:-1], edges[1:])):
        m = (freq >= a) & (freq < z)
        if np.any(m):
            band[:, b] = np.sum(power[:, m], axis=1)
    frame_center_samples = starts + FRAME / 2.0
    pre_samples = PRE_MS * SAMPLE_RATE / 1000.0
    rel_ms = (frame_center_samples - pre_samples) * 1000.0 / SAMPLE_RATE
    return band, centers, rel_ms


def _entropy_rows(x):
    x = np.maximum(np.asarray(x, np.float64), 0.0)
    p = x / (np.sum(x, axis=1, keepdims=True) + 1e-12)
    return -np.sum(p * np.log(p + 1e-12), axis=1) / math.log(x.shape[1])


def _weighted_center(x, centers):
    x = np.maximum(np.asarray(x, np.float64), 0.0)
    return np.sum(x * centers[None, :], axis=1) / (np.sum(x, axis=1) + 1e-12)


def _stat(d, name, x):
    x = np.asarray(x, np.float64)
    d[name + "_mean"] = float(np.mean(x))
    d[name + "_std"] = float(np.std(x))
    d[name + "_min"] = float(np.min(x))
    d[name + "_max"] = float(np.max(x))


def _slope(y, t):
    y = np.asarray(y, np.float64)
    t = np.asarray(t, np.float64)
    if len(y) < 2:
        return 0.0
    return float(np.polyfit(t, y, 1)[0])


def _corr(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _effective_rank(x):
    x = np.asarray(x, np.float64)
    if min(x.shape) == 0 or np.max(np.abs(x)) <= 1e-12:
        return 0.0, 0.0, 0.0
    s = np.linalg.svd(x, compute_uv=False)
    p = s / (np.sum(s) + 1e-12)
    erank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
    first = float(p[0]) if len(p) else 0.0
    second = float(p[1]) if len(p) > 1 else 0.0
    return erank, first, second


def _islands(mass, q):
    mass = np.asarray(mass, np.float64)
    positive = mass[mass > 0]
    if not len(positive):
        return 0.0, 0.0, 0.0
    threshold = float(np.quantile(positive, q))
    mask = mass >= threshold
    labs, n = cc_label(mask, structure=np.ones((3, 3), np.int8))
    if n == 0:
        return 0.0, 0.0, 0.0
    masses = np.asarray([mass[labs == i].sum() for i in range(1, n + 1)], np.float64)
    masses.sort()
    total = float(masses.sum()) + 1e-12
    return float(n), float(masses[-1] / total), float(np.sum(masses[-min(3, n):]) / total)


def energy_flow_features(samples, start):
    pre = int(round(PRE_MS * SAMPLE_RATE / 1000.0))
    post = int(round(POST_MS * SAMPLE_RATE / 1000.0))
    seg = _safe_segment(samples, int(start), pre, post)
    E, centers, t = _band_energy(seg)
    require(np.isfinite(E).all() and np.all(E >= 0), "bad band energy")

    total = np.sum(E, axis=1)
    scale = float(np.median(total) + 1e-12)
    En = E / scale
    Tn = total / scale

    pre_m = (t >= -60.0) & (t < -5.0)
    onset_m = (t >= -5.0) & (t <= 50.0)
    post_m = (t > 50.0) & (t <= 130.0)
    require(pre_m.any() and onset_m.any() and post_m.any(), "time mask empty")

    feat = {}

    # Energy state: how much energy exists and how its distribution changes.
    for label, m in (("pre", pre_m), ("on", onset_m), ("post", post_m)):
        _stat(feat, "energy__total_" + label, Tn[m])
        ent = _entropy_rows(En[m])
        cen = _weighted_center(En[m], centers)
        _stat(feat, "energy__entropy_" + label, ent)
        _stat(feat, "energy__centroid_" + label, cen)
        shares = En[m] / (np.sum(En[m], axis=1, keepdims=True) + 1e-12)
        _stat(feat, "energy__dominant_share_" + label, np.max(shares, axis=1))

    pre_mean = float(np.mean(Tn[pre_m])) + 1e-12
    on_mean = float(np.mean(Tn[onset_m])) + 1e-12
    post_mean = float(np.mean(Tn[post_m])) + 1e-12
    feat["energy__log_on_pre_ratio"] = float(np.log(on_mean / pre_mean))
    feat["energy__log_post_pre_ratio"] = float(np.log(post_mean / pre_mean))
    feat["energy__log_post_on_ratio"] = float(np.log(post_mean / on_mean))
    feat["energy__on_slope_per_ms"] = _slope(Tn[onset_m], t[onset_m])
    feat["energy__post_slope_per_ms"] = _slope(Tn[post_m], t[post_m])

    # Signed dE/dt across log-frequency bands.
    flow = np.diff(En, axis=0)
    ft = (t[:-1] + t[1:]) / 2.0
    fm = (ft >= -5.0) & (ft <= 100.0)
    require(fm.any(), "flow window empty")
    F = flow[fm]
    pos = np.maximum(F, 0.0)
    neg = np.maximum(-F, 0.0)
    pos_frame = np.sum(pos, axis=1)
    neg_frame = np.sum(neg, axis=1)
    net_frame = np.sum(F, axis=1)
    gross_frame = pos_frame + neg_frame
    redistribution = (gross_frame - np.abs(net_frame)) / (gross_frame + 1e-12)
    turnover = 2.0 * np.minimum(pos_frame, neg_frame) / (gross_frame + 1e-12)

    for name, vals in (
        ("positive", pos_frame),
        ("negative", neg_frame),
        ("net", net_frame),
        ("gross", gross_frame),
        ("redistribution", redistribution),
        ("turnover", turnover),
    ):
        _stat(feat, "flux__" + name, vals)

    feat["flux__positive_mass"] = float(np.sum(pos))
    feat["flux__negative_mass"] = float(np.sum(neg))
    feat["flux__net_mass"] = float(np.sum(F))
    feat["flux__gross_mass"] = float(np.sum(np.abs(F)))
    feat["flux__birth_fraction"] = float(np.sum(pos) / (np.sum(np.abs(F)) + 1e-12))
    feat["flux__death_fraction"] = float(np.sum(neg) / (np.sum(np.abs(F)) + 1e-12))
    feat["flux__cancellation_fraction"] = float(
        (np.sum(np.abs(F)) - abs(np.sum(F))) / (np.sum(np.abs(F)) + 1e-12)
    )
    feat["flux__positive_band_count_mean"] = float(np.mean(np.sum(pos > 0, axis=1)))
    feat["flux__negative_band_count_mean"] = float(np.mean(np.sum(neg > 0, axis=1)))

    # Flow geometry: multiple independent directions can indicate simultaneous
    # injection, dissipation, or redistribution rather than one scalar attack.
    erank, s1, s2 = _effective_rank(F)
    feat["geometry__effective_rank"] = erank
    feat["geometry__sv1_fraction"] = s1
    feat["geometry__sv2_fraction"] = s2
    feat["geometry__sv12_gap"] = s1 - s2

    if len(F) > 1:
        cos = []
        for a, b in zip(F[:-1], F[1:]):
            den = float(np.linalg.norm(a) * np.linalg.norm(b))
            cos.append(float(np.dot(a, b) / den) if den > 1e-12 else 0.0)
        _stat(feat, "geometry__consecutive_cosine", np.asarray(cos))
    else:
        _stat(feat, "geometry__consecutive_cosine", np.asarray([0.0]))

    for q in (0.75, 0.90):
        n, top, top3 = _islands(pos, q)
        feat[f"geometry__birth_islands_q{int(q*100)}"] = n
        feat[f"geometry__birth_top_fraction_q{int(q*100)}"] = top
        feat[f"geometry__birth_top3_fraction_q{int(q*100)}"] = top3
        n, top, top3 = _islands(neg, q)
        feat[f"geometry__death_islands_q{int(q*100)}"] = n
        feat[f"geometry__death_top_fraction_q{int(q*100)}"] = top
        feat[f"geometry__death_top3_fraction_q{int(q*100)}"] = top3

    # Energy-flow interaction: does injected energy persist, and does removed
    # energy actually stay removed? This is different from a single onset peak.
    pre_band = np.mean(En[pre_m], axis=0)
    on_band = np.mean(En[onset_m], axis=0)
    post_band = np.mean(En[post_m], axis=0)
    pos_band = np.sum(pos, axis=0)
    neg_band = np.sum(neg, axis=0)
    gain = post_band - pre_band
    loss = pre_band - post_band
    feat["interaction__positive_flow_post_gain_corr"] = _corr(pos_band, gain)
    feat["interaction__negative_flow_post_loss_corr"] = _corr(neg_band, loss)

    pos_total = float(np.sum(pos_band)) + 1e-12
    neg_total = float(np.sum(neg_band)) + 1e-12
    feat["interaction__persistent_birth_fraction"] = float(
        np.sum(pos_band[gain > 0]) / pos_total
    )
    feat["interaction__persistent_death_fraction"] = float(
        np.sum(neg_band[loss > 0]) / neg_total
    )
    feat["interaction__birth_weighted_post_energy"] = float(
        np.sum(pos_band * post_band) / pos_total
    )
    feat["interaction__death_weighted_post_energy"] = float(
        np.sum(neg_band * post_band) / neg_total
    )
    feat["interaction__birth_weighted_gain"] = float(np.sum(pos_band * gain) / pos_total)
    feat["interaction__death_weighted_loss"] = float(np.sum(neg_band * loss) / neg_total)

    # Frequency location of injection/dissipation. These do not encode note
    # identity; they describe where energy enters/leaves the observed signal.
    feat["interaction__birth_centroid_hz"] = float(
        np.sum(pos_band * centers) / (np.sum(pos_band) + 1e-12)
    )
    feat["interaction__death_centroid_hz"] = float(
        np.sum(neg_band * centers) / (np.sum(neg_band) + 1e-12)
    )
    feat["interaction__birth_death_centroid_gap_hz"] = float(
        abs(feat["interaction__birth_centroid_hz"] - feat["interaction__death_centroid_hz"])
    )

    values = np.asarray(list(feat.values()), np.float64)
    require(np.isfinite(values).all(), "nonfinite energy-flow feature")
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
    require(len(cases) == 488, f"cohort drift: {len(cases)}")
    rows = [r for r in cases if int(r["fold"]) == a.fold]
    require(rows, "empty fold")
    require(all(int(r["true_K"]) in (2, 3) for r in rows), "cohort must be K2/K3")

    wanted = {r["recording_id"] for r in rows}
    tracks = {
        t.annotation_member: t
        for t in index_guitarset(a.dataset)
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
        feat = energy_flow_features(
            audio[row["recording_id"]],
            int(row["start_sample"]),
        )
        out.append({
            "row_id": int(row["row_id"]),
            "fold": int(a.fold),
            "true_k": int(row["true_K"]),
            "recording_id": row["recording_id"],
            "start_sample": int(row["start_sample"]),
            "features": feat,
        })

    names = sorted(out[0]["features"])
    require(all(sorted(r["features"]) == names for r in out), "feature schema drift")

    a.output.mkdir(parents=True)
    (a.output / "rows.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in out)
    )
    report = {
        "status": "completed",
        "experiment": "v273_energy_flow_extract",
        "fold": int(a.fold),
        "rows": len(out),
        "K2": int(sum(r["true_k"] == 2 for r in out)),
        "K3": int(sum(r["true_k"] == 3 for r in out)),
        "feature_count": len(names),
        "feature_families": {
            family: int(sum(n.startswith(family + "__") for n in names))
            for family in ("energy", "flux", "geometry", "interaction")
        },
        "window_ms": {"pre": PRE_MS, "post": POST_MS},
        "stft": {"frame": FRAME, "hop": HOP, "bands": BANDS, "fmin": FMIN, "fmax": FMAX},
        "physical_scope": (
            "Mono guitar pickup energy/flow proxy only; not direct acoustic intensity "
            "and not a Navier-Stokes state reconstruction."
        ),
        "labels_used_during_feature_extraction": False,
        "fold3_used": False,
        "prediction_changes": False,
    }
    (a.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
