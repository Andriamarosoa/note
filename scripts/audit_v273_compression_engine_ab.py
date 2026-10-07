"""A/B quality audit for x2/x4 pitch compression: Librosa vs Rubber Band.

The experiment keeps the SAME B_low/base-K3 population and the SAME x2/x4
views used by the prior waveform pitch-shift audit:
  x1 = original
  x2 = +12 semitones
  x4 = +24 semitones

It changes only the waveform pitch-shift engine. We measure:
- K2/K3 feature separability on compressed low-register rows;
- transfer of a classifier trained on NATURAL high-register originals;
- how far the transformed register actually moves;
- transient-envelope preservation relative to the original waveform.

No Exact-K correction is applied here. This isolates the compression transform
before deciding whether it should feed the model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from causal_note.guitarset import ALLOWED_PLAYERS, SAMPLE_RATE, index_guitarset
from scripts.audit_v273_internal_b_like_boundary_corrector import (
    FOLDS,
    NEURONS,
    SEED,
    discover_reports,
    fold_ids,
    nested_base,
    predict,
    structural_features,
)
from scripts.audit_v273_internal_b_low_harmonic_strata import (
    BASE_FEATURES,
    build_blow,
    extract_one,
)
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.train_v273_group_gate_ab import build_model
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import load_bundle, require

FFT_SIZE = 8192
WINDOW = 2048
MIN_HZ = 65.0
MAX_ANALYSIS_HZ = 6000.0
PAD = 4096
VIEWS = (("x1", 0.0), ("x2", 12.0), ("x4", 24.0))
ENGINES = ("librosa", "rubberband")

# Attack-envelope comparison around the candidate onset.
ATTACK_PRE = int(round(0.020 * SAMPLE_RATE))
ATTACK_POST = int(round(0.080 * SAMPLE_RATE))
ENV_FRAME = 128
ENV_HOP = 64
ENV_MAX_LAG_FRAMES = 5


def auc(y, s):
    y = np.asarray(y, np.int32)
    s = np.asarray(s, np.float64)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None


def model():
    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(
            C=1.0,
            max_iter=3000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=28231,
        )),
    ])


def pcm_window(samples, start, length):
    out = np.zeros(length, np.float64)
    l = max(0, int(start))
    r = min(len(samples), int(start) + length)
    if r > l:
        out[l - int(start):l - int(start) + r - l] = samples[l:r]
    return out


def pitch_shift_local(samples, start, n_steps, engine):
    """Duration-preserving local pitch shift using one selected engine."""
    left = PAD + WINDOW
    right = PAD + WINDOW
    local = pcm_window(samples, int(start) - left, left + right).astype(np.float32)
    if not n_steps:
        return local.astype(np.float64), left

    if engine == "librosa":
        import librosa
        shifted = librosa.effects.pitch_shift(
            local,
            sr=SAMPLE_RATE,
            n_steps=float(n_steps),
            bins_per_octave=12,
            res_type="soxr_hq",
            scale=False,
        )
    elif engine == "rubberband":
        import pyrubberband as pyrb
        shifted = pyrb.pitch_shift(local, SAMPLE_RATE, float(n_steps))
    else:
        raise ValueError(engine)

    shifted = np.asarray(shifted, np.float64)
    if len(shifted) < len(local):
        shifted = np.pad(shifted, (0, len(local) - len(shifted)))
    elif len(shifted) > len(local):
        shifted = shifted[:len(local)]
    require(np.isfinite(shifted).all(), f"non-finite {engine} output")
    return shifted, left


def transition_from_local(local, onset):
    taper = np.hanning(WINDOW)
    pre = pcm_window(local, onset - WINDOW, WINDOW) * taper
    post = pcm_window(local, onset, WINDOW) * taper
    p = np.abs(np.fft.rfft(pre, n=FFT_SIZE)) ** 2
    q = np.abs(np.fft.rfft(post, n=FFT_SIZE)) ** 2
    freq = np.fft.rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE)
    keep = (freq >= MIN_HZ) & (freq <= MAX_ANALYSIS_HZ)
    freq = freq[keep]
    x = np.maximum(q[keep] - p[keep], 0.0)
    x /= float(np.sum(x)) + 1e-12
    return freq, x


def rms_envelope(local, onset):
    seg = pcm_window(
        local,
        int(onset) - ATTACK_PRE,
        ATTACK_PRE + ATTACK_POST,
    )
    values = []
    for start in range(0, max(1, len(seg) - ENV_FRAME + 1), ENV_HOP):
        frame = seg[start:start + ENV_FRAME]
        if len(frame) < ENV_FRAME:
            frame = np.pad(frame, (0, ENV_FRAME - len(frame)))
        values.append(float(np.sqrt(np.mean(frame * frame) + 1e-12)))
    env = np.asarray(values, np.float64)
    env /= np.linalg.norm(env) + 1e-12
    return env


def best_envelope_corr(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    best = -1.0
    best_lag = 0
    for lag in range(-ENV_MAX_LAG_FRAMES, ENV_MAX_LAG_FRAMES + 1):
        if lag < 0:
            x, y = a[-lag:], b[:len(b) + lag]
        elif lag > 0:
            x, y = a[:len(a) - lag], b[lag:]
        else:
            x, y = a, b
        n = min(len(x), len(y))
        if n < 4:
            continue
        x = x[:n] - np.mean(x[:n])
        y = y[:n] - np.mean(y[:n])
        den = np.linalg.norm(x) * np.linalg.norm(y)
        c = float(np.dot(x, y) / den) if den > 1e-12 else 0.0
        if c > best:
            best, best_lag = c, lag
    return best, best_lag


def audio_views(cache, ids, dataset_dir, engine):
    indexed = tuple(t for t in index_guitarset(dataset_dir) if t.player_id in ALLOWED_PLAYERS)
    by = {t.annotation_member: t for t in indexed}
    aud = {}
    views = {name: [] for name, _ in VIEWS}
    envcorr = {name: [] for name, _ in VIEWS}
    envlag = {name: [] for name, _ in VIEWS}

    for row in np.asarray(ids, np.int64):
        member = str(cache["members"][row])
        if member not in aud:
            t = by[member]
            wav = decode_pcm16_mono_wav(t.audio_zip, t.audio_member)
            aud[member] = np.asarray(wav.samples, np.float64) / 32768.0
        start = int(cache["cluster_start_samples"][row])

        original, onset = pitch_shift_local(aud[member], start, 0.0, engine)
        original_env = rms_envelope(original, onset)

        for name, steps in VIEWS:
            local, shifted_onset = pitch_shift_local(aud[member], start, steps, engine)
            freq, x = transition_from_local(local, shifted_onset)
            views[name].append(extract_one(freq, x))
            if steps == 0:
                envcorr[name].append(1.0)
                envlag[name].append(0)
            else:
                c, lag = best_envelope_corr(original_env, rms_envelope(local, shifted_onset))
                envcorr[name].append(c)
                envlag[name].append(lag)

    return views, envcorr, envlag


def matrix(rows):
    good = np.asarray([x is not None for x in rows], bool)
    rr = [x for x in rows if x is not None]
    if not rr:
        return good, [], np.zeros((0, 0)), {}
    keys = list(rr[0].keys())
    a = np.asarray([[r[k] for k in keys] for r in rr], np.float64)
    return good, keys, a, {k: i for i, k in enumerate(keys)}


def eval_same(af, yf, av, yv, cols):
    require(len(np.unique(yf)) == 2 and len(np.unique(yv)) == 2, "binary collapse")
    m = model()
    m.fit(af[:, cols], yf)
    p = m.predict_proba(av[:, cols])[:, 1]
    return auc(yv, p)


def save_examples(cache, ids, dataset_dir, output):
    """Save a few deterministic validation examples for listening."""
    import soundfile as sf

    output.mkdir(parents=True, exist_ok=True)
    indexed = tuple(t for t in index_guitarset(dataset_dir) if t.player_id in ALLOWED_PLAYERS)
    by = {t.annotation_member: t for t in indexed}
    aud = {}
    for n, row in enumerate(np.asarray(ids, np.int64)[:2]):
        member = str(cache["members"][row])
        if member not in aud:
            t = by[member]
            wav = decode_pcm16_mono_wav(t.audio_zip, t.audio_member)
            aud[member] = np.asarray(wav.samples, np.float64) / 32768.0
        start = int(cache["cluster_start_samples"][row])
        # Longer audible excerpt than the analysis window.
        clip = pcm_window(aud[member], start - int(0.10*SAMPLE_RATE), int(0.70*SAMPLE_RATE))
        sf.write(output / f"row{int(row)}_original.wav", clip, SAMPLE_RATE)
        for engine in ENGINES:
            # Use a 0.1s pre-roll and enough post-roll for listening.
            # The analysis transformer already protects boundaries with PAD.
            local_start = start + int(0.10*SAMPLE_RATE)
            for label, steps in (("x2", 12.0), ("x4", 24.0)):
                transformed, onset = pitch_shift_local(clip, local_start, steps, engine)
                audible = pcm_window(
                    transformed,
                    onset - int(0.10*SAMPLE_RATE),
                    int(0.70*SAMPLE_RATE),
                )
                sf.write(
                    output / f"row{int(row)}_{engine}_{label}.wav",
                    np.clip(audible, -1.0, 1.0),
                    SAMPLE_RATE,
                )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--fold-root", type=Path, required=True)
    ap.add_argument("--val-fold", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    require(a.val_fold in FOLDS, "bad fold")
    require(not a.output.exists(), "refusing overwrite")
    a.output.mkdir(parents=True)

    reports = discover_reports(a.fold_root)
    fd = reports[a.val_fold][0].parent
    cfg = load_config(a.config)
    cache, _, _ = load_bundle(a.bundle, a.config)
    fit, val = fold_ids(cache, cfg, a.val_fold)
    y = np.minimum(cache["exact"].astype(np.int32), 6)
    yf, yv = y[fit], y[val]

    mu = build_model("learned_gate", SEED)
    mu.load_weights(fd / "uniform.weights.h5")
    mg = build_model("learned_gate", SEED)
    mg.load_weights(fd / "freeze_local_combo.weights.h5")
    bu, bg = nested_base(mu), nested_base(mg)
    ku, b1 = [np.asarray(x).copy() for x in bu.get_layer("candidate_hidden1").get_weights()]
    kg, b2 = [np.asarray(x).copy() for x in bg.get_layer("candidate_hidden1").get_weights()]
    ii = np.asarray(NEURONS, np.int64)
    kg[:, ii] = ku[:, ii]
    b2[ii] = b1[ii]
    bg.get_layer("candidate_hidden1").set_weights([kg, b2])

    pf, gf = predict(mg, cache, fit)
    pv, gv = predict(mg, cache, val)
    xcf = structural_features(cache, fit, pf)
    xcv = structural_features(cache, val, pv)
    fb, vb = build_blow(xcf, xcv, yf, gf, a.val_fold)

    popf = fb & (gf == 3) & np.isin(yf, (2, 3))
    popv = vb & (gv == 3) & np.isin(yv, (2, 3))
    idsf, idsv = fit[popf], val[popv]
    yyf = (yf[popf] == 2).astype(np.int32)
    yyv = (yv[popv] == 2).astype(np.int32)
    require(len(idsf) >= 40 and len(idsv) >= 10, "small K2/K3 population")

    # Original x1 is engine-independent. Use Librosa path only as a carrier.
    base_f, _, _ = audio_views(cache, idsf, a.dataset_dir, "librosa")
    base_v, _, _ = audio_views(cache, idsv, a.dataset_dir, "librosa")
    g1f, k1f, a1f, i1f = matrix(base_f["x1"])
    g1v, k1v, a1v, i1v = matrix(base_v["x1"])
    require(k1f == k1v and len(k1f) > 0, "original keys drift")

    med_idx = i1f["median_triplet_f0"]
    low_cut = float(np.quantile(a1f[:, med_idx], 1/3))
    high_cut = float(np.quantile(a1f[:, med_idx], 2/3))

    pop_index_f = np.flatnonzero(g1f)
    pop_index_v = np.flatnonzero(g1v)
    low_orig_f = a1f[:, med_idx] < low_cut
    low_orig_v = a1v[:, med_idx] < low_cut
    high_orig_f = a1f[:, med_idx] >= high_cut
    high_orig_v = a1v[:, med_idx] >= high_cut

    cols = [i1f[k] for k in BASE_FEATURES]
    natural_high_auc = None
    if (
        high_orig_f.sum() >= 20
        and high_orig_v.sum() >= 8
        and len(np.unique(yyf[pop_index_f][high_orig_f])) == 2
        and len(np.unique(yyv[pop_index_v][high_orig_v])) == 2
    ):
        natural_high_auc = eval_same(
            a1f[high_orig_f],
            yyf[pop_index_f][high_orig_f],
            a1v[high_orig_v],
            yyv[pop_index_v][high_orig_v],
            cols,
        )

    results = {}
    audio_cache = {}
    for engine in ENGINES:
        vf, ef, lf = audio_views(cache, idsf, a.dataset_dir, engine)
        vv, ev, lv = audio_views(cache, idsv, a.dataset_dir, engine)
        results[engine] = {}

        for name, steps in VIEWS:
            gf_ok, keysf, af, idxf = matrix(vf[name])
            gv_ok, keysv, av, idxv = matrix(vv[name])
            require(keysf == keysv and keysf == k1f, f"{engine}/{name} keys drift")

            global_f = np.flatnonzero(gf_ok)
            global_v = np.flatnonzero(gv_ok)
            low_ids_f = set(pop_index_f[low_orig_f].tolist())
            low_ids_v = set(pop_index_v[low_orig_v].tolist())
            mf = np.asarray([q in low_ids_f for q in global_f], bool)
            mv = np.asarray([q in low_ids_v for q in global_v], bool)
            yff = yyf[global_f][mf]
            yvv = yyv[global_v][mv]
            c = [idxf[k] for k in BASE_FEATURES]

            same = None
            if (
                mf.sum() >= 20
                and mv.sum() >= 8
                and len(np.unique(yff)) == 2
                and len(np.unique(yvv)) == 2
            ):
                same = eval_same(af[mf], yff, av[mv], yvv, c)

            transfer = None
            if (
                name != "x1"
                and high_orig_f.sum() >= 20
                and mv.sum() >= 8
                and len(np.unique(yvv)) == 2
            ):
                tm = model()
                tm.fit(a1f[high_orig_f][:, cols], yyf[pop_index_f][high_orig_f])
                p = tm.predict_proba(av[mv][:, c])[:, 1]
                transfer = auc(yvv, p)

            shifted_med = av[mv, idxf["median_triplet_f0"]] if mv.any() else np.asarray([])

            # Envelope arrays are indexed by the original population; align with valid rows.
            valid_original_v = np.asarray(ev[name], np.float64)[gv_ok]
            env_low = valid_original_v[mv]

            results[engine][name] = {
                "n_steps": steps,
                "factor": float(2 ** (steps / 12.0)),
                "low_fit_rows": int(mf.sum()),
                "low_val_rows": int(mv.sum()),
                "same_representation_auc": same,
                "natural_high_model_transfer_auc": transfer,
                "shifted_low_median_triplet_f0_mean": float(np.mean(shifted_med)) if len(shifted_med) else None,
                "fraction_shifted_low_above_original_high_cut": float(np.mean(shifted_med >= high_cut)) if len(shifted_med) else None,
                "attack_envelope_corr_mean": float(np.mean(env_low)) if len(env_low) else None,
                "attack_envelope_corr_median": float(np.median(env_low)) if len(env_low) else None,
            }

    # Deterministic listening examples from validation low-register population.
    example_population_positions = pop_index_v[low_orig_v]
    example_ids = idsv[example_population_positions[:2]]
    save_examples(cache, example_ids, a.dataset_dir, a.output / "audio_examples")

    report = {
        "status": "completed",
        "training": False,
        "protocol": {
            "experiment": "v273_compression_engine_ab",
            "validation_fold": a.val_fold,
            "fit_folds": [x for x in FOLDS if x != a.val_fold],
            "outer_fold_3_used": False,
            "population": "B_low + base K3 + true K in {2,3}; low register from original FIT x1 bottom tertile",
            "compression_views": {"x1": 0, "x2": 12, "x4": 24},
            "engines": {
                "librosa": "librosa.effects.pitch_shift, soxr_hq",
                "rubberband": "pyrubberband -> Rubber Band CLI",
            },
            "features": list(BASE_FEATURES),
            "quality_proxy": "natural-high classifier transfer + attack-envelope preservation",
            "exact_k_model_evaluated": False,
            "automatic_promotion": False,
        },
        "register_cuts_hz": {"low_upper": low_cut, "high_lower": high_cut},
        "natural_high_auc": natural_high_auc,
        "results": results,
        "audio_example_global_indices": [int(x) for x in example_ids],
    }
    (a.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    def ff(v):
        return "n/a" if v is None else f"{v:.3f}"

    lines = [
        f"# Compression engine A/B — fold {a.val_fold}",
        "",
        f"Original low/high cuts: **{low_cut:.1f} / {high_cut:.1f} Hz**.",
        f"Natural-high AUC: **{ff(natural_high_auc)}**.",
        "",
        "| engine | view | same-low AUC | natural-high transfer AUC | attack envelope corr | above old high cut |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for engine in ENGINES:
        for name in ("x2", "x4"):
            x = results[engine][name]
            lines.append(
                f"| {engine} | {name} | {ff(x['same_representation_auc'])} | "
                f"{ff(x['natural_high_model_transfer_auc'])} | "
                f"{ff(x['attack_envelope_corr_mean'])} | "
                f"{ff(x['fraction_shifted_low_above_original_high_cut'])} |"
            )
    lines += [
        "",
        "This audit tests the compression transform only; it does not apply an Exact-K correction.",
        "Audio examples are exported for direct listening.",
    ]
    (a.output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
