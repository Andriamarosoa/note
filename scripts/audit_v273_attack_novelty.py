"""Diagnose new attacks versus components already present before the event.

This is annotation-conditioned diagnosis only. It does not fit a classifier,
change an Exact-K decision, tune a threshold, or read fold 3.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform

import numpy as np

from causal_note.guitarset import SAMPLE_RATE, index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts import audit_v273_internal_b_low_harmonic_strata as h
from scripts.audit_v273_internal_residual_acoustics import (
    DATA_MD5, GROUPS, match_frequencies,
)
from scripts.v273_residual_audit import FOLDS, require, sha256_file, write_json

PROTOCOL = Path("analysis/v273-attack-novelty-protocol.md")
EPS = 1e-12
METRICS = (
    "pre_raw", "post1_raw", "post2_raw",
    "pre_norm", "post1_norm", "post2_norm",
    "gain_raw", "gain_norm",
    "onset_contrast_raw", "onset_contrast_norm",
    "log_ratio_raw", "late_retention_raw",
    "positive_support", "positive_retained_fraction",
)


def raw_powers(samples, start):
    taper = np.hanning(h.WINDOW)
    all_freq = np.fft.rfftfreq(h.FFT_SIZE, 1 / SAMPLE_RATE)
    keep = (all_freq >= h.MIN_HZ) & (all_freq <= h.MAX_ANALYSIS_HZ)
    powers = []
    padding = []
    for offset in (-h.WINDOW, 0, h.WINDOW):
        wave = h.pcm_window(samples, start + offset, h.WINDOW) * taper
        powers.append((np.abs(np.fft.rfft(wave, n=h.FFT_SIZE)) ** 2)[keep])
        left = max(0, -(start + offset))
        right = max(0, start + offset + h.WINDOW - len(samples))
        padding.append(min(h.WINDOW, left + right))
    return all_freq[keep], np.asarray(powers), np.asarray(padding, dtype=np.int64)


def harmonic_mask(freq, f0):
    mask = np.zeros(len(freq), dtype=bool)
    for harmonic in range(1, h.MAX_HARMONICS + 1):
        center = harmonic * float(f0)
        if center > h.MAX_ANALYSIS_HZ:
            break
        mask |= np.abs(freq - center) <= h.KERNEL_HZ
    return mask


def component_metrics(freq, powers, f0):
    require(powers.shape == (3, len(freq)), "power shape mismatch")
    mask = harmonic_mask(freq, f0)
    require(mask.any(), "empty harmonic support")
    support = powers[:, mask].sum(axis=1)
    total = powers.sum(axis=1)
    norm = support / (total + EPS)
    positive = np.maximum(powers[1] - powers[0], 0.0)
    positive_support = float(positive[mask].sum())
    pre, post1, post2 = map(float, support)
    pren, post1n, post2n = map(float, norm)
    return {
        "f0": float(f0),
        "pre_raw": pre,
        "post1_raw": post1,
        "post2_raw": post2,
        "pre_norm": pren,
        "post1_norm": post1n,
        "post2_norm": post2n,
        "gain_raw": post1 - pre,
        "gain_norm": post1n - pren,
        "onset_contrast_raw": (post1 - pre) / (post1 + pre + EPS),
        "onset_contrast_norm": (post1n - pren) / (post1n + pren + EPS),
        "log_ratio_raw": math.log((post1 + EPS) / (pre + EPS)),
        "late_retention_raw": post2 / (post1 + EPS),
        "positive_support": positive_support,
        "positive_retained_fraction": positive_support / (post1 + EPS),
    }


def case_measurement(row, freq, powers, padding):
    expected_f0 = np.asarray([n["frequency_hz"] for n in row["owned_notes"]], np.float64)
    selected_f0 = np.asarray(row["decomposition"]["triplet_f0"], np.float64)
    require(len(expected_f0) == row["true_K"] and len(selected_f0) == 3, "case schema drift")
    count, matches = match_frequencies(expected_f0, selected_f0)
    expected_to_selected = {m["note"]: m["component"] for m in matches}
    selected_to_expected = {m["component"]: m["note"] for m in matches}

    expected = []
    for i, f0 in enumerate(expected_f0):
        item = component_metrics(freq, powers, f0)
        item.update(role="expected", note_index=i,
                    matched_selected_index=expected_to_selected.get(i))
        expected.append(item)

    selected = []
    for j, f0 in enumerate(selected_f0):
        item = component_metrics(freq, powers, f0)
        matched = j in selected_to_expected
        item.update(role="selected_matched" if matched else "selected_unmatched",
                    component_index=j,
                    matched_note_index=selected_to_expected.get(j))
        selected.append(item)

    unmatched = [v for v in selected if v["role"] == "selected_unmatched"]
    summary = {}
    for metric in ("pre_norm", "onset_contrast_raw", "onset_contrast_norm",
                   "log_ratio_raw", "late_retention_raw", "positive_retained_fraction"):
        ev = np.asarray([v[metric] for v in expected], np.float64)
        summary["expected_" + metric + "_mean"] = float(ev.mean())
        summary["expected_" + metric + "_min"] = float(ev.min())
        if unmatched:
            uv = np.asarray([v[metric] for v in unmatched], np.float64)
            summary["unmatched_" + metric + "_mean"] = float(uv.mean())
            summary["expected_minus_unmatched_" + metric] = float(ev.mean() - uv.mean())
        else:
            summary["unmatched_" + metric + "_mean"] = None
            summary["expected_minus_unmatched_" + metric] = None

    return {
        "row_id": row["row_id"],
        "fold": row["fold"],
        "group": row["group"],
        "true_K": row["true_K"],
        "recording_id": row["recording_id"],
        "start_sample": row["start_sample"],
        "matched_selected_count": count,
        "padding_samples": padding.tolist(),
        "expected": expected,
        "selected": selected,
        "summary": summary,
    }


def numeric_summary(values):
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], np.float64)
    if not len(x):
        return None
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "q25": float(np.quantile(x, .25)),
        "q75": float(np.quantile(x, .75)),
        "positive": int(np.sum(x > 0)),
        "negative": int(np.sum(x < 0)),
        "zero": int(np.sum(x == 0)),
    }


def component_role_summary(rows, role):
    values = defaultdict(list)
    count = 0
    for row in rows:
        for item in row["expected"] + row["selected"]:
            if item["role"] != role:
                continue
            count += 1
            for metric in METRICS:
                values[metric].append(item[metric])
    return {"components": count, "metrics": {k: numeric_summary(v) for k, v in values.items()}}


def aggregate(rows):
    out = {"rows": len(rows)}
    for role in ("expected", "selected_matched", "selected_unmatched"):
        out[role] = component_role_summary(rows, role)
    paired = defaultdict(list)
    for row in rows:
        for metric in ("pre_norm", "onset_contrast_raw", "onset_contrast_norm",
                       "log_ratio_raw", "late_retention_raw", "positive_retained_fraction"):
            value = row["summary"]["expected_minus_unmatched_" + metric]
            if value is not None:
                paired[metric].append(value)
    out["expected_minus_unmatched"] = {k: numeric_summary(v) for k, v in paired.items()}
    out["selected_match_counts"] = {
        str(k): sum(r["matched_selected_count"] == k for r in rows)
        for k in range(4)
    }
    return out


def format_cell(value):
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(value)
    return f"{float(value):.6f}"


def run(a):
    require(not a.output.exists(), "refusing overwrite")
    require(PROTOCOL.exists(), "missing preregistered protocol")
    cases = [json.loads(line) for line in a.cases.read_text().splitlines()]
    require(len(cases) == 488, "cohort changed")
    require(all(r["fold"] in FOLDS and r["fold"] != 3 for r in cases), "outer fold leak")
    require({r["group"] for r in cases} == set(GROUPS), "group schema changed")
    expected_counts = {"K3_regressed": 125, "K3_preserved": 147,
                       "K2_corrected": 108, "K2_missed": 108}
    require({g: sum(r["group"] == g for r in cases) for g in GROUPS} == expected_counts,
            "group counts changed")

    config = json.loads(a.config.read_text())
    require(all(config["member_folds"][r["recording_id"]] == r["fold"] for r in cases),
            "recording fold mismatch")

    for name, expected in DATA_MD5.items():
        with (a.dataset / name).open("rb") as stream:
            require(hashlib.file_digest(stream, "md5").hexdigest() == expected,
                    "dataset checksum changed: " + name)

    wanted = {r["recording_id"] for r in cases}
    tracks = {t.annotation_member: t for t in index_guitarset(a.dataset)
              if t.annotation_member in wanted}
    require(set(tracks) == wanted, "audio coverage incomplete")

    by_member = defaultdict(list)
    for row in cases:
        by_member[row["recording_id"]].append(row)

    results = []
    padded = np.zeros(3, dtype=np.int64)
    for member in sorted(by_member):
        require(config["member_folds"][member] in FOLDS, "forbidden recording")
        track = tracks[member]
        audio = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        samples = np.asarray(audio.samples, np.float64) / 32768.0
        for row in sorted(by_member[member], key=lambda q: q["row_id"]):
            freq, powers, padding = raw_powers(samples, int(row["start_sample"]))
            result = case_measurement(row, freq, powers, padding)
            results.append(result)
            padded += padding > 0
        print(json.dumps({"recording": member, "cases": len(results)}), flush=True)

    require(len(results) == 488, "incomplete result")

    groups = {g: aggregate([r for r in results if r["group"] == g]) for g in GROUPS}
    groups["K3"] = aggregate([r for r in results if r["true_K"] == 3])
    groups["K2"] = aggregate([r for r in results if r["true_K"] == 2])
    groups["all"] = aggregate(results)

    by_fold = {
        str(f): {
            "K3": aggregate([r for r in results if r["fold"] == f and r["true_K"] == 3]),
            "K2": aggregate([r for r in results if r["fold"] == f and r["true_K"] == 2]),
            "K3_regressed": aggregate([r for r in results if r["fold"] == f and r["group"] == "K3_regressed"]),
            "K3_preserved": aggregate([r for r in results if r["fold"] == f and r["group"] == "K3_preserved"]),
        }
        for f in FOLDS
    }

    report = {
        "status": "completed",
        "experiment": "v273_attack_novelty_diagnostic",
        "cases": len(results),
        "recordings": len(wanted),
        "folds": list(FOLDS),
        "outer_fold_3_used": False,
        "annotation_use": "diagnostic only; never an inference feature",
        "prediction_changes": False,
        "classifier_training": False,
        "threshold_search": False,
        "normal_audio_only": True,
        "window_samples": h.WINDOW,
        "window_ms": h.WINDOW * 1000 / SAMPLE_RATE,
        "padded_windows": {"pre": int(padded[0]), "post1": int(padded[1]), "post2": int(padded[2])},
        "groups": groups,
        "by_fold": by_fold,
        "source_sha256": {
            "cases": sha256_file(a.cases),
            "config": sha256_file(a.config),
            "protocol": sha256_file(PROTOCOL),
            "script": sha256_file(__file__),
        },
        "source_md5": DATA_MD5,
        "runtime": {
            "python": platform.python_version(),
            **{p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn")},
        },
        "limitations": [
            "Harmonic bands can overlap between sources and do not isolate physical notes.",
            "Selected-unmatched components are labels from annotation-conditioned diagnosis only.",
            "These internal folds were already inspected and are not an untouched validation set.",
            "No compressed-path conclusion is supported.",
        ],
    }

    a.output.mkdir(parents=True)
    (a.output / "cases.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in results),
        encoding="utf-8",
    )
    report["cases_sha256"] = sha256_file(a.output / "cases.jsonl")
    write_json(a.output / "report.json", report)

    lines = [
        "# Attack novelty diagnostic",
        "",
        "No Exact-K decision is changed in this audit.",
        "",
        "| group | rows | expected pre norm median | expected onset contrast median | unmatched onset contrast median | paired contrast delta median |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for g in ("K3_regressed", "K3_preserved", "K2_corrected", "K2_missed", "K3", "K2"):
        q = groups[g]
        def med(role, metric):
            v = q[role]["metrics"].get(metric)
            return None if v is None else v["median"]
        pd = q["expected_minus_unmatched"].get("onset_contrast_raw")
        vals = [
            g, q["rows"],
            med("expected", "pre_norm"),
            med("expected", "onset_contrast_raw"),
            med("selected_unmatched", "onset_contrast_raw"),
            None if pd is None else pd["median"],
        ]
        lines.append("| " + " | ".join(format_cell(v) for v in vals) + " |")
    lines += ["", "Fold 3 excluded. Normal-audio path only. Diagnostic annotations are not inference inputs."]
    (a.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cases", "dataset", "config", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    run(p.parse_args())
