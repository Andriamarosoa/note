"""Audit exclusive versus shared harmonic attack support on frozen K2/K3 cases.

Annotation-conditioned diagnostic only: no classifier fit, no threshold tuning,
no Exact-K action, and no fold-3 data.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np

from causal_note.guitarset import SAMPLE_RATE, index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts import audit_v273_internal_b_low_harmonic_strata as h
from scripts.audit_v273_internal_residual_acoustics import DATA_MD5, GROUPS, match_frequencies
from scripts.audit_v273_attack_novelty import raw_powers, harmonic_mask
from scripts.v273_residual_audit import FOLDS, require, sha256_file, write_json

PROTOCOL = Path("analysis/v273-exclusive-harmonic-support-protocol.md")
EPS = 1e-12


def attack_spectrum(powers):
    require(powers.shape[0] == 3, "power shape mismatch")
    return np.maximum(powers[1] - powers[0], 0.0)


def support_against_set(freq, attack, f0, peers):
    own = harmonic_mask(freq, f0)
    peer = np.zeros(len(freq), dtype=bool)
    for g in peers:
        peer |= harmonic_mask(freq, g)
    total = float(attack[own].sum())
    shared_mask = own & peer
    unique_mask = own & ~peer
    shared = float(attack[shared_mask].sum())
    unique = float(attack[unique_mask].sum())
    return {
        "f0": float(f0),
        "attack_energy_total": total,
        "attack_energy_shared": shared,
        "attack_energy_unique": unique,
        "shared_attack_fraction": shared / (total + EPS),
        "unique_attack_fraction": unique / (total + EPS),
        "harmonic_bins": int(own.sum()),
        "shared_bins": int(shared_mask.sum()),
        "unique_bins": int(unique_mask.sum()),
    }


def overlap_with_reference(freq, attack, f0, reference_f0):
    own = harmonic_mask(freq, f0)
    ref = np.zeros(len(freq), dtype=bool)
    for g in reference_f0:
        ref |= harmonic_mask(freq, g)
    total = float(attack[own].sum())
    overlap = float(attack[own & ref].sum())
    outside = float(attack[own & ~ref].sum())
    return {
        "overlap_with_expected_energy": overlap,
        "outside_expected_energy": outside,
        "overlap_with_expected_fraction": overlap / (total + EPS),
        "outside_expected_fraction": outside / (total + EPS),
    }


def measure_case(row, freq, powers):
    attack = attack_spectrum(powers)
    expected_f0 = np.asarray([n["frequency_hz"] for n in row["owned_notes"]], np.float64)
    selected_f0 = np.asarray(row["decomposition"]["triplet_f0"], np.float64)
    require(len(expected_f0) == row["true_K"], "expected count drift")
    require(len(selected_f0) == 3, "triplet schema drift")

    matched_count, matches = match_frequencies(expected_f0, selected_f0)
    selected_to_expected = {m["component"]: m["note"] for m in matches}

    expected = []
    for i, f0 in enumerate(expected_f0):
        peers = [g for j, g in enumerate(expected_f0) if j != i]
        item = support_against_set(freq, attack, f0, peers)
        item.update(role="expected", note_index=i)
        expected.append(item)

    selected = []
    for j, f0 in enumerate(selected_f0):
        peers = [g for q, g in enumerate(selected_f0) if q != j]
        item = support_against_set(freq, attack, f0, peers)
        matched = j in selected_to_expected
        item.update(role="selected_matched" if matched else "selected_unmatched",
                    component_index=j,
                    matched_note_index=selected_to_expected.get(j))
        if not matched:
            item.update(overlap_with_reference(freq, attack, f0, expected_f0))
        selected.append(item)

    unmatched = [x for x in selected if x["role"] == "selected_unmatched"]
    summary = {}
    for metric in ("unique_attack_fraction", "shared_attack_fraction", "attack_energy_total"):
        ev = np.asarray([x[metric] for x in expected], np.float64)
        summary["expected_" + metric + "_mean"] = float(ev.mean())
        if unmatched:
            uv = np.asarray([x[metric] for x in unmatched], np.float64)
            summary["unmatched_" + metric + "_mean"] = float(uv.mean())
            summary["expected_minus_unmatched_" + metric] = float(ev.mean() - uv.mean())
        else:
            summary["unmatched_" + metric + "_mean"] = None
            summary["expected_minus_unmatched_" + metric] = None
    if unmatched:
        for metric in ("overlap_with_expected_fraction", "outside_expected_fraction"):
            summary["unmatched_" + metric + "_mean"] = float(np.mean([x[metric] for x in unmatched]))
    else:
        summary["unmatched_overlap_with_expected_fraction_mean"] = None
        summary["unmatched_outside_expected_fraction_mean"] = None

    return {
        "row_id": row["row_id"],
        "fold": row["fold"],
        "group": row["group"],
        "true_K": row["true_K"],
        "recording_id": row["recording_id"],
        "start_sample": row["start_sample"],
        "matched_selected_count": matched_count,
        "expected": expected,
        "selected": selected,
        "summary": summary,
    }


def summary(values):
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


def aggregate(rows):
    roles = {}
    for role in ("expected", "selected_matched", "selected_unmatched"):
        items = [item for row in rows for item in row["expected"] + row["selected"]
                 if item["role"] == role]
        metrics = {}
        for metric in ("attack_energy_total", "attack_energy_shared", "attack_energy_unique",
                       "shared_attack_fraction", "unique_attack_fraction"):
            metrics[metric] = summary([x[metric] for x in items])
        if role == "selected_unmatched":
            for metric in ("overlap_with_expected_fraction", "outside_expected_fraction"):
                metrics[metric] = summary([x[metric] for x in items])
        roles[role] = {"components": len(items), "metrics": metrics}

    paired = {}
    for metric in ("unique_attack_fraction", "shared_attack_fraction", "attack_energy_total"):
        paired[metric] = summary([
            r["summary"]["expected_minus_unmatched_" + metric]
            for r in rows
            if r["summary"]["expected_minus_unmatched_" + metric] is not None
        ])
    unmatched_overlap = summary([
        r["summary"]["unmatched_overlap_with_expected_fraction_mean"]
        for r in rows
        if r["summary"]["unmatched_overlap_with_expected_fraction_mean"] is not None
    ])
    return {
        "rows": len(rows),
        "roles": roles,
        "expected_minus_unmatched": paired,
        "unmatched_overlap_with_expected": unmatched_overlap,
        "match_counts": {str(k): int(sum(r["matched_selected_count"] == k for r in rows)) for k in range(4)},
    }


def fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return str(v)
    return f"{float(v):.6f}"


def run(a):
    require(not a.output.exists(), "refusing overwrite")
    require(PROTOCOL.exists(), "missing preregistered protocol")
    rows = [json.loads(s) for s in a.cases.read_text().splitlines()]
    require(len(rows) == 488, "cohort changed")
    require(all(r["fold"] in FOLDS and r["fold"] != 3 for r in rows), "fold-3 leak")
    expected_counts = {"K3_regressed": 125, "K3_preserved": 147,
                       "K2_corrected": 108, "K2_missed": 108}
    require({g: sum(r["group"] == g for r in rows) for g in GROUPS} == expected_counts,
            "group counts changed")

    config = json.loads(a.config.read_text())
    require(all(config["member_folds"][r["recording_id"]] == r["fold"] for r in rows),
            "recording fold mismatch")
    for name, expected in DATA_MD5.items():
        with (a.dataset / name).open("rb") as stream:
            require(hashlib.file_digest(stream, "md5").hexdigest() == expected,
                    "dataset checksum changed: " + name)

    wanted = {r["recording_id"] for r in rows}
    tracks = {t.annotation_member: t for t in index_guitarset(a.dataset)
              if t.annotation_member in wanted}
    require(set(tracks) == wanted, "audio coverage incomplete")
    by_member = defaultdict(list)
    for row in rows:
        by_member[row["recording_id"]].append(row)

    results = []
    for member in sorted(by_member):
        require(config["member_folds"][member] in FOLDS, "forbidden recording")
        track = tracks[member]
        audio = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        samples = np.asarray(audio.samples, np.float64) / 32768.0
        for row in sorted(by_member[member], key=lambda q: q["row_id"]):
            freq, powers, _ = raw_powers(samples, int(row["start_sample"]))
            results.append(measure_case(row, freq, powers))
        print(json.dumps({"recording": member, "cases": len(results)}), flush=True)

    require(len(results) == 488, "incomplete audit")

    groups = {g: aggregate([r for r in results if r["group"] == g]) for g in GROUPS}
    groups["K3"] = aggregate([r for r in results if r["true_K"] == 3])
    groups["K2"] = aggregate([r for r in results if r["true_K"] == 2])
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
        "experiment": "v273_exclusive_harmonic_attack_support",
        "cases": len(results),
        "recordings": len(wanted),
        "folds": list(FOLDS),
        "outer_fold_3_used": False,
        "annotation_use": "diagnostic only",
        "prediction_changes": False,
        "classifier_training": False,
        "threshold_search": False,
        "normal_audio_only": True,
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
            "Harmonic masks can overlap between genuine physical sources.",
            "Annotation frequencies are used only to label diagnostic relationships.",
            "Previously inspected internal folds are not untouched final validation.",
            "Normal-audio path only; no compressed-path conclusion.",
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
        "# Exclusive harmonic attack support diagnostic",
        "",
        "No Exact-K prediction is changed.",
        "",
        "| group | rows | expected unique median | unmatched unique median | paired unique delta median | unmatched overlap-with-expected median |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for g in ("K3_regressed", "K3_preserved", "K2_corrected", "K2_missed", "K3", "K2"):
        q = groups[g]
        em = q["roles"]["expected"]["metrics"]["unique_attack_fraction"]
        um = q["roles"]["selected_unmatched"]["metrics"]["unique_attack_fraction"]
        pd = q["expected_minus_unmatched"]["unique_attack_fraction"]
        ov = q["roles"]["selected_unmatched"]["metrics"]["overlap_with_expected_fraction"]
        vals = [g, q["rows"],
                None if em is None else em["median"],
                None if um is None else um["median"],
                None if pd is None else pd["median"],
                None if ov is None else ov["median"]]
        lines.append("| " + " | ".join(fmt(v) for v in vals) + " |")
    lines += ["", "Fold 3 excluded. Diagnostic annotations are not inference inputs."]
    (a.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cases", "dataset", "config", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    run(p.parse_args())
