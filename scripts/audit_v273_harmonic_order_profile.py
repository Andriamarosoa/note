"""Diagnostic harmonic-order profile audit on frozen acoustic cases.

No inference model, no Exact-K action, no fold 3.
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
from sklearn.metrics import roc_auc_score

from causal_note.guitarset import SAMPLE_RATE, index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts import audit_v273_internal_b_low_harmonic_strata as h
from scripts.audit_v273_attack_novelty import raw_powers
from scripts.audit_v273_internal_residual_acoustics import DATA_MD5, GROUPS, match_frequencies
from scripts.v273_residual_audit import FOLDS, require, sha256_file, write_json

PROTOCOL = Path("analysis/v273-harmonic-order-profile-protocol.md")
EPS = 1e-12
ORDERS = tuple(range(1, h.MAX_HARMONICS + 1))
METRICS = (
    "global_fraction", "profile_fraction", "pre_fraction",
    "post1_fraction", "onset_ratio"
)


def order_metrics(freq, powers, f0):
    pre, post1 = powers[0], powers[1]
    attack = np.maximum(post1 - pre, 0.0)
    total_attack = float(attack.sum())
    total_pre = float(pre.sum())
    total_post1 = float(post1.sum())
    raw = []
    for order in ORDERS:
        center = order * float(f0)
        if center > freq[-1]:
            raw.append(None)
            continue
        mask = np.abs(freq - center) <= h.KERNEL_HZ
        e_attack = float(attack[mask].sum())
        e_pre = float(pre[mask].sum())
        e_post1 = float(post1[mask].sum())
        raw.append((e_attack, e_pre, e_post1))
    component_attack = sum(v[0] for v in raw if v is not None)
    out = []
    for order, value in zip(ORDERS, raw):
        if value is None:
            out.append({
                "order": order,
                "valid": False,
                "center_hz": order * float(f0),
            })
            continue
        e_attack, e_pre, e_post1 = value
        out.append({
            "order": order,
            "valid": True,
            "center_hz": order * float(f0),
            "attack_energy": e_attack,
            "pre_energy": e_pre,
            "post1_energy": e_post1,
            "global_fraction": e_attack / (total_attack + EPS),
            "profile_fraction": e_attack / (component_attack + EPS),
            "pre_fraction": e_pre / (total_pre + EPS),
            "post1_fraction": e_post1 / (total_post1 + EPS),
            "onset_ratio": math.log((e_post1 + EPS) / (e_pre + EPS)),
        })
    return out


def overlaps_expected(order_entry, expected_f0):
    if not order_entry.get("valid"):
        return False
    center = float(order_entry["center_hz"])
    for f0 in expected_f0:
        for order in ORDERS:
            hz = order * float(f0)
            if hz > h.MAX_ANALYSIS_HZ:
                break
            if abs(center - hz) <= h.KERNEL_HZ:
                return True
    return False


def measure_case(row, freq, powers):
    expected_f0 = np.asarray([n["frequency_hz"] for n in row["owned_notes"]], np.float64)
    selected_f0 = np.asarray(row["decomposition"]["triplet_f0"], np.float64)
    require(len(expected_f0) == row["true_K"], "expected count drift")
    require(len(selected_f0) == 3, "selected triplet drift")

    _, matches = match_frequencies(expected_f0, selected_f0)
    selected_to_expected = {m["component"]: m["note"] for m in matches}

    components = []
    for i, f0 in enumerate(expected_f0):
        components.append({
            "role": "expected",
            "index": i,
            "f0": float(f0),
            "orders": order_metrics(freq, powers, f0),
        })
    for j, f0 in enumerate(selected_f0):
        matched = j in selected_to_expected
        orders = order_metrics(freq, powers, f0)
        if not matched:
            for item in orders:
                item["overlaps_expected"] = overlaps_expected(item, expected_f0)
        components.append({
            "role": "selected_matched" if matched else "selected_unmatched",
            "index": j,
            "f0": float(f0),
            "orders": orders,
        })
    return {
        "row_id": row["row_id"],
        "fold": row["fold"],
        "group": row["group"],
        "true_K": row["true_K"],
        "recording_id": row["recording_id"],
        "start_sample": row["start_sample"],
        "components": components,
    }


def stat(values):
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], np.float64)
    if not len(x):
        return None
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "q25": float(np.quantile(x, .25)),
        "q75": float(np.quantile(x, .75)),
    }


def auc_expected_vs_unmatched(rows, metric, order):
    vals, labels = [], []
    for row in rows:
        for comp in row["components"]:
            if comp["role"] not in ("expected", "selected_unmatched"):
                continue
            item = comp["orders"][order - 1]
            if not item.get("valid"):
                continue
            vals.append(item[metric])
            labels.append(1 if comp["role"] == "expected" else 0)
    if len(set(labels)) < 2:
        return None
    auc = float(roc_auc_score(labels, vals))
    oriented = max(auc, 1.0 - auc)
    direction = "expected_higher" if auc >= .5 else "expected_lower"
    return {"raw_auc": auc, "oriented_auc": oriented, "direction": direction, "n": len(vals)}


def role_stats(rows, role, metric, order):
    values = []
    for row in rows:
        for comp in row["components"]:
            if comp["role"] != role:
                continue
            item = comp["orders"][order - 1]
            if item.get("valid"):
                values.append(item[metric])
    return stat(values)


def overlap_stats(rows, order):
    energies, flags = [], []
    for row in rows:
        for comp in row["components"]:
            if comp["role"] != "selected_unmatched":
                continue
            item = comp["orders"][order - 1]
            if not item.get("valid"):
                continue
            flags.append(bool(item.get("overlaps_expected", False)))
            energies.append((float(item["attack_energy"]), bool(item.get("overlaps_expected", False))))
    if not flags:
        return None
    total_energy = sum(v for v, _ in energies)
    overlap_energy = sum(v for v, flag in energies if flag)
    return {
        "n": len(flags),
        "count_overlap_rate": float(np.mean(flags)),
        "energy_overlap_fraction": overlap_energy / (total_energy + EPS),
    }


def summarize(rows):
    report = {"rows": len(rows), "orders": {}}
    for order in ORDERS:
        by_metric = {}
        for metric in METRICS:
            e = role_stats(rows, "expected", metric, order)
            u = role_stats(rows, "selected_unmatched", metric, order)
            by_metric[metric] = {
                "expected": e,
                "selected_unmatched": u,
                "median_delta_expected_minus_unmatched":
                    None if e is None or u is None else e["median"] - u["median"],
                "auc": auc_expected_vs_unmatched(rows, metric, order),
            }
        report["orders"][str(order)] = {
            "metrics": by_metric,
            "unmatched_overlap_with_expected": overlap_stats(rows, order),
        }
    return report


def run(a):
    require(not a.output.exists(), "refusing overwrite")
    require(PROTOCOL.exists(), "missing preregistered protocol")
    cases = [json.loads(line) for line in a.cases.read_text().splitlines()]
    require(len(cases) == 488, "cohort changed")
    require(all(r["fold"] in FOLDS and r["fold"] != 3 for r in cases), "fold-3 leak")
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

    measured = []
    for member in sorted(by_member):
        track = tracks[member]
        audio = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        samples = np.asarray(audio.samples, np.float64) / 32768.0
        for row in sorted(by_member[member], key=lambda q: q["row_id"]):
            freq, powers, _ = raw_powers(samples, int(row["start_sample"]))
            measured.append(measure_case(row, freq, powers))
        print(json.dumps({"recording": member, "cases": len(measured)}), flush=True)

    require(len(measured) == 488, "incomplete measurement")
    groups = {
        "K3": summarize([r for r in measured if r["true_K"] == 3]),
        "K2": summarize([r for r in measured if r["true_K"] == 2]),
        "K3_regressed": summarize([r for r in measured if r["group"] == "K3_regressed"]),
        "K3_preserved": summarize([r for r in measured if r["group"] == "K3_preserved"]),
        "K2_corrected": summarize([r for r in measured if r["group"] == "K2_corrected"]),
        "K2_missed": summarize([r for r in measured if r["group"] == "K2_missed"]),
    }
    by_fold = {
        str(f): {
            "K3": summarize([r for r in measured if r["fold"] == f and r["true_K"] == 3]),
            "K3_regressed": summarize([r for r in measured if r["fold"] == f and r["group"] == "K3_regressed"]),
            "K3_preserved": summarize([r for r in measured if r["fold"] == f and r["group"] == "K3_preserved"]),
        }
        for f in FOLDS
    }

    out = {
        "status": "completed",
        "experiment": "v273_harmonic_order_profile",
        "cases": len(measured),
        "recordings": len(wanted),
        "folds": list(FOLDS),
        "outer_fold_3_used": False,
        "prediction_changes": False,
        "classifier_training": False,
        "annotation_use": "diagnostic only",
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
            "Per-order bands can overlap across true physical sources.",
            "Expected/unmatched labels require annotations and are diagnostic only.",
            "Internal folds have already been inspected.",
            "Normal-audio path only.",
        ],
    }

    a.output.mkdir(parents=True)
    (a.output / "components.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in measured),
        encoding="utf-8",
    )
    out["components_sha256"] = sha256_file(a.output / "components.jsonl")
    write_json(a.output / "report.json", out)

    lines = [
        "# Harmonic-order profile diagnostic",
        "",
        "K3 expected vs selected-unmatched, profile_fraction:",
        "",
        "| h | expected median | unmatched median | delta | oriented AUC | direction | overlap rate |",
        "|---:|---:|---:|---:|---:|---|---:|",
    ]
    k3 = groups["K3"]
    for order in ORDERS:
        q = k3["orders"][str(order)]
        m = q["metrics"]["profile_fraction"]
        e = m["expected"]; u = m["selected_unmatched"]; auc = m["auc"]
        ov = q["unmatched_overlap_with_expected"]
        lines.append(
            f"| {order} | "
            f"{'n/a' if e is None else f'{e['median']:.6f}'} | "
            f"{'n/a' if u is None else f'{u['median']:.6f}'} | "
            f"{'n/a' if m['median_delta_expected_minus_unmatched'] is None else f'{m['median_delta_expected_minus_unmatched']:.6f}'} | "
            f"{'n/a' if auc is None else f'{auc['oriented_auc']:.3f}'} | "
            f"{'n/a' if auc is None else auc['direction']} | "
            f"{'n/a' if ov is None else f'{ov['count_overlap_rate']:.3f}'} |"
        )
    lines += ["", "Fold 3 excluded. No Exact-K changes. Diagnostic only."]
    (a.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cases", "dataset", "config", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    run(p.parse_args())
