"""S817 recovery audit of *completed fold results* from interrupted CI.

The full S817 scientific run successfully fitted/scored all 80 recordings,
then raised NameError while serializing TeacherEvidence. This script is
strictly read-only: no fitting, no recalibration, no access to labels and
no new choice of a selector. It validates the archived fold checkpoints,
all output predictions, count totals, source checksums and reconstructs
a clearly labelled post-run audit, not a forged successful training run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rollup(rows):
    fields = ("correct", "frames", "undercount", "overcount")
    s = {k: sum(int(row[k]) for row in rows) for k in fields}
    if s["correct"] + s["undercount"] + s["overcount"] != s["frames"]:
        raise AssertionError("Active-K confusion accounting violated")
    s["accuracy"] = s["correct"] / max(1, s["frames"])
    return s


def recover(archive_root: Path):
    candidates = list(archive_root.rglob("report.json"))
    reports = []
    for path in candidates:
        try:
            v = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if v.get("schema") == "S817_REAL_FIT80_FIVE_PRIVATE_ACTIVEK_HEADS_MIDI256_V1":
            reports.append((path, v))
    if len(reports) != 1:
        raise ValueError("exactly one genuine S817 raw report required")
    path, r = reports[0]
    root = path.parent
    if (r.get("complete") is not False or r.get("source_MIDI_modified")
        or r.get("other_characteristics_modified")
        or r.get("production_LIVE_modified")
        or r.get("promotion") or r.get("CAL_opened")
        or r.get("HELD_opened") or r.get("TEST_opened")):
        raise AssertionError("S817 original job must remain incomplete/unpromoted")
    if (len(r["tracks"]) != 80 or len(r["folds"]) != 5
        or r["feature_hop_samples"] != 256
        or r["decision_hop_samples"] != 256
        or r["metric_hop_samples"] != 256):
        raise AssertionError("partial S817 research evidence is insufficient")
    tracks = r["tracks"]
    comps = ("BN1", "Funk2", "Jazz1", "Rock3", "SS1")
    if tuple(sorted(r["folds"])) != comps or len(set(x["index"] for x in tracks)) != 80:
        raise AssertionError("composition-fold or track identity mismatch")
    by_comp = {}
    changed = 0
    frames = 0
    for comp in comps:
        f = r["folds"][comp]
        model_path = root / f["checkpoint"]
        if sha(model_path) != f["checkpoint_sha256"]:
            raise AssertionError("S817 private model checkpoint SHA mismatch")
        comp_rows = [v for v in tracks if v["composition"] == comp]
        if len(comp_rows) != 16 or f["test_tracks"] != 16 or f["train_tracks"] != 48 or f["calibration_tracks"] != 16:
            raise AssertionError("composition-disjoint 48/16/16 split mismatch")
        if comp in f["model_metadata"]["train_compositions"] or comp in f["model_metadata"]["calibration_compositions"]:
            raise AssertionError("test composition leakage")
        by_comp[comp] = dict(
            new=rollup([v["new"] for v in comp_rows]),
            frozen_s815=rollup([v["frozen_s815"] for v in comp_rows]),
            chosen=f["model_metadata"]["selected"],
        )
    per_k = {str(k): rollup([v["by_true_k"][str(k)] for v in tracks]) for k in range(1, 7)}
    for rec in tracks:
        decisions = root / rec["decisions_file"]
        if sha(decisions) != rec["decisions_sha256"]:
            raise AssertionError("S817 decision file integrity mismatch")
        with np.load(decisions, allow_pickle=False) as z:
            prior = z["original_s815_k"]
            new = z["new_active_k"]
            ends = z["metric_ends"]
            if prior.shape != new.shape or len(new) != len(ends):
                raise AssertionError("decision output shape mismatch")
            if not np.all((new <= 6) & (new >= 0)):
                raise AssertionError("invalid cardinality K0..6")
            if not np.array_equal(ends, np.arange(1, len(ends) + 1) * 256):
                raise AssertionError("acoustic decision stream is not MIDI256")
            changed += int(np.count_nonzero(prior != new))
            frames += len(new)
    if frames != 306536:
        raise AssertionError("true FIT80 MIDI256 frame total drift")
    h0 = rollup([v["source"] for v in tracks])
    s815 = rollup([v["frozen_s815"] for v in tracks])
    s817 = rollup([v["new"] for v in tracks])
    if h0["correct"] != 142213 or s815["correct"] != 141946 or h0["frames"] != 249411:
        raise AssertionError("S815 and S708 verified original scores not reproduced")
    if s817["frames"] != 249411:
        raise AssertionError("Active-K evaluated denominator changed")
    if sum(per_k[str(k)]["correct"] for k in range(1, 7)) != s817["correct"]:
        raise AssertionError("per K1..K6 correctness differs from Active-K")
    if sum(per_k[str(k)]["overcount"] for k in range(1, 7)) != s817["overcount"]:
        raise AssertionError("per K1..K6 overcount differs from Active-K")
    if sum(per_k[str(k)]["undercount"] for k in range(1, 7)) != s817["undercount"]:
        raise AssertionError("per K1..K6 undercount differs from Active-K")
    return {
        "schema": "S817_RECOVERED_FROM_ARCHIVED_COMPLETED_FOLDS_V1",
        "original_failed_run_id": 39540692270,
        "original_run_status": "failure",
        "original_report_complete": False,
        "recovery_verified": True,
        "original_report_sha256": sha(path),
        "tracks": 80,
        "total_acoustic_frames": frames,
        "decision_differences_from_frozen_s815": changed,
        "historical_fit_active_k": h0,
        "frozen_s815_fit_active_k": s815,
        "recovered_s817_fit_active_k": s817,
        "delta_vs_s815_pp": 100 * (s817["accuracy"] - s815["accuracy"]),
        "delta_vs_h0_pp": 100 * (s817["accuracy"] - h0["accuracy"]),
        "by_comp": by_comp,
        "by_true_active_k": per_k,
        "new_independent_generalization": False,
        "fit80_previously_explored": True,
        "other_characteristics_changed": False,
        "live_changed": False,
        "promotion": False,
        "unification_allowed": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = recover(a.artifact_root)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print("S817_RECOVERY_COMPLETE", json.dumps({
        k: result[k] for k in ("recovered_s817_fit_active_k",
                              "frozen_s815_fit_active_k",
                              "historical_fit_active_k",
                              "delta_vs_s815_pp","delta_vs_h0_pp",
                              "decision_differences_from_frozen_s815")
    }), flush=True)
