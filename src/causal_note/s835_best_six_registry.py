"""S835: inspect and protect six independently verified best-known axis references.

Registry and non-regression gates ONLY. Never merges trained heads, starts
unavailable checkpoints, changes the MIDI publisher, or fabricates native256
Onset/Offset evidence. Every score retains its metric, cohort and clock.
"""
from __future__ import annotations

import json
from pathlib import Path

SCHEMA = "S835_BEST_VERIFIED_INDEPENDENT_AXIS_REFERENCES_V1"
AXES = ("exact_k", "active_exact_k", "onset", "offset",
        "frame_pitch", "full_note")
ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs" / "best_six_axes_preserved.json"


def load_registry(path: Path = MANIFEST, *, validate: bool = True):
    with Path(path).open(encoding="utf-8") as file:
        record = json.load(file)
    if validate:
        validate_registry(record, root=ROOT)
    return record


def validate_registry(registry, *, root: Path = ROOT):
    if registry.get("schema") != SCHEMA:
        raise ValueError("unexpected best-six registry schema")
    if set(registry.get("axes", {})) != set(AXES):
        raise ValueError("all six axes must be present exactly once")
    policy = registry["policy"]
    if (policy["hop_samples"] != 256 or policy["sampling_hz"] != 44100 or
        policy["added_lookahead_samples"] != 0 or
        policy["added_hold_samples"] != 0 or
        policy["public_midi_modified"] or policy["merged_midi_enabled"] or
        policy["runtime_native256_complete"]):
        raise ValueError("live/no-regression policy cannot be bypassed")
    if registry["cohorts"]["FIT80"]["native_frames"] != 306536:
        raise ValueError("source scoring cohort drift")
    seen = set()
    for axis, record in registry["axes"].items():
        if axis in seen:
            raise ValueError("duplicate axis")
        seen.add(axis)
        selected = record["chosen"]
        mode = selected.get("mode")
        if mode not in ("strict_256_shadow_candidate", "protected_legacy_keep"):
            raise ValueError("unknown independent axis reference mode")
        score = float(selected["score"])
        if not 0 <= score <= 1:
            raise ValueError("invalid score")
        if selected["cohort"] != "FIT80":
            raise ValueError("do not mix CAL32 and FIT80 winning comparisons")
        if selected["ready_for_live"]:
            raise ValueError("candidate promoted without MIDI/runtime certification")
        if not selected.get("code_paths"):
            raise ValueError("candidate lacks referenced implementation")
        for filename in selected["code_paths"]:
            if (filename.startswith("/") or ".." in Path(filename).parts or
                not (root / filename).is_file()):
                raise ValueError(f"missing/unsafe source {filename}")
        prior = record.get("same_cohort_h0")
        if mode == "strict_256_shadow_candidate":
            if selected["hop_samples"] != 256 or prior is None:
                raise ValueError("selected true-native source not verified")
            if (prior["cohort"] != selected["cohort"] or
                prior["metric"] != selected["metric"] or
                score < float(prior["score"])):
                raise ValueError("regressive or incomparable candidate")
        else:
            if selected["series"] != "S708/H0" or axis not in ("onset", "offset"):
                raise ValueError("protected legacy replacement prohibited")
            if selected["hop_samples"] == 256:
                raise ValueError("do not invent native256 protected H0")
        historical = record.get("historical_record_other_grid")
        if historical and historical["hop_samples"] == 256:
            raise ValueError("old-grid record falsely advertised native256")
    for filename in registry["evidence"]["source_reports"]:
        if filename.startswith("/") or not (root / filename).is_file():
            raise ValueError(f"referenced scientific report not found {filename}")
    return True


def six_native256_ready(registry):
    """False until a non-regressive 256-only Onset and Offset are validated."""
    return all(
        registry["axes"][axis]["chosen"]["mode"] == "strict_256_shadow_candidate"
        and registry["axes"][axis]["chosen"]["ready_for_live"]
        for axis in AXES
    )


def assert_single_axis_noninterference(before: dict, after: dict, changed_axis: str):
    """Hard gate for any later mutation of a private head/selector/decoder.

    Compare output *and* selector state. This validation cannot substitute
    for paired musical metrics, which require genuine separately scored GT.
    """
    if changed_axis not in AXES:
        raise ValueError("unknown axis")
    if set(before) != set(AXES) or set(after) != set(AXES):
        raise ValueError("missing independent axis")
    for axis in AXES:
        if axis != changed_axis and before[axis] != after[axis]:
            raise AssertionError(f"{changed_axis} changed protected {axis}")
    return True


def assert_paired_no_regression(before: dict, after: dict, *, tolerance: float = 0.):
    """Six scores on exact same metric, cohort and frame set are required."""
    if set(before) != set(AXES) or set(after) != set(AXES):
        raise ValueError("six independently evaluated axes required")
    for axis in AXES:
        original, trial = before[axis], after[axis]
        required = ("metric", "cohort", "frame_count", "score")
        if any(k not in original or k not in trial for k in required):
            raise ValueError("comparison incomplete")
        if any(original[k] != trial[k] for k in required[:3]):
            raise ValueError(f"mismatched paired scoring for {axis}")
        if float(trial["score"]) + tolerance < float(original["score"]):
            raise AssertionError(f"musical regression in {axis}")
    return True
