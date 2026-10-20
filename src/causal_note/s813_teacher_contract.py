"""S813: independent teachers first, distillation only after six proven >85%.

This gate checks declared evaluation evidence; it does not create that evidence
or certify a model from its score alone. Teacher K/pitch scores are direct
outputs, not scores of an already unified MIDI stream. The historical S704
score definitions and 80% reports are deliberately unchanged.

All acoustic, decision and evaluation hops must be genuinely 256 samples at
44100 Hz. Scoring held 1024-sample features on a 256 grid does not qualify.
This module permits an *unification experiment*, never a LIVE deployment.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Iterable, Mapping


SR = 44100
HOP_SAMPLES = 256
RESOLUTION_MS = 1000.0 * HOP_SAMPLES / SR
STRICT_THRESHOLD = 0.85
AXES = ("exact_k", "active_exact_k", "onset", "offset", "frame_pitch", "full_note")
REQUIRED_METRICS = {
    "exact_k": "teacher_exact_k_accuracy_midi256",
    "active_exact_k": "teacher_active_exact_k_accuracy_midi256",
    "onset": "teacher_onset_event_f1_midi256_pitch80ms",
    "offset": "teacher_offset_accuracy_midi256_fixed_onset_anchors_50ms",
    "frame_pitch": "teacher_frame_pitch_f1_midi256",
    "full_note": "teacher_full_note_event_f1_midi256_pitch80ms_offset50ms",
}


@dataclass(frozen=True)
class TeacherEvidence:
    axis: str
    score: float | None
    metric_name: str | None = None
    # Canonical identity of the evaluation cohort AND split, shared by all six
    # teachers. This is not the model/checkpoint identity.
    source_id: str | None = None
    # The individual evaluation/run identity may differ between teachers.
    evaluation_id: str | None = None
    score_scope: str = "independent_teacher"
    evaluation_role: str = "fit_research"
    sample_rate_hz: int = SR
    feature_hop_samples: int = HOP_SAMPLES
    decision_hop_samples: int = HOP_SAMPLES
    metric_hop_samples: int = HOP_SAMPLES
    generalization_verified: bool | None = None
    inference_uses_labels: bool | None = None
    private_parameters_verified: bool | None = None
    private_state_verified: bool | None = None
    causal_observations_verified: bool | None = None
    offset_anchors_fixed: bool | None = None


def _integer(value):
    return isinstance(value, Integral) and not isinstance(value, bool)


def validate_observation_timing(*, available_sample, decision_sample,
                                latest_audio_sample):
    """Reject future evidence/backdating; evidence may only be held forward.

    `available_sample` is when the observation actually became available, not
    a timestamp retrospectively attached to it. Decisions lie on MIDI256.
    This check alone does not establish fresh acoustic extraction every hop.
    """
    for name, value in (("available_sample", available_sample),
                        ("decision_sample", decision_sample),
                        ("latest_audio_sample", latest_audio_sample)):
        if not _integer(value) or value < 0:
            raise ValueError(f"invalid {name}")
    if decision_sample % HOP_SAMPLES:
        raise ValueError("decision is not on the MIDI256 grid")
    if available_sample > decision_sample:
        raise ValueError("future observation or backdated decision")
    if decision_sample > latest_audio_sample:
        raise ValueError("decision precedes receipt of its audio")
    return True


def assess_teacher(evidence: TeacherEvidence):
    """Return PASS / FAIL / UNKNOWN for one teacher; never change a score."""
    if not isinstance(evidence, TeacherEvidence):
        raise TypeError("TeacherEvidence required")
    if evidence.axis not in AXES:
        raise ValueError("unknown teacher axis")
    failures, missing = [], []
    score = evidence.score
    if score is None:
        missing.append("score_not_measured")
    elif (isinstance(score, bool) or not isinstance(score, Real)
          or not math.isfinite(score) or not 0 <= score <= 1):
        failures.append("invalid_score")
    elif not score > STRICT_THRESHOLD:
        failures.append("score_not_strictly_above_85_percent")

    for name in ("source_id", "evaluation_id", "metric_name"):
        value = getattr(evidence, name)
        if not isinstance(value, str) or not value.strip():
            missing.append(name + "_missing")
    if evidence.metric_name is not None and evidence.metric_name != REQUIRED_METRICS[evidence.axis]:
        failures.append("wrong_metric_or_proxy")
    if evidence.score_scope != "independent_teacher":
        failures.append("not_an_independent_teacher_score")
    if evidence.evaluation_role != "independent_validation":
        failures.append("independent_validation_required")

    for name, expected in (("sample_rate_hz", SR),
                           ("feature_hop_samples", HOP_SAMPLES),
                           ("decision_hop_samples", HOP_SAMPLES),
                           ("metric_hop_samples", HOP_SAMPLES)):
        value = getattr(evidence, name)
        if not _integer(value) or value != expected:
            failures.append(name + "_violates_midi256")

    flags = ("generalization_verified", "private_parameters_verified",
             "private_state_verified", "causal_observations_verified")
    if evidence.axis == "offset":
        flags += ("offset_anchors_fixed",)
    for name in flags:
        value = getattr(evidence, name)
        if value is None:
            missing.append(name + "_missing")
        elif value is not True:
            failures.append(name + "_failed")
    if evidence.inference_uses_labels is None:
        missing.append("inference_label_audit_missing")
    elif evidence.inference_uses_labels is not False:
        failures.append("inference_uses_labels")

    status = "FAIL" if failures else ("UNKNOWN" if missing else "PASS")
    return {"axis": evidence.axis, "status": status,
            "metric_name": evidence.metric_name, "score_scope": evidence.score_scope,
            "score": float(score) if isinstance(score, Real) and math.isfinite(score) else None,
            "source_id": evidence.source_id, "evaluation_id": evidence.evaluation_id,
            "failures": failures, "missing": missing}


def unification_gate(evidence: Iterable[TeacherEvidence], *,
                     cpu_evidence: Mapping | None = None):
    """Allow distillation research only after all six teachers qualify.

    CPU measurements do not affect teacher research eligibility. Even good CPU
    timings cannot establish capture-to-MIDI latency: `live_ready` stays false
    until a separate S793 paired hardware validation and deployment decision.
    Missing teachers remain UNKNOWN; duplicates and unknown axes are rejected.
    Six passing scores must describe the same evaluation cohort and split.
    """
    rows = {}
    for row in evidence:
        result = assess_teacher(row)
        if row.axis in rows:
            raise ValueError("duplicate teacher axis: " + row.axis)
        rows[row.axis] = result
    for axis in AXES:
        if axis not in rows:
            rows[axis] = {"axis": axis, "status": "UNKNOWN", "score": None,
                          "metric_name": REQUIRED_METRICS[axis],
                          "score_scope": "independent_teacher", "failures": [],
                          "missing": ["teacher_evaluation_missing"]}
    statuses = [rows[axis]["status"] for axis in AXES]
    six_pass = all(status == "PASS" for status in statuses)
    failures = []
    if six_pass and len({rows[axis]["source_id"] for axis in AXES}) != 1:
        failures.append("teacher_evaluation_cohort_or_split_mismatch")
    allowed = six_pass and not failures
    status = "PASS" if allowed else ("FAIL" if failures or "FAIL" in statuses else "UNKNOWN")
    return {"allow_unification": allowed, "status": status,
            "failures": failures,
            "strict_threshold": STRICT_THRESHOLD,
            "sample_rate_hz": SR, "hop_samples": HOP_SAMPLES,
            "resolution_ms": RESOLUTION_MS,
            "teachers": {axis: rows[axis] for axis in AXES},
            "unified_midi_evaluated": False,
            "cpu_evidence_supplied": cpu_evidence is not None,
            "live_ready": False,
            "live_reason": "separate_S793_paired_hardware_validation_required"}
