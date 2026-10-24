"""S839 hard gate: strict *training and inference* PCM256, not just 256 output clocks.

Legacy S708 inference every 256 was historically trained at 1024. It is NOT
eligible for the user's strict PCM256 record. This module cannot train or
certify a model on its own; it rejects incompatible provenance and validates
the training/inference tensors actually passed by a new trainer.

Zero future, zero lookahead, zero added hold, independent Onset weights.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Mapping, Any

import numpy as np

SR = 44100
HOP = 256
PERIOD_MS = 1000 * HOP / SR
CONTRACT = "NOTE_S839_NATIVE_TRAIN_AND_INFERENCE_PCM256_V1"
REQUIRED = ("source_pcm_rate_hz", "train_input_samples", "train_hop_samples",
            "inference_input_samples", "inference_hop_samples",
            "largest_direct_pcm_window_samples", "lookahead_samples",
            "added_wait_samples", "pretraining_source_hop_samples",
            "trained_from_random_initialization", "uses_frozen_1024_teacher",
            "target_axis", "provenance_contract")


class Native256Violation(ValueError):
    pass


def verify_native256(metadata: Mapping[str, Any], *,
                     live_input_samples: int | None = None,
                     model_tensor_samples: int | None = None) -> None:
    """Fail closed on unknown/absent provenance and every 1024-trained teacher.

    The metadata MUST be backed by new training instrumentation: it is not
    sufficient on its own as a proof that a model was trained correctly.
    """
    if not isinstance(metadata, Mapping):
        raise Native256Violation("no structured provenance: not native PCM256")
    absent = [key for key in REQUIRED if key not in metadata]
    if absent:
        raise Native256Violation("missing required native256 provenance: " +
                                 ", ".join(absent))
    expected = {
        "provenance_contract": CONTRACT,
        "source_pcm_rate_hz": SR,
        "train_input_samples": HOP,
        "train_hop_samples": HOP,
        "inference_input_samples": HOP,
        "inference_hop_samples": HOP,
        "largest_direct_pcm_window_samples": HOP,
        "lookahead_samples": 0,
        "added_wait_samples": 0,
        "pretraining_source_hop_samples": HOP,
        "trained_from_random_initialization": True,
        "uses_frozen_1024_teacher": False,
        "target_axis": "onset",
    }
    for key, value in expected.items():
        observed = metadata[key]
        if type(observed) is not type(value) or observed != value:
            raise Native256Violation(
                f"{key}: got {observed!r}, require {value!r}")
    for key, val in (("live_input_samples",live_input_samples),
                     ("model_tensor_samples",model_tensor_samples)):
        if val is not None and (type(val) is not int or val != HOP):
            raise Native256Violation(
                f"{key}: {val!r} incompatible with input block 256")


def provenance_from_observed_training(*, observed_train_shapes,
                                      actual_step_samples,
                                      pretrained: bool = False,
                                      teacher_training_hop_samples=None):
    """Create metadata only from real observed training mini-batch tensor shapes.

    A future trainer must call this at the end of training, not manufacture a
    constant metadata dictionary to 'upgrade' existing frozen S708 weights.
    """
    shapes = list(observed_train_shapes)
    if not shapes:
        raise Native256Violation("no actual training mini-batches observed")
    if type(actual_step_samples) is not int or actual_step_samples != HOP:
        raise Native256Violation("training hop must be exactly 256")
    if pretrained or teacher_training_hop_samples not in (None, HOP):
        raise Native256Violation("reused/pretrained historical checkpoint")
    for shape in shapes:
        if len(shape) != 3 or shape[1] != 1 or shape[2] != HOP or shape[0] < 1:
            raise Native256Violation(f"training tensor is not [batch,1,256]: {shape}")
    info = dict(provenance_contract=CONTRACT,
                source_pcm_rate_hz=SR,
                train_input_samples=HOP,
                train_hop_samples=HOP,
                inference_input_samples=HOP,
                inference_hop_samples=HOP,
                largest_direct_pcm_window_samples=HOP,
                lookahead_samples=0, added_wait_samples=0,
                pretraining_source_hop_samples=HOP,
                trained_from_random_initialization=True,
                uses_frozen_1024_teacher=False,
                target_axis="onset",
                observed_train_batches=len(shapes),
                observed_train_shapes_unique=sorted(set(tuple(s) for s in shapes)))
    verify_native256(info)
    return info


class Exact256Stream:
    """Enforce consecutive PCM256 blocks. No buffering/aggregation/lookahead.

    caller provides causal predict_onset_256(block); this never turns an old
    1024-trained network into a native model.
    """
    def __init__(self, predictor, provenance):
        verify_native256(provenance)
        if not callable(predictor):
            raise TypeError("predictor must be callable")
        self.predictor = predictor
        self.next_sample = 0

    def step(self, block, *, start_sample):
        samples = np.asarray(block)
        if samples.shape != (HOP,) or not np.isfinite(samples).all():
            raise Native256Violation("expected exactly 256 finite mono PCM samples")
        if type(start_sample) is not int or start_sample != self.next_sample:
            raise Native256Violation("missing/discontinuous/delayed PCM256 block")
        decision = self.predictor(samples)
        self.next_sample += HOP
        return self.next_sample, decision
