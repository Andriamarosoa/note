"""Verified TensorFlow Lite runtime for the polyphonic bundle."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PolyphonicPrediction:
    frame_probability: np.ndarray
    onset_probability: np.ndarray
    harmonic_amplitude: np.ndarray
    harmonic_offset_cents: np.ndarray
    inference_ms: float


class PolyphonicBundle:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.metadata = json.loads(
            (self.root / "metadata.json").read_text(encoding="utf-8")
        )
        self.model_path = self.root / self.metadata["artifact"]["tflite"]
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        if _sha256(self.model_path) != self.metadata["artifact"]["sha256"]:
            raise ValueError("Invalid polyphonic model SHA256.")


class TFLitePolyphonicModel:
    def __init__(self, bundle: PolyphonicBundle, threads: int | None = None) -> None:
        import tensorflow as tf

        threads = int(
            bundle.metadata["recommended_tflite_threads"]
            if threads is None else threads
        )
        self.interpreter = tf.lite.Interpreter(
            model_path=str(bundle.model_path), num_threads=threads
        )
        self.interpreter.allocate_tensors()
        self.runner = self.interpreter.get_signature_runner("serving_default")
        self.gain = float(bundle.metadata["normalization_gain"])
        self.audio = np.zeros((1, 4096, 1), np.float32)
        self.mask = np.zeros((1, 4096), np.float32)

    def infer(self, waveform: np.ndarray, visible_window: int = 4096) -> PolyphonicPrediction:
        values = np.asarray(waveform, np.float32).reshape(-1)
        if values.shape != (4096,):
            raise ValueError("A mono 4096-sample window is required.")
        visible = int(np.clip(visible_window, 1, 4096))
        self.audio.fill(0.0)
        self.mask.fill(0.0)
        self.audio[0, -visible:, 0] = values[-visible:] * self.gain
        np.clip(self.audio, -1.0, 1.0, out=self.audio)
        self.mask[0, -visible:] = 1.0
        started = time.perf_counter()
        result = self.runner(audio=self.audio, time_mask=self.mask)
        elapsed = (time.perf_counter() - started) * 1000.0
        return PolyphonicPrediction(
            frame_probability=np.asarray(result["frame"], np.float32).reshape(-1),
            onset_probability=np.asarray(result["onset"], np.float32).reshape(-1),
            harmonic_amplitude=np.asarray(
                result["harmonic_amplitude"], np.float32
            ).reshape(37, 20),
            harmonic_offset_cents=np.asarray(
                result["harmonic_offset_cents"], np.float32
            ).reshape(37, 20),
            inference_ms=elapsed,
        )
