"""Streaming predictor adapter for the saved V8 causal stream model."""
import math
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

from .v8_runtime import V8ScoreChunk


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be an integer > 0")
    return value


def _finite_samples(samples: Iterable[float]) -> Tuple[float, ...]:
    result = []
    for sample in samples:
        if isinstance(sample, bool):
            raise ValueError("audio samples must be finite numbers")
        value = float(sample)
        if not math.isfinite(value):
            raise ValueError("audio samples must be finite numbers")
        result.append(value)
    return tuple(result)


def _to_list(value: Any):
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


class V8KerasPredictor:
    """Exact causal chunk inference using bounded receptive-field context.

    V8's gated residual topology is intentionally not forced through V7's
    specialized stateful Conv1D runner. This adapter first establishes exact
    correctness by recomputing at most receptive_field-1 past samples. A future
    optimized runner can replace it only after equality tests pass.
    """

    def __init__(self, model: Any, *, receptive_field: int = 4093) -> None:
        if not callable(model):
            raise ValueError("model must be callable")
        self._model = model
        self._receptive_field = _positive_integer("receptive_field", receptive_field)
        self._context: Tuple[float, ...] = ()
        self._next_sample: Optional[int] = None

    @classmethod
    def from_path(
        cls,
        path: str,
        *,
        receptive_field: int = 4093,
    ) -> "V8KerasPredictor":
        model_path = Path(path)
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        try:
            from tensorflow import keras
        except ImportError as exc:
            raise RuntimeError("TensorFlow is required to load a V8 model") from exc
        model = keras.models.load_model(str(model_path), compile=False)
        return cls(model, receptive_field=receptive_field)

    @property
    def receptive_field(self) -> int:
        return self._receptive_field

    @property
    def next_sample(self) -> Optional[int]:
        return self._next_sample

    def reset(self) -> None:
        self._context = ()
        self._next_sample = None

    def _batch(self, values: Tuple[float, ...]):
        try:
            import numpy as np
        except ImportError:
            return [[[sample] for sample in values]]
        return np.asarray(values, dtype=np.float32).reshape(1, -1, 1)

    def _outputs(self, values: Tuple[float, ...]):
        outputs = self._model(self._batch(values), training=False)
        if not isinstance(outputs, dict):
            raise ValueError("V8 stream model must return named outputs")
        required = (
            "onset_presence",
            "offset_presence",
            "onset_multiplicity",
            "offset_multiplicity",
        )
        missing = [name for name in required if name not in outputs]
        if missing:
            raise ValueError(f"V8 stream model missing outputs: {missing}")
        return {name: _to_list(outputs[name])[0] for name in required}

    def warm_up(self, chunk_size: int = 512) -> None:
        size = _positive_integer("chunk_size", chunk_size)
        outputs = self._outputs((0.0,) * size)
        if any(len(outputs[name]) != size for name in outputs):
            raise ValueError("V8 warm-up returned inconsistent temporal length")

    def predict_chunk(
        self,
        samples: Iterable[float],
        *,
        start_sample: int,
    ) -> V8ScoreChunk:
        values = _finite_samples(samples)
        if isinstance(start_sample, bool) or not isinstance(start_sample, int) or start_sample < 0:
            raise ValueError("start_sample must be an integer >= 0")
        if self._next_sample is not None and start_sample != self._next_sample:
            raise ValueError(
                f"expected contiguous chunk at {self._next_sample}, got {start_sample}"
            )
        if not values:
            self._next_sample = start_sample
            return V8ScoreChunk(start_sample, (), (), (), ())

        model_input = self._context + values
        outputs = self._outputs(model_input)
        count = len(values)
        for name in outputs:
            if len(outputs[name]) < count:
                raise ValueError(f"V8 output {name} is shorter than input chunk")

        onset_presence = tuple(float(row[0]) for row in outputs["onset_presence"][-count:])
        offset_presence = tuple(float(row[0]) for row in outputs["offset_presence"][-count:])
        onset_multiplicity = tuple(
            tuple(float(value) for value in row)
            for row in outputs["onset_multiplicity"][-count:]
        )
        offset_multiplicity = tuple(
            tuple(float(value) for value in row)
            for row in outputs["offset_multiplicity"][-count:]
        )
        scores = V8ScoreChunk(
            start_sample,
            onset_presence,
            offset_presence,
            onset_multiplicity,
            offset_multiplicity,
        )
        context_size = self._receptive_field - 1
        self._context = model_input[-context_size:] if context_size else ()
        self._next_sample = start_sample + count
        return scores


__all__ = ["V8KerasPredictor"]
