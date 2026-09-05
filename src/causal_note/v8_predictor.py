"""Streaming predictor adapter for the saved V8 causal stream model."""
import math
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

from .v8_runtime import V8ScoreChunk


_REQUIRED_OUTPUTS = (
    "onset_presence",
    "offset_presence",
    "onset_multiplicity",
    "offset_multiplicity",
)


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


def _causal_cache_length(layer: Any) -> int:
    if (
        getattr(layer, "padding", None) != "causal"
        or tuple(getattr(layer, "strides", ())) != (1,)
    ):
        raise ValueError(f"unsupported non-causal or strided layer {layer.name}")
    return (int(layer.kernel_size[0]) - 1) * int(layer.dilation_rate[0])


class _V8StatefulCausalRunner:
    """Exact streaming runner for V8's transient + gated residual topology."""

    def __init__(self, model: Any, tf: Any) -> None:
        self._tf = tf
        self._transient = model.get_layer("v8_transient_conv")
        self._input_projection = model.get_layer("v8_input_projection")
        self._fusion_projection = model.get_layer("v8_fusion_projection")
        self._heads = {
            "onset_presence": model.get_layer("onset_presence_sequence"),
            "offset_presence": model.get_layer("offset_presence_sequence"),
            "onset_multiplicity": model.get_layer("onset_multiplicity_sequence"),
            "offset_multiplicity": model.get_layer("offset_multiplicity_sequence"),
        }

        feature_layers = {
            int(layer.name.rsplit("_", 1)[1]): layer
            for layer in model.layers
            if layer.name.startswith("v8_feature_")
        }
        if not feature_layers:
            raise ValueError("V8 model has no residual feature blocks")
        indices = tuple(sorted(feature_layers))
        if indices != tuple(range(1, len(indices) + 1)):
            raise ValueError("V8 residual block indices must be contiguous")

        blocks = []
        for index in indices:
            feature = feature_layers[index]
            gate = model.get_layer(f"v8_gate_{index}")
            mix = model.get_layer(f"v8_mix_{index}")
            if _causal_cache_length(feature) != _causal_cache_length(gate):
                raise ValueError("V8 feature/gate cache lengths differ")
            if tuple(feature.kernel_size) != tuple(gate.kernel_size):
                raise ValueError("V8 feature/gate kernels differ")
            if tuple(feature.dilation_rate) != tuple(gate.dilation_rate):
                raise ValueError("V8 feature/gate dilations differ")
            blocks.append((feature, gate, mix))
        self._blocks = tuple(blocks)

        cache_specs = []
        initial_caches = []

        transient_cache = _causal_cache_length(self._transient)
        transient_channels = int(self._transient.kernel.shape[1])
        transient_spec = tf.TensorSpec(
            (1, transient_cache, transient_channels), tf.float32
        )
        cache_specs.append(transient_spec)
        initial_caches.append(tf.zeros(transient_spec.shape, dtype=tf.float32))

        for feature, _gate, _mix in self._blocks:
            cache_length = _causal_cache_length(feature)
            input_channels = int(feature.kernel.shape[1])
            spec = tf.TensorSpec((1, cache_length, input_channels), tf.float32)
            cache_specs.append(spec)
            initial_caches.append(tf.zeros(spec.shape, dtype=tf.float32))

        self._initial_caches = tuple(initial_caches)
        self._caches = self._initial_caches
        transient = self._transient
        input_projection = self._input_projection
        fusion_projection = self._fusion_projection
        blocks = self._blocks
        heads = self._heads

        @tf.function(
            input_signature=[tf.TensorSpec((1, None, 1), tf.float32)] + cache_specs,
            reduce_retracing=True,
        )
        def infer(chunk, *caches):
            chunk_length = tf.shape(chunk)[1]

            transient_input = tf.concat((caches[0], chunk), axis=1)
            transient_full = transient(transient_input, training=False)
            transient_hidden = transient_full[:, -chunk_length:, :]
            transient_cache_length = int(caches[0].shape[1])
            new_caches = [
                transient_input[:, -transient_cache_length:, :]
                if transient_cache_length
                else transient_input[:, :0, :]
            ]

            hidden = input_projection(chunk, training=False)
            for (feature, gate, mix), cache in zip(blocks, caches[1:]):
                layer_input = tf.concat((cache, hidden), axis=1)
                feature_full = feature(layer_input, training=False)
                gate_full = gate(layer_input, training=False)
                current_length = tf.shape(hidden)[1]
                gated = (
                    feature_full[:, -current_length:, :]
                    * gate_full[:, -current_length:, :]
                )
                hidden = hidden + mix(gated, training=False)
                cache_length = int(cache.shape[1])
                new_caches.append(
                    layer_input[:, -cache_length:, :]
                    if cache_length
                    else layer_input[:, :0, :]
                )

            fused = tf.concat((hidden, transient_hidden), axis=-1)
            fused = fusion_projection(fused, training=False)
            outputs = {
                name: head(fused, training=False)
                for name, head in heads.items()
            }
            return outputs, tuple(new_caches)

        self._infer = infer

    def predict(self, batch: Any):
        return self._infer(batch, *self._caches)

    def commit(self, caches: Tuple[Any, ...]) -> None:
        self._caches = tuple(caches)

    def reset(self) -> None:
        self._caches = self._initial_caches


def _stateful_runner(model: Any):
    try:
        import tensorflow as tf
    except ImportError:
        return None
    if not isinstance(model, tf.keras.Model):
        return None
    try:
        return _V8StatefulCausalRunner(model, tf)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


class V8KerasPredictor:
    """Exact causal chunk inference with an optimized stateful fast path.

    The fallback recomputes at most ``receptive_field - 1`` past samples and is
    retained as a correctness oracle. TensorFlow V8 models use a per-layer
    cache runner only when the topology is recognized exactly.
    """

    def __init__(
        self,
        model: Any,
        *,
        receptive_field: int = 4093,
        use_stateful: bool = True,
    ) -> None:
        if not callable(model):
            raise ValueError("model must be callable")
        self._model = model
        self._receptive_field = _positive_integer("receptive_field", receptive_field)
        self._stateful = _stateful_runner(model) if use_stateful else None
        self._context: Tuple[float, ...] = ()
        self._next_sample: Optional[int] = None

    @classmethod
    def from_path(
        cls,
        path: str,
        *,
        receptive_field: int = 4093,
        use_stateful: bool = True,
    ) -> "V8KerasPredictor":
        model_path = Path(path)
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        try:
            from tensorflow import keras
        except ImportError as exc:
            raise RuntimeError("TensorFlow is required to load a V8 model") from exc
        model = keras.models.load_model(str(model_path), compile=False)
        return cls(
            model,
            receptive_field=receptive_field,
            use_stateful=use_stateful,
        )

    @property
    def receptive_field(self) -> int:
        return self._receptive_field

    @property
    def next_sample(self) -> Optional[int]:
        return self._next_sample

    @property
    def stateful_enabled(self) -> bool:
        return self._stateful is not None

    def reset(self) -> None:
        self._context = ()
        self._next_sample = None
        if self._stateful is not None:
            self._stateful.reset()

    def _batch(self, values: Tuple[float, ...]):
        try:
            import numpy as np
        except ImportError:
            return [[[sample] for sample in values]]
        return np.asarray(values, dtype=np.float32).reshape(1, -1, 1)

    def _validate_outputs(self, outputs: Any):
        if not isinstance(outputs, dict):
            raise ValueError("V8 stream model must return named outputs")
        missing = [name for name in _REQUIRED_OUTPUTS if name not in outputs]
        if missing:
            raise ValueError(f"V8 stream model missing outputs: {missing}")
        return outputs

    def _slow_outputs(self, values: Tuple[float, ...]):
        outputs = self._validate_outputs(
            self._model(self._batch(values), training=False)
        )
        return outputs

    def _score_chunk(self, outputs: Any, start_sample: int, count: int) -> V8ScoreChunk:
        converted = {
            name: _to_list(outputs[name])[0]
            for name in _REQUIRED_OUTPUTS
        }
        for name in converted:
            if len(converted[name]) < count:
                raise ValueError(f"V8 output {name} is shorter than input chunk")

        return V8ScoreChunk(
            start_sample,
            tuple(float(row[0]) for row in converted["onset_presence"][-count:]),
            tuple(float(row[0]) for row in converted["offset_presence"][-count:]),
            tuple(
                tuple(float(value) for value in row)
                for row in converted["onset_multiplicity"][-count:]
            ),
            tuple(
                tuple(float(value) for value in row)
                for row in converted["offset_multiplicity"][-count:]
            ),
        )

    def warm_up(self, chunk_size: int = 512) -> None:
        size = _positive_integer("chunk_size", chunk_size)
        values = (0.0,) * size
        if self._stateful is None:
            outputs = self._slow_outputs(values)
        else:
            outputs, _pending = self._stateful.predict(self._batch(values))
            outputs = self._validate_outputs(outputs)
        scores = self._score_chunk(outputs, 0, size)
        if scores.sample_count != size:
            raise ValueError("V8 warm-up returned inconsistent temporal length")

    def predict_chunk(
        self,
        samples: Iterable[float],
        *,
        start_sample: int,
    ) -> V8ScoreChunk:
        values = _finite_samples(samples)
        if (
            isinstance(start_sample, bool)
            or not isinstance(start_sample, int)
            or start_sample < 0
        ):
            raise ValueError("start_sample must be an integer >= 0")
        if self._next_sample is not None and start_sample != self._next_sample:
            raise ValueError(
                f"expected contiguous chunk at {self._next_sample}, got {start_sample}"
            )
        if not values:
            self._next_sample = start_sample
            return V8ScoreChunk(start_sample, (), (), (), ())

        pending_caches = None
        if self._stateful is None:
            model_input = self._context + values
            outputs = self._slow_outputs(model_input)
        else:
            model_input = values
            outputs, pending_caches = self._stateful.predict(self._batch(values))
            outputs = self._validate_outputs(outputs)

        scores = self._score_chunk(outputs, start_sample, len(values))

        if self._stateful is None:
            context_size = self._receptive_field - 1
            self._context = model_input[-context_size:] if context_size else ()
        else:
            self._stateful.commit(pending_caches)
        self._next_sample = start_sample + len(values)
        return scores


__all__ = ["V8KerasPredictor"]
