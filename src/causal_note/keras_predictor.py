"""Streaming adapter from a causal Keras model to boundary score chunks."""

import math
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

from .detector import BoundaryScoreChunk


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be an integer > 0")
    return value


def _finite_samples(samples: Iterable[float]) -> Tuple[float, ...]:
    converted = []
    for sample in samples:
        if isinstance(sample, bool):
            raise ValueError("audio samples must be finite numbers")
        try:
            value = float(sample)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("audio samples must be finite numbers") from exc
        if not math.isfinite(value):
            raise ValueError("audio samples must be finite numbers")
        converted.append(value)
    return tuple(converted)


def _output_slot_count(model: Any) -> int:
    output_shape = getattr(model, "output_shape", None)
    if isinstance(output_shape, dict):
        shapes = output_shape
    else:
        output_names = getattr(model, "output_names", None)
        if not isinstance(output_names, (tuple, list)) or not isinstance(
            output_shape, (tuple, list)
        ):
            raise ValueError("model must expose named onset and offset outputs")
        if len(output_names) != len(output_shape):
            raise ValueError("model output names and shapes are inconsistent")
        shapes = dict(zip(output_names, output_shape))

    slot_counts = []
    for name in ("onset", "offset"):
        shape = shapes.get(name)
        if shape is None:
            raise ValueError(f"model must expose a named {name} output")
        try:
            slot_counts.append(_positive_integer(f"model {name} slot count", shape[-1]))
        except (TypeError, IndexError) as exc:
            raise ValueError(
                f"model {name} output must have a slot dimension"
            ) from exc
    if slot_counts[0] != slot_counts[1]:
        raise ValueError("model onset and offset slot counts must match")
    return slot_counts[0]


def _to_nested_lists(value: Any) -> Any:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _inference_callable(model: Any):
    """Compile real TensorFlow models while keeping dependency-free fakes usable."""

    try:
        import tensorflow as tf
    except ImportError:
        return lambda batch: model(batch, training=False)
    if not isinstance(model, tf.keras.Model):
        return lambda batch: model(batch, training=False)

    @tf.function(
        input_signature=[tf.TensorSpec((1, None, 1), tf.float32)],
        reduce_retracing=True,
    )
    def infer(batch):
        return model(batch, training=False)

    return infer


class _StatefulCausalRunner:
    """Fast exact streaming for the convolution stack built by this project."""

    def __init__(self, model: Any, tf: Any) -> None:
        self._tf = tf
        self._layers = tuple(
            layer
            for layer in model.layers
            if layer.name.startswith("causal_conv_")
        )
        if not self._layers:
            raise ValueError("model has no supported causal convolution stack")
        self._heads = {
            name: model.get_layer(name) for name in ("onset", "offset")
        }

        cache_specs = []
        initial_caches = []
        for layer in self._layers:
            if (
                getattr(layer, "padding", None) != "causal"
                or tuple(getattr(layer, "strides", ())) != (1,)
            ):
                raise ValueError("unsupported non-causal or strided model layer")
            kernel_size = int(layer.kernel_size[0])
            dilation = int(layer.dilation_rate[0])
            cache_length = (kernel_size - 1) * dilation
            input_channels = int(layer.kernel.shape[1])
            spec = tf.TensorSpec(
                (1, cache_length, input_channels),
                tf.float32,
            )
            cache_specs.append(spec)
            initial_caches.append(tf.zeros(spec.shape, dtype=spec.dtype))
        self._initial_caches = tuple(initial_caches)
        self._caches = self._initial_caches
        layers = self._layers
        heads = self._heads

        @tf.function(
            input_signature=[tf.TensorSpec((1, None, 1), tf.float32)]
            + cache_specs,
            reduce_retracing=True,
        )
        def infer(chunk, *caches):
            hidden = chunk
            new_caches = []
            for layer, cache in zip(layers, caches):
                layer_input = tf.concat((cache, hidden), axis=1)
                cache_length = int(cache.shape[1])
                full_output = layer(layer_input, training=False)
                hidden = full_output[:, -tf.shape(hidden)[1] :, :]
                new_caches.append(
                    layer_input[:, -cache_length:, :]
                    if cache_length
                    else layer_input[:, :0, :]
                )
            outputs = {
                name: head(hidden, training=False)
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
        return _StatefulCausalRunner(model, tf)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


class KerasBoundaryPredictor:
    """Run a causal Keras model over contiguous chunks without losing context.

    Only the last ``receptive_field - 1`` samples are retained.  Recomputing
    that bounded context makes the result for a chunk identical to running the
    same strictly causal model over the complete stream.
    """

    def __init__(
        self,
        model: Any,
        *,
        receptive_field: Optional[int] = None,
    ) -> None:
        if not callable(model):
            raise ValueError("model must be callable")
        inferred = getattr(model, "receptive_field", None)
        field = inferred if receptive_field is None else receptive_field
        self._receptive_field = _positive_integer("receptive_field", field)
        self._slot_count = _output_slot_count(model)
        self._model = model
        self._infer = _inference_callable(model)
        self._stateful = _stateful_runner(model)
        self._context: Tuple[float, ...] = ()
        self._next_sample: Optional[int] = None

    @classmethod
    def from_path(
        cls,
        path: str,
        *,
        receptive_field: int,
    ) -> "KerasBoundaryPredictor":
        """Load a saved model lazily so core tests need no ML dependency."""

        model_path = Path(path)
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        try:
            from tensorflow import keras
        except ImportError as exc:
            raise RuntimeError(
                "TensorFlow is required to load a trained boundary model"
            ) from exc
        return cls(
            keras.models.load_model(str(model_path), compile=False),
            receptive_field=receptive_field,
        )

    @property
    def slot_count(self) -> int:
        return self._slot_count

    @property
    def receptive_field(self) -> int:
        return self._receptive_field

    def reset(self) -> None:
        """Forget stream context before starting an unrelated audio stream."""

        self._context = ()
        self._next_sample = None
        if self._stateful is not None:
            self._stateful.reset()

    def warm_up(self, chunk_size: int = 512) -> None:
        """Compile the inference graph without consuming stream state."""

        size = _positive_integer("chunk_size", chunk_size)
        values = (0.0,) * size
        try:
            import numpy as np
        except ImportError:
            batch = [[[sample] for sample in values]]
        else:
            batch = np.asarray(values, dtype=np.float32).reshape(1, -1, 1)
        if self._stateful is None:
            outputs = self._infer(batch)
        else:
            outputs, _ = self._stateful.predict(batch)
        try:
            onset = _to_nested_lists(outputs["onset"])[0]
            offset = _to_nested_lists(outputs["offset"])[0]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("model must return onset and offset sequences") from exc
        scores = BoundaryScoreChunk(
            0,
            tuple(tuple(row) for row in onset[-size:]),
            tuple(tuple(row) for row in offset[-size:]),
        )
        if scores.sample_count != size or scores.slot_count != self._slot_count:
            raise ValueError("model warm-up returned inconsistent boundary scores")

    def predict_chunk(
        self,
        samples: Tuple[float, ...],
        *,
        start_sample: int,
    ) -> BoundaryScoreChunk:
        values = _finite_samples(samples)
        if isinstance(start_sample, bool) or not isinstance(start_sample, int):
            raise ValueError("start_sample must be an integer >= 0")
        if start_sample < 0:
            raise ValueError("start_sample must be an integer >= 0")
        if self._next_sample is not None and start_sample != self._next_sample:
            raise ValueError(
                f"expected contiguous chunk at {self._next_sample}, "
                f"got {start_sample}"
            )
        if not values:
            self._next_sample = start_sample
            return BoundaryScoreChunk(start_sample, (), ())

        model_input = values if self._stateful is not None else self._context + values
        try:
            import numpy as np
        except ImportError:
            # Keeps lightweight protocol tests usable without ML dependencies.
            batch = [[[sample] for sample in model_input]]
        else:
            batch = np.asarray(model_input, dtype=np.float32).reshape(1, -1, 1)
        pending_caches = None
        if self._stateful is None:
            outputs = self._infer(batch)
        else:
            outputs, pending_caches = self._stateful.predict(batch)
        if not isinstance(outputs, dict):
            raise ValueError("model must return named boundary outputs")
        try:
            onset = _to_nested_lists(outputs["onset"])[0]
            offset = _to_nested_lists(outputs["offset"])[0]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("model must return onset and offset sequences") from exc

        sample_count = len(values)
        onset_tail = onset[-sample_count:]
        offset_tail = offset[-sample_count:]
        scores = BoundaryScoreChunk(
            start_sample,
            tuple(tuple(row) for row in onset_tail),
            tuple(tuple(row) for row in offset_tail),
        )
        if scores.sample_count != sample_count:
            raise ValueError("model returned the wrong temporal length")
        if scores.slot_count != self._slot_count:
            raise ValueError("model returned the wrong slot count")

        if self._stateful is None:
            context_size = self._receptive_field - 1
            self._context = (
                model_input[-context_size:] if context_size else ()
            )
        else:
            self._stateful.commit(pending_caches)
        self._next_sample = start_sample + sample_count
        return scores
