"""Exact causal stateful predictor for the split-task V8.3 stream model."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

from .v8_predictor import V8KerasPredictor, _causal_cache_length


class _V83StatefulCausalRunner:
    """Per-layer cached runner for shared stem plus private onset/offset towers."""

    def __init__(self, model: Any, tf: Any) -> None:
        self._tf = tf
        self._transient = model.get_layer("v8_transient_conv")
        self._input_projection = model.get_layer("v8_input_projection")
        self._onset_fusion = model.get_layer("v83_onset_fusion_projection")
        self._offset_fusion = model.get_layer("v83_offset_fusion_projection")
        self._heads = {
            "onset_presence": model.get_layer("onset_presence_sequence"),
            "offset_presence": model.get_layer("offset_presence_sequence"),
            "onset_multiplicity": model.get_layer("onset_multiplicity_sequence"),
            "offset_multiplicity": model.get_layer("offset_multiplicity_sequence"),
        }

        shared_features = {
            int(layer.name.rsplit("_", 1)[1]): layer
            for layer in model.layers
            if layer.name.startswith("v8_feature_")
        }
        if not shared_features:
            raise ValueError("V8.3 model has no shared residual blocks")
        shared_indices = tuple(sorted(shared_features))
        if shared_indices != tuple(range(1, len(shared_indices) + 1)):
            raise ValueError("V8.3 shared block indices must be contiguous")
        split_after = shared_indices[-1]

        self._shared_blocks = self._collect_blocks(
            model,
            shared_indices,
            feature_prefix="v8_feature_",
            gate_prefix="v8_gate_",
            mix_prefix="v8_mix_",
        )

        private = {}
        for kind in ("onset", "offset"):
            feature_prefix = f"v83_{kind}_feature_"
            features = {
                int(layer.name.rsplit("_", 1)[1]): layer
                for layer in model.layers
                if layer.name.startswith(feature_prefix)
            }
            indices = tuple(sorted(features))
            if not indices or indices[0] != split_after + 1:
                raise ValueError(f"V8.3 {kind} private tower does not continue shared tower")
            if indices != tuple(range(indices[0], indices[-1] + 1)):
                raise ValueError(f"V8.3 {kind} private block indices must be contiguous")
            private[kind] = self._collect_blocks(
                model,
                indices,
                feature_prefix=feature_prefix,
                gate_prefix=f"v83_{kind}_gate_",
                mix_prefix=f"v83_{kind}_mix_",
            )
        if len(private["onset"]) != len(private["offset"]):
            raise ValueError("V8.3 private towers have different depths")
        self._private = private

        cache_specs = []
        initial_caches = []

        transient_cache = _causal_cache_length(self._transient)
        transient_channels = int(self._transient.kernel.shape[1])
        spec = tf.TensorSpec((1, transient_cache, transient_channels), tf.float32)
        cache_specs.append(spec)
        initial_caches.append(tf.zeros(spec.shape, dtype=tf.float32))

        all_blocks = (
            tuple(self._shared_blocks)
            + tuple(self._private["onset"])
            + tuple(self._private["offset"])
        )
        for feature, _gate, _mix in all_blocks:
            cache_length = _causal_cache_length(feature)
            input_channels = int(feature.kernel.shape[1])
            spec = tf.TensorSpec((1, cache_length, input_channels), tf.float32)
            cache_specs.append(spec)
            initial_caches.append(tf.zeros(spec.shape, dtype=tf.float32))

        self._initial_caches = tuple(initial_caches)
        self._caches = self._initial_caches
        transient = self._transient
        input_projection = self._input_projection
        shared_blocks = tuple(self._shared_blocks)
        onset_blocks = tuple(self._private["onset"])
        offset_blocks = tuple(self._private["offset"])
        onset_fusion = self._onset_fusion
        offset_fusion = self._offset_fusion
        heads = self._heads
        shared_count = len(shared_blocks)
        onset_count = len(onset_blocks)

        def run_block(hidden, block, cache):
            feature, gate, mix = block
            layer_input = tf.concat((cache, hidden), axis=1)
            feature_full = feature(layer_input, training=False)
            gate_full = gate(layer_input, training=False)
            current_length = tf.shape(hidden)[1]
            gated = (
                feature_full[:, -current_length:, :]
                * gate_full[:, -current_length:, :]
            )
            next_hidden = hidden + mix(gated, training=False)
            cache_length = int(cache.shape[1])
            next_cache = (
                layer_input[:, -cache_length:, :]
                if cache_length
                else layer_input[:, :0, :]
            )
            return next_hidden, next_cache

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
            cursor = 1
            for block in shared_blocks:
                hidden, next_cache = run_block(hidden, block, caches[cursor])
                new_caches.append(next_cache)
                cursor += 1

            onset_hidden = hidden
            for block in onset_blocks:
                onset_hidden, next_cache = run_block(onset_hidden, block, caches[cursor])
                new_caches.append(next_cache)
                cursor += 1

            offset_hidden = hidden
            offset_start = 1 + shared_count + onset_count
            cursor = offset_start
            for block in offset_blocks:
                offset_hidden, next_cache = run_block(offset_hidden, block, caches[cursor])
                new_caches.append(next_cache)
                cursor += 1

            onset_fused = onset_fusion(
                tf.concat((onset_hidden, transient_hidden), axis=-1), training=False
            )
            offset_fused = offset_fusion(
                tf.concat((offset_hidden, transient_hidden), axis=-1), training=False
            )
            outputs = {
                "onset_presence": heads["onset_presence"](onset_fused, training=False),
                "onset_multiplicity": heads["onset_multiplicity"](onset_fused, training=False),
                "offset_presence": heads["offset_presence"](offset_fused, training=False),
                "offset_multiplicity": heads["offset_multiplicity"](offset_fused, training=False),
            }
            return outputs, tuple(new_caches)

        self._infer = infer

    @staticmethod
    def _collect_blocks(model, indices, *, feature_prefix, gate_prefix, mix_prefix):
        blocks = []
        for index in indices:
            feature = model.get_layer(f"{feature_prefix}{index}")
            gate = model.get_layer(f"{gate_prefix}{index}")
            mix = model.get_layer(f"{mix_prefix}{index}")
            if _causal_cache_length(feature) != _causal_cache_length(gate):
                raise ValueError("V8.3 feature/gate cache lengths differ")
            if tuple(feature.kernel_size) != tuple(gate.kernel_size):
                raise ValueError("V8.3 feature/gate kernels differ")
            if tuple(feature.dilation_rate) != tuple(gate.dilation_rate):
                raise ValueError("V8.3 feature/gate dilations differ")
            blocks.append((feature, gate, mix))
        return tuple(blocks)

    def predict(self, batch: Any):
        return self._infer(batch, *self._caches)

    def commit(self, caches: Tuple[Any, ...]) -> None:
        self._caches = tuple(caches)

    def reset(self) -> None:
        self._caches = self._initial_caches


class V83KerasPredictor(V8KerasPredictor):
    """V8 predictor API with the V8.3 split-task cached runner."""

    def __init__(self, model: Any, *, receptive_field: int = 4093, use_stateful: bool = True):
        super().__init__(model, receptive_field=receptive_field, use_stateful=False)
        if use_stateful:
            try:
                import tensorflow as tf
                if isinstance(model, tf.keras.Model):
                    self._stateful = _V83StatefulCausalRunner(model, tf)
            except (ImportError, AttributeError, KeyError, TypeError, ValueError):
                self._stateful = None

    @classmethod
    def from_path(cls, path: str, *, receptive_field: int = 4093, use_stateful: bool = True):
        model_path = Path(path)
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        try:
            from tensorflow import keras
        except ImportError as exc:
            raise RuntimeError("TensorFlow is required to load a V8.3 model") from exc
        model = keras.models.load_model(str(model_path), compile=False)
        return cls(model, receptive_field=receptive_field, use_stateful=use_stateful)


__all__ = ["V83KerasPredictor"]
