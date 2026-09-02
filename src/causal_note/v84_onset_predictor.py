"""Onset-only streaming predictor for V8.4 diagnostics.

V8.4 deployment evaluates two independent streams.  Onset-only diagnostics do
not need to execute the frozen offset stream, so this wrapper loads only the
nested onset stream while preserving the exact V8 onset outputs.
"""
from pathlib import Path

from .v8_predictor import V8KerasPredictor


class V84OnsetOnlyKerasPredictor:
    def __init__(self, model, *, receptive_field: int = 4093, use_stateful: bool = True):
        self._onset = V8KerasPredictor(
            model.get_layer("v84_onset_stream"),
            receptive_field=receptive_field,
            use_stateful=use_stateful,
        )
        self._receptive_field = receptive_field

    @classmethod
    def from_path(cls, path: str, *, receptive_field: int = 4093, use_stateful: bool = True):
        model_path = Path(path)
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        try:
            from tensorflow import keras
        except ImportError as exc:
            raise RuntimeError("TensorFlow is required to load a V8.4 model") from exc
        model = keras.models.load_model(str(model_path), compile=False)
        return cls(model, receptive_field=receptive_field, use_stateful=use_stateful)

    @property
    def receptive_field(self) -> int:
        return self._receptive_field

    @property
    def next_sample(self):
        return self._onset.next_sample

    @property
    def stateful_enabled(self) -> bool:
        return self._onset.stateful_enabled

    def reset(self) -> None:
        self._onset.reset()

    def warm_up(self, chunk_size: int = 512) -> None:
        self._onset.warm_up(chunk_size)

    def predict_chunk(self, samples, *, start_sample: int):
        return self._onset.predict_chunk(tuple(samples), start_sample=start_sample)


__all__ = ["V84OnsetOnlyKerasPredictor"]
