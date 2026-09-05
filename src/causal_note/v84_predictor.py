"""Streaming predictor for the V8.4 two-stream topology."""
from pathlib import Path

from .v8_predictor import V8KerasPredictor
from .v8_runtime import V8ScoreChunk


class V84KerasPredictor:
    def __init__(self, model, *, receptive_field: int = 4093, use_stateful: bool = True):
        self._onset = V8KerasPredictor(
            model.get_layer("v84_onset_stream"),
            receptive_field=receptive_field,
            use_stateful=use_stateful,
        )
        self._offset = V8KerasPredictor(
            model.get_layer("v84_offset_stream"),
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
        if self._onset.next_sample != self._offset.next_sample:
            raise RuntimeError("V8.4 onset/offset predictors lost stream alignment")
        return self._onset.next_sample

    @property
    def stateful_enabled(self) -> bool:
        return self._onset.stateful_enabled and self._offset.stateful_enabled

    def reset(self) -> None:
        self._onset.reset()
        self._offset.reset()

    def warm_up(self, chunk_size: int = 512) -> None:
        self._onset.warm_up(chunk_size)
        self._offset.warm_up(chunk_size)

    def predict_chunk(self, samples, *, start_sample: int) -> V8ScoreChunk:
        values = tuple(samples)
        onset = self._onset.predict_chunk(values, start_sample=start_sample)
        offset = self._offset.predict_chunk(values, start_sample=start_sample)
        if onset.sample_count != offset.sample_count:
            raise RuntimeError("V8.4 onset/offset predictors returned different lengths")
        return V8ScoreChunk(
            start_sample,
            onset.onset_presence,
            offset.offset_presence,
            onset.onset_multiplicity,
            offset.offset_multiplicity,
        )


__all__ = ["V84KerasPredictor"]
