"""TensorFlow smoke checks for V8 architecture and exact stateful streaming."""
from pathlib import Path
import tempfile

import numpy as np

from causal_note.v8_model import build_v8_point_model
from causal_note.v8_predictor import V8KerasPredictor


def _flatten_scores(scores):
    return np.concatenate(
        (
            np.asarray(scores.onset_presence, dtype=np.float32)[:, None],
            np.asarray(scores.offset_presence, dtype=np.float32)[:, None],
            np.asarray(scores.onset_multiplicity, dtype=np.float32),
            np.asarray(scores.offset_multiplicity, dtype=np.float32),
        ),
        axis=1,
    )


def main() -> None:
    model = build_v8_point_model(
        filters=4,
        kernel_size=3,
        dilation_rates=(1, 2, 4),
    )
    receptive_field = int(model.receptive_field)
    batch = np.zeros((2, receptive_field, 1), dtype=np.float32)
    outputs = model(batch, training=False)
    expected = {
        "onset_presence": (2, 1),
        "offset_presence": (2, 1),
        "onset_multiplicity": (2, 3),
        "offset_multiplicity": (2, 3),
    }
    actual = {name: tuple(value.shape) for name, value in outputs.items()}
    if actual != expected:
        raise AssertionError(f"unexpected V8 point output shapes: {actual}")

    stream = model.get_layer(model.stream_model_name)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "v8-smoke.keras"
        stream.save(path)
        fast = V8KerasPredictor.from_path(
            str(path),
            receptive_field=receptive_field,
            use_stateful=True,
        )
        slow = V8KerasPredictor.from_path(
            str(path),
            receptive_field=receptive_field,
            use_stateful=False,
        )
        if not fast.stateful_enabled:
            raise AssertionError("V8 saved model did not enable the stateful runner")

        fast.warm_up(32)
        slow.warm_up(32)
        rng = np.random.default_rng(1337)
        audio = rng.normal(0.0, 0.1, size=97).astype(np.float32)
        chunk_sizes = (1, 7, 13, 5, 32, 11, 28)
        if sum(chunk_sizes) != len(audio):
            raise AssertionError("smoke chunk partition changed")

        position = 0
        max_error = 0.0
        for size in chunk_sizes:
            chunk = tuple(float(value) for value in audio[position : position + size])
            fast_scores = fast.predict_chunk(chunk, start_sample=position)
            slow_scores = slow.predict_chunk(chunk, start_sample=position)
            fast_values = _flatten_scores(fast_scores)
            slow_values = _flatten_scores(slow_scores)
            error = float(np.max(np.abs(fast_values - slow_values)))
            max_error = max(max_error, error)
            np.testing.assert_allclose(
                fast_values,
                slow_values,
                rtol=2e-5,
                atol=2e-6,
            )
            position += size

        if fast.next_sample != len(audio) or slow.next_sample != len(audio):
            raise AssertionError("V8 predictors did not advance stream position")

    print(
        "V8 TensorFlow smoke PASS:",
        f"receptive_field={receptive_field}",
        f"point_outputs={actual}",
        "stateful=True",
        f"stateful_max_abs_error={max_error:.9g}",
    )


if __name__ == "__main__":
    main()
