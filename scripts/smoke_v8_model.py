"""TensorFlow smoke check for the V8 architecture and saved stream model."""
from pathlib import Path
import tempfile

import numpy as np

from causal_note.v8_model import build_v8_point_model
from causal_note.v8_predictor import V8KerasPredictor


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
        predictor = V8KerasPredictor.from_path(
            str(path),
            receptive_field=receptive_field,
        )
        predictor.warm_up(32)
        scores = predictor.predict_chunk((0.0,) * 32, start_sample=0)
        if scores.sample_count != 32:
            raise AssertionError("reloaded V8 stream model returned wrong length")
        if predictor.next_sample != 32:
            raise AssertionError("V8 predictor did not advance stream position")

    print(
        "V8 TensorFlow smoke PASS:",
        f"receptive_field={receptive_field}",
        f"point_outputs={actual}",
    )


if __name__ == "__main__":
    main()
