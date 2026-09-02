"""V10.3 clean two-group wrapper.

The historical V10.1/V10.2 holdout intersection contains only two composition
groups. Keep the no-leakage requirement, but reduce the meta-model to a single
regularized logistic gate so the tiny group diversity cannot support a hidden
MLP. All frozen experts, features, fusion equation and locked12 protocol remain
unchanged.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_v103_residual_soft_fusion as v

v.MIN_COMMON_GROUPS = 2
v.GATE_EPOCHS = 60
v.GATE_L2 = 1e-2


def _build_linear_gate(feature_dim: int):
    from tensorflow import keras

    feat = keras.Input((feature_dim,), name="fusion_features")
    p1 = keras.Input((v.SLOT_COUNT + 1,), name="v101_anchor")
    p2 = keras.Input((v.SLOT_COUNT + 1,), name="v102_count")
    alpha = keras.layers.Dense(
        1,
        activation="sigmoid",
        kernel_regularizer=keras.regularizers.l2(v.GATE_L2),
        name="gate_alpha",
    )(feat)
    final = keras.layers.Lambda(
        lambda z: (1.0 - z[0]) * z[1] + z[0] * z[2],
        name="final_count",
    )([alpha, p1, p2])
    model = keras.Model(
        {"fusion_features": feat, "v101_anchor": p1, "v102_count": p2},
        final,
        name="v103_linear_residual_soft_fusion",
    )
    model.compile(optimizer=keras.optimizers.Adam(5e-4), loss="categorical_crossentropy")
    return model, keras.Model(model.inputs, alpha)


v._build_gate = _build_linear_gate

if __name__ == "__main__":
    raise SystemExit(v.main())
