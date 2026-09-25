"""Synthetic shape/gradient checks only; never train on outer-fold examples."""
import argparse
from pathlib import Path
import numpy as np
from scripts.rebuild_v273_sources import write_json
from scripts import train_v260_count_weighting as v260
from scripts import train_v272_poly_conditional_count as v272


def check(output):
    import tensorflow as tf
    reports = []
    for name, builder, classes in (('full_count', v260.build_model, 7), ('poly_count', v272.build_model, 5)):
        default = builder('uniform', 273)
        rng = np.random.default_rng(273)
        inputs = {t.name.split(':')[0]: rng.normal(size=(2, *t.shape[1:])).astype(np.float32)
                  for t in default.inputs}
        inputs['candidate_mask'][:] = 1
        reference = np.asarray(default(inputs, training=False))
        reference_weights = default.get_weights()
        explicit = builder('uniform', 273, time_frames=23)
        np.testing.assert_array_equal(reference, np.asarray(explicit(inputs, training=False)))
        model = builder('uniform', 273, time_frames=31)
        assert len(model.get_weights()) == len(reference_weights)
        for a, b in zip(reference_weights, model.get_weights()):
            np.testing.assert_array_equal(a, b)
        x = dict(inputs)
        x['spectral_map'] = rng.normal(size=(2, 31, 64, 3)).astype(np.float32)
        p = np.asarray(model(x, training=False))
        assert p.shape == (2, classes) and np.isfinite(p).all()
        np.testing.assert_allclose(p.sum(1), 1., atol=1e-6)
        before = [v.numpy().copy() for v in model.trainable_variables]
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                tf.constant([0, 1]), model(x, training=True)))
        gradients = tape.gradient(loss, model.trainable_variables)
        assert all(g is not None and np.isfinite(g.numpy()).all() for g in gradients)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        assert any(not np.array_equal(a, b.numpy()) for a, b in zip(before, model.trainable_variables))
        reports.append(dict(component=name, output_classes=classes, covered_frames=31,
                            historical_default_identical=True, initial_weights_23_and_31_identical=True, probabilities_normalized=True,
                            finite_gradient_and_update=True, synthetic_loss=float(loss)))
    result = dict(status='passed', examples='random synthetic only', outer_examples_used=0,
                  actual_training_run=False, models=reports)
    write_json(output, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    check(p.parse_args().output)
