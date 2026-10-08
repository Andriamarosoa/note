import numpy as np
from scipy.optimize._numdiff import approx_derivative

from scripts.v273_probability_calibration import (
    IDENTITY, availability, decode, fit, objective, transform,
)


def fixture():
    rng = np.random.default_rng(27404)
    n = 350
    baseline = rng.integers(2, 5, n)
    proposal = rng.integers(2, 7, (n, 9))
    proposal[:, 0] = baseline
    proposal[:15] = baseline[:15, None]  # no alternative, including OTHER labels
    return dict(baseline=baseline, proposal=proposal,
                truth=rng.integers(0, 7, n), baseline_logits=rng.normal(size=n),
                pooled_class_logits=rng.normal(size=(n, 7)))


def test_gradient_including_missing_destinations_and_other():
    d = fixture(); av = availability(d['proposal'], d['baseline'])
    target = np.where(av[np.arange(len(av)), d['truth']], d['truth'], 7)
    args = (d['baseline_logits'], d['pooled_class_logits'], av,
            (d['truth'] == d['baseline']).astype(float), target)
    p = np.array([.6, -.4, 1.4, .8])
    value, gradient = objective(p, *args)
    numerical = approx_derivative(lambda x: objective(x, *args)[0], p).ravel()
    assert np.isfinite(value)
    np.testing.assert_allclose(gradient, numerical, atol=1e-8, rtol=1e-6)


def test_transform_is_normalized_and_preserves_destination_ranking():
    d = fixture()
    outputs = []
    for p in [IDENTITY, [.2, .7, .05, -3], [2, -1, 4, 2]]:
        out = transform(d['baseline_logits'], d['pooled_class_logits'],
                        d['proposal'], d['baseline'], p)
        np.testing.assert_allclose(out['class_probability'].sum(1)+out['other_probability'], 1)
        prediction, best, gain = decode(dict(d, **out))
        assert np.isfinite(gain).all()
        assert np.array_equal(prediction[:15], d['baseline'][:15])
        outputs.append(best)
    assert all(np.array_equal(outputs[0], x) for x in outputs[1:])


def test_convex_fit_converges_and_improves_its_objective():
    d = fixture(); learned = fit(d)
    assert learned['success'] and learned['objective_fitted'] < learned['objective_identity']
