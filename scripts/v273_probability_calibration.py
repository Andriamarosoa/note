"""Four-parameter convex calibration of a frozen joint verdict distribution."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

IDENTITY = np.array([1., 0., 1., 0.])
RIDGE = .001
BOUNDS = ((.05, 20.), (-10., 10.), (.05, 20.), (-10., 10.))


def availability(proposal, baseline):
    return np.stack([(proposal == k).any(1) & (baseline != k)
                     for k in range(7)], axis=1)


def objective(parameters, z, logits, available, baseline_correct, target):
    """Event-mean joint NLL plus fixed identity ridge, with analytic gradient."""
    ar, br, aq, bq = parameters
    x = ar*z + br
    safe_logits = np.where(available, logits, 0.)
    raw = np.column_stack([np.where(available, aq*safe_logits+bq, -np.inf),
                           np.zeros(len(z))])
    logq = raw-logsumexp(raw, axis=1, keepdims=True)
    q = np.exp(logq)
    wrong = 1-baseline_correct
    loss = np.mean(np.logaddexp(0., x)-baseline_correct*x
                   - wrong*logq[np.arange(len(z)), target])
    dr = expit(x)-baseline_correct
    residual = q.copy()
    residual[np.arange(len(z)), target] -= 1
    dq = residual[:, :7]*wrong[:, None]
    gradient = np.array([np.mean(dr*z), np.mean(dr),
                         np.mean(np.sum(dq*safe_logits, axis=1)),
                         np.mean(np.sum(dq, axis=1))])
    delta = np.asarray(parameters)-IDENTITY
    return float(loss+RIDGE*np.dot(delta, delta)), gradient+2*RIDGE*delta


def fit(data):
    """Receives calibration rows only; no evaluated labels or features."""
    y, b = data['truth'], data['baseline']
    av = availability(data['proposal'], b)
    target = np.where(av[np.arange(len(y)), y], y, 7)
    args = (data['baseline_logits'].astype(float),
            data['pooled_class_logits'].astype(float), av,
            (y == b).astype(float), target)
    result = minimize(objective, IDENTITY.copy(), args=args, method='L-BFGS-B',
                      jac=True, bounds=BOUNDS,
                      options=dict(maxiter=1000, ftol=1e-13, gtol=1e-8,
                                   maxls=40))
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError('Calibration failed: '+str(result.message))
    before = objective(IDENTITY, *args)[0]
    if result.fun > before+1e-9:
        raise ValueError('Calibrator is worse than identity on its fit objective')
    return dict(parameters=result.x.tolist(), success=bool(result.success),
                iterations=int(result.nit), message=str(result.message),
                objective_identity=float(before), objective_fitted=float(result.fun),
                penalty=float(RIDGE*np.sum((result.x-IDENTITY)**2)),
                bounds_hit=[bool(np.isclose(v, lo) or np.isclose(v, hi))
                            for v, (lo, hi) in zip(result.x, BOUNDS)])


def transform(z, logits, proposal, baseline, parameters):
    """Label-free transformation; the available destination ranking is fixed."""
    ar, br, aq, bq = parameters
    if ar <= 0 or aq <= 0:
        raise ValueError('Calibration slopes must be positive')
    z = ar*np.asarray(z, float)+br
    logits = aq*np.asarray(logits, float)+bq
    av = availability(proposal, baseline)
    raw = np.column_stack([np.where(av, logits, -np.inf), np.zeros(len(z))])
    q = np.exp(raw-logsumexp(raw, axis=1, keepdims=True))
    r = expit(z)
    return dict(baseline_logits=z, pooled_class_logits=logits,
                baseline_correct=r,
                class_probability=(1-r[:, None])*q[:, :7]+np.eye(7)[baseline]*r[:, None],
                other_probability=(1-r)*q[:, 7])


def decode(data):
    b = data['baseline']
    av = availability(data['proposal'], b)
    # Choosing by logits avoids artificial ties from saturated probabilities.
    best = np.where(av, data['pooled_class_logits'], -np.inf).argmax(1)
    has = av.any(1)
    gain = data['class_probability'][np.arange(len(b)), best]-data['baseline_correct']
    gain = np.where(has, gain, 0.)
    selected = has & (gain > 0)
    return np.where(selected, best, b), best, gain


def subset(data, mask):
    return {key: value[mask] for key, value in data.items()}
