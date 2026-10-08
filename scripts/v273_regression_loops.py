"""Convex, reproducible calibration with full-group context and nested policy choice."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

FAMILIES = ('identity', 'q_affine', 'joint_affine', 'transition',
            'group_audit', 'acoustic43', 'acoustic58')
COSTS = (1., 1.15, 1.3, 1.5, 2., 3., 5.)
RETAIN = .90


def available(proposal, baseline):
    return np.stack([(proposal == k).any(1) & (baseline != k) for k in range(7)], 1)


def group_context(proposal, audits, baseline):
    """All complete groups for a destination; no singleton gate or chosen-mask proxy."""
    n, groups = proposal.shape
    if groups != 255 or audits.shape != (n, 255, 9):
        raise ValueError('Expected 255 complete groups and nine audit descriptors')
    membership = np.array([[bool(mask & (1 << j)) for j in range(8)]
                           for mask in range(1, 256)], float)
    names = ['audit_mean_' + str(j) for j in range(9)] + [
        'local_gain_min', 'local_gain_max', 'fraction_of_groups'] + [
        'full_group_member_fraction_S' + str(j+1) for j in range(8)]
    out = np.zeros((n, 7, len(names)))
    for k in range(2, 7):
        mask = (proposal == k) & (baseline[:, None] != k)
        count = mask.sum(1); den = np.maximum(count, 1)
        out[:, k, :9] = np.einsum('ng,ngd->nd', mask, audits) / den[:, None]
        local_gain = audits[:, :, 3] - audits[:, :, 4]
        for col, fn, fill in [(9, np.min, np.inf), (10, np.max, -np.inf)]:
            out[:, k, col] = np.where(count > 0, fn(np.where(mask, local_gain, fill), 1), 0)
        out[:, k, 11] = count / 255
        out[:, k, 12:] = mask @ membership / den[:, None]
    return out, names


def design(data, family):
    """Only observables enter this function. Labels and split identities are unused."""
    stage = FAMILIES.index(family); b = data['baseline']; n = len(b)
    xr = np.zeros((n, 0)); xq = np.zeros((n, 7, 0)); rn = []; qn = []
    if stage >= 3:
        xr = np.eye(7)[b][:, 2:5]; rn = ['initial_K'+str(k) for k in (2, 3, 4)]
        dest = np.broadcast_to(np.eye(7)[:, 2:7], (n, 7, 5))
        interaction = (xr[:, None, :, None] * dest[:, :, None, :]).reshape(n, 7, 15)
        xq = np.concatenate([dest, interaction], 2)
        qn = ['destination_K'+str(k) for k in range(2, 7)] + [
            f'transition_{j}_{k}' for j in (2, 3, 4) for k in range(2, 7)]
    if stage >= 4:
        g = data['group_context'][:, 2:7]
        xr = np.concatenate([xr, g.reshape(n, -1)], 1)
        rn += [f'K{k}__{name}' for k in range(2, 7) for name in data['group_names']]
        xq = np.concatenate([xq, data['group_context']], 2)
        qn += list(data['group_names'])
    if stage >= 5:
        use = [j for j, name in enumerate(data['feature_names']) if stage == 6 or
               name.startswith(('spectral__', 'birth__', 'persistence__', 'damping__'))]
        acoustic = data['features'][:, use]
        xr = np.concatenate([xr, acoustic], 1)
        rn += [str(data['feature_names'][j]) for j in use]
        dest = np.broadcast_to(np.eye(7)[:, 2:7], (n, 7, 5))
        ax = (dest[:, :, :, None] * acoustic[:, None, None, :]).reshape(n, 7, -1)
        xq = np.concatenate([xq, ax], 2)
        qn += [f'K{k}__{data["feature_names"][j]}' for k in range(2, 7) for j in use]
    return xr, xq, rn, qn


def standardizer(x):
    if not x.shape[-1]:
        return dict(mean=[], scale=[])
    mean = x.mean(0); scale = x.std(0)
    scale = np.where(scale > 1e-8, scale, 1.)
    return dict(mean=mean.tolist(), scale=scale.tolist())


def standardize(x, state):
    return np.clip((x - np.asarray(state['mean'])) / np.asarray(state['scale']), -6, 6)


def binary_objective(theta, z, x, target):
    v = theta[0]*z + theta[1] + x @ theta[2:]
    delta = expit(v)-target
    value = np.mean(np.logaddexp(0., v)-target*v)
    grad = np.r_[np.mean(delta*z), np.mean(delta), x.T @ delta/len(z)]
    prior = np.r_[1., np.zeros(len(theta)-1)]
    weight = np.r_[.001, .001, np.full(len(theta)-2, .05)]
    return float(value+np.sum(weight*(theta-prior)**2)), grad+2*weight*(theta-prior)


def conditional_objective(theta, logits, x, av, target, wrong):
    v = theta[0]*logits + theta[1] + np.einsum('nkd,d->nk', x, theta[2:])
    full = np.column_stack([np.where(av, v, -np.inf), np.zeros(len(v))])
    logq = full-logsumexp(full, axis=1, keepdims=True)
    delta = np.exp(logq); delta[np.arange(len(v)), target] -= 1
    delta = delta[:, :7] * wrong[:, None]
    value = -np.mean(wrong*logq[np.arange(len(v)), target])
    grad = np.r_[np.mean(np.sum(delta*logits, 1)), np.mean(delta.sum(1)),
                 np.einsum('nk,nkd->d', delta, x)/len(v)]
    prior = np.r_[1., np.zeros(len(theta)-1)]
    weight = np.r_[.001, .001, np.full(len(theta)-2, .05)]
    return float(value+np.sum(weight*(theta-prior)**2)), grad+2*weight*(theta-prior)


def optimize(objective, dimension, args):
    initial = np.r_[1., np.zeros(dimension-1)]
    result = minimize(objective, initial, args=args, jac=True, method='L-BFGS-B',
                      bounds=[(.05, 20.), (-10., 10.)]+[(None, None)]*(dimension-2),
                      options=dict(maxiter=1000, ftol=1e-12, gtol=1e-7, maxls=40))
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(str(result.message))
    if result.fun > objective(initial, *args)[0]+1e-8:
        raise ValueError('Optimizer worsened its training objective')
    return dict(parameters=result.x.tolist(), iterations=int(result.nit),
                objective=float(result.fun), success=True,
                identity_objective=float(objective(initial, *args)[0]))


def fit(data, xr, xq, train, family):
    """All statistics and labels are restricted to explicit calibration indices."""
    train = np.asarray(train, int)
    if not len(train):
        raise ValueError('Empty calibration set')
    sr = standardizer(xr[train]); sq = standardizer(xq[train][data['available'][train]])
    result = dict(family=family, r_scale=sr, q_scale=sq, r=None, q=None)
    if family == 'identity':
        return result
    y, b = data['truth'][train], data['baseline'][train]
    av = data['available'][train]
    target = np.where(av[np.arange(len(train)), y], y, 7)
    result['q'] = optimize(conditional_objective, 2+xq.shape[2], (
        data['logits'][train], standardize(xq[train], sq), av, target, (y != b).astype(float)))
    if family != 'q_affine':
        result['r'] = optimize(binary_objective, 2+xr.shape[1], (
            data['z'][train], standardize(xr[train], sr), (y == b).astype(float)))
    return result


def predict(data, xr, xq, at, model):
    """No truth, piece, fold or event ID is consulted at inference."""
    at = np.asarray(at, int)
    z = data['z'][at].copy(); logits = data['logits'][at].copy()
    if model['r'] is not None:
        p = np.asarray(model['r']['parameters'])
        z = p[0]*z+p[1]+standardize(xr[at], model['r_scale']) @ p[2:]
    if model['q'] is not None:
        p = np.asarray(model['q']['parameters'])
        logits = p[0]*logits+p[1]+np.einsum('nkd,d->nk', standardize(xq[at], model['q_scale']), p[2:])
    full = np.column_stack([np.where(data['available'][at], logits, -np.inf), np.zeros(len(at))])
    q = np.exp(full-logsumexp(full, axis=1, keepdims=True)); r = expit(z)
    cp = (1-r[:, None])*q[:, :7]+np.eye(7)[data['baseline'][at]]*r[:, None]
    return r, cp, (1-r)*q[:, 7], logits


def decode(baseline, av, r, cp, logits, cost):
    best = np.where(av, logits, -np.inf).argmax(1)
    gain = cp[np.arange(len(best)), best]-cost*r
    take = av.any(1) & (gain > 0)
    return np.where(take, best, baseline)


def counts(y, b, p):
    change = p != b
    cor = int((change & (p == y)).sum()); reg = int((change & (b == y)).sum())
    return dict(changes=int(change.sum()), corrections=cor, regressions=reg, net=cor-reg)


def select_policy(y, b, predictions, original):
    """Call only with inner held-out rows; external labels are never parameters."""
    base = counts(y, b, original); candidates = []
    for index, pred in enumerate(predictions.T):
        stat = counts(y, b, pred)
        feasible = (stat['corrections'] >= RETAIN*base['corrections'] and
                    stat['net'] >= base['net'] and stat['regressions'] <= base['regressions'])
        if feasible:
            candidates.append((stat['regressions'], -stat['net'], -stat['corrections'], index))
    if not candidates:
        raise ValueError('Missing mandatory original-policy fallback')
    chosen = min(candidates)[-1]
    return chosen, dict(reference=base, chosen=counts(y, b, predictions[:, chosen]),
                        feasible=len(candidates), considered=predictions.shape[1])
