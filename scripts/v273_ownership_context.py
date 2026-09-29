"""Exact proposal ownership geometry, without annotations or model scores.

This is a tested input-building component, not a trained count correction.
Call separately for each track, with full (untruncated) proposal groups in
their original ID order. A complete_through watermark certifies proposal
enumeration, not merely how many audio samples have been read.
"""
import numpy as np

RADIUS = 882


def _integers(values, name):
    a = np.asarray(values)
    if a.ndim != 1 or a.dtype.kind not in 'iu':
        raise ValueError(name + ' must be a one-dimensional integer array')
    if a.dtype.kind == 'u' and a.size and a.max() > np.iinfo(np.int64).max:
        raise ValueError(name + ' exceeds int64')
    return a.astype(np.int64, copy=False)


def _groups(groups):
    result = [_integers(g, 'candidate positions') for g in groups]
    if not result or not any(len(g) for g in result):
        raise ValueError('at least one proposal is required')
    if any(np.any(g < 0) for g in result):
        raise ValueError('proposal positions must be nonnegative')
    return result


def _radius(radius):
    if isinstance(radius, (bool, np.bool_)) or not isinstance(radius, (int, np.integer)) or radius < 0:
        raise ValueError('radius must be a nonnegative integer')
    return int(radius)


def owners_at_samples(groups, query_samples, *, radius=RADIUS):
    """Offline exact nearest-group rule, including radius and lower-ID ties.

    Query positions are geometric coordinates. This function never accesses
    annotations; using annotated onsets as queries belongs only to an audit.
    """
    groups, q, radius = _groups(groups), _integers(query_samples, 'queries'), _radius(radius)
    best = np.full(len(q), np.iinfo(np.int64).max, np.int64)
    owner = np.full(len(q), -1, np.int64)
    # Strict improvement preserves the earlier group ID on an exact tie.
    for cid, values in enumerate(groups):
        if not len(values):
            continue
        values = np.sort(values)
        j = np.searchsorted(values, q)
        left = values[np.clip(j - 1, 0, len(values) - 1)]
        right = values[np.clip(j, 0, len(values) - 1)]
        distance = np.minimum(np.abs(q - left), np.abs(q - right))
        update = (distance < best) & (distance <= radius)
        best[update], owner[update] = distance[update], cid
    return owner


def required_proposal_watermark(current_candidates, *, radius=RADIUS):
    """Conservative completeness bound for every onset eligible to this group."""
    values, radius = _integers(current_candidates, 'current candidates'), _radius(radius)
    if not len(values) or np.any(values < 0):
        raise ValueError('current group must contain nonnegative proposals')
    return int(values.max()) + 2 * radius


def owned_sample_mask(groups, current_group, query_samples, *, complete_through, radius=RADIUS):
    """Label-free native input mask; refuse an uncertified future context.

    Feed this geometry to an ownership-aware model before decoding. Applying
    it after a global K prediction cannot recover the missing event locations.
    No existing trainer calls this experimental helper yet.
    """
    groups, q, radius = _groups(groups), _integers(query_samples, 'queries'), _radius(radius)
    if not isinstance(current_group, (int, np.integer)) or not 0 <= current_group < len(groups):
        raise ValueError('invalid current group')
    if not isinstance(complete_through, (int, np.integer)):
        raise ValueError('proposal completeness watermark must be an integer')
    required = required_proposal_watermark(groups[current_group], radius=radius)
    if complete_through < required:
        raise ValueError(f'incomplete proposal context: need {required}, got {complete_through}')
    if np.any(q > complete_through):
        raise ValueError('query beyond certified proposal context')
    available = [g[g <= complete_through] for g in groups]
    return owners_at_samples(available, q, radius=radius) == current_group
