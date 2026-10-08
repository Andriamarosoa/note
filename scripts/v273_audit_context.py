"""Select audit inputs explicitly without destroying the stored observations."""
from __future__ import annotations

import hashlib
import numpy as np

LOCAL_COLUMNS = (3, 4, 5, 7, 8)


def audit_context(inputs, mode):
    if mode not in ('global', 'local'):
        raise ValueError('Unknown audit context mode')
    count = inputs['member_votes'].shape[1]
    direct = 6*count+13
    features = np.array(inputs['group_features'], copy=True)
    if features.shape[-1] != direct+9:
        raise ValueError('Full group descriptors and nine audit columns required')
    if mode == 'global':
        features[..., [direct+k for k in LOCAL_COLUMNS]] = 0
    audits = features[..., direct:]
    local = audits[..., LOCAL_COLUMNS]
    evidence = dict(mode=mode, local_columns=list(LOCAL_COLUMNS),
        local_values_nonzero=int(np.count_nonzero(local)),
        rows_with_local_evidence=int(np.any(local != 0, axis=(1, 2)).sum()),
        effective_audit_sha256=hashlib.sha256(np.ascontiguousarray(audits).tobytes()).hexdigest())
    return dict(inputs, group_features=features), evidence
