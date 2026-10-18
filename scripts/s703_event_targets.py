"""TRAIN labels only; never imported by the causal inference head module."""
import numpy as np
from scripts.s613_train_strict_live import event_score


def component_targets(rows, truth):
    """Keep the baseline's own TP labels; learn rescues on its missed notes.

    A closer proposal must not relabel a correct native event as a negative.
    Ground truth is used exclusively for supervised fitting, never inference.
    """
    y = np.zeros(len(rows), np.uint8)
    native_ids = [i for i, r in enumerate(rows) if r['native']]
    rescue_ids = [i for i, r in enumerate(rows) if not r['native']]
    _, native_pairs = event_score([rows[i] for i in native_ids], truth)
    matched = {j for _, j in native_pairs}
    for i, _ in native_pairs:
        y[native_ids[i]] = 1
    missed = [r for i, r in enumerate(truth) if i not in matched]
    _, rescue_pairs = event_score([rows[i] for i in rescue_ids], missed)
    for i, _ in rescue_pairs:
        y[rescue_ids[i]] = 1
    return y
