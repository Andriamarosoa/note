"""Train an OPTIONAL genuine-upper-onset logistic head (never native S613).

Inputs come from build_live_harmonic_pairs, with train/dev GROUPS disjoint.
A checkpoint is created only if the held-out development classification has
enough positive/negative examples. Reported scores are PAIR classification,
not NOTE F1, Exact-K, or end-to-end latency. Both datasets must exclude the
frozen S613 test set and any hidden prospective test recording.

Example:
  python -m scripts.train_live_harmonic_expert \\
    --train-pairs data/pairs/train_harmonic_pairs.npz \\
    --dev-pairs data/pairs/dev_harmonic_pairs.npz \\
    --out model/experimental/harmonic_expert.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from causal_note.harmonic_expert import SCHEMA, FEATURE_NAMES, N_FEATURES


def load_dataset(path):
    with np.load(path, allow_pickle=False) as z:
        X = np.asarray(z["X"], np.float32)
        y = np.asarray(z["y"], np.int32)
        groups = set(str(v) for v in z["groups"].tolist())
        names = tuple(z["feature_names"].tolist())
    if X.ndim != 2 or X.shape[1] != N_FEATURES or y.shape != (len(X),):
        raise ValueError(f"Bad harmonic-pair dataset dimensions: {path}")
    if names != FEATURE_NAMES or not (np.isfinite(X).all()
                                      and np.isin(y, (0, 1)).all()):
        raise ValueError(f"Invalid harmonic training samples: {path}")
    return X, y, groups


def train(train_path, dev_path, output, *, min_recall=0.99):
    from sklearn.linear_model import LogisticRegression
    xtr, ytr, train_groups = load_dataset(train_path)
    xdev, ydev, dev_groups = load_dataset(dev_path)
    if train_groups & dev_groups:
        raise ValueError("Train/dev source groups overlap; cannot fit")
    if len(train_groups) < 3 or len(dev_groups) < 2:
        raise ValueError("Need >=3 train source groups and >=2 disjoint dev groups")
    if min(int(ytr.sum()), int((1 - ytr).sum())) < 25:
        raise ValueError("Insufficient genuine and false upper onsets in training")
    if min(int(ydev.sum()), int((1 - ydev).sum())) < 15:
        raise ValueError("Insufficient independent genuine and false upper onsets in development")
    mean = xtr.mean(0, dtype=np.float64).astype(np.float32)
    scale = xtr.std(0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, 1e-4)
    fitted = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1500,
                                random_state=613)
    fitted.fit((xtr - mean) / scale, ytr)
    W = np.asarray(fitted.coef_[0], np.float32)
    bias = float(fitted.intercept_[0])
    logits = ((xdev - mean) / scale) @ W + bias
    pd = 1. / (1. + np.exp(-np.clip(logits, -30, 30)))
    # Select the MOST aggressive veto that preserves the requested recall
    # of actual upper note attacks on disjoint DEV. This is not proof of
    # zero regression: must still verify complete NoteOn/Exact-K replays.
    feasible = []
    for threshold in (0.005, 0.01, 0.02, 0.03, 0.05, 0.075,
                      0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
        true_genuine = ydev == 1
        negative = ~true_genuine
        recall = float(np.mean(pd[true_genuine] >= threshold))
        suppress_fpr = float(np.mean(pd[negative] < threshold))
        if recall >= min_recall:
            feasible.append((suppress_fpr, threshold, recall))
    if not feasible:
        raise ValueError(
            "No safe-enough threshold on dev: expert is NOT fit for guarded promotion"
        )
    # Prefer lower threshold when the same number of false onsets is vetoed.
    veto_recall, threshold, genuine_recall = sorted(
        feasible, key=lambda v: (v[0], -v[1]))[-1]
    if output.exists():
        raise FileExistsError(f"Checkpoint already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, schema=np.asarray(SCHEMA),
             feature_names=np.asarray(FEATURE_NAMES),
             mean=mean, scale=scale, weights=W,
             bias=np.asarray(bias, np.float32),
             recommended_threshold=np.asarray(threshold, np.float32))
    report = dict(
        train_sources=len(train_groups), dev_sources=len(dev_groups),
        train_samples=len(ytr), train_genuine=int(ytr.sum()),
        dev_samples=len(ydev), dev_genuine=int(ydev.sum()),
        threshold=threshold, genuine_dev_recall=genuine_recall,
        false_onset_veto_fraction_on_dev=veto_recall,
        status="EXPERIMENTAL_NOT_PROMOTED",
        warning="Pair recall is not Exact-K or NoteOn F1. Validate real polyphony and time/CPU before enabling guard.",
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pairs", type=Path, required=True)
    parser.add_argument("--dev-pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-genuine-recall", type=float, default=0.99)
    args = parser.parse_args()
    if not 0.0 < args.min_genuine_recall <= 1.0:
        parser.error("min-genuine-recall must be inside (0,1]")
    print(json.dumps(train(args.train_pairs, args.dev_pairs, args.out,
                           min_recall=args.min_genuine_recall), indent=2))


if __name__ == "__main__":
    main()
