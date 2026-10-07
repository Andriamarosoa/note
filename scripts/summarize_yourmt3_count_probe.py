"""Cross-fold linear probes for frozen YourMT3+ Exact-K representations.

For each held-out fold, fit a fixed linear classifier on the other three folds.
The full autoregressive YourMT3+ predictions are only replayed for comparison.
No hyperparameter is selected on held-out labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.yourmt3_exactk_common import FOLDS, metrics, paired, require, write_json

SEED = 17064


def load_feature_folds(root: Path):
    out = {}
    for fold in FOLDS:
        paths = list(root.rglob(f"features-fold-{fold}.npz"))
        require(len(paths) == 1, f"expected one feature file for fold {fold}, got {paths}")
        with np.load(paths[0], allow_pickle=False) as z:
            out[fold] = {k: z[k] for k in z.files}
    return out


def load_full_predictions(root: Path):
    records = []
    for path in root.rglob("predictions.npz"):
        with np.load(path, allow_pickle=False) as z:
            if "predicted" not in z.files or "global_index" not in z.files:
                continue
            records.append((z["global_index"].astype(np.int64), z["predicted"].astype(np.int32)))
    require(len(records) == 4, f"expected four full-decoder prediction files, got {len(records)}")
    ids = np.concatenate([x[0] for x in records])
    pred = np.concatenate([x[1] for x in records])
    order = np.argsort(ids)
    ids, pred = ids[order], pred[order]
    require(len(np.unique(ids)) == len(ids), "duplicate full-decoder row ids")
    return ids, pred


def fit_probe(train_x, train_y, test_x):
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        class_weight="balanced",
        max_iter=2000,
        tol=1e-4,
        shuffle=True,
        random_state=SEED,
        average=True,
        n_jobs=1,
    )
    clf.fit(train_x, train_y)
    return clf.predict(test_x).astype(np.int32)


def format_score(rec):
    return f"{rec['correct']}/{rec['rows']} ({100.0 * rec['exact']:.4f}%)"


def main(a):
    folds = load_feature_folds(a.features)
    full_ids, full_pred = load_full_predictions(a.full)
    all_ids = np.concatenate([folds[f]["global_index"].astype(np.int64) for f in FOLDS])
    all_y = np.concatenate([folds[f]["k"].astype(np.int32) for f in FOLDS])
    all_base = np.concatenate([folds[f]["baseline"].astype(np.int32) for f in FOLDS])
    require(len(all_ids) == 59309 and len(np.unique(all_ids)) == 59309, "cohort identity drift")

    full_lookup = dict(zip(full_ids.tolist(), full_pred.tolist()))
    require(all(int(i) in full_lookup for i in all_ids), "full-decoder prediction coverage drift")
    all_full = np.asarray([full_lookup[int(i)] for i in all_ids], np.int32)

    aggregate_pred = {"encoder": [], "predecoder": []}
    aggregate_y, aggregate_base, aggregate_full, aggregate_ids = [], [], [], []
    by_fold = {}

    for held in FOLDS:
        test = folds[held]
        train_folds = [f for f in FOLDS if f != held]
        train_y = np.concatenate([folds[f]["k"].astype(np.int32) for f in train_folds])
        test_y = test["k"].astype(np.int32)
        test_base = test["baseline"].astype(np.int32)
        test_ids = test["global_index"].astype(np.int64)
        require(not set(test_ids.tolist()) & set(np.concatenate([folds[f]["global_index"] for f in train_folds]).tolist()),
                "train/test row leakage")

        fold_rec = {}
        for representation in ("encoder", "predecoder"):
            train_x = np.concatenate([folds[f][representation].astype(np.float32) for f in train_folds], axis=0)
            test_x = test[representation].astype(np.float32)
            pred = fit_probe(train_x, train_y, test_x)
            aggregate_pred[representation].append(pred)
            fold_rec[representation] = {
                "metrics": metrics(test_y, pred),
                "paired_vs_freeze": paired(test_y, test_base, pred),
            }

        test_full = np.asarray([full_lookup[int(i)] for i in test_ids], np.int32)
        fold_rec["freeze_local_combo"] = metrics(test_y, test_base)
        fold_rec["full_yourmt3_decoder"] = {
            "metrics": metrics(test_y, test_full),
            "paired_vs_freeze": paired(test_y, test_base, test_full),
        }
        by_fold[str(held)] = fold_rec
        aggregate_y.append(test_y)
        aggregate_base.append(test_base)
        aggregate_full.append(test_full)
        aggregate_ids.append(test_ids)

    y = np.concatenate(aggregate_y)
    base = np.concatenate(aggregate_base)
    full = np.concatenate(aggregate_full)
    ids = np.concatenate(aggregate_ids)
    encoder = np.concatenate(aggregate_pred["encoder"])
    predecoder = np.concatenate(aggregate_pred["predecoder"])
    require(len(y) == 59309, "aggregate row count drift")

    report = {
        "status": "completed",
        "protocol": {
            "folds": list(FOLDS),
            "held_out_training": "train on three folds, evaluate on fourth",
            "probe": "StandardScaler + fixed SGDClassifier(log_loss, balanced, alpha=1e-4, average=True)",
            "hyperparameter_selection_on_test": False,
            "full_decoder_source_run": 37605163312,
            "fold3_used": False,
            "player05_used": False,
        },
        "aggregate": {
            "freeze_local_combo": metrics(y, base),
            "encoder_linear_probe": {
                "metrics": metrics(y, encoder),
                "paired_vs_freeze": paired(y, base, encoder),
            },
            "predecoder_linear_probe": {
                "metrics": metrics(y, predecoder),
                "paired_vs_freeze": paired(y, base, predecoder),
            },
            "full_yourmt3_decoder": {
                "metrics": metrics(y, full),
                "paired_vs_freeze": paired(y, base, full),
            },
        },
        "by_fold": by_fold,
    }

    a.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        a.output / "predictions.npz",
        global_index=ids,
        k=y,
        freeze_local_combo=base,
        encoder_linear_probe=encoder,
        predecoder_linear_probe=predecoder,
        full_yourmt3_decoder=full,
    )
    write_json(a.output / "report.json", report)

    agg = report["aggregate"]
    rows = [
        ("freeze_local_combo", agg["freeze_local_combo"]),
        ("encoder linear probe", agg["encoder_linear_probe"]["metrics"]),
        ("pre-decoder linear probe", agg["predecoder_linear_probe"]["metrics"]),
        ("full YourMT3+ decoder", agg["full_yourmt3_decoder"]["metrics"]),
    ]
    lines = [
        "# YourMT3+ representation ablation for Exact-K",
        "",
        "All probes are cross-fold: three folds train, the fourth is held out. No held-out tuning.",
        "The full YourMT3+ row predictions are replayed from run 37605163312.",
        "",
        "| Stage | Global Exact-K | Poly K2-K6 | Poly under | Poly over |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, rec in rows:
        poly = rec["poly"]
        lines.append(
            f"| {name} | {100*rec['exact']:.4f}% | {100*poly['exact']:.4f}% | {poly['under']} | {poly['over']} |"
        )
    lines += ["", "## Per-K Exact-K", "", "| K | freeze | encoder probe | pre-decoder probe | full decoder |",
              "|---:|---:|---:|---:|---:|"]
    for k in range(7):
        vals = [
            agg["freeze_local_combo"]["by_k"][str(k)]["exact"],
            agg["encoder_linear_probe"]["metrics"]["by_k"][str(k)]["exact"],
            agg["predecoder_linear_probe"]["metrics"]["by_k"][str(k)]["exact"],
            agg["full_yourmt3_decoder"]["metrics"]["by_k"][str(k)]["exact"],
        ]
        lines.append("| " + str(k) + " | " + " | ".join(f"{100*v:.3f}%" for v in vals) + " |")

    lines += [
        "",
        "Interpretation rule:",
        "- If encoder probe is already near the full decoder, the main gain is in the long-context Perceiver-TF/MoE representation.",
        "- If pre-decoder jumps over encoder, the 13-channel learned projection carries substantial count information.",
        "- If both probes remain far below the full decoder, autoregressive Multi-T5 event decoding is the dominant contributor.",
    ]
    (a.output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--full", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
