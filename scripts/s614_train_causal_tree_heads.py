"""Second native head family: small boosted trees, unchanged causal PCM ABI.

Uses only S613 train/dev sources. The previously reserved test stays closed.
Preserves the first model family and all of its development results.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from causal_note.strict_live import NativeLiveHeads, NativeTreeLiveHeads, Selection, SCHEMA, count_features
from scripts.s613_train_strict_live import build_training, load_source, targets, real_notes, decode, event_score, summary
from scripts.s613_prepare_strict_live import digest, save


def train(root, previous, out):
    assert not (previous / "test_predictions_before_labels.json").exists()
    if out.exists():
        raise FileExistsError("Immutable second-family experiment")
    out.mkdir(parents=True)
    protocol = json.loads((root / "protocol.json").read_text())
    x, y, weights, cx, cy, proof = build_training(root, protocol)
    del cx, cy
    count_head = NativeLiveHeads(previous / "selected_heads.npz")
    rng = np.random.default_rng(613)
    cp = []
    for spec in protocol["sources"]:
        if spec["split"] != "train":
            continue
        ends, features, meta = load_source(root, spec)
        target, _ = targets(ends, meta["labels"]["notes"])
        rate = np.where(np.maximum(target[..., 0], target[..., 2]) > 0, 1., np.where(target[..., 1] > 0, .5, .08))
        chosen = rng.random(rate.shape) < rate
        k = count_head._net("count", count_features(features))
        k = np.exp(k - k.max(-1, keepdims=True)); k /= k.sum(-1, keepdims=True)
        cp.append(np.broadcast_to(k[:, None], (len(features), 49, 7))[chosen].copy())
    cp = np.concatenate(cp)
    assert len(cp) == len(x)
    x = np.concatenate((x, cp), -1)
    del cp, features
    # Fixed uniform thinning, with the existing inverse sampling weights.
    ids = np.random.default_rng(614).choice(len(x), min(len(x), 600000), replace=False)
    x, y, weights = x[ids], y[ids], weights[ids]
    models = []
    for column, balance in enumerate((6., 1.5, 6.)):
        model = HistGradientBoostingClassifier(max_iter=140, max_leaf_nodes=23,
                  learning_rate=.09, min_samples_leaf=40, l2_regularization=1.,
                  early_stopping=False, random_state=614 + column)
        before = time.perf_counter()
        model.fit(x, y[:, column].astype(int), sample_weight=weights * np.where(y[:, column] > 0, balance, 1.))
        models.append(model)
        print("TREE HEAD", column, "seconds", time.perf_counter() - before, flush=True)
    metadata = dict(count_head.metadata)
    metadata.update(architecture="native_causal_tree_heads", tree_iterations=140, tree_leaves=23,
                    previous_count_model_sha256=count_head.sha256, selection=asdict(Selection()))
    bundle = dict(schema=SCHEMA, architecture="native_causal_tree_heads", metadata=metadata,
                  count_arrays={k: v for k, v in count_head.arrays.items() if k.startswith("count.")}, forests=models)
    path = out / "heads_tree.joblib"
    joblib.dump(bundle, path, compress=3)
    save(out / "training_sources.json", proof)
    tune(root, out, protocol, bundle)


def tune(root, out, protocol, bundle):
    assert not (out / "test_predictions_before_labels.json").exists()
    heads = NativeTreeLiveHeads(out / "heads_tree.joblib")
    configs = [Selection(onset=on, active_on=active, confirmation_frames=confirm)
               for on, active, confirm in ((.3,.3,1),(.5,.3,1),(.7,.3,1),(.85,.3,1),
                                           (.3,.5,1),(.5,.5,1),(.7,.5,1),(.85,.5,1),
                                           (.5,.7,1),(.7,.7,1),(.85,.7,1),(.7,.5,2))]
    scores = [[] for _ in configs]
    for spec in protocol["sources"]:
        if spec["split"] != "dev":
            continue
        ends, features, meta = load_source(root, spec)
        before = time.perf_counter()
        p, k = heads.predict(features)
        truth = real_notes(meta["labels"]["notes"], meta["prepared_samples"] / 44100)
        for i, config in enumerate(configs):
            events = decode(ends, p, k, config, meta["prepared_samples"])
            score, _ = event_score(events, truth)
            scores[i].append(dict(source=spec["name"], corpus=spec["corpus"], **score))
        print("TREE DEV SOURCE", spec["name"], "seconds", time.perf_counter() - before, flush=True)
    results = [dict(selection=asdict(config), overall=summary(items), by_corpus={c: summary([s for s in items if s["corpus"] == c])
               for c in ("GuitarSet", "URMP")}) for config, items in zip(configs, scores)]
    for row in results:
        print("TREE DEV LOOP", json.dumps(row), flush=True)
    save(out / "development_loops.json", results)
    best = max(results, key=lambda row: row["overall"]["F1"])
    bundle["metadata"]["selection"] = best["selection"]
    bundle["metadata"]["development_F1"] = best["overall"]["F1"]
    joblib.dump(bundle, out / "selected_heads.joblib", compress=3)
    save(out / "selection_frozen.json", dict(best=best, model_filename="selected_heads.joblib", model_sha256=digest((out / "selected_heads.joblib").read_bytes()),
           protocol_sha256=digest((root / "protocol.json").read_bytes()), test_labels_read=False, total_development_loops=len(results)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    train(args.root, args.previous, args.out)
