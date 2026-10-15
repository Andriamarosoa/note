"""Causal consensus of native NOTE neural/tree heads, selected on dev only."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from causal_note.strict_live import NativeLiveHeads, NativeONNXLiveHeads, Selection
from scripts.s613_train_strict_live import load_source, real_notes, decode, event_score, summary
from scripts.s613_prepare_strict_live import digest, save


def run(root, neural_dir, tree_dir, out):
    if out.exists():
        raise FileExistsError("Immutable consensus experiment")
    assert not any((p / "test_predictions_before_labels.json").exists() for p in (neural_dir, tree_dir))
    protocol = json.loads((root / "protocol.json").read_text())
    neural = NativeLiveHeads(neural_dir / "selected_heads.npz")
    tree = NativeONNXLiveHeads(tree_dir / "selected_heads.npz")
    configs = [(weight, Selection(onset=on, active_on=active)) for weight in (.25, .5, .75)
               for on in (.45, .6, .75, .85) for active in (.5, .7)]
    scores = [[] for _ in configs]
    out.mkdir(parents=True)
    for spec in protocol["sources"]:
        if spec["split"] != "dev":
            continue
        ends, features, meta = load_source(root, spec)
        npred, k = neural.predict(features)
        tpred, tk = tree.predict(features)
        assert np.array_equal(k, tk)
        truth = real_notes(meta["labels"]["notes"], meta["prepared_samples"] / 44100)
        for i, (weight, selection) in enumerate(configs):
            p = np.power(np.clip(npred, 1e-8, 1), 1 - weight) * np.power(np.clip(tpred, 1e-8, 1), weight)
            events = decode(ends, p, k, selection, meta["prepared_samples"])
            score, _ = event_score(events, truth)
            scores[i].append(dict(source=spec["name"], corpus=spec["corpus"], **score))
        print("CONSENSUS DEV SOURCE", spec["name"], flush=True)
    results = [dict(tree_weight=weight, selection=asdict(config), overall=summary(items), by_corpus={c: summary([s for s in items if s["corpus"] == c])
               for c in ("GuitarSet", "URMP")}) for (weight, config), items in zip(configs, scores)]
    for row in results:
        print("CONSENSUS LOOP", json.dumps(row), flush=True)
    save(out / "development_loops.json", results)
    best = max(results, key=lambda row: row["overall"]["F1"])
    with np.load(neural_dir / "selected_heads.npz", allow_pickle=False) as z:
        arrays = {key: z[key].copy() for key in z.files}
    with np.load(tree_dir / "selected_heads.npz", allow_pickle=False) as z:
        for i in range(3):
            arrays[f"tree{i}"] = z[f"tree{i}"].copy()
    metadata = dict(neural.metadata, architecture="native_causal_hybrid", tree_weight=best["tree_weight"],
                    neural_model_sha256=neural.sha256, tree_model_sha256=tree.sha256,
                    selection=best["selection"], development_F1=best["overall"]["F1"])
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(out / "selected_heads.npz", **arrays)
    save(out / "selection_frozen.json", dict(best=best, model_filename="selected_heads.npz",
           model_sha256=digest((out / "selected_heads.npz").read_bytes()),
           protocol_sha256=digest((root / "protocol.json").read_bytes()), test_labels_read=False, total_development_loops=len(results)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--trees", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.neural, args.trees, args.out)
