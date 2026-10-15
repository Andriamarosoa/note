"""Export NOTE-trained trees to CPU ONNX and verify development parity."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from threadpoolctl import threadpool_limits
from causal_note.strict_live import SCHEMA, NativeTreeLiveHeads, NativeONNXLiveHeads, Selection, StrictLiveStream, HOP
from scripts.s613_prepare_strict_live import digest, save
from scripts.s613_train_strict_live import load_source, decode


def export(root, experiment):
    assert not (experiment / "test_predictions_before_labels.json").exists()
    frozen = json.loads((experiment / "selection_frozen.json").read_text())
    native = NativeTreeLiveHeads(experiment / "selected_heads.joblib", expected_sha256=frozen["model_sha256"])
    path = experiment / "selected_heads.npz"
    metadata = dict(native.metadata, architecture="native_causal_tree_heads_onnx", sklearn_source_sha256=native.sha256)
    arrays = dict(native.arrays, schema=np.asarray(SCHEMA), metadata=np.asarray(json.dumps(metadata)))
    for i, model in enumerate(native.forests):
        graph = convert_sklearn(model, initial_types=[("features", FloatTensorType([None, 133]))],
                                options={id(model): {"zipmap": False}}, target_opset=17)
        # HGB compares float32 inputs promoted to double against double
        # thresholds. skl2onnx 1.18 rounds these thresholds to nearest float,
        # which can change an equality branch. Floor to the largest float32
        # <= the original threshold: x <= double_t iff x <= floor32(t).
        # This repairs export semantics; it does not alter fitted trees.
        for node in graph.graph.node:
            if node.op_type != "TreeEnsembleClassifier":
                continue
            attributes = {a.name: a for a in node.attribute}
            values = attributes["nodes_values"]
            for j, (tree_id, node_id, mode) in enumerate(zip(attributes["nodes_treeids"].ints,
                    attributes["nodes_nodeids"].ints, attributes["nodes_modes"].strings)):
                if mode != b"BRANCH_LEQ":
                    continue
                threshold = float(model._predictors[tree_id][0].nodes[node_id]["num_threshold"])
                value = np.float32(threshold)
                if float(value) > threshold:
                    value = np.nextafter(value, np.float32(-np.inf))
                values.floats[j] = float(value)
        arrays[f"tree{i}"] = np.frombuffer(graph.SerializeToString(), np.uint8)
    np.savez_compressed(path, **arrays)
    runtime = NativeONNXLiveHeads(path)
    protocol = json.loads((root / "protocol.json").read_text())
    proof = []
    with threadpool_limits(limits=3):
        for spec in protocol["sources"]:
            if spec["split"] != "dev":
                continue
            ends, features, meta = load_source(root, spec)
            # Export equivalence probe: bounded real-audio prefixes from all
            # dev sources. Final consensus selection reruns every dev frame
            # through ONNX; held-out scoring uses only the deployed runtime.
            ends, features = ends[:512], features[:512]
            old, oldk = native.predict(features)
            new, newk = runtime.predict(features)
            error = float(np.max(np.abs(old - new)))
            assert error < 3e-6 and np.array_equal(oldk, newk), (spec["name"], error, float(np.max(np.abs(oldk-newk))))
            config = Selection(**metadata["selection"])
            a = decode(ends, old, oldk, config, int(ends[-1]))
            b = decode(ends, new, newk, config, int(ends[-1]))
            assert a == b, "ONNX changed a selected development event"
            proof.append(dict(source=spec["name"], frames=len(features), seconds=int(ends[-1]) / 44100,
                              max_probability_error=error, events=len(a), exact_events=True))
            print("ONNX PARITY", json.dumps(proof[-1]), flush=True)
    save(experiment / "onnx_export_parity.json", dict(native_sha256=native.sha256, onnx_sha256=runtime.sha256, sources=proof))
    save(experiment / "selection_native_before_onnx.json", frozen)
    frozen.update(model_filename="selected_heads.npz", model_sha256=runtime.sha256, native_model_sha256=native.sha256,
                  onnx_parity_sha256=digest((experiment / "onnx_export_parity.json").read_bytes()))
    save(experiment / "selection_frozen.json", frozen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    export(args.root, args.experiment)
