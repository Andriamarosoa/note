"""V28.0-H: frozen prior correction and equal fusion on a second internal split.

Discovery reuses G predictions. H trains two fresh models for the two epochs
selected by G, then infers fold 2 once per model after both states are frozen.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_v280_balanced_internal as g

f, e = g.f, g.e
EPOCHS = 2
G_SOURCE = {
    "run_id": 34684806334, "run_attempt": 1,
    "head_sha": "bff41aec998a798055687c94705629eb98626aee",
    "comparison_json_sha256": "680e54d783bae6894d42058fbe0f3f9845a70b2c64914c4124910ae78edbe2a9",
    "implementation_sha256": "9f4180e23ded17493777694f2b112e48ffc74e73af081fb862bc488b0de583fb",
    "artifacts": {
        "v280-g-internal-comparison": {"id": 10304691260, "digest": "sha256:8508188bffde11afda96690b2ba3309f0d0e9fc6b08c3592fe5b0368f65fe62d"},
        "v280-g-control-2": {"id": 10299622000, "digest": "sha256:b2ee3f974ba64f4e7dd4541c185b7353b3fc09d6323a2021d4c6c2f364d903e9"},
        "v280-g-balanced-2": {"id": 10304227222, "digest": "sha256:c1523848c628ecaffde7dbe4d1ffb77cc65f9f7de983e45d708d2ef6f52c6b26"},
    },
}
CONTRACT = {
    "experiment": "v280_h_prior_corrected_fusion", "schema_version": 1,
    "fit_folds": [1, 3, 4], "validation_fold": 2, "excluded_outer_fold": 0,
    "epochs": EPOCHS, "epoch_choice": "both_G_best_checkpoints_were_epoch_2",
    "checkpoint_selection_on_H": False, "validation_during_training": False,
    "architecture_per_arm": "harmonic", "arms": list(g.ARMS),
    "seed": e.SEED, "batch_size": e.BATCH_SIZE, "learning_rate": 2e-4,
    "loss_weights": e.LOSS_WEIGHTS, "sampling": f.CONTRACT["sampling"],
    "augmentation": "none", "initialization": f.CONTRACT["initialization"],
    "weighting": "control=1; balanced=N_fit/(2*N_fit_in_K_group)",
    "decoder": "0.5*p_control + 0.5*normalize(p_balanced/fit_class_weights)",
    "prior_correction_exponent": 1.0, "fusion_coefficient": 0.5,
    "learned_calibrator": False, "threshold_tuning": False,
    "validation_inference_passes_per_arm": 1,
    "gate": "poly_correct_strictly_up AND global_correct_not_down AND false_poly_not_up",
    "scope": "second_internal_development_partition_after_G_discovery",
    "independent_validation_claim": False, "automatic_promotion": False,
    "outer_evaluation": False, "automatic_next_training": False,
}


class FusionError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise FusionError(message)


def load_data(directory, *, features_required=True):
    require(e.smoke._sha256_file(ROOT / "scripts/train_v280_balanced_internal.py") ==
            G_SOURCE["implementation_sha256"], "G implementation changed")
    return g.load_data(directory, features_required=features_required)


def roles(arrays):
    # Reuse F's composition-disjointness check, but define H roles explicitly.
    f.partitions(arrays["row_fold"], arrays["member"], 0)
    fold = arrays["row_fold"]
    return {"fit": np.flatnonzero(np.isin(fold, CONTRACT["fit_folds"])),
            "validation": np.flatnonzero(fold == 2), "outer": np.flatnonzero(fold == 0)}


def partition_record(arrays):
    return {name: {"rows": len(rows), "indices_sha256": e.digest_array(arrays["global_index"][rows])}
            for name, rows in roles(arrays).items()}


def weighting(arrays, arm):
    require(arm in g.ARMS, "unknown arm")
    fit = roles(arrays)["fit"]
    k = arrays["target_cardinality"][fit]
    n, poly = len(k), int(np.sum(k >= 2))
    require(0 < poly < n, "both fit groups required")
    nonpoly = n - poly
    lo, hi = (1.0, 1.0) if arm == "control" else (n / (2 * nonpoly), n / (2 * poly))
    weights = np.where(k >= 2, hi, lo).astype(np.float32)
    return {"fit_rows": n, "fit_poly_rows": poly, "fit_nonpoly_rows": nonpoly,
            "nonpoly_weight": lo, "poly_weight": hi,
            "fit_weight_sha256": e.digest_array(weights)}


def normalize(probability):
    p = np.asarray(probability, dtype=np.float64)
    require(p.ndim == 2 and np.isfinite(p).all() and np.all(p >= 0), "invalid probability matrix")
    total = p.sum(axis=1, keepdims=True)
    require(np.all(total > 0), "empty probability row")
    return p / total


def correct_prior(probability, spec, exponent=1.0):
    require(probability.ndim == 2 and probability.shape[1] == 7, "seven count classes required")
    weights = np.asarray([spec["nonpoly_weight"]] * 2 + [spec["poly_weight"]] * 5)
    require(np.isfinite(weights).all() and np.all(weights > 0), "positive finite weights required")
    return normalize(probability / weights ** exponent)


def fuse(control, balanced, spec):
    require(control.shape == balanced.shape, "unaligned posterior shapes")
    return .5 * normalize(control) + .5 * correct_prior(balanced, spec)


def score(k, p):
    result = e.count_metrics(k, p)
    prediction = p.argmax(axis=1)
    result["false_poly_rows"] = int(np.sum((k < 2) & (prediction >= 2)))
    result["nonpoly_rows"] = int(np.sum(k < 2))
    result["poly_undercount_rows"] = int(np.sum((k >= 2) & (prediction < k)))
    return result


def passes_gate(control, candidate):
    return (candidate["poly_correct"] > control["poly_correct"] and
            candidate["correct"] >= control["correct"] and
            candidate["false_poly_rows"] <= control["false_poly_rows"])


def table(methods):
    lines = ["| Method | Poly Exact-K | Global Exact-K | False polyphonic rows | Poly NLL |",
             "|---|---:|---:|---:|---:|"]
    for name, m in methods.items():
        nll = "n/a" if m["poly_nll"] is None else f"{m['poly_nll']:.5f}"
        lines.append(f"| {name} | {m['poly_exact_k']:.4%} | {m['exact_k']:.4%} | {m['false_poly_rows']} | {nll} |")
    return lines


def audit_g(args):
    _, arrays, _ = load_data(args.prepared_dir, features_required=False)
    source = args.input_dir
    require(e.smoke._sha256_file(source / "comparison/comparison.json") ==
            G_SOURCE["comparison_json_sha256"], "G comparison changed")
    report = f.read_json(source / "comparison/comparison.json")
    require(report["contract"] == g.CONTRACT and report["status"] == "complete", "invalid G contract")
    predictions, states = {}, {}
    for arm in g.ARMS:
        directory = source / arm
        state = f.read_json(directory / "state.json")
        require(state == report["arms"][arm], "G state differs from audited comparison")
        for name, expected in {"arm": arm, "epochs_completed": 12, "chunk": 2, "epoch_budget": 12,
                               "source_head_sha": G_SOURCE["head_sha"], "source_run_id": str(G_SOURCE["run_id"]),
                               "contract": g.CONTRACT, "source": g.SOURCE,
                               "weighting": g.weighting(arrays, arm)[1]}.items():
            require(state.get(name) == expected, f"G state mismatch: {name}")
        f.verify_files(directory, state["files"])
        f.validate_training_history(state, arrays)
        best = f.choose_epoch(state["history"])
        require(best["epoch"] == state["best_epoch"] == EPOCHS, "G fixed epoch evidence changed")
        for prefix, expected in (("best", best), ("last", state["history"][-1])):
            p = f.read_npz(directory / (prefix + "-validation.npz"))
            m, pb = g.verify_prediction(p, arrays)
            require(f.same_count_metrics(m, expected["validation"]) and
                    f.same_count_metrics(pb, expected["poibin_validation"]), "G metrics changed")
            if prefix == "best":
                predictions[arm] = p
        states[arm] = state
    for name in ("initial_all_weights_sha256", "initial_shared_weights_sha256", "partitions", "optimizer_updates"):
        require(states["control"][name] == states["balanced"][name], "G pairing changed")
    c, b = (predictions[arm]["probability"].astype(np.float64) for arm in g.ARMS)
    k, members = predictions["control"]["k"], predictions["control"]["member"]
    spec = states["balanced"]["weighting"]
    methods = {"control": c, "balanced": b, "inverse_weights": correct_prior(b, spec),
               "soft_hierarchical": np.concatenate([c[:, :2], c[:, 2:].sum(axis=1, keepdims=True) * normalize(b[:, 2:])], axis=1),
               "mixture_half": fuse(c, b, spec)}
    for alpha in (.25, .5, .75):
        methods[f"prior_alpha_{alpha}"] = correct_prior(b, spec, alpha)
    routed = c.argmax(axis=1)
    eligible = routed >= 2
    routed[eligible] = b[eligible, 2:].argmax(axis=1) + 2
    methods["hard_routed"] = np.eye(7)[routed]
    metrics = {name: score(k, p) for name, p in methods.items()}
    for name in ("nll", "poly_nll", "brier"):
        metrics["hard_routed"][name] = None  # Hard decisions are not calibrated posteriors.
    selected = "mixture_half"  # Frozen after this disclosed nine-candidate discovery.
    bootstrap = f.track_bootstrap(members, k, c.argmax(axis=1), methods[selected].argmax(axis=1))
    result = {"status": "complete", "source": G_SOURCE, "data_source": g.SOURCE,
              "scope": "posthoc_nine_candidate_discovery_on_G_selected_validation_predictions",
              "independent_validation_claim": False, "new_neural_training": False,
              "methods": metrics, "selected_for_H": selected, "H_contract": CONTRACT,
              "paired_track_bootstrap_selected": bootstrap,
              "selection_multiplicity_adjusted": False,
              "poly_delta_pp": 100 * (metrics[selected]["poly_exact_k"] - metrics["control"]["poly_exact_k"]),
              "global_delta_pp": 100 * (metrics[selected]["exact_k"] - metrics["control"]["exact_k"]),
              "passes_development_gate": passes_gate(metrics["control"], metrics[selected]),
              "parameters_per_arm": states["control"]["parameters"],
              "parameters_fusion": sum(s["parameters"] for s in states.values()),
              "reference_promoted": False}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    e.smoke._atomic_json(args.output_dir / "discovery.json", result)
    lines = ["# V28.0-H — discovery on frozen G outputs", "",
             "Exploratory results on G's already-selected internal checkpoints, not independent confirmation.", "",
             *table(metrics), "", f"Frozen candidate: {selected}; equal fusion after full inverse-weight correction.",
             f"Poly delta {result['poly_delta_pp']:+.4f} pp; global delta {result['global_delta_pp']:+.4f} pp.",
             f"Descriptive paired track interval: [{bootstrap['lower_95_pp']:.4f}, {bootstrap['upper_95_pp']:.4f}] pp.",
             "Nine candidates were examined; the interval is not corrected for selection.",
             "Fusion uses two 110402-parameter networks (220804 total), with two forward passes.",
             "H freezes this method and epoch 2 before evaluating a second internal split. V27.3 remains the reference."]
    (args.output_dir / "discovery.md").write_text("\n".join(lines) + "\n")
    e.emit("G_discovery_audited", poly_delta_pp=result["poly_delta_pp"], global_delta_pp=result["global_delta_pp"])


def preflight(args):
    _, arrays, _ = load_data(args.prepared_dir)
    specs = {arm: weighting(arrays, arm) for arm in g.ARMS}
    require(specs["control"]["fit_rows"] == 45640 and specs["control"]["fit_poly_rows"] == 5594, "H fit counts changed")
    require(len(roles(arrays)["validation"]) == 15282, "H validation identities changed")
    e.smoke._atomic_json(args.output, {"status": "complete", "contract": CONTRACT,
        "data_source": g.SOURCE, "discovery_source": G_SOURCE, **f.provenance(),
        "partitions": partition_record(arrays), "weighting": specs})
    e.emit("H_preflight_complete", weighting=specs, training_started=False)


def validate_state(directory, arrays, arm):
    state = f.read_json(directory / "state.json")
    for name, expected in {"status": "complete", "arm": arm, "contract": CONTRACT,
                           "data_source": g.SOURCE, "discovery_source": G_SOURCE,
                           "epochs_completed": EPOCHS, "weighting": weighting(arrays, arm),
                           "partitions": partition_record(arrays), "validation_inference_passes": 0,
                           "parameters": e.expected_parameter_count(harmonic=True), **f.provenance()}.items():
        require(state.get(name) == expected, f"H training state mismatch: {name}")
    f.verify_files(directory, state["files"])
    fit = roles(arrays)["fit"]
    groups = np.asarray([e.group_stem(str(m)) for m in arrays["member"][fit]])
    steps = (len(fit) + e.BATCH_SIZE - 1) // e.BATCH_SIZE
    require([h["epoch"] for h in state["history"]] == list(range(1, EPOCHS + 1)), "incomplete H history")
    for h in state["history"]:
        order = fit[e.epoch_order(groups, arrays["target_cardinality"][fit], epoch=h["epoch"])]
        require(h["updates"] == h["epoch"] * steps and
                h["training_order_sha256"] == e.digest_array(arrays["global_index"][order]), "H coverage changed")
        require(np.isfinite(list(h["train_losses"].values())).all(), "nonfinite H losses")
        require("validation" not in h, "H validation was inspected during training")
    require(state["optimizer_updates"] == EPOCHS * steps, "H optimizer budget changed")
    return state


def train(args):
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    features, arrays, _ = load_data(args.prepared_dir)
    fit = roles(arrays)["fit"]
    spec = weighting(arrays, args.arm)
    model, shared_sha = e.initialize_arm("harmonic")
    initial_sha = g.full_initial_digest(model)
    model.optimizer.build(model.trainable_variables)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    groups = np.asarray([e.group_stem(str(m)) for m in arrays["member"][fit]])
    history = []
    e.emit("H_training_started", arm=args.arm, epoch_budget=EPOCHS, fit_rows=len(fit), validation_inference=False)
    for epoch in range(1, EPOCHS + 1):
        start, losses = time.monotonic(), {}
        order = fit[e.epoch_order(groups, arrays["target_cardinality"][fit], epoch=epoch)]
        for offset in range(0, len(fit), e.BATCH_SIZE):
            rows = order[offset:offset + e.BATCH_SIZE]
            current = model.train_on_batch(np.asarray(features[rows], dtype=np.float32),
                {name: arrays["target_" + name][rows] for name in e.LOSS_WEIGHTS},
                sample_weight=g.batch_weights(arrays, rows, spec), return_dict=True)
            require(np.isfinite(list(current.values())).all(), "nonfinite H training loss")
            for name, value in current.items():
                losses[name] = losses.get(name, 0.0) + float(value) * len(rows)
            updates = int(model.optimizer.iterations.numpy())
            if offset == 0 or updates % 50 == 0:
                e.emit("H_training_batch_completed", arm=args.arm, epoch=epoch, updates=updates)
        history.append({"epoch": epoch, "updates": updates,
            "training_order_sha256": e.digest_array(arrays["global_index"][order]),
            "train_losses": {name: value / len(fit) for name, value in losses.items()},
            "epoch_seconds": time.monotonic() - start})
        optimizer = f.save_training_state(model, args.output_dir)
        e.smoke._atomic_json(args.output_dir / "progress.json", {"arm": args.arm, "history": history, "contract": CONTRACT})
        e.emit("H_epoch_completed", arm=args.arm, epoch=epoch, updates=updates, validation_inference=False)
    state = {"status": "complete", "arm": args.arm, "contract": CONTRACT,
        "data_source": g.SOURCE, "discovery_source": G_SOURCE, **f.provenance(),
        "epochs_completed": EPOCHS, "weighting": spec, "partitions": partition_record(arrays),
        "initial_all_weights_sha256": initial_sha, "initial_shared_weights_sha256": shared_sha,
        "parameters": model.count_params(), "history": history, "optimizer_updates": updates,
        "optimizer_variables": optimizer, "validation_inference_passes": 0,
        "files": f.file_records(args.output_dir, ["last.weights.h5", "optimizer.npz"])}
    e.smoke._atomic_json(args.output_dir / "state.json", state)
    e.emit("H_training_complete", arm=args.arm, epochs=EPOCHS, updates=updates)


def evaluate(args):
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    features, arrays, _ = load_data(args.prepared_dir)
    states = {arm: validate_state(args.input_dir / arm, arrays, arm) for arm in g.ARMS}
    for name in ("initial_all_weights_sha256", "initial_shared_weights_sha256", "optimizer_updates", "partitions"):
        require(states["control"][name] == states["balanced"][name], f"unpaired H states: {name}")
    require([h["training_order_sha256"] for h in states["control"]["history"]] ==
            [h["training_order_sha256"] for h in states["balanced"]["history"]], "unpaired H batch orders")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # This record is written before any validation feature is indexed for inference.
    e.smoke._atomic_json(args.output_dir / "freeze-before-validation.json", {
        "contract": CONTRACT, "states": states, **f.provenance(), "validation_inference_started": False})
    val = roles(arrays)["validation"]
    identity = {"global_index": arrays["global_index"][val], "member": arrays["member"][val],
                "k": arrays["target_cardinality"][val]}
    probabilities, diagnostic_metrics, timings = {}, {}, {}
    for arm in g.ARMS:
        model, _ = e.initialize_arm("harmonic")
        model.load_weights(args.input_dir / arm / "last.weights.h5")
        diagnostic = tf.keras.Model(model.inputs, {"probability": model.output["cardinality"],
            "poibin": model.output["poibin_cardinality"], "string_birth": model.output["string_birth"],
            "residual_logits": model.get_layer("v280_cardinality_residual_logits").output})
        predict = tf.function(lambda x: diagnostic(x, training=False),
            input_signature=[tf.TensorSpec((None, *e.FEATURE_SHAPE), tf.float32)])
        prediction = {**identity, **{name: np.empty((len(val), columns), dtype=np.float32) for name, columns in
            (("probability", 7), ("poibin", 7), ("string_birth", 6), ("residual_logits", 7))}}
        start = time.monotonic()
        for offset in range(0, len(val), e.BATCH_SIZE):
            rows = val[offset:offset + e.BATCH_SIZE]
            for name, values in predict(np.asarray(features[rows], dtype=np.float32)).items():
                prediction[name][offset:offset + len(rows)] = values.numpy()
        timings[arm] = time.monotonic() - start
        require(np.allclose(g.model_code.poisson_binomial_numpy(prediction["string_birth"]), prediction["poibin"],
                            rtol=2e-5, atol=2e-6), "H string/Poisson-binomial disagreement")
        require(np.allclose(g.model_code.residual_cardinality_numpy(prediction["poibin"], prediction["residual_logits"]),
                            prediction["probability"], rtol=2e-5, atol=2e-6), "H residual/count disagreement")
        np.savez_compressed(args.output_dir / f"{arm}-validation.npz", **prediction)
        probabilities[arm] = prediction["probability"]
        diagnostic_metrics[arm] = score(identity["k"], prediction["poibin"])
    probabilities["fusion"] = fuse(probabilities["control"], probabilities["balanced"], states["balanced"]["weighting"])
    np.savez_compressed(args.output_dir / "fusion-validation.npz", **identity, probability=probabilities["fusion"])
    metrics = {name: score(identity["k"], p) for name, p in probabilities.items()}
    bootstrap = f.track_bootstrap(identity["member"], identity["k"], probabilities["control"].argmax(axis=1), probabilities["fusion"].argmax(axis=1))
    gate = passes_gate(metrics["control"], metrics["fusion"])
    result = {"status": "complete", "contract": CONTRACT, **f.provenance(),
        "states": states, "metrics": metrics, "poibin_metrics": diagnostic_metrics,
        "development_gate_passed": gate, "paired_track_bootstrap": bootstrap,
        "inference_wall_seconds_including_compilation": timings,
        "validation_inference_passes_per_arm": 1,
        "parameters_fusion": sum(s["parameters"] for s in states.values()),
        "files": f.file_records(args.output_dir, ["freeze-before-validation.json", "control-validation.npz", "balanced-validation.npz", "fusion-validation.npz"]),
        "reference_promoted": False, "outer_evaluation_launched": False}
    e.smoke._atomic_json(args.output_dir / "comparison.json", result)
    lines = ["# V28.0-H — second internal development split", "", *table(metrics), "",
        f"Frozen development gate passed: {gate}.",
        f"Poly delta {bootstrap['delta_pp']:+.4f} pp; descriptive track interval [{bootstrap['lower_95_pp']:.4f}, {bootstrap['upper_95_pp']:.4f}] pp.",
        "Each fresh model trained exactly 2 epochs; no H validation checkpoint or decoder tuning.",
        "Second internal development partition, not independent external confirmation. V27.3 remains the reference.",
        "Fusion requires two networks and two forward passes. No further training or promotion is triggered."]
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    e.emit("H_comparison_complete", gate_passed=gate, poly_delta_pp=bootstrap["delta_pp"], reference_promoted=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (("audit-g", audit_g), ("preflight", preflight), ("train", train), ("evaluate", evaluate)):
        command = commands.add_parser(name)
        command.add_argument("--prepared-dir", type=Path, required=True)
        if name in ("audit-g", "evaluate"):
            command.add_argument("--input-dir", type=Path, required=True)
        if name == "preflight":
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument("--output-dir", type=Path, required=True)
        if name == "train":
            command.add_argument("--arm", choices=g.ARMS, required=True)
        command.set_defaults(func=function)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
