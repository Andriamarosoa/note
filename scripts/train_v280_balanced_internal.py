"""V28.0-G: paired internal test of uniform versus 50/50 cardinality loss.

Both arms are fresh harmonic models. Only the cardinality sample weights
change. Fold 1 is internal validation; folds 2/3/4 train; fold 0 is not inferred.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_v280_nested_outer as f
from scripts import train_v280_harmonic_count as model_code
from scripts.audit_v280_undercount import CODE_SHA256

e = f.e
ARMS = ("control", "balanced")
SOURCE = {
    "run_id": 34462604665, "run_attempt": 1,
    "head_sha": "50e615bfb6cbce6ac5bbc57843ef3d1378e1bf79",
    "artifact": "v280-f-prepared-all", "artifact_id": 10146298232,
    "digest": "sha256:0f747055fb3c6acef2104d67a17b036b1252fe58a73bb840754cded5eab00cb1",
    "manifest_sha256": "e43567e3bd0945911915b4b11d88f960e4d95c0519809045dc9c7b056f0e5226",
}
CONTRACT = {
    "experiment": "v280_g_balanced_internal", "schema_version": 1,
    "architecture": "harmonic", "arms": list(ARMS), "outer_fold_excluded": 0,
    "fit_folds": [2, 3, 4], "validation_fold": 1,
    "epochs": e.EPOCHS, "chunk_epochs": f.CHUNK_EPOCHS,
    "batch_size": e.BATCH_SIZE, "seed": e.SEED, "learning_rate": 2e-4,
    "loss_weights": e.LOSS_WEIGHTS, "feature_sha256": e.CONFIG.sha256,
    "cardinality_weight_control": "1",
    "cardinality_weight_balanced": "N_fit/(2*N_fit_in_true_K_group)",
    "groups": ["K<2", "K>=2"], "weight_counts_from": "fit_rows_only",
    "auxiliary_sample_weights": "unchanged_F_masks",
    "sampling": f.CONTRACT["sampling"], "augmentation": "none",
    "initialization": f.CONTRACT["initialization"],
    "resume": f.CONTRACT["resume"], "checkpoint_selection": f.CONTRACT["checkpoint_selection"],
    "validation_weighting": "unweighted", "fresh_control_required": True,
    "saved_prediction_heads": ["probability", "poibin", "string_birth", "residual_logits"],
    "outer_evaluation": False, "refit": False, "automatic_promotion": False,
    "development_after_posthoc_F_diagnosis": True, "independent_validation_claim": False,
}


class BalancedError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise BalancedError(message)


def load_data(directory, *, features_required=True):
    for name, digest in CODE_SHA256.items():
        require(e.smoke._sha256_file(ROOT / name) == digest, f"frozen implementation changed: {name}")
    require(e.smoke._sha256_file(directory / "manifest.json") == SOURCE["manifest_sha256"], "prepared manifest changed")
    return f.load_prepared(directory, features_required=features_required)


def weighting(arrays, arm):
    require(arm in ARMS, "unknown weighting arm")
    roles = f.partitions(arrays["row_fold"], arrays["member"], 0)
    fit_k = arrays["target_cardinality"][roles["fit"]]
    require(np.all((fit_k >= 0) & (fit_k <= 6)), "invalid fit cardinality")
    n = len(fit_k)
    n_poly = int(np.sum(fit_k >= 2))
    n_nonpoly = n - n_poly
    require(n_poly > 0 and n_nonpoly > 0, "both fit groups are required")
    nonpoly, poly = (1.0, 1.0) if arm == "control" else (n / (2 * n_nonpoly), n / (2 * n_poly))
    weights = np.where(fit_k >= 2, poly, nonpoly).astype(np.float32)
    spec = {"fit_rows": n, "fit_poly_rows": n_poly, "fit_nonpoly_rows": n_nonpoly,
            "nonpoly_weight": nonpoly, "poly_weight": poly,
            "fit_weight_sha256": e.digest_array(weights), "fit_mean_weight_float64": float(weights.astype(np.float64).mean())}
    return roles, spec


def batch_weights(arrays, rows, spec):
    result = {name: arrays["weight_" + name][rows].copy() for name in e.LOSS_WEIGHTS}
    require(np.all(result["cardinality"] == 1), "unexpected base cardinality mask")
    result["cardinality"] = np.where(arrays["target_cardinality"][rows] >= 2,
        spec["poly_weight"], spec["nonpoly_weight"]).astype(np.float32)
    result["string_fret_onset"] = result["string_fret_onset"][:, None]
    return result


def preflight(args):
    _, arrays, _ = load_data(args.prepared_dir)
    specs = {arm: weighting(arrays, arm)[1] for arm in ARMS}
    require(specs["control"]["fit_rows"] == 46921 and specs["control"]["fit_poly_rows"] == 5514, "unexpected frozen fit counts")
    result = {"status": "complete", "contract": CONTRACT, "source": SOURCE,
              "scientific_code_sha256": CODE_SHA256, **f.provenance(),
              "partitions": f.partition_record(arrays, 0), "weighting": specs}
    e.smoke._atomic_json(args.output, result)
    e.emit("balanced_preflight_complete", **specs, training_started=False)


def state_from(directory, arrays, arm, chunk):
    state = f.read_json(directory / "state.json")
    _, spec = weighting(arrays, arm)
    for name, expected in {"status": "complete", "contract": CONTRACT, "source": SOURCE,
                           "arm": arm, "fold": 0, "phase": "probe", "chunk": chunk,
                           "weighting": spec, "epoch_budget": e.EPOCHS,
                           "epochs_completed": chunk * f.CHUNK_EPOCHS, **f.provenance()}.items():
        require(state.get(name) == expected, f"state mismatch: {name}")
    require(state["prepared_manifest_sha256"] == SOURCE["manifest_sha256"], "state data mismatch")
    f.verify_files(directory, state["files"])
    f.validate_training_history(state, arrays)
    return state


def verify_prediction(prediction, arrays):
    val = f.partitions(arrays["row_fold"], arrays["member"], 0)["validation"]
    for name, expected in {"global_index": arrays["global_index"][val],
                           "member": arrays["member"][val], "k": arrays["target_cardinality"][val]}.items():
        require(np.array_equal(prediction[name], expected), "validation identities changed")
    n = len(val)
    for name, columns in (("probability", 7), ("poibin", 7), ("string_birth", 6), ("residual_logits", 7)):
        require(prediction[name].shape == (n, columns) and np.isfinite(prediction[name]).all(), "invalid diagnostic head")
    expected_pb = model_code.poisson_binomial_numpy(prediction["string_birth"])
    expected_count = model_code.residual_cardinality_numpy(prediction["poibin"], prediction["residual_logits"])
    require(np.allclose(expected_pb, prediction["poibin"], rtol=2e-5, atol=2e-6), "string/Poisson-binomial heads disagree")
    require(np.allclose(expected_count, prediction["probability"], rtol=2e-5, atol=2e-6), "residual/count heads disagree")
    return e.count_metrics(prediction["k"], prediction["probability"]), e.count_metrics(prediction["k"], prediction["poibin"])


def full_initial_digest(model):
    digest = hashlib.sha256()
    for variable in model.weights:
        digest.update(variable.name.encode())
        digest.update(np.asarray(variable.numpy()).tobytes())
    return digest.hexdigest()


def train_chunk(args):
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    require((args.chunk == 2) == (args.previous_dir is not None), "only chunk 2 restores a previous state")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    features, arrays, _ = load_data(args.prepared_dir)
    roles, spec = weighting(arrays, args.arm)
    fit, val = roles["fit"], roles["validation"]
    model, shared_sha = e.initialize_arm("harmonic")
    initial_sha = full_initial_digest(model)
    model.optimizer.build(model.trainable_variables)
    history, previous_sha = [], None
    output = args.output_dir
    output.mkdir(parents=True)
    if args.chunk == 2:
        previous = state_from(args.previous_dir, arrays, args.arm, 1)
        require(previous["initial_all_weights_sha256"] == initial_sha, "resume initialization changed")
        f.restore_training_state(model, args.previous_dir, previous["optimizer_variables"])
        history = previous["history"]
        previous_sha = e.smoke._sha256_file(args.previous_dir / "state.json")
        for name in ("best.weights.h5", "best-validation.npz"):
            shutil.copy2(args.previous_dir / name, output / name)
    steps = (len(fit) + e.BATCH_SIZE - 1) // e.BATCH_SIZE
    require(int(model.optimizer.iterations.numpy()) == len(history) * steps, "optimizer resume iteration mismatch")
    outputs = {"probability": model.output["cardinality"], "poibin": model.output["poibin_cardinality"],
               "string_birth": model.output["string_birth"],
               "residual_logits": model.get_layer("v280_cardinality_residual_logits").output}
    diagnostic_model = tf.keras.Model(model.inputs, outputs)
    predict = tf.function(lambda x: diagnostic_model(x, training=False),
        input_signature=[tf.TensorSpec((None, *e.FEATURE_SHAPE), tf.float32)])
    groups = np.asarray([e.group_stem(str(m)) for m in arrays["member"][fit]])
    best = max(history, key=lambda h: e.checkpoint_key(h["validation"])) if history else None
    e.emit("training_started", arm=args.arm, chunk=args.chunk, completed_epochs=len(history),
           epoch_budget=e.EPOCHS, training_rows=len(fit), validation_rows=len(val), weighting=spec)
    for epoch in range(len(history) + 1, args.chunk * f.CHUNK_EPOCHS + 1):
        start, losses = time.monotonic(), {}
        order = fit[e.epoch_order(groups, arrays["target_cardinality"][fit], epoch=epoch)]
        for offset in range(0, len(order), e.BATCH_SIZE):
            rows = order[offset:offset + e.BATCH_SIZE]
            current = model.train_on_batch(np.asarray(features[rows], dtype=np.float32),
                {name: arrays["target_" + name][rows] for name in e.LOSS_WEIGHTS},
                sample_weight=batch_weights(arrays, rows, spec), return_dict=True)
            require(np.isfinite(list(current.values())).all(), "nonfinite training loss")
            for name, value in current.items():
                losses[name] = losses.get(name, 0.0) + float(value) * len(rows)
            updates = int(model.optimizer.iterations.numpy())
            if offset == 0 or updates % 50 == 0:
                e.emit("training_batch_completed", arm=args.arm, epoch=epoch, updates=updates, rows_seen=offset + len(rows))
            if epoch == 1 and offset == 0 and len(fit) == 46921 and len(val) == 14001:
                print(f"::notice title=V28.0-G gradient update completed::{args.arm}: first real training update completed", flush=True)
        prediction = {"global_index": arrays["global_index"][val], "member": arrays["member"][val],
                      "k": arrays["target_cardinality"][val],
                      **{name: np.empty((len(val), columns), dtype=np.float32) for name, columns in
                         (("probability", 7), ("poibin", 7), ("string_birth", 6), ("residual_logits", 7))}}
        for offset in range(0, len(val), e.BATCH_SIZE):
            rows = val[offset:offset + e.BATCH_SIZE]
            head_values = predict(np.asarray(features[rows], dtype=np.float32))
            for name, value in head_values.items():
                prediction[name][offset:offset + len(rows)] = value.numpy()
        metrics, pb_metrics = verify_prediction(prediction, arrays)
        entry = {"epoch": epoch, "updates": int(model.optimizer.iterations.numpy()),
                 "training_order_sha256": e.digest_array(arrays["global_index"][order]),
                 "train_losses": {name: value / len(fit) for name, value in losses.items()},
                 "validation": metrics, "poibin_validation": pb_metrics,
                 "epoch_seconds": time.monotonic() - start}
        if best is None or e.checkpoint_key(metrics) > e.checkpoint_key(best["validation"]):
            best = entry
            model.save_weights(output / "best.weights.h5")
            np.savez_compressed(output / "best-validation.npz", **prediction)
        np.savez_compressed(output / "last-validation.npz", **prediction)
        history.append(entry)
        optimizer_variables = f.save_training_state(model, output)
        e.smoke._atomic_json(output / "progress.json", {"arm": args.arm, "history": history,
                            "epochs_completed": epoch, "weighting": spec, "contract": CONTRACT})
        e.emit("epoch_completed", arm=args.arm, epoch=epoch, best_epoch=best["epoch"],
               updates=entry["updates"], validation=metrics, poibin_validation=pb_metrics)
    state = {"status": "complete", "arm": args.arm, "fold": 0, "phase": "probe",
             "contract": CONTRACT, "source": SOURCE, **f.provenance(),
             "prepared_manifest_sha256": SOURCE["manifest_sha256"], "weighting": spec,
             "chunk": args.chunk, "epoch_budget": e.EPOCHS, "epochs_completed": len(history),
             "parameters": model.count_params(), "partitions": f.partition_record(arrays, 0),
             "initial_all_weights_sha256": initial_sha, "initial_shared_weights_sha256": shared_sha,
             "previous_state_sha256": previous_sha, "optimizer_variables": optimizer_variables,
             "optimizer_updates": int(model.optimizer.iterations.numpy()), "history": history,
             "best_epoch": best["epoch"], "selected_validation": best["validation"],
             "files": f.file_records(output, [p.name for p in output.iterdir() if p.suffix in (".h5", ".npz")])}
    e.smoke._atomic_json(output / "state.json", state)
    e.emit("chunk_complete", arm=args.arm, chunk=args.chunk, epochs_completed=len(history))


def compare(args):
    _, arrays, _ = load_data(args.prepared_dir, features_required=False)
    states, best_predictions = {}, {}
    for arm in ARMS:
        root = args.input_dir / f"v280-g-{arm}-2"
        state = state_from(root, arrays, arm, 2)
        best = f.choose_epoch(state["history"])
        require(state["best_epoch"] == best["epoch"] and state["selected_validation"] == best["validation"], "checkpoint choice changed")
        for prefix, expected in (("best", best), ("last", state["history"][-1])):
            prediction = f.read_npz(root / (prefix + "-validation.npz"))
            metrics, pb = verify_prediction(prediction, arrays)
            require(f.same_count_metrics(metrics, expected["validation"]), "recomputed cardinality metrics differ")
            require(f.same_count_metrics(pb, expected["poibin_validation"]), "recomputed Poisson-binomial metrics differ")
            if prefix == "best":
                best_predictions[arm] = prediction
        states[arm] = state
    left, right = (states[arm] for arm in ARMS)
    for name in ("initial_all_weights_sha256", "initial_shared_weights_sha256", "optimizer_updates", "partitions", "parameters"):
        require(left[name] == right[name], f"unpaired comparison: {name}")
    require([h["training_order_sha256"] for h in left["history"]] ==
            [h["training_order_sha256"] for h in right["history"]], "unpaired batch orders")
    lm, rm = left["selected_validation"], right["selected_validation"]
    delta = rm["poly_correct"] - lm["poly_correct"]
    preference = "balanced" if delta > 0 else "control" if delta < 0 else "tie"
    k = best_predictions["control"]["k"]
    c = best_predictions["control"]["probability"].argmax(axis=1)
    b = best_predictions["balanced"]["probability"].argmax(axis=1)
    bootstrap = f.track_bootstrap(best_predictions["control"]["member"], k, c, b)
    result = {"status": "complete", "contract": CONTRACT, "source": SOURCE, **f.provenance(),
              "scope": "single_internal_development_split_after_F_diagnosis",
              "internal_preference": preference, "poly_correct_delta_rows": delta,
              "poly_delta_pp": 100.0 * (rm["poly_exact_k"] - lm["poly_exact_k"]),
              "global_delta_pp": 100.0 * (rm["exact_k"] - lm["exact_k"]),
              "paired_track_bootstrap": bootstrap, "arms": states,
              "outer_evaluation_launched": False, "reference_promoted": False,
              "v273_reference_preserved": True}
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    e.smoke._atomic_json(args.output_dir / "comparison.json", result)
    rows = ["# V28.0-G — internal loss weighting comparison", "",
            "| Arm | Selected epoch | Poly Exact-K | Global Exact-K | Poly NLL | Poly undercount |",
            "|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        m = states[arm]["selected_validation"]
        under = sum(m["by_true_k"][str(i)]["undercount"] for i in range(2, 7)) / m["poly_rows"]
        rows.append(f"| {arm} | {states[arm]['best_epoch']} | {m['poly_exact_k']:.4%} | {m['exact_k']:.4%} | {m['poly_nll']:.5f} | {under:.4%} |")
    rows += ["", f"Balanced minus control: {result['poly_delta_pp']:+.4f} pp / {delta:+d} correct polyphonic rows.",
             f"Internal preference: {preference}. Single development split; V27.3 remains the reference.",
             "No external fold evaluation or automatic next training is triggered."]
    (args.output_dir / "summary.md").write_text("\n".join(rows) + "\n")
    e.emit("balanced_comparison_complete", preference=preference, delta_poly_pp=result["poly_delta_pp"], reference_promoted=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("preflight")
    prep.add_argument("--prepared-dir", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.set_defaults(func=preflight)
    train = commands.add_parser("train-chunk")
    train.add_argument("--prepared-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--previous-dir", type=Path)
    train.add_argument("--arm", choices=ARMS, required=True)
    train.add_argument("--chunk", type=int, choices=(1, 2), required=True)
    train.set_defaults(func=train_chunk)
    comparison = commands.add_parser("compare")
    for name in ("prepared-dir", "input-dir", "output-dir"):
        comparison.add_argument("--" + name, type=Path, required=True)
    comparison.set_defaults(func=compare)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
