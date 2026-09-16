"""V28.0-I: final single-model 35%-poly loss trial against frozen G control.

Exactly 12 epochs, unchanged G model/data/initialization/batches/checkpoint rule.
The only learning change is the main loss's group weights. No fusion or tuning.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_v280_balanced_internal as g

f, e = g.f, g.e
POLY_MASS = .35
G_IMPLEMENTATION_SHA = "9f4180e23ded17493777694f2b112e48ffc74e73af081fb862bc488b0de583fb"
CONTROL_SOURCE = {
    "run_id": 34684806334, "run_attempt": 1,
    "head_sha": "bff41aec998a798055687c94705629eb98626aee",
    "comparison_json_sha256": "680e54d783bae6894d42058fbe0f3f9845a70b2c64914c4124910ae78edbe2a9",
    "artifacts": {
        "v280-g-internal-comparison": {"id": 10304691260, "digest": "sha256:8508188bffde11afda96690b2ba3309f0d0e9fc6b08c3592fe5b0368f65fe62d"},
        "v280-g-control-2": {"id": 10299622000, "digest": "sha256:b2ee3f974ba64f4e7dd4541c185b7353b3fc09d6323a2021d4c6c2f364d903e9"},
    },
}
CONTRACT = {
    "experiment": "v280_i_final_moderate_trial", "schema_version": 1,
    "architecture": "harmonic", "parameters": 110402, "model_count": 1,
    "fit_folds": [2, 3, 4], "validation_fold": 1, "outer_fold_excluded": 0,
    "epochs": 12, "chunk_epochs": 6, "batch_size": e.BATCH_SIZE, "seed": e.SEED,
    "learning_rate": 2e-4, "loss_weights": e.LOSS_WEIGHTS,
    "poly_loss_mass": POLY_MASS, "nonpoly_loss_mass": 1 - POLY_MASS,
    "weighting": "w_poly=.35*N_fit/N_poly; w_nonpoly=.65*N_fit/N_nonpoly",
    "weight_counts_from": "fit_rows_only", "auxiliary_masks": "unchanged_G",
    "initialization": f.CONTRACT["initialization"], "sampling": f.CONTRACT["sampling"],
    "resume": f.CONTRACT["resume"], "checkpoint_selection": f.CONTRACT["checkpoint_selection"],
    "augmentation": "none", "validation_weighting": "unweighted", "decoder": "raw_argmax",
    "control": "frozen_G_uniform_12_epoch_run_no_new_control_training",
    "selection_of_35_percent": "fixed_intermediate_hypothesis_after_G_H_not_an_optimized_value",
    "gate": {"minimum_poly_gain_pp": 5.0, "poly_bootstrap_lower95_strictly_positive": True,
             "global_correct_not_down": True, "false_poly_not_up": True},
    "on_failure": "stop_v28_return_to_v273", "on_pass": "material_internal_candidate_requires_confirmation",
    "scope": "final_internal_development_trial_after_F_G_H", "independent_validation_claim": False,
    "automatic_promotion": False, "outer_evaluation": False, "automatic_next_training": False,
}


class FinalTrialError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise FinalTrialError(message)


def load_data(directory, *, features_required=True):
    require(e.smoke._sha256_file(ROOT / "scripts/train_v280_balanced_internal.py") == G_IMPLEMENTATION_SHA,
            "frozen G implementation changed")
    return g.load_data(directory, features_required=features_required)


def weighting(arrays):
    roles, counts = g.weighting(arrays, "control")
    n, poly, nonpoly = (counts[k] for k in ("fit_rows", "fit_poly_rows", "fit_nonpoly_rows"))
    lo, hi = (1 - POLY_MASS) * n / nonpoly, POLY_MASS * n / poly
    k = arrays["target_cardinality"][roles["fit"]]
    weights = np.where(k >= 2, hi, lo).astype(np.float32)
    return roles, {"fit_rows": n, "fit_poly_rows": poly, "fit_nonpoly_rows": nonpoly,
                   "nonpoly_weight": lo, "poly_weight": hi,
                   "fit_weight_sha256": e.digest_array(weights),
                   "fit_mean_weight_float64": float(weights.astype(np.float64).mean())}


def validate_predictions(directory, state, arrays):
    history = state["history"]
    best = max(history, key=lambda entry: e.checkpoint_key(entry["validation"]))
    require(state["best_epoch"] == best["epoch"] and state["selected_validation"] == best["validation"],
            "checkpoint selection changed")
    best_prediction = None
    for prefix, entry in (("best", best), ("last", history[-1])):
        prediction = f.read_npz(directory / (prefix + "-validation.npz"))
        count, pb = g.verify_prediction(prediction, arrays)
        require(f.same_count_metrics(count, entry["validation"]) and
                f.same_count_metrics(pb, entry["poibin_validation"]), "saved metrics changed")
        if prefix == "best":
            best_prediction = prediction
    return best_prediction


def load_control(directory, comparison_dir, arrays):
    path = comparison_dir / "comparison.json"
    require(e.smoke._sha256_file(path) == CONTROL_SOURCE["comparison_json_sha256"], "G comparison changed")
    comparison = f.read_json(path)
    require(comparison["status"] == "complete" and comparison["contract"] == g.CONTRACT, "invalid G comparison")
    state = f.read_json(directory / "state.json")
    require(state == comparison["arms"]["control"], "control differs from frozen G report")
    for name, expected in {"status": "complete", "contract": g.CONTRACT, "arm": "control",
                           "fold": 0, "phase": "probe", "chunk": 2, "epoch_budget": 12,
                           "epochs_completed": 12, "source": g.SOURCE,
                           "prepared_manifest_sha256": g.SOURCE["manifest_sha256"],
                           "source_head_sha": CONTROL_SOURCE["head_sha"],
                           "source_run_id": str(CONTROL_SOURCE["run_id"]),
                           "weighting": g.weighting(arrays, "control")[1]}.items():
        require(state.get(name) == expected, f"invalid G control: {name}")
    f.verify_files(directory, state["files"])
    f.validate_training_history(state, arrays)
    prediction = validate_predictions(directory, state, arrays)
    return state, prediction


def preflight(args):
    _, arrays, _ = load_data(args.prepared_dir)
    control, _ = load_control(args.control_dir, args.control_comparison_dir, arrays)
    _, spec = weighting(arrays)
    require(spec["fit_rows"] == 46921 and spec["fit_poly_rows"] == 5514, "frozen fit counts changed")
    e.smoke._atomic_json(args.output, {"status": "complete", "contract": CONTRACT,
        "source": g.SOURCE, "control_source": CONTROL_SOURCE, **f.provenance(),
        "partitions": f.partition_record(arrays, 0), "weighting": spec,
        "control_selected_validation": control["selected_validation"], "control_best_epoch": control["best_epoch"]})
    e.emit("I_preflight_complete", weighting=spec, training_started=False)


def state_from(directory, arrays, chunk):
    state = f.read_json(directory / "state.json")
    for name, expected in {"status": "complete", "contract": CONTRACT, "arm": "moderate",
                           "fold": 0, "phase": "probe", "chunk": chunk, "epoch_budget": 12,
                           "epochs_completed": chunk * 6, "source": g.SOURCE,
                           "control_source": CONTROL_SOURCE, "weighting": weighting(arrays)[1],
                           "prepared_manifest_sha256": g.SOURCE["manifest_sha256"], **f.provenance()}.items():
        require(state.get(name) == expected, f"I training state mismatch: {name}")
    f.verify_files(directory, state["files"])
    f.validate_training_history(state, arrays)
    validate_predictions(directory, state, arrays)
    return state


def train_chunk(args):
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    require((args.chunk == 2) == (args.previous_dir is not None), "only chunk 2 resumes")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    features, arrays, _ = load_data(args.prepared_dir)
    roles, spec = weighting(arrays)
    fit, val = roles["fit"], roles["validation"]
    model, shared_sha = e.initialize_arm("harmonic")
    initial_sha = g.full_initial_digest(model)
    model.optimizer.build(model.trainable_variables)
    history, previous_sha = [], None
    output = args.output_dir
    output.mkdir(parents=True)
    if args.chunk == 2:
        previous = state_from(args.previous_dir, arrays, 1)
        require(previous["initial_all_weights_sha256"] == initial_sha, "resume initialization changed")
        f.restore_training_state(model, args.previous_dir, previous["optimizer_variables"])
        history = previous["history"]
        previous_sha = e.smoke._sha256_file(args.previous_dir / "state.json")
        for name in ("best.weights.h5", "best-validation.npz"):
            shutil.copy2(args.previous_dir / name, output / name)
    steps = (len(fit) + e.BATCH_SIZE - 1) // e.BATCH_SIZE
    require(int(model.optimizer.iterations.numpy()) == len(history) * steps, "Adam iteration mismatch")
    diagnostic = tf.keras.Model(model.inputs, {"probability": model.output["cardinality"],
        "poibin": model.output["poibin_cardinality"], "string_birth": model.output["string_birth"],
        "residual_logits": model.get_layer("v280_cardinality_residual_logits").output})
    predict = tf.function(lambda x: diagnostic(x, training=False),
        input_signature=[tf.TensorSpec((None, *e.FEATURE_SHAPE), tf.float32)])
    groups = np.asarray([e.group_stem(str(m)) for m in arrays["member"][fit]])
    best = max(history, key=lambda entry: e.checkpoint_key(entry["validation"])) if history else None
    e.emit("I_training_started", chunk=args.chunk, completed_epochs=len(history), epoch_budget=12,
           fit_rows=len(fit), validation_rows=len(val), weighting=spec)
    for epoch in range(len(history) + 1, args.chunk * 6 + 1):
        start, losses = time.monotonic(), {}
        order = fit[e.epoch_order(groups, arrays["target_cardinality"][fit], epoch=epoch)]
        for offset in range(0, len(fit), e.BATCH_SIZE):
            rows = order[offset:offset + e.BATCH_SIZE]
            current = model.train_on_batch(np.asarray(features[rows], dtype=np.float32),
                {name: arrays["target_" + name][rows] for name in e.LOSS_WEIGHTS},
                sample_weight=g.batch_weights(arrays, rows, spec), return_dict=True)
            require(np.isfinite(list(current.values())).all(), "nonfinite I training loss")
            for name, value in current.items():
                losses[name] = losses.get(name, 0.0) + float(value) * len(rows)
            updates = int(model.optimizer.iterations.numpy())
            if offset == 0 or updates % 50 == 0:
                e.emit("I_training_batch_completed", epoch=epoch, updates=updates)
        prediction = {"global_index": arrays["global_index"][val], "member": arrays["member"][val],
            "k": arrays["target_cardinality"][val], **{name: np.empty((len(val), columns), dtype=np.float32)
            for name, columns in (("probability", 7), ("poibin", 7), ("string_birth", 6), ("residual_logits", 7))}}
        for offset in range(0, len(val), e.BATCH_SIZE):
            rows = val[offset:offset + e.BATCH_SIZE]
            for name, values in predict(np.asarray(features[rows], dtype=np.float32)).items():
                prediction[name][offset:offset + len(rows)] = values.numpy()
        metrics, pb = g.verify_prediction(prediction, arrays)
        entry = {"epoch": epoch, "updates": int(model.optimizer.iterations.numpy()),
            "training_order_sha256": e.digest_array(arrays["global_index"][order]),
            "train_losses": {name: value / len(fit) for name, value in losses.items()},
            "validation": metrics, "poibin_validation": pb, "epoch_seconds": time.monotonic() - start}
        if best is None or e.checkpoint_key(metrics) > e.checkpoint_key(best["validation"]):
            best = entry
            model.save_weights(output / "best.weights.h5")
            np.savez_compressed(output / "best-validation.npz", **prediction)
        np.savez_compressed(output / "last-validation.npz", **prediction)
        history.append(entry)
        optimizer = f.save_training_state(model, output)
        e.smoke._atomic_json(output / "progress.json", {"history": history, "contract": CONTRACT, "weighting": spec})
        e.emit("I_epoch_completed", epoch=epoch, best_epoch=best["epoch"], updates=entry["updates"], validation=metrics)
    state = {"status": "complete", "arm": "moderate", "fold": 0, "phase": "probe",
        "contract": CONTRACT, "source": g.SOURCE, "control_source": CONTROL_SOURCE, **f.provenance(),
        "prepared_manifest_sha256": g.SOURCE["manifest_sha256"], "weighting": spec,
        "chunk": args.chunk, "epoch_budget": 12, "epochs_completed": len(history),
        "parameters": model.count_params(), "partitions": f.partition_record(arrays, 0),
        "initial_all_weights_sha256": initial_sha, "initial_shared_weights_sha256": shared_sha,
        "previous_state_sha256": previous_sha, "optimizer_variables": optimizer,
        "optimizer_updates": int(model.optimizer.iterations.numpy()), "history": history,
        "best_epoch": best["epoch"], "selected_validation": best["validation"],
        "files": f.file_records(output, [p.name for p in output.iterdir() if p.suffix in (".h5", ".npz")])}
    e.smoke._atomic_json(output / "state.json", state)
    e.emit("I_chunk_complete", chunk=args.chunk, epochs_completed=len(history))


def false_poly(metrics):
    return sum(sum(row[2:]) for row in metrics["confusion_matrix_true_by_predicted"][:2])


def decide(control, candidate, bootstrap):
    require(control["rows"] == candidate["rows"] and control["poly_rows"] == candidate["poly_rows"], "unpaired denominators")
    gates = {"poly_gain_at_least_5pp": 100 * (candidate["poly_correct"] - control["poly_correct"]) >= 5 * control["poly_rows"],
             "poly_lower95_strictly_positive": bootstrap["lower_95_pp"] > 0,
             "global_correct_not_down": candidate["correct"] >= control["correct"],
             "false_poly_not_up": false_poly(candidate) <= false_poly(control)}
    return {"gates": gates, "passed": all(gates.values()),
            "decision": CONTRACT["on_pass"] if all(gates.values()) else CONTRACT["on_failure"],
            "official_reference": "V27.3", "further_v28_weight_search": False,
            "reference_promoted": False, "automatic_next_training": False}


def compare(args):
    _, arrays, _ = load_data(args.prepared_dir, features_required=False)
    control, c = load_control(args.control_dir, args.control_comparison_dir, arrays)
    candidate = state_from(args.candidate_dir, arrays, 2)
    p = validate_predictions(args.candidate_dir, candidate, arrays)
    for name in ("initial_all_weights_sha256", "initial_shared_weights_sha256", "optimizer_updates", "parameters", "partitions"):
        require(control[name] == candidate[name], f"unpaired I/G comparison: {name}")
    require([x["training_order_sha256"] for x in control["history"]] ==
            [x["training_order_sha256"] for x in candidate["history"]], "training orders differ from G control")
    bootstrap = f.track_bootstrap(c["member"], c["k"], c["probability"].argmax(axis=1), p["probability"].argmax(axis=1))
    cm, pm = control["selected_validation"], candidate["selected_validation"]
    decision = decide(cm, pm, bootstrap)
    result = {"status": "complete", "contract": CONTRACT, **f.provenance(),
        "source": g.SOURCE, "control_source": CONTROL_SOURCE,
        "control": control, "moderate": candidate, "paired_track_bootstrap": bootstrap,
        "poly_delta_pp": bootstrap["delta_pp"], "global_delta_pp": 100 * (pm["exact_k"] - cm["exact_k"]),
        "decision": decision, "outer_evaluation_launched": False}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    e.smoke._atomic_json(args.output_dir / "comparison.json", result)
    e.smoke._atomic_json(args.output_dir / "decision.json", decision)
    rows = ["# V28.0-I — final moderate weighting trial", "",
        "| Model | Selected epoch | Poly Exact-K | Global Exact-K | False poly rows | Poly NLL |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, state in (("G uniform control", control), ("I moderate 35%", candidate)):
        m = state["selected_validation"]
        rows.append(f"| {name} | {state['best_epoch']} | {m['poly_exact_k']:.4%} | {m['exact_k']:.4%} | {false_poly(m)} | {m['poly_nll']:.5f} |")
    rows += ["", f"Decision: **{decision['decision']}**. Gates: {decision['gates']}.",
        f"Poly delta {bootstrap['delta_pp']:+.4f} pp; descriptive 95% interval [{bootstrap['lower_95_pp']:.4f}, {bootstrap['upper_95_pp']:.4f}] pp.",
        "Both arms completed 12 epochs with identical initial weights, data and batch orders. The G control was reused, not retrained.",
        "Internal development comparison after checkpoint selection; not independent confirmation against V27.3.",
        "V27.3 remains the official reference. No further weighting search, outer evaluation or training is launched."]
    (args.output_dir / "summary.md").write_text("\n".join(rows) + "\n")
    e.emit("I_final_verdict", decision=decision["decision"], gates=decision["gates"], poly_delta_pp=bootstrap["delta_pp"], global_delta_pp=result["global_delta_pp"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (("preflight", preflight), ("train-chunk", train_chunk), ("compare", compare)):
        command = commands.add_parser(name)
        command.add_argument("--prepared-dir", type=Path, required=True)
        if name in ("preflight", "compare"):
            command.add_argument("--control-dir", type=Path, required=True)
            command.add_argument("--control-comparison-dir", type=Path, required=True)
        if name == "preflight":
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument("--output-dir", type=Path, required=True)
        if name == "train-chunk":
            command.add_argument("--chunk", type=int, choices=(1, 2), required=True)
            command.add_argument("--previous-dir", type=Path)
        if name == "compare":
            command.add_argument("--candidate-dir", type=Path, required=True)
        command.set_defaults(func=function)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
