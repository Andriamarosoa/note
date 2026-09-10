"""V28.0-F: nested, composition-group OOF evaluation of the harmonic count expert.

Each fold has a fresh 12-epoch inner probe, then a fresh four-fold refit for
the inner-selected epoch budget. Six-epoch chunks preserve Adam state across
Actions jobs. The held fold is predicted once, after the refit is complete.
This is development OOF: architecture selection and earlier experiments have
already used these compositions. No deployment or reference promotion occurs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts import train_v280_internal_ablation as e
from scripts.evaluate_boundaries import match_boundaries, milliseconds_to_samples
from scripts.evaluate_v8_boundaries import _reference_positions
from scripts.train_v100_spectral_string_slots import _cached_prediction_map

FOLD_COUNT = 5
EXPECTED_ROWS = 76768
EXPECTED_POLY_ROWS = 9401
CHUNK_EPOCHS = 6
BOOTSTRAP_REPLICATES = 10000
TOLERANCES_MS = (5, 10, 20, 50)
SELECTION_SHA = "7f40e09cd67ba0c49cfead12384603e9069e8c978e294ef260c4550b376d76aa"
REFERENCE_PREDICTIONS_SHA = "a353887c6ef1b60d5f6a3854627e99a5cf110e0cb7db0ed4ec50643b26bf805b"
REFERENCE_REPORT_SHA = "53a59f53910484bf86858390c33d53359a72eaa9004275e3ec7f0f0814925564"
FLOAT_METRIC_ABS_TOLERANCE = 1e-12
RECOVERY_SOURCE = {
    "run_id": 34437608635,
    "head_sha": "a203e1afa95264c14d3eef5bf0e63eafcec2eb4e",
    "run_attempt": 1, "failed_job_id": 102795200081,
    "prepared_manifest_sha256": "e43567e3bd0945911915b4b11d88f960e4d95c0519809045dc9c7b056f0e5226",
    "state_sha256": {
        "probe": "67c7bf41c3a7db72f85e35c8007462b7e9483dce4f34379eaeb8bf9878c89138",
        "refit": "4ae51101b5e43209ee763353e86f1e4e9cdbd874951fb08788df9bfc191aa2e0",
    },
    "artifacts": {
        "v280-f-prepared-all": {"id": 10136891720,
            "digest": "sha256:e4c7187d29c68747f3c53712f52d705c06b15cc98b1873083b295b963de52be6"},
        "v280-f-preparation-audit": {"id": 10136892010,
            "digest": "sha256:626b0fd97e516d21d4a2469e762ad0050f8a4dd20c6a2a8a45b934f1dbdf9d81"},
        "v280-f-fold-0-probe-2": {"id": 10141403654,
            "digest": "sha256:e9e239c579ff53bc905606307b5e2a17f5a0d4081067ed1cdab228de6b19b2c0"},
        "v280-f-fold-0-refit-2": {"id": 10142571624,
            "digest": "sha256:b240e5e790f6726ec206d7263ba632d1d09e7413c2b651ce84e113731440c041"},
    },
}
SOURCES = {
    **e.SOURCES,
    "internal_selection": {
        "run_id": 34351021229,
        "head_sha": "7473be3c0a6b4dbb5344ef90744f66a872720f69",
        "artifact": "v280-e-internal-comparison",
        "digest": "sha256:67a9c1bbd6d134e6fb26eff5c84bf0b7416c5b6e7248680c3a1053a5219f46f4",
    },
    "reference": {
        "run_id": 34297767492,
        "head_sha": "dd43b4f92b234dc7c84377e18389fd34350cd1b4",
        "artifact": "v273-selective-transition-summary",
        "digest": "sha256:199972f9df5c05391b8d39761fe3a1ca5982cd8b5c91b7168da2623675e1e60b",
    },
}
CONTRACT = {
    "schema_version": 1, "experiment": "v280_f_nested_harmonic_development_oof",
    "arm": "harmonic", "fold_count": FOLD_COUNT,
    "seed": e.SEED, "inner_epochs": e.EPOCHS, "batch_size": e.BATCH_SIZE,
    "chunk_epochs": CHUNK_EPOCHS, "learning_rate": 2e-4,
    "loss_weights": e.LOSS_WEIGHTS, "feature_sha256": e.CONFIG.sha256,
    "meta_validation": "smallest_canonical_fold_except_outer",
    "checkpoint_selection": "max_inner_poly_correct_then_min_poly_nll_then_earlier_epoch",
    "refit": "fresh_model_and_Adam_on_four_nonouter_folds_for_inner_selected_epochs",
    "sampling": e.CONTRACT["sampling"], "initialization": e.CONTRACT["shared_layer_initialization"],
    "resume": "last_model_weights_and_all_Adam_variables_at_epoch_boundary",
    "end_of_stream": e.CONTRACT["end_of_stream"],
    "augmentation": "none", "class_weighting": "none", "early_stopping": False,
    "outer_inference_passes_per_fold": 1,
    "outer_labels_used_for_current_epoch_selection": False,
    "prior_architecture_selection_used_development_compositions": True,
    "independent_validation_claim": False,
    "historical_validation_jams_read": False, "locked12_indexed": False,
    "external_checkpoint_or_teacher_loaded": False,
    "ranking": "unchanged_V10_frozen_top_samples_as_used_by_V27.3",
    "event_tolerances_ms": list(TOLERANCES_MS),
    "bootstrap": {"unit": "track", "replicates": BOOTSTRAP_REPLICATES, "seed": e.SEED},
    "material_gain_gate": {"poly_gain_pp_at_least": 5.0, "positive_folds_at_least": 4,
                           "track_bootstrap_95_lower_pp_above": 0.0},
    "automatic_promotion": False, "reference": "V27.3",
}


class OuterError(RuntimeError):
    pass


def read_json(path):
    return json.loads(Path(path).read_text())


def provenance():
    return {"source_head_sha": os.environ.get("GITHUB_SHA"),
            "source_run_id": os.environ.get("GITHUB_RUN_ID")}


def same_count_metrics(actual, recorded):
    """Keep all discrete metrics exact; allow only float64 reduction noise."""
    if actual.keys() != recorded.keys():
        return False
    for key, value in actual.items():
        expected = recorded[key]
        if key in ("nll", "poly_nll", "brier"):
            if (not math.isfinite(value) or not math.isfinite(expected)
                    or not math.isclose(value, expected, rel_tol=0.0,
                                        abs_tol=FLOAT_METRIC_ABS_TOLERANCE)):
                return False
        elif value != expected:
            return False
    return True


def validate_state_origin(state, path, *, fold, phase, prepared_sha, allow_recovery=False):
    if all(state.get(key) == value for key, value in provenance().items()):
        return
    source = RECOVERY_SOURCE
    if (allow_recovery and fold == 0 and phase in ("probe", "refit")
            and state.get("source_head_sha") == source["head_sha"]
            and str(state.get("source_run_id")) == str(source["run_id"])
            and prepared_sha == source["prepared_manifest_sha256"]
            and e.smoke._sha256_file(path) == source["state_sha256"][phase]):
        return
    raise OuterError("training state producer differs from the current run or frozen fold-0 recovery")


def file_records(root, names):
    return {name: {"sha256": e.smoke._sha256_file(root / name),
                   "bytes": (root / name).stat().st_size} for name in names}


def verify_files(root, records):
    for name, record in records.items():
        if Path(name).name != name:
            raise OuterError("unexpected nested artifact path")
        if e.smoke._sha256_file(root / name) != record["sha256"]:
            raise OuterError(f"artifact digest mismatch: {name}")


def partitions(row_fold, members, fold):
    """No target values enter the nested partition rule."""
    row_fold, members = np.asarray(row_fold), np.asarray(members)
    if fold not in range(FOLD_COUNT) or row_fold.shape != members.shape:
        raise OuterError("invalid fold or member shape")
    if set(np.unique(row_fold)) != set(range(FOLD_COUNT)):
        raise OuterError("canonical folds are missing")
    groups = np.asarray([e.group_stem(str(m)) for m in members])
    for group in set(groups):
        if len(np.unique(row_fold[groups == group])) != 1:
            raise OuterError("composition crosses fold boundaries")
    validation_fold = min(f for f in range(FOLD_COUNT) if f != fold)
    roles = {
        "outer": np.flatnonzero(row_fold == fold),
        "validation": np.flatnonzero(row_fold == validation_fold),
        "fit": np.flatnonzero((row_fold != fold) & (row_fold != validation_fold)),
        "refit": np.flatnonzero(row_fold != fold),
    }
    if any(not len(indices) for indices in roles.values()):
        raise OuterError("empty nested partition")
    return roles


def partition_record(arrays, fold):
    return {name: {"rows": len(rows),
                   "indices_sha256": e.digest_array(arrays["global_index"][rows])}
            for name, rows in partitions(arrays["row_fold"], arrays["member"], fold).items()}


def verify_selection(directory):
    path = directory / "comparison.json"
    if e.smoke._sha256_file(path) != SELECTION_SHA:
        raise OuterError("internal selection artifact changed")
    report = read_json(path)
    if (report["status"] != "complete" or report["contract"] != e.CONTRACT
            or report["internal_preference"] != "harmonic"):
        raise OuterError("completed harmonic selection is required")
    return {"sha256": SELECTION_SHA, "internal_preference": "harmonic",
            "poly_correct_delta_rows": report["poly_correct_delta_rows"],
            "weights_or_epoch_budget_reused": False}


def prepare(args):
    output = args.output_dir
    if output.exists():
        raise FileExistsError(output)
    selection = verify_selection(args.selection_dir)
    if e.smoke._sha256_file(args.cluster_dir / "v280-cluster-metadata.npz") != e.CLUSTER_PAYLOAD_SHA256:
        raise OuterError("frozen cluster payload changed")
    metadata = e.smoke.load_distilled_cluster_cache(args.cluster_dir)
    indexed, tracks, historical = e.mining._dataset_split(args.dataset_dir)
    membership = e.mining.V100Membership(metadata.track_members, metadata.shard_paths,
                                         e.mining._member_digest(metadata.track_members))
    selected = e.mining.validate_outer_clean_selection(indexed, tracks, historical, membership)
    row_fold, original_split = e.split_rows(metadata, selected.tracks)
    n = metadata.row_count
    if n != EXPECTED_ROWS or int(np.sum(metadata.exact >= 2)) != EXPECTED_POLY_ROWS:
        raise OuterError("unexpected source row coverage")
    arrays = {"global_index": np.arange(n, dtype=np.int64), "row_fold": row_fold,
              "member": metadata.members}
    split = {"groups_per_fold": original_split["groups_per_fold"],
             "row_counts_per_fold": original_split["row_counts_per_fold"],
             "roles_by_fold": {str(f): partition_record(arrays, f) for f in range(FOLD_COUNT)}}
    output.mkdir(parents=True)
    e.smoke._atomic_json(output / "split.json", split)
    e.emit("outer_partitions_frozen", **split)
    verification = e.mining.verify_cache(args.cqt_dir)
    cqt_manifest = read_json(args.cqt_dir / "manifest.json")
    if tuple(r["annotation_member"] for r in cqt_manifest["cache"]["tracks"]) != metadata.track_members:
        raise OuterError("CQT and cluster memberships differ")
    features = np.lib.format.open_memmap(output / "features.npy", mode="w+", dtype=np.float16,
                                         shape=(n, *e.FEATURE_SHAPE))
    targets, weights, per_track, reference_members, reference_samples = {}, {}, [], [], []
    tracks_by_member = {t.annotation_member: t for t in selected.tracks}
    for number, member in enumerate(metadata.track_members, start=1):
        rows = np.flatnonzero(metadata.members == member)
        part = e.smoke._subset(metadata, rows)
        candidates, reconstruction = e.smoke.v92._reconstruct_candidates(
            {"sequence": part.sequence, "mask": part.mask, "top_samples": part.top_samples})
        match = reconstruction["top_sample_match_fraction"]
        if match is None or match < .999:
            raise OuterError(f"candidate reconstruction failed: {member}")
        midi, midi_mask, supervision = e.smoke.derive_midi_supervision(
            tracks_by_member, part.members, candidates, part.slot_targets)
        local_targets, local_weights, diagnostic = e.training_targets(
            part.exact, part.slot_targets, midi, midi_mask)
        for name, values in local_targets.items():
            if name not in targets:
                targets[name] = np.empty((n, *values.shape[1:]), dtype=values.dtype)
                weights[name] = np.empty(n, dtype=np.float32)
            targets[name][rows], weights[name][rows] = values, local_weights[name]
        crops, feature_diagnostic = e.track_features(args.cqt_dir, cqt_manifest, member, candidates)
        features[rows] = crops
        if not np.isfinite(features[rows]).all():
            raise OuterError("non-finite float16 features")
        refs, _ = _reference_positions(tracks_by_member[member], feature_diagnostic["audio_sample_count"])
        reference_members.extend([member] * len(refs))
        reference_samples.extend(refs)
        per_track.append({"member": member, "targets": diagnostic, "supervision": supervision,
                          "features": feature_diagnostic})
        e.emit("prepared_track", track=number, total=len(metadata.track_members), member=member)
    features.flush()
    del features
    np.savez_compressed(output / "labels.npz", **arrays,
                        **{"target_" + name: value for name, value in targets.items()},
                        **{"weight_" + name: value for name, value in weights.items()})
    # Training commands never deserialize this evaluation-only file.
    np.savez_compressed(output / "evaluation.npz", top_samples=metadata.top_samples,
                        candidate_count=np.sum(metadata.mask, axis=1).astype(np.int32),
                        reference_member=np.asarray(reference_members),
                        reference_sample=np.asarray(reference_samples, dtype=np.int64))
    manifest = {"status": "complete", "contract": CONTRACT, "sources": SOURCES, **provenance(),
                "split": split, "selection": selection, "feature_shape": [n, *e.FEATURE_SHAPE],
                "track_count": len(metadata.track_members), "per_track": per_track,
                "historical_members_indexed_for_exclusion_only": len(historical),
                "all_development_targets_materialized_before_training": True,
                "cqt_verification": verification, "cqt_source_manifest": cqt_manifest,
                "files": file_records(output, ["features.npy", "labels.npz", "evaluation.npz", "split.json"])}
    e.smoke._atomic_json(output / "manifest.json", manifest)
    e.emit("preparation_complete", rows=n, tracks=len(metadata.track_members))


def load_prepared(directory, *, features_required=True):
    manifest = read_json(directory / "manifest.json")
    if manifest.get("status") != "complete" or manifest.get("contract") != CONTRACT or manifest.get("sources") != SOURCES:
        raise OuterError("prepared contract mismatch")
    names = {"features.npy", "labels.npz", "evaluation.npz", "split.json"}
    if set(manifest["files"]) != names:
        raise OuterError("unexpected prepared file set")
    # Aggregate jobs download only audit/evaluation files, not the large features.
    verify_files(directory, {k: v for k, v in manifest["files"].items()
                              if features_required or k != "features.npy"})
    if read_json(directory / "split.json") != manifest["split"]:
        raise OuterError("split manifest mismatch")
    with np.load(directory / "labels.npz", allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    n = len(arrays["global_index"])
    if n != EXPECTED_ROWS or not np.array_equal(arrays["global_index"], np.arange(n)):
        raise OuterError("prepared rows do not cover the full canonical index")
    if len(set(arrays["member"])) != 240 or int(np.sum(arrays["target_cardinality"] >= 2)) != EXPECTED_POLY_ROWS:
        raise OuterError("prepared track or polyphonic coverage changed")
    for name in e.LOSS_WEIGHTS:
        if len(arrays["target_" + name]) != n or arrays["weight_" + name].shape != (n,):
            raise OuterError("unaligned prepared targets")
    for f in range(FOLD_COUNT):
        if partition_record(arrays, f) != manifest["split"]["roles_by_fold"][str(f)]:
            raise OuterError("nested partition digest mismatch")
    features = np.load(directory / "features.npy", mmap_mode="r", allow_pickle=False) if features_required else None
    if features_required and (features.shape != (n, *e.FEATURE_SHAPE) or features.dtype != np.float16):
        raise OuterError("prepared feature shape or dtype mismatch")
    return features, arrays, manifest


def save_training_state(model, output):
    """Save model and Adam, including iteration and first/second moments."""
    model.save_weights(output / "last.weights.h5")
    variables = model.optimizer.variables()
    np.savez_compressed(output / "optimizer.npz", **{f"v{i}": v.numpy() for i, v in enumerate(variables)})
    return [{"name": v.name, "shape": list(v.shape), "dtype": v.dtype.name} for v in variables]


def restore_training_state(model, directory, specification):
    model.load_weights(directory / "last.weights.h5")
    model.optimizer.build(model.trainable_variables)
    variables = model.optimizer.variables()
    actual = [{"name": v.name, "shape": list(v.shape), "dtype": v.dtype.name} for v in variables]
    if actual != specification:
        raise OuterError("optimizer variable contract changed")
    with np.load(directory / "optimizer.npz", allow_pickle=False) as data:
        if set(data.files) != {f"v{i}" for i in range(len(variables))}:
            raise OuterError("optimizer state is incomplete")
        for i, variable in enumerate(variables):
            value = data[f"v{i}"]
            if value.shape != tuple(variable.shape) or not np.isfinite(value).all():
                raise OuterError("invalid optimizer state")
            variable.assign(value)


def load_state(directory, *, fold, phase, prepared_sha, allow_recovery=False):
    state = read_json(directory / "state.json")
    for name, expected in {"status": "complete", "contract": CONTRACT, "fold": fold,
                           "sources": SOURCES, "phase": phase,
                           "prepared_manifest_sha256": prepared_sha}.items():
        if state.get(name) != expected:
            raise OuterError(f"training state mismatch: {name}")
    validate_state_origin(state, directory / "state.json", fold=fold, phase=phase,
                          prepared_sha=prepared_sha, allow_recovery=allow_recovery)
    verify_files(directory, state["files"])
    if len(state["history"]) != state["epochs_completed"]:
        raise OuterError("incomplete epoch history")
    return state


def choose_epoch(history):
    if [entry["epoch"] for entry in history] != list(range(1, e.EPOCHS + 1)):
        raise OuterError("a complete 12-epoch inner probe is required")
    return max(history, key=lambda entry: e.checkpoint_key(entry["validation"]))


def train_chunk(args):
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    features, arrays, manifest = load_prepared(args.prepared_dir)
    prepared_sha = e.smoke._sha256_file(args.prepared_dir / "manifest.json")
    roles = partitions(arrays["row_fold"], arrays["member"], args.fold)
    fit = roles["fit" if args.phase == "probe" else "refit"]
    val = roles["validation"] if args.phase == "probe" else np.asarray([], dtype=np.int64)
    if (args.chunk == 2) != (args.previous_dir is not None):
        raise OuterError("only second chunks must restore the previous chunk")
    output = args.output_dir
    output.mkdir(parents=True)
    model, initial_sha = e.initialize_arm("harmonic")
    # Build slots explicitly, so their names and ordering are stable on resume.
    model.optimizer.build(model.trainable_variables)
    selection, history, previous_sha = None, [], None
    if args.chunk == 2:
        state = load_state(args.previous_dir, fold=args.fold, phase=args.phase, prepared_sha=prepared_sha)
        if state["chunk"] != 1 or state["epochs_completed"] != min(CHUNK_EPOCHS, state["epoch_budget"]):
            raise OuterError("second chunk requires the completed first chunk")
        restore_training_state(model, args.previous_dir, state["optimizer_variables"])
        selection, history = state["selection"], state["history"]
        previous_sha = e.smoke._sha256_file(args.previous_dir / "state.json")
        if (state["initial_shared_weights_sha256"] != initial_sha
                or state["partitions"] != manifest["split"]["roles_by_fold"][str(args.fold)]):
            raise OuterError("resume model initialization or partitions changed")
        for name in ("best.weights.h5", "validation-predictions.npz"):
            if (args.previous_dir / name).exists():
                shutil.copy2(args.previous_dir / name, output / name)
    elif args.phase == "refit":
        if args.probe_dir is None:
            raise OuterError("refit requires this fold's completed inner probe")
        probe = load_state(args.probe_dir, fold=args.fold, phase="probe", prepared_sha=prepared_sha)
        best = choose_epoch(probe["history"])
        if probe["epochs_completed"] != e.EPOCHS or probe["chunk"] != 2:
            raise OuterError("refit cannot start from a partial inner probe")
        selection = {"epoch": best["epoch"], "validation": best["validation"],
                     "probe_state_sha256": e.smoke._sha256_file(args.probe_dir / "state.json")}
        # No probe weights or optimizer state are loaded for this fresh refit.
        e.smoke._atomic_json(output / "selection.json", selection)
        e.emit("refit_budget_frozen", fold=args.fold, **selection)
    budget = e.EPOCHS if args.phase == "probe" else int(selection["epoch"])
    if budget not in range(1, e.EPOCHS + 1):
        raise OuterError("invalid inner-selected refit epoch budget")
    completed = len(history)
    expected_updates = completed * ((len(fit) + e.BATCH_SIZE - 1) // e.BATCH_SIZE)
    if int(model.optimizer.iterations.numpy()) != expected_updates:
        raise OuterError("optimizer iteration does not match completed epochs")
    k = arrays["target_cardinality"]
    groups = np.asarray([e.group_stem(str(m)) for m in arrays["member"][fit]])
    predict = tf.function(lambda x: model(x, training=False)["cardinality"],
                          input_signature=[tf.TensorSpec((None, *e.FEATURE_SHAPE), tf.float32)])
    best = max(history, key=lambda entry: e.checkpoint_key(entry["validation"])) if history and len(val) else None
    started = time.monotonic()
    e.emit("training_started", fold=args.fold, phase=args.phase, chunk=args.chunk,
           epochs_already_completed=completed, epoch_budget=budget, training_rows=len(fit),
           validation_rows=len(val), parameters=model.count_params())
    for epoch in range(completed + 1, min(args.chunk * CHUNK_EPOCHS, budget) + 1):
        epoch_start, losses = time.monotonic(), {}
        order = fit[e.epoch_order(groups, k[fit], epoch=epoch)]
        for offset in range(0, len(order), e.BATCH_SIZE):
            rows = order[offset:offset + e.BATCH_SIZE]
            weights = {name: arrays["weight_" + name][rows] for name in e.LOSS_WEIGHTS}
            weights["string_fret_onset"] = weights["string_fret_onset"][:, None]
            current = model.train_on_batch(np.asarray(features[rows], dtype=np.float32),
                {name: arrays["target_" + name][rows] for name in e.LOSS_WEIGHTS},
                sample_weight=weights, return_dict=True)
            if not np.isfinite(list(current.values())).all():
                raise OuterError("non-finite training loss")
            for name, value in current.items():
                losses[name] = losses.get(name, 0.0) + float(value) * len(rows)
            updates = int(model.optimizer.iterations.numpy())
            if offset == 0 or updates % 50 == 0:
                e.emit("training_batch_completed", fold=args.fold, phase=args.phase, epoch=epoch,
                       updates=updates, rows_seen=offset + len(rows), loss=float(current["loss"]))
            if epoch == completed + 1 and offset == 0 and os.environ.get("GITHUB_ACTIONS") == "true":
                print(f"::notice title=V28.0-F gradient update completed::Fold {args.fold}, "
                      f"{args.phase}, epoch {epoch}, {len(fit)} training rows", flush=True)
        entry = {"epoch": epoch, "updates": int(model.optimizer.iterations.numpy()),
                 "train_losses": {name: value / len(fit) for name, value in losses.items()},
                 "training_order_sha256": e.digest_array(arrays["global_index"][order])}
        if len(val):
            probability = np.empty((len(val), 7), dtype=np.float32)
            for offset in range(0, len(val), e.BATCH_SIZE):
                rows = val[offset:offset + e.BATCH_SIZE]
                probability[offset:offset + len(rows)] = predict(np.asarray(features[rows], dtype=np.float32)).numpy()
            entry["validation"] = e.count_metrics(k[val], probability)
            if best is None or e.checkpoint_key(entry["validation"]) > e.checkpoint_key(best["validation"]):
                best = entry
                model.save_weights(output / "best.weights.h5")
                np.savez_compressed(output / "validation-predictions.npz", global_index=arrays["global_index"][val],
                                    member=arrays["member"][val], k=k[val], probability=probability)
        entry["epoch_seconds"] = time.monotonic() - epoch_start
        history.append(entry)
        optimizer_variables = save_training_state(model, output)
        e.smoke._atomic_json(output / "progress.json", {"fold": args.fold, "phase": args.phase,
            "epochs_completed": epoch, "history": history, "optimizer_variables": optimizer_variables})
        e.emit("epoch_completed", fold=args.fold, phase=args.phase, **entry)
    optimizer_variables = save_training_state(model, output)
    state = {"status": "complete", "contract": CONTRACT, "sources": SOURCES, **provenance(),
             "fold": args.fold, "phase": args.phase, "chunk": args.chunk,
             "epoch_budget": budget, "epochs_completed": len(history), "history": history,
             "selection": selection, "previous_state_sha256": previous_sha,
             "prepared_manifest_sha256": prepared_sha,
             "partitions": manifest["split"]["roles_by_fold"][str(args.fold)],
             "initial_shared_weights_sha256": initial_sha, "parameters": model.count_params(),
             "optimizer_updates": int(model.optimizer.iterations.numpy()),
             "optimizer_variables": optimizer_variables,
             "chunk_seconds": time.monotonic() - started,
             "files": file_records(output, [p.name for p in output.iterdir() if p.suffix in (".h5", ".npz")])}
    e.smoke._atomic_json(output / "state.json", state)
    e.emit("chunk_complete", fold=args.fold, phase=args.phase, chunk=args.chunk,
           epochs_completed=len(history), epoch_budget=budget)


def validate_training_history(state, arrays):
    roles = partitions(arrays["row_fold"], arrays["member"], state["fold"])
    fit = roles["fit" if state["phase"] == "probe" else "refit"]
    groups = np.asarray([e.group_stem(str(m)) for m in arrays["member"][fit]])
    steps = (len(fit) + e.BATCH_SIZE - 1) // e.BATCH_SIZE
    if state["partitions"] != partition_record(arrays, state["fold"]):
        raise OuterError("training partition identities changed")
    if state["parameters"] != e.expected_parameter_count(harmonic=True):
        raise OuterError("unexpected model capacity")
    if [h["epoch"] for h in state["history"]] != list(range(1, state["epochs_completed"] + 1)):
        raise OuterError("epoch history has gaps or duplicates")
    for h in state["history"]:
        order = fit[e.epoch_order(groups, arrays["target_cardinality"][fit], epoch=h["epoch"])]
        if (h["updates"] != h["epoch"] * steps
                or h["training_order_sha256"] != e.digest_array(arrays["global_index"][order])):
            raise OuterError("training coverage or optimizer budget mismatch")
        if not np.isfinite(list(h["train_losses"].values())).all():
            raise OuterError("non-finite training history")
    if state["optimizer_updates"] != state["epochs_completed"] * steps:
        raise OuterError("final optimizer budget mismatch")


def verify_probe_predictions(probe, prediction, arrays):
    best = choose_epoch(probe["history"])
    val = partitions(arrays["row_fold"], arrays["member"], probe["fold"])["validation"]
    for key, expected in {"global_index": arrays["global_index"][val],
                          "member": arrays["member"][val], "k": arrays["target_cardinality"][val]}.items():
        if not np.array_equal(prediction[key], expected):
            raise OuterError("inner validation prediction identities mismatch")
    if not same_count_metrics(e.count_metrics(prediction["k"], prediction["probability"]), best["validation"]):
        raise OuterError("inner-selected checkpoint metrics do not reproduce")
    return best


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def evaluate(args):
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    recover = getattr(args, "recover_fold0", False)
    if recover and args.fold != 0:
        raise OuterError("saved-run recovery is limited to fold 0")
    features, arrays, _ = load_prepared(args.prepared_dir)
    prepared_sha = e.smoke._sha256_file(args.prepared_dir / "manifest.json")
    probe = load_state(args.probe_dir, fold=args.fold, phase="probe", prepared_sha=prepared_sha,
                       allow_recovery=recover)
    refit = load_state(args.refit_dir, fold=args.fold, phase="refit", prepared_sha=prepared_sha,
                       allow_recovery=recover)
    for state in (probe, refit):
        validate_training_history(state, arrays)
        if state["chunk"] != 2 or state["epochs_completed"] != state["epoch_budget"]:
            raise OuterError("outer inference requires both completed phases")
    best = verify_probe_predictions(probe, read_npz(args.probe_dir / "validation-predictions.npz"), arrays)
    selection = {"epoch": best["epoch"], "validation": best["validation"],
                 "probe_state_sha256": e.smoke._sha256_file(args.probe_dir / "state.json")}
    if refit["selection"] != selection or refit["epoch_budget"] != best["epoch"]:
        raise OuterError("refit does not match the frozen inner-only selection")
    output = args.output_dir
    output.mkdir(parents=True)
    # A reviewable freeze record is written before accessing any outer logits.
    e.smoke._atomic_json(output / "selection.json", selection)
    model, initial_sha = e.initialize_arm("harmonic")
    if initial_sha != refit["initial_shared_weights_sha256"]:
        raise OuterError("evaluation model initialization differs")
    model.load_weights(args.refit_dir / "last.weights.h5")
    outer = partitions(arrays["row_fold"], arrays["member"], args.fold)["outer"]
    predict = tf.function(lambda x: model(x, training=False)["cardinality"],
                          input_signature=[tf.TensorSpec((None, *e.FEATURE_SHAPE), tf.float32)])
    probability = np.empty((len(outer), 7), dtype=np.float32)
    batch_ms, started = [], time.monotonic()
    for offset in range(0, len(outer), e.BATCH_SIZE):
        rows = outer[offset:offset + e.BATCH_SIZE]
        batch = np.asarray(features[rows], dtype=np.float32)
        tick = time.perf_counter()
        probability[offset:offset + len(rows)] = predict(batch).numpy()
        batch_ms.append(1000.0 * (time.perf_counter() - tick))
    metrics = e.count_metrics(arrays["target_cardinality"][outer], probability)
    np.savez_compressed(output / "predictions.npz", global_index=arrays["global_index"][outer],
                        member=arrays["member"][outer], row_fold=arrays["row_fold"][outer],
                        k=arrays["target_cardinality"][outer], probability=probability)
    for source, name in ((args.probe_dir / "state.json", "probe-state.json"),
                         (args.probe_dir / "validation-predictions.npz", "probe-validation.npz"),
                         (args.probe_dir / "best.weights.h5", "inner-best.weights.h5"),
                         (args.refit_dir / "state.json", "refit-state.json"),
                         (args.refit_dir / "last.weights.h5", "outer.weights.h5")):
        shutil.copy2(source, output / name)
    report = {"status": "complete", "contract": CONTRACT, "sources": SOURCES, **provenance(),
              "recovery_source": RECOVERY_SOURCE if recover else None,
              "float_metric_abs_tolerance": FLOAT_METRIC_ABS_TOLERANCE,
              "fold": args.fold, "selection": selection, "outer": metrics,
              "prepared_manifest_sha256": prepared_sha,
              "outer_inference_passes": 1, "parameters": model.count_params(),
              "inference": {"seconds_including_compile_and_IO": time.monotonic() - started,
                  "first_batch_ms_including_compile": batch_ms[0],
                  "batch_wall_ms_p50_after_first": float(np.median(batch_ms[1:])) if len(batch_ms) > 1 else None,
                  "batch_wall_ms_p95_after_first": float(np.percentile(batch_ms[1:], 95)) if len(batch_ms) > 1 else None,
                  "amortized_network_ms_per_cluster": sum(batch_ms[1:]) / (len(outer) - e.BATCH_SIZE) if len(batch_ms) > 1 else None,
                  "streaming_single_cluster_latency_measured": False,
                  "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0},
              "files": file_records(output, [p.name for p in output.iterdir()])}
    e.smoke._atomic_json(output / "report.json", report)
    e.emit("outer_fold_complete", fold=args.fold, selected_epoch=best["epoch"], rows=len(outer),
           aggregate_decision_pending=True)


def validate_oof_parts(parts, arrays):
    """Reject incomplete, duplicated, permuted or mislabeled fold results."""
    if len(parts) != FOLD_COUNT:
        raise OuterError("all five folds are required")
    keys = {"global_index", "member", "row_fold", "k", "probability"}
    for fold, part in enumerate(parts):
        if set(part) != keys:
            raise OuterError("unexpected outer prediction fields")
        rows = partitions(arrays["row_fold"], arrays["member"], fold)["outer"]
        for key, expected in {"global_index": arrays["global_index"][rows],
                              "member": arrays["member"][rows], "row_fold": arrays["row_fold"][rows],
                              "k": arrays["target_cardinality"][rows]}.items():
            if not np.array_equal(part[key], expected):
                raise OuterError(f"outer prediction identities mismatch: fold {fold}, {key}")
        e.count_metrics(part["k"], part["probability"])
    merged = {key: np.concatenate([p[key] for p in parts]) for key in keys}
    order = np.argsort(merged["global_index"], kind="stable")
    merged = {key: value[order] for key, value in merged.items()}
    if not np.array_equal(merged["global_index"], arrays["global_index"]):
        raise OuterError("OOF rows must have exact one-time coverage")
    return merged


def discrete_metrics(k, prediction):
    # NLL/Brier are undefined for the selective V27.3 discrete override.
    if np.asarray(prediction).shape != np.asarray(k).shape or np.any((prediction < 0) | (prediction > 6)):
        raise OuterError("invalid discrete reference predictions")
    result = e.count_metrics(k, np.eye(7)[np.asarray(prediction, dtype=np.int64)])
    return {key: value for key, value in result.items() if key not in ("nll", "poly_nll", "brier")}


def event_metrics(arrays, evaluation, counts):
    predicted = _cached_prediction_map({"members": arrays["member"], "top_samples": evaluation["top_samples"]},
                                       np.arange(len(counts)), counts)
    reference_members = evaluation["reference_member"]
    result = {}
    for tolerance in TOLERANCES_MS:
        tp, refs, preds = 0, 0, 0
        for member in sorted(set(arrays["member"])):
            reference = evaluation["reference_sample"][reference_members == member].tolist()
            values = predicted.get(str(member), ())
            tp += len(match_boundaries(reference, values, milliseconds_to_samples(tolerance)))
            refs += len(reference)
            preds += len(values)
        result[str(tolerance)] = {"tp": tp, "fp": preds - tp, "fn": refs - tp,
                                  "precision": tp / preds if preds else 0.0,
                                  "recall": tp / refs if refs else 0.0,
                                  "f1": 2.0 * tp / (refs + preds) if refs + preds else 0.0}
    return result


def track_bootstrap(members, k, reference, prediction, *, replicates=BOOTSTRAP_REPLICATES):
    """Paired, micro-averaged exact-K delta; resample whole tracks."""
    tracks = sorted(set(members))
    totals, deltas = [], []
    correct_delta = (prediction == k).astype(np.int64) - (reference == k).astype(np.int64)
    for member in tracks:
        mask = (members == member) & (k >= 2)
        totals.append(int(np.sum(mask)))
        deltas.append(int(np.sum(correct_delta[mask])))
    totals, deltas = np.asarray(totals), np.asarray(deltas)
    rng = np.random.default_rng(e.SEED)
    chosen = rng.integers(0, len(tracks), size=(replicates, len(tracks)))
    denominators = totals[chosen].sum(axis=1)
    valid = denominators > 0
    values = 100.0 * deltas[chosen[valid]].sum(axis=1) / denominators[valid]
    if not len(values):
        raise OuterError("track bootstrap has no polyphonic observations")
    return {"unit": "track", "seed": e.SEED, "replicates": replicates,
            "valid_replicates": int(valid.sum()), "tracks": len(tracks),
            "delta_pp": 100.0 * int(deltas.sum()) / int(totals.sum()),
            "lower_95_pp": float(np.percentile(values, 2.5)),
            "upper_95_pp": float(np.percentile(values, 97.5)),
            "scope": "conditional_development_uncertainty_not_independent_confirmation"}


def aggregate(args):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    _, arrays, manifest = load_prepared(args.prepared_dir, features_required=False)
    prepared_sha = e.smoke._sha256_file(args.prepared_dir / "manifest.json")
    reports, parts = [], []
    for fold in range(FOLD_COUNT):
        root = args.input_dir / f"v280-f-fold-{fold}"
        report = read_json(root / "report.json")
        for name, expected in {"status": "complete", "fold": fold, "contract": CONTRACT,
                               "sources": SOURCES, "prepared_manifest_sha256": prepared_sha,
                               "outer_inference_passes": 1, **provenance()}.items():
            if report.get(name) != expected:
                raise OuterError(f"fold report mismatch: {fold}, {name}")
        verify_files(root, report["files"])
        probe, refit = read_json(root / "probe-state.json"), read_json(root / "refit-state.json")
        for state, phase in ((probe, "probe"), (refit, "refit")):
            if (state["status"] != "complete" or state["sources"] != SOURCES
                    or state["prepared_manifest_sha256"] != prepared_sha
                    or state["fold"] != fold or state["phase"] != phase or state["contract"] != CONTRACT
                    or state["chunk"] != 2 or state["epochs_completed"] != state["epoch_budget"]):
                raise OuterError("incomplete or mismatched nested phase")
            validate_state_origin(state, root / f"{phase}-state.json", fold=fold, phase=phase,
                                  prepared_sha=prepared_sha,
                                  allow_recovery=getattr(args, "recover_fold0", False))
            validate_training_history(state, arrays)
        best = verify_probe_predictions(probe, read_npz(root / "probe-validation.npz"), arrays)
        selection = {"epoch": best["epoch"], "validation": best["validation"],
                     "probe_state_sha256": e.smoke._sha256_file(root / "probe-state.json")}
        if report["selection"] != selection or refit["selection"] != selection or refit["epoch_budget"] != best["epoch"]:
            raise OuterError("inner-only refit selection mismatch")
        if (e.smoke._sha256_file(root / "outer.weights.h5") != refit["files"]["last.weights.h5"]["sha256"]
                or e.smoke._sha256_file(root / "probe-validation.npz") != probe["files"]["validation-predictions.npz"]["sha256"]):
            raise OuterError("evaluated weights or selected validation predictions changed")
        part = read_npz(root / "predictions.npz")
        if not same_count_metrics(e.count_metrics(part["k"], part["probability"]), report["outer"]):
            raise OuterError("outer metrics fail independent recomputation")
        reports.append(report)
        parts.append(part)
    merged = validate_oof_parts(parts, arrays)
    reference_root = args.reference_dir / "v273-summary"
    for name, digest in (("predictions.npz", REFERENCE_PREDICTIONS_SHA), ("report.json", REFERENCE_REPORT_SHA)):
        if e.smoke._sha256_file(reference_root / name) != digest:
            raise OuterError("frozen V27.3 payload changed")
    reference = read_npz(reference_root / "predictions.npz")
    for key, expected in {"global_index": merged["global_index"], "member": merged["member"],
                          "k": merged["k"], "outer_fold": merged["row_fold"]}.items():
        if not np.array_equal(reference[key], expected):
            raise OuterError("V27.3 and V28 identities are not paired")
    k, pred = merged["k"], merged["probability"].argmax(axis=1)
    base = reference["pred273_selective_transition"]
    baseline, current = discrete_metrics(k, base), e.count_metrics(k, merged["probability"])
    if baseline["poly_correct"] != 4005 or baseline["correct"] != 62558 or baseline["poly_rows"] != EXPECTED_POLY_ROWS:
        raise OuterError("V27.3 reference count metrics fail to reproduce")
    evaluation = read_npz(args.prepared_dir / "evaluation.npz")
    events = {"v273": event_metrics(arrays, evaluation, base),
              "v280": event_metrics(arrays, evaluation, pred),
              "true_k_oracle_same_ranking": event_metrics(arrays, evaluation, k)}
    if [events["v273"]["50"][key] for key in ("tp", "fp", "fn")] != [34404, 7859, 9816]:
        raise OuterError("V27.3 event counts fail to reproduce")
    per_fold = {}
    positive_folds = 0
    for f in range(FOLD_COUNT):
        mask = merged["row_fold"] == f
        b, c = discrete_metrics(k[mask], base[mask]), e.count_metrics(k[mask], merged["probability"][mask])
        delta = c["poly_correct"] - b["poly_correct"]
        positive_folds += int(delta > 0)
        per_fold[str(f)] = {"v273": b, "v280": c, "poly_correct_delta_rows": delta,
                            "inner_selected_epoch": reports[f]["selection"]["epoch"]}
    bootstrap = track_bootstrap(merged["member"], k, base, pred)
    delta_pp = 100.0 * (current["poly_exact_k"] - baseline["poly_exact_k"])
    gates = {"gain_at_least_5pp": delta_pp >= 5.0, "at_least_four_positive_folds": positive_folds >= 4,
             "bootstrap_lower_above_zero": bootstrap["lower_95_pp"] > 0.0}
    result = {"status": "complete", "contract": CONTRACT, "sources": SOURCES, **provenance(),
              "recovery_source": RECOVERY_SOURCE if getattr(args, "recover_fold0", False) else None,
              "float_metric_abs_tolerance": FLOAT_METRIC_ABS_TOLERANCE,
              "reference": baseline, "v280": current, "per_fold": per_fold,
              "delta_poly_pp": delta_pp, "delta_poly_correct_rows": current["poly_correct"] - baseline["poly_correct"],
              "track_bootstrap": bootstrap, "event_metrics": events,
              "candidate_shortage": {"rows_below_true_k": int(np.sum(evaluation["candidate_count"] < k)),
                  "rows_below_predicted_k": int(np.sum(evaluation["candidate_count"] < pred)),
                  "frozen_ranking_rows_below_true_k": int(np.sum((evaluation["top_samples"] >= 0).sum(axis=1) < k))},
              "material_gain_gates": gates, "material_gain_gates_passed": all(gates.values()),
              "reference_promoted": False, "v273_reference_preserved": True,
              "compute": {"fold_inference": {str(r["fold"]): r["inference"] for r in reports},
                          "cqt_source_verification": manifest["cqt_verification"]},
              "prepared_manifest_sha256": prepared_sha,
              "limitations": ["development compositions previously observed during architecture research",
                              "single seed; track bootstrap does not remove architecture-selection bias",
                              "batched CPU timing is not streaming deployment latency"]}
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(args.output_dir / "predictions.npz", **merged, reference_prediction=base)
    result["predictions_sha256"] = e.smoke._sha256_file(args.output_dir / "predictions.npz")
    e.smoke._atomic_json(args.output_dir / "report.json", result)
    lines = ["# V28.0-F — five-fold development OOF", "",
             "| Model | Polyphonic exact-K | Global exact-K | Event F1 @ 50 ms |",
             "|---|---:|---:|---:|"]
    for label, metric, event_key in (("V27.3", baseline, "v273"), ("V28 harmonic", current, "v280")):
        lines.append(f"| {label} | {metric['poly_exact_k']:.4%} | {metric['exact_k']:.4%} | {events[event_key]['50']['f1']:.4%} |")
    lines += ["", f"Polyphonic delta: {delta_pp:+.4f} percentage points; {positive_folds}/5 positive folds.",
              f"Paired track bootstrap 95% interval: [{bootstrap['lower_95_pp']:+.4f}, {bootstrap['upper_95_pp']:+.4f}] pp.",
              "", "Development OOF after prior architecture selection; not independent validation.",
              "V27.3 remains the reference. No automatic promotion or deployment."]
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    e.emit("outer_comparison_complete", poly_exact=current["poly_exact_k"], delta_pp=delta_pp,
           positive_folds=positive_folds, reference_promoted=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for name in ("dataset-dir", "cluster-dir", "cqt-dir", "selection-dir", "output-dir"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.set_defaults(func=prepare)
    fit = commands.add_parser("train-chunk")
    for name in ("prepared-dir", "output-dir"):
        fit.add_argument("--" + name, type=Path, required=True)
    fit.add_argument("--previous-dir", type=Path)
    fit.add_argument("--probe-dir", type=Path)
    fit.add_argument("--fold", type=int, choices=range(FOLD_COUNT), required=True)
    fit.add_argument("--phase", choices=("probe", "refit"), required=True)
    fit.add_argument("--chunk", type=int, choices=(1, 2), required=True)
    fit.set_defaults(func=train_chunk)
    evaluation = commands.add_parser("evaluate")
    for name in ("prepared-dir", "probe-dir", "refit-dir", "output-dir"):
        evaluation.add_argument("--" + name, type=Path, required=True)
    evaluation.add_argument("--fold", type=int, choices=range(FOLD_COUNT), required=True)
    evaluation.add_argument("--recover-fold0", action="store_true")
    evaluation.set_defaults(func=evaluate)
    summary = commands.add_parser("aggregate")
    for name in ("prepared-dir", "input-dir", "reference-dir", "output-dir"):
        summary.add_argument("--" + name, type=Path, required=True)
    summary.add_argument("--recover-fold0", action="store_true")
    summary.set_defaults(func=aggregate)
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
