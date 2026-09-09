"""V28.0-E: full internal training, harmonic versus no-harmonic control.

The canonical composition folds are reproduced using row identities/counts.
Fold 0 is reserved, fold 1 is internal validation, folds 2/3/4 are training.
Only the internal partitions receive MIDI targets and cluster crops. Both arms
visit every training row once per epoch, from scratch, with the same order.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts import mine_v280_causal_cqt as mining
from scripts import train_v280_real_smoke as smoke
from scripts.train_boundaries import group_stem
from scripts.train_v104_oof_fold import _balanced_group_folds
from scripts.train_v280_harmonic_count import (
    CARDINALITY_CLASSES, FRETS_PER_STRING, LOSS_WEIGHTS, STANDARD_TUNING_MIDI,
    build_model, expected_parameter_count, guitar_pitch_count,
)
from causal_note.v280_causal_cqt import CausalCQTConfig, cluster_feature_map, read_track_cache

SCHEMA = 1
SEED = 28035
EPOCHS = 12
BATCH_SIZE = 32
ARMS = ("harmonic", "no_harmonic")
CLUSTER_PAYLOAD_SHA256 = "666a5f55e62a9d8f78f45a432e771a998341ea2662980c400aa807edbe97f8ba"
CONFIG = CausalCQTConfig()
FEATURE_SHAPE = (CONFIG.cluster_frames, len(CONFIG.center_frequencies_hz), 3)
SOURCES = {
    "cluster_metadata": {
        "run_id": 34346175521,
        "head_sha": "97c158e8c8b110b85eeac80a336a06768a95b936",
        "artifact": "v280-cluster-metadata",
        "digest": "sha256:dccf400794cd8d50a9aab8b13a1292a1d6e97696315a96ba21beb5ec8e423d8c",
    },
    "guitarset": {
        "run_id": smoke.V272_RUN_ID, "head_sha": smoke.V272_HEAD_SHA,
        "artifact": smoke.V272_ARTIFACT, "digest": smoke.V272_ARTIFACT_DIGEST,
    },
    "causal_cqt": {
        "run_id": smoke.V280_CQT_RUN_ID, "head_sha": smoke.V280_CQT_HEAD_SHA,
        "artifact": smoke.V280_CQT_ARTIFACT, "digest": smoke.V280_CQT_ARTIFACT_DIGEST,
    },
}
CONTRACT = {
    "schema_version": SCHEMA,
    "experiment": "v280_e_internal_harmonic_ablation",
    "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
    "learning_rate": 2e-4, "loss_weights": LOSS_WEIGHTS,
    "reserved_fold": 0, "validation_fold": 1, "training_folds": [2, 3, 4],
    "feature_sha256": CONFIG.sha256,
    "end_of_stream": "flush_at_last_available_causal_frame_without_future_padding",
    "sampling": "each_training_row_once_group_interleaved_with_polyphonic_anchors",
    "augmentation": "none", "early_stopping": False,
    "checkpoint_selection": "max_internal_poly_exact_then_min_poly_nll_then_earlier_epoch",
    "arm_selection": "harmonic_only_if_strictly_more_internal_poly_correct",
    "shared_layer_initialization": "GlorotUniform_seeded_by_layer_name_zero_bias",
    "parameter_matched_control": False,
    "outer_predictions_computed": False,
    "historical_validation_jams_read": False, "locked12_indexed": False,
    "external_checkpoint_or_teacher_loaded": False, "promotion_authorized": False,
}


class AblationError(RuntimeError):
    pass


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def digest_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def split_rows(metadata, tracks):
    """Use the exact V10.4/V24 folding function without consulting target values."""
    assignment, groups, loads, _ = _balanced_group_folds(
        {"members": metadata.members}, tracks
    )
    row_fold = np.asarray(
        [assignment[group_stem(str(member))] for member in metadata.members], dtype=np.int16
    )
    roles = {
        "reserved": np.flatnonzero(row_fold == 0),
        "validation": np.flatnonzero(row_fold == 1),
        "train": np.flatnonzero(row_fold >= 2),
    }
    for name, rows in roles.items():
        if not len(rows):
            raise AblationError(f"empty {name} partition")
    for left, right in (("train", "validation"), ("train", "reserved"), ("validation", "reserved")):
        left_groups = {group_stem(str(metadata.members[i])) for i in roles[left]}
        right_groups = {group_stem(str(metadata.members[i])) for i in roles[right]}
        if left_groups & right_groups:
            raise AblationError("composition leakage between partitions")
    if len(np.unique(np.concatenate(list(roles.values())))) != metadata.row_count:
        raise AblationError("partitions do not cover the source rows exactly once")
    return row_fold, {
        "groups_per_fold": groups, "row_counts_per_fold": list(map(int, loads)),
        "global_row_count": metadata.row_count,
        "roles": {name: {"rows": len(rows), "indices_sha256": digest_array(rows)}
                  for name, rows in roles.items()},
    }


def training_targets(exact, strings, midi, midi_mask):
    """Retain every exact-K target, masking only incompatible auxiliary labels."""
    k = np.asarray(exact, dtype=np.int32)
    strings = np.asarray(strings, dtype=np.float32)
    midi = np.asarray(midi, dtype=np.float32)
    mask = np.asarray(midi_mask) > 0.5
    if k.ndim != 1 or strings.shape != (len(k), 6) or midi.shape != strings.shape or mask.shape != strings.shape:
        raise AblationError("unaligned target shapes")
    if np.any((k < 0) | (k >= CARDINALITY_CLASSES)):
        raise AblationError("invalid frozen exact-K labels")
    if not np.isfinite(midi[mask]).all():
        raise AblationError("non-finite active MIDI labels")
    consistent = np.sum(strings > 0.5, axis=1) == k
    aligned = np.all(mask == (strings > 0.5), axis=1)
    rounded = np.rint(midi).astype(np.int32)
    cents = 100.0 * np.abs(midi.astype(np.float64) - rounded)
    fret_index = rounded - np.asarray(STANDARD_TUNING_MIDI)[None, :]
    pitch_index = rounded - int(CONFIG.input_min_midi)
    representable = (
        (cents < 50.0 - smoke.MIDI_TIE_TOLERANCE_CENTS)
        & (fret_index >= 0) & (fret_index < FRETS_PER_STRING)
        & (pitch_index >= 0) & (pitch_index < guitar_pitch_count(CONFIG))
    )
    complete_pitch = aligned & np.all(~mask | representable, axis=1)
    usable = consistent & complete_pitch
    targets = {
        "cardinality": k, "string_birth": strings,
        "string_fret_onset": np.zeros((len(k), 6, FRETS_PER_STRING), dtype=np.float32),
        "pitch_onset": np.zeros((len(k), guitar_pitch_count(CONFIG)), dtype=np.float32),
        "poibin_cardinality": k,
    }
    valid_rows = np.flatnonzero(usable)
    if len(valid_rows):
        valid, _, _ = smoke.real_targets(k[valid_rows], strings[valid_rows], midi[valid_rows], mask[valid_rows])
        for name in ("string_fret_onset", "pitch_onset"):
            targets[name][valid_rows] = valid[name]
    weights = {
        "cardinality": np.ones(len(k), dtype=np.float32),
        "string_birth": consistent.astype(np.float32),
        "poibin_cardinality": consistent.astype(np.float32),
        "string_fret_onset": usable.astype(np.float32),
        "pitch_onset": usable.astype(np.float32),
    }
    active_cents = cents[mask]
    diagnostics = {
        "rows": len(k), "count_rows_discarded": 0,
        "inconsistent_count_string_rows": int(np.sum(~consistent)),
        "unaligned_midi_rows": int(np.sum(~aligned)),
        "incomplete_or_ambiguous_pitch_rows": int(np.sum(~complete_pitch)),
        "pitch_auxiliary_rows_masked": int(np.sum(~usable)),
        "active_midi_labels": int(np.sum(mask)),
        "max_absolute_quantization_cents": float(np.max(active_cents)) if len(active_cents) else None,
    }
    return targets, weights, diagnostics


def epoch_order(groups, k, *, epoch, batch_size=BATCH_SIZE):
    """All rows once; interleave groups and seed batches with polyphonic rows."""
    groups, k = np.asarray(groups), np.asarray(k)
    if groups.shape != k.shape or groups.ndim != 1 or not len(k) or batch_size < 2:
        raise AblationError("invalid sampler inputs")
    rng = np.random.default_rng(SEED + epoch)
    queues = {str(g): deque(rng.permutation(np.flatnonzero(groups == g)).tolist())
              for g in sorted(set(groups))}
    base = []
    while queues:
        for group in rng.permutation(sorted(queues)):
            base.append(queues[group].popleft())
            if not queues[group]:
                del queues[group]
    base = np.asarray(base, dtype=np.int64)
    batches = (len(k) + batch_size - 1) // batch_size
    anchors = base[k[base] >= 2][:batches]
    reserved = set(anchors.tolist())
    rest = deque(int(i) for i in base if int(i) not in reserved)
    order = []
    for batch in range(batches):
        size = min(batch_size, len(k) - len(order))
        if batch < len(anchors):
            order.append(int(anchors[batch]))
            size -= 1
        order.extend(rest.popleft() for _ in range(size))
    return np.asarray(order, dtype=np.int64)


def count_metrics(k, probability):
    k, p = np.asarray(k, dtype=np.int64), np.asarray(probability, dtype=np.float64)
    if k.ndim != 1 or not len(k) or p.shape != (len(k), 7):
        raise AblationError("invalid probability/target shapes")
    if np.any((k < 0) | (k > 6)) or not np.isfinite(p).all() or np.any(p < 0):
        raise AblationError("invalid probability/target values")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-5):
        raise AblationError("probabilities are not normalized")
    pred = p.argmax(axis=1)
    correct, poly = pred == k, k >= 2
    if not np.any(poly):
        raise AblationError("internal validation has no polyphonic rows")
    nll = -np.log(np.maximum(p[np.arange(len(k)), k], 1e-7))
    matrix = np.zeros((7, 7), dtype=np.int64)
    np.add.at(matrix, (k, pred), 1)
    return {
        "rows": len(k), "poly_rows": int(np.sum(poly)),
        "correct": int(np.sum(correct)), "poly_correct": int(np.sum(correct & poly)),
        "exact_k": float(np.mean(correct)), "poly_exact_k": float(np.mean(correct[poly])),
        "nll": float(np.mean(nll)), "poly_nll": float(np.mean(nll[poly])),
        "brier": float(np.mean(np.sum((p - np.eye(7)[k]) ** 2, axis=1))),
        "confusion_matrix_true_by_predicted": matrix.tolist(),
        "by_true_k": {str(i): {"rows": int(np.sum(k == i)),
                               "correct": int(np.sum(correct & (k == i))),
                               "undercount": int(np.sum((pred < k) & (k == i))),
                               "overcount": int(np.sum((pred > k) & (k == i)))}
                      for i in range(7)},
    }


def checkpoint_key(metrics):
    return metrics["poly_correct"], -metrics["poly_nll"]


def track_features(cache_dir, manifest, member, candidates, config=CONFIG):
    """Keep terminal clusters by flushing the decision at the actual audio EOF."""
    records = [record for record in manifest["cache"]["tracks"] if record["annotation_member"] == member]
    if len(records) != 1:
        raise AblationError(f"missing or duplicate CQT track: {member}")
    track = read_track_cache(Path(cache_dir) / records[0]["cache_path"], config)
    features = np.empty((len(candidates), config.cluster_frames, len(config.center_frequencies_hz), 3),
                        dtype=np.float32)
    starts, ends, clipped, missing = [], [], [], []
    for row, samples in enumerate(candidates):
        if not len(samples):
            raise AblationError(f"empty candidate cluster for {member}")
        start = int(np.min(samples))
        if start < 0 or start > track.sample_count:
            raise AblationError(f"cluster starts outside the available recording: {member}")
        requested_end = start + config.cluster_post_samples
        end = min(requested_end, track.sample_count)
        if end < requested_end:
            clipped.append(row)
            missing.append(requested_end - end)
        features[row] = cluster_feature_map(track, start, config, decision_end=int(end))
        starts.append(start)
        ends.append(end)
    return features, {
        "selected_track_count": 1, "cluster_start_min": min(starts), "cluster_start_max": max(starts),
        "decision_end_min": min(ends), "decision_end_max": max(ends),
        "audio_sample_count": int(track.sample_count),
        "end_of_stream_clipped_rows": len(clipped), "end_of_stream_clipped_local_indices": clipped,
        "missing_post_samples_max": max(missing, default=0),
        "feature_min": float(np.min(features)), "feature_max": float(np.max(features)),
        "feature_mean": float(np.mean(features)),
    }


def prepare(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if smoke._sha256_file(args.cluster_dir / "v280-cluster-metadata.npz") != CLUSTER_PAYLOAD_SHA256:
        raise AblationError("cluster payload differs from the preserved V28.0-D source")
    metadata = smoke.load_distilled_cluster_cache(args.cluster_dir)
    indexed, tracks, historical = mining._dataset_split(args.dataset_dir)
    membership = mining.V100Membership(metadata.track_members, metadata.shard_paths,
                                       mining._member_digest(metadata.track_members))
    selection = mining.validate_outer_clean_selection(indexed, tracks, historical, membership)
    row_fold, split = split_rows(metadata, selection.tracks)
    keep = np.flatnonzero(row_fold != 0)
    inner = smoke._subset(metadata, keep)
    output.mkdir(parents=True)
    smoke._atomic_json(output / "split.json", split)
    emit("internal_split_frozen", **split)
    verification = mining.verify_cache(args.cqt_dir)
    cqt_manifest = json.loads((args.cqt_dir / "manifest.json").read_text())
    if tuple(r["annotation_member"] for r in cqt_manifest["cache"]["tracks"]) != metadata.track_members:
        raise AblationError("cluster and CQT memberships differ")
    features = np.lib.format.open_memmap(output / "features.npy", mode="w+", dtype=np.float16,
                                         shape=(len(keep), *FEATURE_SHAPE))
    targets, weights, per_track = {}, {}, []
    tracks_by_member = {track.annotation_member: track for track in selection.tracks}
    for number, member in enumerate(inner.track_members, start=1):
        rows = np.flatnonzero(inner.members == member)
        part = smoke._subset(inner, rows)
        candidates, reconstruction = smoke.v92._reconstruct_candidates(
            {"sequence": part.sequence, "mask": part.mask, "top_samples": part.top_samples}
        )
        match = reconstruction["top_sample_match_fraction"]
        if match is None or match < 0.999:
            raise AblationError(f"candidate reconstruction failed for {member}")
        midi, midi_mask, supervision = smoke.derive_midi_supervision(
            tracks_by_member, part.members, candidates, part.slot_targets
        )
        local_targets, local_weights, diagnostics = training_targets(
            part.exact, part.slot_targets, midi, midi_mask
        )
        for name, values in local_targets.items():
            if name not in targets:
                targets[name] = np.empty((len(keep), *values.shape[1:]), dtype=values.dtype)
                weights[name] = np.empty(len(keep), dtype=np.float32)
            targets[name][rows] = values
            weights[name][rows] = local_weights[name]
        crops, feature_diagnostics = track_features(args.cqt_dir, cqt_manifest, member, candidates)
        features[rows] = crops
        per_track.append({"member": member, "targets": diagnostics,
                          "supervision": supervision, "features": feature_diagnostics})
        emit("prepared_track", track=number, total=len(inner.track_members), member=member, rows=len(rows))
    features.flush()
    del features
    np.savez_compressed(output / "labels.npz", global_index=keep, row_fold=row_fold[keep],
                        member=inner.members,
                        **{"target_" + name: values for name, values in targets.items()},
                        **{"weight_" + name: values for name, values in weights.items()})
    manifest = {
        "status": "complete", "contract": CONTRACT, "sources": SOURCES,
        "prepared_at_utc": smoke._utc_now(), "source_head_sha": os.environ.get("GITHUB_SHA"),
        "source_run_id": os.environ.get("GITHUB_RUN_ID"),
        "split": split, "feature_shape": [len(keep), *FEATURE_SHAPE],
        "full_outer_clean_supervision_arrays_deserialized": True,
        "reserved_fold_jams_read_or_rows_materialized": False,
        "historical_validation_member_names_indexed_for_exclusion_only": True,
        "outer_clean_track_count": len(metadata.track_members),
        "historical_validation_track_count": len(historical),
        "inner_track_count": len(inner.track_members), "per_track": per_track,
        "cqt_verification": verification,
        "cluster_manifest": json.loads((args.cluster_dir / "manifest.json").read_text()),
        "files": {name: {"sha256": smoke._sha256_file(output / name),
                         "bytes": (output / name).stat().st_size}
                  for name in ("features.npy", "labels.npz", "split.json")},
    }
    smoke._atomic_json(output / "manifest.json", manifest)
    emit("preparation_complete", rows=len(keep), tracks=len(inner.track_members))


def load_prepared(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("contract") != CONTRACT or manifest.get("sources") != SOURCES:
        raise AblationError("prepared source contract differs from the frozen experiment")
    if set(manifest["files"]) != {"features.npy", "labels.npz", "split.json"}:
        raise AblationError("unexpected prepared file set")
    for name, record in manifest["files"].items():
        if smoke._sha256_file(directory / name) != record["sha256"]:
            raise AblationError(f"prepared file digest mismatch: {name}")
    with np.load(directory / "labels.npz", allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    features = np.load(directory / "features.npy", mmap_mode="r", allow_pickle=False)
    n = len(arrays["global_index"])
    if features.shape != (n, *FEATURE_SHAPE) or list(features.shape) != manifest["feature_shape"]:
        raise AblationError("prepared feature shape mismatch")
    if len(np.unique(arrays["global_index"])) != n or set(np.unique(arrays["row_fold"])) != {1, 2, 3, 4}:
        raise AblationError("duplicate row or reserved fold in prepared data")
    fit, val = np.flatnonzero(arrays["row_fold"] >= 2), np.flatnonzero(arrays["row_fold"] == 1)
    for name, indices in (("train", fit), ("validation", val)):
        if digest_array(arrays["global_index"][indices]) != manifest["split"]["roles"][name]["indices_sha256"]:
            raise AblationError("prepared partition digest mismatch")
    fit_groups = {group_stem(str(m)) for m in arrays["member"][fit]}
    val_groups = {group_stem(str(m)) for m in arrays["member"][val]}
    if fit_groups & val_groups:
        raise AblationError("prepared composition leakage")
    return features, arrays, manifest, fit, val


def initialize_arm(arm):
    """Match shared-layer initial weights despite arm-specific extra layers."""
    import tensorflow as tf
    model = build_model(CONFIG, seed=SEED, harmonic=arm == "harmonic")
    shared = hashlib.sha256()
    for layer in sorted(model.layers, key=lambda item: item.name):
        if not layer.weights:
            continue
        if not hasattr(layer, "kernel") or not hasattr(layer, "bias"):
            raise AblationError(f"unhandled weighted layer: {layer.name}")
        layer_seed = int(hashlib.sha256(f"{SEED}:{layer.name}".encode()).hexdigest()[:8], 16) % (2**31 - 1)
        if layer.name == "v280_cardinality_residual_logits":
            kernel = np.zeros(layer.kernel.shape, dtype=np.float32)
        else:
            kernel = tf.keras.initializers.GlorotUniform(seed=layer_seed)(layer.kernel.shape).numpy()
        bias = np.zeros(layer.bias.shape, dtype=np.float32)
        layer.set_weights([kernel, bias])
        if "_harmonic_" not in layer.name:
            shared.update(layer.name.encode())
            shared.update(kernel.tobytes())
            shared.update(bias.tobytes())
    return model, shared.hexdigest()


def train(args):
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    features, arrays, manifest, fit, val = load_prepared(args.prepared_dir)
    output.mkdir(parents=True)
    model, initial_sha = initialize_arm(args.arm)
    k = arrays["target_cardinality"]
    groups = np.asarray([group_stem(str(m)) for m in arrays["member"][fit]])
    targets = {name: arrays["target_" + name] for name in LOSS_WEIGHTS}
    weights = {name: arrays["weight_" + name] for name in LOSS_WEIGHTS}
    history, best = [], None
    updates = 0
    start = time.monotonic()
    predict = tf.function(lambda x: model(x, training=False)["cardinality"],
                          input_signature=[tf.TensorSpec((None, *FEATURE_SHAPE), tf.float32)])
    emit("training_started", arm=args.arm, train_rows=len(fit), validation_rows=len(val),
         epochs=EPOCHS, batch_size=BATCH_SIZE, parameters=model.count_params())
    for epoch in range(1, EPOCHS + 1):
        order = fit[epoch_order(groups, k[fit], epoch=epoch)]
        losses = {}
        epoch_start = time.monotonic()
        for offset in range(0, len(order), BATCH_SIZE):
            rows = order[offset:offset + BATCH_SIZE]
            batch_weights = {name: values[rows] for name, values in weights.items()}
            batch_weights["string_fret_onset"] = batch_weights["string_fret_onset"][:, None]
            current = model.train_on_batch(np.asarray(features[rows], dtype=np.float32),
                {name: values[rows] for name, values in targets.items()},
                sample_weight=batch_weights, return_dict=True)
            if not np.isfinite(list(current.values())).all():
                raise AblationError("non-finite training loss")
            for name, value in current.items():
                losses[name] = losses.get(name, 0.0) + float(value) * len(rows)
            updates += 1
            if updates == 1 and os.environ.get("GITHUB_ACTIONS") == "true":
                print(f"::notice title=V28.0-E training started::{args.arm}: first gradient update completed; "
                      f"{len(fit)} training rows, {EPOCHS} epochs, batch size {BATCH_SIZE}", flush=True)
            if offset == 0 or updates % 50 == 0:
                emit("training_batch_completed", arm=args.arm, epoch=epoch, updates=updates,
                     rows_seen_in_epoch=offset + len(rows), loss=float(current["loss"]))
        probability = np.empty((len(val), 7), dtype=np.float32)
        for offset in range(0, len(val), BATCH_SIZE):
            rows = val[offset:offset + BATCH_SIZE]
            probability[offset:offset + len(rows)] = predict(np.asarray(features[rows], dtype=np.float32)).numpy()
        metrics = count_metrics(k[val], probability)
        entry = {"epoch": epoch, "updates": updates,
                 "train_losses": {name: value / len(fit) for name, value in losses.items()},
                 "validation": metrics, "epoch_seconds": time.monotonic() - epoch_start,
                 "training_order_sha256": digest_array(arrays["global_index"][order])}
        history.append(entry)
        if best is None or checkpoint_key(metrics) > checkpoint_key(best["validation"]):
            best = entry
            model.save_weights(output / "best.weights.h5")
            np.savez_compressed(output / "validation-predictions.npz",
                global_index=arrays["global_index"][val], member=arrays["member"][val],
                k=k[val], probability=probability)
        model.save_weights(output / "last.weights.h5")
        smoke._atomic_json(output / "progress.json", {
            "arm": args.arm, "epochs_completed": epoch, "contract": CONTRACT,
            "best_epoch": best["epoch"], "history": history,
        })
        emit("epoch_completed", arm=args.arm, epoch=epoch, best_epoch=best["epoch"],
             train_loss=entry["train_losses"]["loss"], validation=metrics,
             epoch_seconds=entry["epoch_seconds"])
    report = {
        "status": "complete", "arm": args.arm, "contract": CONTRACT, "sources": SOURCES,
        "source_head_sha": os.environ.get("GITHUB_SHA"), "source_run_id": os.environ.get("GITHUB_RUN_ID"),
        "prepared_manifest_sha256": smoke._sha256_file(args.prepared_dir / "manifest.json"),
        "shared_initial_weights_sha256": initial_sha,
        "trainable_parameters": model.count_params(), "epochs_completed": len(history),
        "training_rows": len(fit), "validation_rows": len(val), "optimizer_updates": updates,
        "best_epoch": best["epoch"], "selected_validation": best["validation"],
        "history": history, "elapsed_seconds": time.monotonic() - start,
        "weights_sha256": smoke._sha256_file(output / "best.weights.h5"),
        "predictions_sha256": smoke._sha256_file(output / "validation-predictions.npz"),
    }
    smoke._atomic_json(output / "report.json", report)
    emit("arm_completed", arm=args.arm, best_epoch=best["epoch"], validation=best["validation"])


def compare(args):
    reports, predictions = {}, {}
    for arm in ARMS:
        root = args.input_dir / f"v280-e-{arm}"
        report = json.loads((root / "report.json").read_text())
        if (report["status"] != "complete" or report["arm"] != arm or report["contract"] != CONTRACT
                or report["sources"] != SOURCES or report["epochs_completed"] != EPOCHS):
            raise AblationError(f"incomplete or mismatched arm: {arm}")
        if report["trainable_parameters"] != expected_parameter_count(harmonic=arm == "harmonic"):
            raise AblationError("parameter contract mismatch")
        expected_updates = EPOCHS * ((report["training_rows"] + BATCH_SIZE - 1) // BATCH_SIZE)
        if report["optimizer_updates"] != expected_updates or len(report["history"]) != EPOCHS:
            raise AblationError("incomplete epoch or optimizer budget")
        best = max(report["history"], key=lambda entry: checkpoint_key(entry["validation"]))
        if report["best_epoch"] != best["epoch"] or report["selected_validation"] != best["validation"]:
            raise AblationError("checkpoint selection differs from the frozen rule")
        for name, key in (("validation-predictions.npz", "predictions_sha256"), ("best.weights.h5", "weights_sha256")):
            if smoke._sha256_file(root / name) != report[key]:
                raise AblationError(f"arm file digest mismatch: {name}")
        with np.load(root / "validation-predictions.npz", allow_pickle=False) as data:
            prediction = {name: np.asarray(data[name]) for name in data.files}
        if count_metrics(prediction["k"], prediction["probability"]) != report["selected_validation"]:
            raise AblationError("recomputed validation metrics disagree")
        reports[arm], predictions[arm] = report, prediction
    harmonic, control = reports["harmonic"], reports["no_harmonic"]
    for key in ("prepared_manifest_sha256", "shared_initial_weights_sha256", "training_rows",
                "validation_rows", "optimizer_updates", "source_head_sha", "source_run_id"):
        if harmonic[key] != control[key]:
            raise AblationError(f"unpaired arm comparison: {key}")
    for name in ("global_index", "member", "k"):
        if not np.array_equal(predictions["harmonic"][name], predictions["no_harmonic"][name]):
            raise AblationError("validation identities differ between arms")
    for h, c in zip(harmonic["history"], control["history"]):
        if h["training_order_sha256"] != c["training_order_sha256"]:
            raise AblationError("training row order differs between arms")
    hm, cm = harmonic["selected_validation"], control["selected_validation"]
    selected = "harmonic" if hm["poly_correct"] > cm["poly_correct"] else "no_harmonic"
    result = {
        "status": "complete", "contract": CONTRACT,
        "decision_scope": "single_internal_development_split_only",
        "internal_preference": selected,
        "poly_exact_delta_percentage_points": 100.0 * (hm["poly_exact_k"] - cm["poly_exact_k"]),
        "poly_correct_delta_rows": hm["poly_correct"] - cm["poly_correct"],
        "v273_reference_preserved": True, "outer_evaluation_launched": False,
        "arms": reports,
    }
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    smoke._atomic_json(args.output_dir / "comparison.json", result)
    emit("comparison_complete", preference=selected, harmonic_poly_exact=hm["poly_exact_k"],
         control_poly_exact=cm["poly_exact_k"], delta_rows=result["poly_correct_delta_rows"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for name in ("dataset-dir", "cluster-dir", "cqt-dir", "output-dir"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.set_defaults(func=prepare)
    fit = commands.add_parser("train")
    fit.add_argument("--prepared-dir", type=Path, required=True)
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument("--arm", choices=ARMS, required=True)
    fit.set_defaults(func=train)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--input-dir", type=Path, required=True)
    comparison.add_argument("--output-dir", type=Path, required=True)
    comparison.set_defaults(func=compare)
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
