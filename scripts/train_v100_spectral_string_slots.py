"""V10.0 time-frequency string-slot cardinality decoder.

V9.2 established that the six physical GuitarSet strings are an almost exact
factorization of 40 ms cluster cardinality (99.99% count agreement), but adding
six string heads on the frozen V8.8 cluster representation did not beat V9.1.
The V8.6 acoustic representation aggregates each horizon into pre/post spectra,
which is useful for transition verification but collapses the fine time-frequency
structure needed to separate simultaneous attacks.

V10 keeps V8.4/V8.6/V8.7/V8.8 frozen as proposal/context machinery and adds a
new acoustic branch over a causal 40 ms cluster observation. Each cluster gets
a 23 x 64 x 3 time-frequency transition map:
  * normalized log spectral power,
  * positive change against the pre-cluster spectral baseline,
  * positive frame-to-frame spectral flux.
The map is fused with the frozen candidate-set representation and supervised by
six physical string-birth heads plus the V9.1 ordinal count heads. String labels
are training-only; runtime inputs are audio and frozen model outputs.

Two subcommands are provided. ``mine`` converts one V9.1 cache shard into a
spectral cache without rerunning the frozen neural stack. ``train-eval`` merges
spectral shards, performs a composition-safe train-only holdout calibration,
and evaluates once on the historical locked12 benchmark.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_note.guitarset import ALLOWED_PLAYERS, SAMPLE_RATE, SLOT_COUNT, index_guitarset
from scripts.evaluate_v8_boundaries import DEFAULT_SEED, DEFAULT_VALIDATION_FRACTION
from scripts.train_boundaries import decode_pcm16_mono_wav, group_stem, split_tracks_by_group
from scripts.train_v86_state_transition_proposals import MAX_HORIZON, TOLERANCE_MS, _candidate_ceiling
from scripts.train_v88_regime_moe import _retained_predictions
from scripts.evaluate_v90_cluster_oracles import _metrics
from scripts.train_v90_structured_cluster_cardinality import (
    CLUSTER_WINDOW_MS,
    CLUSTER_WINDOW_SAMPLES,
    _build_cluster_model,
    _cluster_data,
    _model_inputs,
    _prediction_map,
    _represent_full,
)
from scripts.train_v91_ordinal_cardinality import (
    ORDINAL_STAGES,
    _conditional_sample_weights,
    _cumulative,
    _dataset_split,
    _load_caches,
    _load_frozen_stack,
    _split_train_groups,
)
from scripts.train_v92_string_factorized_cardinality import (
    _cardinality_report,
    _derive_cache_slot_targets,
    _reconstruct_candidates,
    _slot_report,
    _slot_targets_for_runtime_clusters,
)

DEFAULT_SEED_V100 = 10031
CACHE_SCHEMA_VERSION = 1
PRE_SAMPLES = 1308
POST_SAMPLES = CLUSTER_WINDOW_SAMPLES  # 1764 ~= 40 ms
SEGMENT_SAMPLES = PRE_SAMPLES + POST_SAMPLES  # 3072
FRAME_LENGTH = 256
FRAME_STEP = 128
FFT_SIZE = 2048
SPECTRAL_BANDS = 64
MIN_HZ = 55.0
MAX_HZ = 6000.0
TIME_FRAMES = 1 + (SEGMENT_SAMPLES - FRAME_LENGTH) // FRAME_STEP
SPECTRAL_CHANNELS = 3
DECODE_MODES = (
    "slot_threshold_040",
    "slot_threshold_045",
    "slot_threshold_050",
    "slot_threshold_055",
    "slot_threshold_060",
    "slot_expected_round",
    "ordinal_cumulative_050",
    "hybrid_expected_25slot",
    "hybrid_expected_50slot",
    "hybrid_expected_75slot",
)


class V100Error(RuntimeError):
    pass


def _pcm_window(samples: np.ndarray, start: int, length: int) -> np.ndarray:
    out = np.zeros(length, dtype=np.float32)
    left = max(0, int(start))
    right = min(len(samples), int(start) + length)
    if right > left:
        dest = left - int(start)
        out[dest:dest + right - left] = samples[left:right]
    return out


def _spectral_map_from_segment(segment: np.ndarray) -> np.ndarray:
    if segment.shape != (SEGMENT_SAMPLES,):
        raise V100Error(f"unexpected segment shape {segment.shape}")
    taper = np.hanning(FRAME_LENGTH).astype(np.float32)
    freqs = np.fft.rfftfreq(FFT_SIZE, d=1.0 / SAMPLE_RATE)
    targets = np.geomspace(MIN_HZ, MAX_HZ, SPECTRAL_BANDS)
    power_rows = []
    for start in range(0, SEGMENT_SAMPLES - FRAME_LENGTH + 1, FRAME_STEP):
        frame = segment[start:start + FRAME_LENGTH] * taper
        power = np.abs(np.fft.rfft(frame, n=FFT_SIZE)) ** 2
        power_rows.append(np.interp(targets, freqs, power))
    power = np.asarray(power_rows, dtype=np.float64)
    if power.shape != (TIME_FRAMES, SPECTRAL_BANDS):
        raise V100Error(f"unexpected spectral map base shape {power.shape}")
    eps = 1e-10
    pre_frame_count = max(2, PRE_SAMPLES // FRAME_STEP - 1)
    pre = power[:pre_frame_count]
    scalar = float(np.median(np.mean(pre, axis=1))) + eps
    pre_band = np.mean(pre, axis=0) + eps
    log_power = np.clip(np.log1p(power / scalar), 0.0, 12.0)
    positive_pre = np.clip(np.log1p(np.maximum(power - pre_band[None, :], 0.0) / pre_band[None, :]), 0.0, 12.0)
    flux = np.zeros_like(log_power)
    flux[1:] = np.maximum(log_power[1:] - log_power[:-1], 0.0)
    result = np.stack((log_power, positive_pre, flux), axis=-1).astype(np.float32)
    return result


def _spectral_maps_for_cache(cache, dataset_dir: Path):
    indexed = tuple(t for t in index_guitarset(dataset_dir) if t.player_id in ALLOWED_PLAYERS)
    by_member = {t.annotation_member: t for t in indexed}
    candidate_samples, reconstruction = _reconstruct_candidates(cache)
    maps = np.zeros((len(cache["target"]), TIME_FRAMES, SPECTRAL_BANDS, SPECTRAL_CHANNELS), dtype=np.float16)
    by_member_rows: Dict[str, List[int]] = defaultdict(list)
    for i, member in enumerate(cache["members"]):
        by_member_rows[str(member)].append(i)
    for ordinal, (member, ids) in enumerate(sorted(by_member_rows.items()), start=1):
        track = by_member.get(member)
        if track is None:
            raise V100Error(f"cache member missing from dataset: {member}")
        audio = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        samples = np.asarray(audio.samples, dtype=np.float32) / 32768.0
        for idx in ids:
            candidates = candidate_samples[idx]
            if not len(candidates):
                raise V100Error("empty reconstructed candidate cluster")
            cluster_start = int(np.min(candidates))
            segment = _pcm_window(samples, cluster_start - PRE_SAMPLES, SEGMENT_SAMPLES)
            maps[idx] = _spectral_map_from_segment(segment).astype(np.float16)
        print(f"spectral {ordinal}/{len(by_member_rows)}: {member} clusters={len(ids)}")
    return maps, reconstruction


def _spectral_maps_for_runtime(tracks, clusters, records):
    maps = np.zeros((len(clusters), TIME_FRAMES, SPECTRAL_BANDS, SPECTRAL_CHANNELS), dtype=np.float32)
    by_member_ids: Dict[str, List[int]] = defaultdict(list)
    for cid, cluster in enumerate(clusters):
        by_member_ids[cluster["member"]].append(cid)
    by_member_track = {t.annotation_member: t for t in tracks}
    for member, ids in by_member_ids.items():
        track = by_member_track[member]
        audio = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        samples = np.asarray(audio.samples, dtype=np.float32) / 32768.0
        for cid in ids:
            cluster = clusters[cid]
            cluster_start = min(int(records[i]["sample"]) for i in cluster["indices"])
            segment = _pcm_window(samples, cluster_start - PRE_SAMPLES, SEGMENT_SAMPLES)
            maps[cid] = _spectral_map_from_segment(segment)
    return maps


def _save_spectral_cache(path: Path, cache, spectral, slots):
    np.savez(
        path,
        schema_version=np.asarray([CACHE_SCHEMA_VERSION], dtype=np.int16),
        spectral=np.asarray(spectral, dtype=np.float16),
        sequence=np.asarray(cache["sequence"], dtype=np.float16),
        mask=np.asarray(cache["mask"], dtype=np.uint8),
        stats=np.asarray(cache["stats"], dtype=np.float16),
        target=np.asarray(cache["target"], dtype=np.uint8),
        exact=np.asarray(cache["exact"], dtype=np.uint8),
        members=np.asarray(cache["members"], dtype="U96"),
        top_samples=np.asarray(cache["top_samples"], dtype=np.int32),
        slot_targets=np.asarray(slots, dtype=np.uint8),
        track_members=np.asarray(cache["track_members"], dtype="U96"),
    )


def mine(args):
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    cache = _load_caches(args.cache_dir)
    slots, _, slot_diag = _derive_cache_slot_targets(cache, args.dataset_dir)
    if slot_diag["top_sample_match_fraction"] < 0.999:
        raise V100Error(f"weak cluster reconstruction {slot_diag}")
    if slot_diag["occupancy_exact_count_match_fraction"] < 0.99:
        raise V100Error(f"string occupancy mismatch {slot_diag}")
    spectral, reconstruction = _spectral_maps_for_cache(cache, args.dataset_dir)
    path = args.output_dir / f"v100-spectral-shard-{args.shard_index:02d}.npz"
    _save_spectral_cache(path, cache, spectral, slots)
    report = {
        "schema_version": 1,
        "architecture": {
            "name": "V10 spectral cache shard",
            "runtime_inputs_use_annotations": False,
            "string_labels_training_only": True,
            "offset_stream_executed": False,
        },
        "configuration": {
            "shard_index": args.shard_index,
            "segment_samples": SEGMENT_SAMPLES,
            "pre_samples": PRE_SAMPLES,
            "post_samples": POST_SAMPLES,
            "time_frames": TIME_FRAMES,
            "spectral_bands": SPECTRAL_BANDS,
            "spectral_channels": SPECTRAL_CHANNELS,
        },
        "data": {
            "track_count": len(cache["track_members"]),
            "cluster_count": len(cache["target"]),
            "slot_diagnostics": slot_diag,
            "reconstruction": reconstruction,
        },
        "cache": {"path": path.name, "spectral_shape": list(spectral.shape)},
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"shard": args.shard_index, "tracks": len(cache["track_members"]), "clusters": len(cache["target"]), "shape": list(spectral.shape)}, indent=2))
    return report


def _load_spectral_caches(cache_dir: Path):
    paths = sorted(cache_dir.rglob("v100-spectral-shard-*.npz"))
    if not paths:
        raise V100Error(f"no spectral caches under {cache_dir}")
    arrays = defaultdict(list)
    track_members = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if int(data["schema_version"][0]) != CACHE_SCHEMA_VERSION:
                raise V100Error(f"bad cache schema in {path}")
            for key in ("spectral", "sequence", "mask", "stats", "target", "exact", "members", "top_samples", "slot_targets"):
                arrays[key].append(np.asarray(data[key]))
            track_members.extend(str(x) for x in data["track_members"])
    merged = {key: np.concatenate(values, axis=0) for key, values in arrays.items()}
    merged["members"] = merged["members"].astype(str)
    merged["target"] = merged["target"].astype(np.int32)
    merged["exact"] = merged["exact"].astype(np.int32)
    merged["top_samples"] = merged["top_samples"].astype(np.int32)
    merged["slot_targets"] = merged["slot_targets"].astype(np.float32)
    merged["track_members"] = tuple(sorted(track_members))
    merged["shard_paths"] = [str(p) for p in paths]
    n = len(merged["target"])
    for key in ("spectral", "sequence", "mask", "stats", "exact", "members", "top_samples", "slot_targets"):
        if len(merged[key]) != n:
            raise V100Error(f"cache row mismatch {key}")
    return merged


def _build_model():
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required") from exc

    scaffold = _build_cluster_model()
    candidate_hidden = scaffold.get_layer("cluster_hidden2").output
    spectral = keras.Input((TIME_FRAMES, SPECTRAL_BANDS, SPECTRAL_CHANNELS), name="spectral_map")
    x = keras.layers.LayerNormalization(axis=-1, name="spectral_channel_norm")(spectral)
    x = keras.layers.Conv2D(24, (3, 7), strides=(1, 2), padding="same", activation="relu", name="tf_conv1")(x)
    x = keras.layers.Conv2D(48, (3, 5), strides=(2, 2), padding="same", activation="relu", name="tf_conv2")(x)
    x = keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation="relu", name="tf_conv3")(x)
    x = keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="tf_conv4")(x)
    x = keras.layers.Flatten(name="tf_flatten")(x)
    x = keras.layers.Dense(192, activation="relu", name="tf_hidden")(x)
    x = keras.layers.Dropout(0.15, name="tf_dropout")(x)
    fused = keras.layers.Concatenate(name="acoustic_candidate_fusion")([candidate_hidden, x])
    fused = keras.layers.LayerNormalization(name="fusion_norm")(fused)
    fused = keras.layers.Dense(256, activation="relu", name="fusion_hidden1")(fused)
    shared = keras.layers.Dense(128, activation="relu", name="fusion_hidden2")(fused)

    string_outputs = []
    for slot in range(SLOT_COUNT):
        slot_hidden = keras.layers.Dense(64, activation="relu", name=f"string_{slot}_hidden")(shared)
        string_outputs.append(keras.layers.Dense(1, activation="sigmoid", name=f"string_{slot}")(slot_hidden))
    string_stack = keras.layers.Concatenate(name="string_birth_vector")(string_outputs)
    slot_count = keras.layers.Lambda(lambda t: tf.reduce_sum(t, axis=1, keepdims=True) / float(SLOT_COUNT), name="slot_count")(string_stack)
    ordinal_outputs = [
        keras.layers.Dense(1, activation="sigmoid", name=f"ge{stage}")(shared)
        for stage in range(1, ORDINAL_STAGES + 1)
    ]
    outputs = {f"string_{slot}": out for slot, out in enumerate(string_outputs)}
    outputs["slot_count"] = slot_count
    outputs.update({f"ge{stage}": out for stage, out in enumerate(ordinal_outputs, start=1)})

    loss = {f"string_{slot}": "binary_crossentropy" for slot in range(SLOT_COUNT)}
    loss["slot_count"] = "mse"
    loss.update({f"ge{stage}": "binary_crossentropy" for stage in range(1, ORDINAL_STAGES + 1)})
    loss_weights = {f"string_{slot}": 0.9 for slot in range(SLOT_COUNT)}
    loss_weights["slot_count"] = 0.35
    loss_weights.update({f"ge{stage}": 0.25 * math.sqrt(stage) for stage in range(1, ORDINAL_STAGES + 1)})
    model = keras.Model(
        {"candidate_set": scaffold.input["candidate_set"], "candidate_mask": scaffold.input["candidate_mask"], "cluster_stats": scaffold.input["cluster_stats"], "spectral_map": spectral},
        outputs,
        name="v100_time_frequency_string_slots",
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-4), loss=loss, loss_weights=loss_weights)
    return model, loss_weights


def _balanced_binary_weights(y):
    y = np.asarray(y).reshape(-1) > 0.5
    pos = int(np.sum(y)); neg = len(y) - pos
    if not pos or not neg:
        raise V100Error(f"binary target lacks class pos={pos} neg={neg}")
    w = np.empty(len(y), dtype=np.float32)
    w[y] = len(y) / (2.0 * pos)
    w[~y] = len(y) / (2.0 * neg)
    return np.clip(w, 0.25, 8.0)


def _targets(slot_targets, k):
    result = {f"string_{slot}": slot_targets[:, slot].reshape(-1, 1) for slot in range(SLOT_COUNT)}
    result["slot_count"] = (np.minimum(k, SLOT_COUNT).astype(np.float32) / float(SLOT_COUNT)).reshape(-1, 1)
    result.update({f"ge{stage}": (k >= stage).astype(np.float32).reshape(-1, 1) for stage in range(1, ORDINAL_STAGES + 1)})
    return result


def _sample_weights(slot_targets, k):
    result = {f"string_{slot}": _balanced_binary_weights(slot_targets[:, slot]) for slot in range(SLOT_COUNT)}
    counts = np.bincount(np.minimum(k, SLOT_COUNT), minlength=SLOT_COUNT + 1).astype(np.float64)
    weights = np.ones(SLOT_COUNT + 1, dtype=np.float64)
    nz = counts > 0
    weights[nz] = np.sqrt(len(k) / ((SLOT_COUNT + 1) * counts[nz]))
    result["slot_count"] = np.clip(weights[np.minimum(k, SLOT_COUNT)], 0.35, 4.0).astype(np.float32)
    ordinal, _ = _conditional_sample_weights(k)
    result.update(ordinal)
    return result


def _inputs(cache, indices=None):
    if indices is None:
        return {
            "candidate_set": cache["sequence"],
            "candidate_mask": cache["mask"],
            "cluster_stats": cache["stats"],
            "spectral_map": cache["spectral"],
        }
    return {
        "candidate_set": cache["sequence"][indices],
        "candidate_mask": cache["mask"][indices],
        "cluster_stats": cache["stats"][indices],
        "spectral_map": cache["spectral"][indices],
    }


def _predict(model, inputs):
    raw = model.predict(inputs, batch_size=192, verbose=0)
    slots = np.stack([np.asarray(raw[f"string_{slot}"]).reshape(-1) for slot in range(SLOT_COUNT)], axis=1)
    conditionals = np.stack([np.asarray(raw[f"ge{stage}"]).reshape(-1) for stage in range(1, ORDINAL_STAGES + 1)], axis=1)
    return slots.astype(np.float64), conditionals.astype(np.float64)


def _decode(slots, conditionals, mode):
    slot_expected = np.sum(slots, axis=1)
    cumulative = _cumulative(conditionals)
    ordinal_expected = np.sum(cumulative, axis=1)
    if mode.startswith("slot_threshold_"):
        threshold = int(mode.rsplit("_", 1)[1]) / 100.0
        return np.sum(slots >= threshold, axis=1).astype(np.int32)
    if mode == "slot_expected_round":
        value = slot_expected
    elif mode == "ordinal_cumulative_050":
        return np.sum(cumulative >= 0.5, axis=1).astype(np.int32)
    elif mode.startswith("hybrid_expected_"):
        slot_weight = int(mode.rsplit("_", 1)[1].replace("slot", "")) / 100.0
        value = slot_weight * slot_expected + (1.0 - slot_weight) * ordinal_expected
    else:
        raise ValueError(mode)
    return np.clip(np.floor(value + 0.5), 0, SLOT_COUNT).astype(np.int32)


def _cached_prediction_map(cache, indices, k_values):
    retained: Dict[str, List[int]] = defaultdict(list)
    for idx, k in zip(indices, k_values):
        samples = [int(v) for v in cache["top_samples"][idx] if int(v) >= 0]
        retained[str(cache["members"][idx])].extend(samples[:int(k)])
    return {m: tuple(sorted(v)) for m, v in retained.items()}


def _slot_report_local(truth, probabilities):
    pred = probabilities >= 0.5
    truth_b = truth >= 0.5
    per_slot = {}
    for slot in range(SLOT_COUNT):
        tp = int(np.sum(pred[:, slot] & truth_b[:, slot])); fp = int(np.sum(pred[:, slot] & ~truth_b[:, slot])); fn = int(np.sum(~pred[:, slot] & truth_b[:, slot]))
        per_slot[str(slot)] = {"precision": tp/(tp+fp) if tp+fp else None, "recall": tp/(tp+fn) if tp+fn else None, "positive_count": int(np.sum(truth_b[:, slot]))}
    return {
        "exact_string_vector_accuracy_at_0_5": float(np.mean(np.all(pred == truth_b, axis=1))),
        "mean_predicted_string_count_at_0_5": float(np.mean(np.sum(pred, axis=1))),
        "mean_true_string_count": float(np.mean(np.sum(truth_b, axis=1))),
        "per_string": per_slot,
    }


def train_eval(args):
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    random.seed(args.seed); np.random.seed(args.seed)
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required") from exc
    tf.random.set_seed(args.seed)

    cache = _load_spectral_caches(args.cache_dir)
    _, train_split, validation = _dataset_split(args.dataset_dir)
    if len(cache["track_members"]) != len(train_split):
        raise V100Error(f"expected {len(train_split)} cached tracks got {len(cache['track_members'])}")
    validation_members = {t.annotation_member for t in validation}
    if set(cache["track_members"]) & validation_members:
        raise V100Error("train cache overlaps validation")
    fit_tracks, holdout_tracks, holdout_groups = _split_train_groups(train_split, args.seed)
    fit_set = {t.annotation_member for t in fit_tracks}; holdout_set = {t.annotation_member for t in holdout_tracks}
    fit_idx = np.asarray([i for i,m in enumerate(cache["members"]) if str(m) in fit_set], dtype=np.int64)
    hold_idx = np.asarray([i for i,m in enumerate(cache["members"]) if str(m) in holdout_set], dtype=np.int64)
    k = np.minimum(cache["exact"], SLOT_COUNT)

    model, loss_weights = _build_model()
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=2e-5),
    ]
    print(f"training V10 spectral slots fit={len(fit_idx)} holdout={len(hold_idx)} tracks={len(train_split)}")
    history = model.fit(
        _inputs(cache, fit_idx),
        _targets(cache["slot_targets"][fit_idx], k[fit_idx]),
        sample_weight=_sample_weights(cache["slot_targets"][fit_idx], k[fit_idx]),
        validation_data=(
            _inputs(cache, hold_idx),
            _targets(cache["slot_targets"][hold_idx], k[hold_idx]),
            _sample_weights(cache["slot_targets"][hold_idx], k[hold_idx]),
        ),
        epochs=args.epochs,
        batch_size=96,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )

    hold_slots, hold_cond = _predict(model, _inputs(cache, hold_idx))
    sweep=[]
    for mode in DECODE_MODES:
        pred=_decode(hold_slots,hold_cond,mode)
        metrics=_metrics(holdout_tracks,_cached_prediction_map(cache,hold_idx,pred))
        macro=0.5*(metrics["solo"]["f1"]+metrics["comp"]["f1"])
        sweep.append({"mode":mode,"macro_f1":float(macro),"metrics":metrics,"cardinality":_cardinality_report(k[hold_idx],pred)})
    best=max(sweep,key=lambda r:(r["macro_f1"],r["metrics"]["global"]["f1"],r["cardinality"]["accuracy"]))
    decode_mode=best["mode"]
    model.save_weights(args.output_dir/"v100-spectral-string-slots.weights.h5")

    locked12=tuple(validation[:12])
    floor,v88_threshold,enc86,enc87,model88=_load_frozen_stack(args)
    print("evaluating V10 once on historical locked12")
    score_streams,records,x88,out88=_represent_full(locked12,args.base_model,floor,enc86,enc87,model88)
    clusters,fused,assignment,sequence,mask,stats,target,exact,truncated=_cluster_data(locked12,records,x88,out88)
    spectral=_spectral_maps_for_runtime(locked12,clusters,records)
    locked_cache={"sequence":sequence,"mask":mask,"stats":stats,"spectral":spectral}
    slots,cond=_predict(model,_inputs(locked_cache))
    pred_k=_decode(slots,cond,decode_mode)
    metrics=_metrics(locked12,_prediction_map(clusters,records,fused,pred_k))
    v88=_metrics(locked12,_retained_predictions(records,fused,v88_threshold))
    oracle_k=np.minimum(np.asarray(exact,dtype=np.int32),SLOT_COUNT)
    oracle=_metrics(locked12,_prediction_map(clusters,records,fused,oracle_k))
    slot_truth,slot_assignment=_slot_targets_for_runtime_clusters(locked12,clusters,records)
    v91=json.loads(args.v91_report.read_text())["locked12"]["v91_metrics"]

    report={
        "schema_version":1,
        "architecture":{
            "name":"V10.0 time-frequency six-string slot decoder",
            "frozen_base":"V8.4 + V8.6 + V8.7 + V8.8",
            "base_trainable":False,
            "v100_trainable_parameters":int(model.count_params()),
            "runtime_inputs_use_annotations":False,
            "string_labels_training_only":True,
            "offset_stream_executed":False,
            "offset_weights_modified":False,
            "spectral_map_shape":[TIME_FRAMES,SPECTRAL_BANDS,SPECTRAL_CHANNELS],
            "spectral_segment_samples":SEGMENT_SAMPLES,
            "spectral_pre_samples":PRE_SAMPLES,
            "spectral_post_samples":POST_SAMPLES,
            "cluster_window_ms":CLUSTER_WINDOW_MS,
            "maximum_verification_delay_ms":(MAX_HORIZON+CLUSTER_WINDOW_SAMPLES)*1000.0/SAMPLE_RATE,
        },
        "configuration":{
            "seed":args.seed,"decode_mode":decode_mode,"decode_mode_selected_on_locked_validation":False,
            "decode_modes_fixed_before_locked_eval":list(DECODE_MODES),"epochs_ran":len(history.history["loss"]),"loss_weights":loss_weights,
            "matching_tolerance_ms":TOLERANCE_MS,
        },
        "data":{
            "full_train_track_count":len(train_split),"full_validation_track_count":len(validation),"cached_cluster_count":len(cache["target"]),
            "cache_shard_count":len(cache["shard_paths"]),"fit_track_count":len(fit_tracks),"holdout_track_count":len(holdout_tracks),
            "fit_cluster_count":len(fit_idx),"holdout_cluster_count":len(hold_idx),"holdout_composition_groups":list(holdout_groups),
            "composition_group_leakage":bool({group_stem(t) for t in fit_tracks}&{group_stem(t) for t in holdout_tracks}),
            "locked_string_assignment":slot_assignment,
        },
        "training_history":{key:[float(v) for v in values] for key,values in history.history.items()},
        "holdout_decode_calibration":{"best":best,"sweep":sweep},
        "locked12":{
            "frozen_v88_baseline_metrics":v88,"v91_metrics":v91,"v100_metrics":metrics,"oracle_exact_cardinality_metrics":oracle,
            "v100_cardinality":_cardinality_report(oracle_k,pred_k),"v100_string_slots":_slot_report_local(slot_truth,slots),
            "candidate_ceiling":_candidate_ceiling(locked12,score_streams,floor),
        },
    }
    (args.output_dir/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"decode_mode":decode_mode,"parameters":int(model.count_params()),"v88":v88,"v91":v91,"v100":metrics,"oracle":oracle,"cardinality":report["locked12"]["v100_cardinality"],"string_slots":report["locked12"]["v100_string_slots"]},indent=2,sort_keys=True))
    return report


def parser():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="command",required=True)
    m=sub.add_parser("mine")
    m.add_argument("dataset_dir",nargs="?",type=Path,default=ROOT/"data"/"GuitarSet")
    m.add_argument("--cache-dir",type=Path,required=True)
    m.add_argument("--shard-index",type=int,required=True)
    m.add_argument("--output-dir",type=Path,required=True)
    t=sub.add_parser("train-eval")
    t.add_argument("dataset_dir",nargs="?",type=Path,default=ROOT/"data"/"GuitarSet")
    t.add_argument("--base-model",type=Path,required=True)
    t.add_argument("--v86-weights",type=Path,required=True); t.add_argument("--v86-report",type=Path,required=True)
    t.add_argument("--v87-weights",type=Path,required=True); t.add_argument("--v87-report",type=Path,required=True)
    t.add_argument("--v88-weights",type=Path,required=True); t.add_argument("--v88-report",type=Path,required=True)
    t.add_argument("--v91-report",type=Path,required=True)
    t.add_argument("--cache-dir",type=Path,required=True); t.add_argument("--output-dir",type=Path,required=True)
    t.add_argument("--epochs",type=int,default=25); t.add_argument("--seed",type=int,default=DEFAULT_SEED_V100)
    return p


def main(argv:Optional[Sequence[str]]=None):
    args=parser().parse_args(argv)
    if args.command=="mine": mine(args)
    else: train_eval(args)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
