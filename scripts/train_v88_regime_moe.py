"""V8.8 causal regime-gated mixture-of-experts onset decoder.

V8.7 improved candidate ranking and reached the best solo onset F1 so far, but a
single global retention decision became too conservative for dense comp/strum
births. V8.8 keeps the entire V8.4 -> V8.6 -> V8.7 stack frozen and learns a
hierarchical decision head that explicitly separates two acoustic regimes:

  * isolated / continuation regime
  * clustered birth / strum regime

The router receives only frozen audio-derived candidate representations. Two
specialized birth experts are trained on their respective regimes, while an
auxiliary local-cardinality head (0, 1, 2, 3+) forces the shared representation
to encode whether the current acoustic transition belongs to a multi-birth
state. The final birth probability is a learned soft mixture of the two experts.

The regime/cardinality labels are training supervision only. At runtime the
inputs remain audio, frozen V8.4 scores, frozen V8.6 acoustic embeddings and
frozen V8.7 causal candidate-memory representations. Offset is never executed
or modified. No future candidate context is introduced; the maximum delay stays
the V8.6 acoustic horizon (1024 samples, ~23.22 ms).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
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

from causal_note.guitarset import ALLOWED_PLAYERS, SAMPLE_RATE, index_guitarset
from scripts.audit_v84_solo_comp_onsets import _auc
from scripts.evaluate_boundaries import _count_metrics, match_boundaries, milliseconds_to_samples
from scripts.evaluate_v8_boundaries import DEFAULT_SEED, DEFAULT_VALIDATION_FRACTION, _arrangement, _reference_positions
from scripts.train_boundaries import decode_pcm16_mono_wav, split_tracks_by_group
from scripts.train_v85_transition_verifier import _split_members
from scripts.train_v86_state_transition_proposals import (
    MAX_HORIZON,
    TOLERANCE_MS,
    TOLERANCE_SAMPLES,
    _candidate_ceiling,
    _predict_score_tracks,
    _records,
    _weights,
)
from scripts.train_v87_causal_candidate_memory import (
    CLASS_NAMES,
    _build_memory_head,
    _encode_records,
    _load_v86_encoder,
    _sequence_arrays,
)

DEFAULT_SEED_V88 = 8831
DEFAULT_TRAIN_MEMBERS = 30
LOCAL_CLUSTER_MS = 20.0
LOCAL_CLUSTER_SAMPLES = milliseconds_to_samples(LOCAL_CLUSTER_MS)
V87_HIDDEN_LAYER = "context_hidden2"
V87_HIDDEN_DIM = 48
V87_CLASS_DIM = len(CLASS_NAMES)
V86_EMBEDDING_DIM = 64
V86_CLASS_DIM = len(CLASS_NAMES)
# V87 hidden + V87 class probs + V86 embedding + V86 class probs + raw proposal
# score + normalized multiplicity + baseline-edge source flag.
FEATURE_DIM = V87_HIDDEN_DIM + V87_CLASS_DIM + V86_EMBEDDING_DIM + V86_CLASS_DIM + 3
CARDINALITY_CLASSES = 4  # 0, 1, 2, 3+


class V88Error(RuntimeError):
    pass


def _load_v87_encoder(weights_path: Path):
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for V8.8 training") from exc
    model = _build_memory_head()
    model.load_weights(weights_path)
    hidden = model.get_layer(V87_HIDDEN_LAYER).output
    encoder = keras.Model(model.input, [hidden, model.output], name="frozen_v87_candidate_memory_encoder")
    encoder.trainable = False
    return model, encoder


def _encode_v87(encoder, sequences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hidden, probabilities = encoder.predict(sequences, batch_size=256, verbose=0)
    return np.asarray(hidden, dtype=np.float32), np.asarray(probabilities, dtype=np.float32)


def _track_references(tracks) -> Dict[str, Tuple[int, ...]]:
    result: Dict[str, Tuple[int, ...]] = {}
    for track in tracks:
        audio = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        references, _ = _reference_positions(track, audio.frame_count)
        result[track.annotation_member] = tuple(int(value) for value in references)
    return result


def _regime_targets(records: Sequence[dict], tracks) -> Tuple[np.ndarray, np.ndarray]:
    references = _track_references(tracks)
    route = np.zeros((len(records), 1), dtype=np.float32)
    cardinality = np.zeros((len(records),), dtype=np.int32)
    for index, record in enumerate(records):
        sample = int(record["sample"])
        local = sum(
            1
            for reference in references[record["member"]]
            if abs(int(reference) - sample) <= LOCAL_CLUSTER_SAMPLES
        )
        cardinality[index] = min(3, int(local))
        route[index, 0] = float(local >= 2)
    return route, cardinality


def _feature_matrix(
    records: Sequence[dict],
    v86_embeddings: np.ndarray,
    v86_probabilities: np.ndarray,
    v87_hidden: np.ndarray,
    v87_probabilities: np.ndarray,
) -> np.ndarray:
    if not (
        len(records)
        == len(v86_embeddings)
        == len(v86_probabilities)
        == len(v87_hidden)
        == len(v87_probabilities)
    ):
        raise V88Error("candidate representation length mismatch")
    rows: List[np.ndarray] = []
    for record, emb86, prob86, hidden87, prob87 in zip(
        records, v86_embeddings, v86_probabilities, v87_hidden, v87_probabilities
    ):
        row = np.concatenate(
            (
                np.asarray(hidden87, dtype=np.float32),
                np.asarray(prob87, dtype=np.float32),
                np.asarray(emb86, dtype=np.float32),
                np.asarray(prob86, dtype=np.float32),
                np.asarray(
                    [
                        float(record["score"]),
                        (float(record["count"]) - 1.0) / 2.0,
                        float("baseline_edge" in record.get("sources", ())),
                    ],
                    dtype=np.float32,
                ),
            )
        )
        if row.shape != (FEATURE_DIM,):
            raise V88Error(f"unexpected V8.8 feature shape {row.shape}")
        rows.append(row)
    return np.stack(rows) if rows else np.zeros((0, FEATURE_DIM), dtype=np.float32)


def _build_moe_head():
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for V8.8 training") from exc

    features = keras.Input((FEATURE_DIM,), name="candidate_state")
    x = keras.layers.LayerNormalization(name="state_norm")(features)
    x = keras.layers.Dense(160, activation="relu", name="shared_hidden1")(x)
    x = keras.layers.Dropout(0.15, name="shared_dropout")(x)
    shared = keras.layers.Dense(96, activation="relu", name="shared_hidden2")(x)

    router_hidden = keras.layers.Dense(48, activation="relu", name="router_hidden")(shared)
    router = keras.layers.Dense(1, activation="sigmoid", name="cluster_router")(router_hidden)

    cardinality_hidden = keras.layers.Dense(48, activation="relu", name="cardinality_hidden")(shared)
    cardinality = keras.layers.Dense(
        CARDINALITY_CLASSES, activation="softmax", name="local_cardinality"
    )(cardinality_hidden)

    isolated_hidden = keras.layers.Dense(64, activation="relu", name="isolated_hidden")(shared)
    isolated = keras.layers.Dense(1, activation="sigmoid", name="isolated_birth")(isolated_hidden)

    cluster_input = keras.layers.Concatenate(name="cluster_expert_context")([shared, cardinality])
    cluster_hidden = keras.layers.Dense(64, activation="relu", name="cluster_hidden")(cluster_input)
    cluster = keras.layers.Dense(1, activation="sigmoid", name="cluster_birth")(cluster_hidden)

    fused = keras.layers.Lambda(
        lambda values: (1.0 - values[0]) * values[1] + values[0] * values[2],
        name="fused_birth",
    )([router, isolated, cluster])

    model = keras.Model(
        features,
        {
            "cluster_router": router,
            "local_cardinality": cardinality,
            "isolated_birth": isolated,
            "cluster_birth": cluster,
            "fused_birth": fused,
        },
        name="v88_regime_gated_birth_moe",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss={
            "cluster_router": "binary_crossentropy",
            "local_cardinality": "sparse_categorical_crossentropy",
            "isolated_birth": "binary_crossentropy",
            "cluster_birth": "binary_crossentropy",
            "fused_birth": "binary_crossentropy",
        },
        loss_weights={
            "cluster_router": 0.35,
            "local_cardinality": 0.35,
            "isolated_birth": 0.65,
            "cluster_birth": 0.65,
            "fused_birth": 1.0,
        },
    )
    return model


def _training_targets(records: Sequence[dict], route: np.ndarray, cardinality: np.ndarray):
    birth = np.asarray([float(record["birth"]) for record in records], dtype=np.float32).reshape(-1, 1)
    return {
        "cluster_router": route,
        "local_cardinality": cardinality,
        "isolated_birth": birth,
        "cluster_birth": birth,
        "fused_birth": birth,
    }


def _sample_weights(records: Sequence[dict], route: np.ndarray):
    base = _weights(records).astype(np.float32)
    cluster = route[:, 0].astype(np.float32)
    isolated = 1.0 - cluster
    return {
        "cluster_router": base,
        "local_cardinality": base,
        "isolated_birth": base * isolated,
        "cluster_birth": base * cluster,
        "fused_birth": base,
    }


def _predictions(model, features: np.ndarray) -> dict:
    raw = model.predict(features, batch_size=256, verbose=0)
    return {key: np.asarray(value) for key, value in raw.items()}


def _fused_scores(predictions: dict) -> np.ndarray:
    return np.asarray(predictions["fused_birth"], dtype=np.float32).reshape(-1)


def _retained_predictions(records: Sequence[dict], scores: np.ndarray, threshold: float) -> Dict[str, Tuple[int, ...]]:
    retained: Dict[str, List[int]] = defaultdict(list)
    for record, score in zip(records, scores):
        if float(score) >= threshold:
            retained[record["member"]].extend([int(record["sample"])] * int(record["count"]))
    return {member: tuple(sorted(values)) for member, values in retained.items()}


def _aggregate(tracks, predictions: Dict[str, Tuple[int, ...]], arrangement: Optional[str] = None) -> dict:
    refs = preds = tp = 0
    for track in tracks:
        arr = _arrangement(track.annotation_member)
        if arrangement is not None and arr != arrangement:
            continue
        audio = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        references, _ = _reference_positions(track, audio.frame_count)
        values = predictions.get(track.annotation_member, ())
        pairs = match_boundaries(references, values, TOLERANCE_SAMPLES)
        refs += len(references)
        preds += len(values)
        tp += len(pairs)
    result = asdict(_count_metrics(refs, preds, tp))
    result["prediction_reference_ratio"] = preds / refs if refs else None
    return result


def _evaluate_threshold(records, scores, tracks, threshold: float) -> dict:
    predictions = _retained_predictions(records, scores, threshold)
    return {
        key: _aggregate(tracks, predictions, arrangement)
        for key, arrangement in (("global", None), ("solo", "solo"), ("comp", "comp"))
    }


def _calibrate(records, scores, tracks) -> dict:
    sweep = []
    for threshold in np.arange(0.10, 0.901, 0.05):
        metrics = _evaluate_threshold(records, scores, tracks, float(threshold))
        macro = (metrics["solo"]["f1"] + metrics["comp"]["f1"]) / 2.0
        sweep.append({"threshold": float(round(threshold, 6)), "macro_f1": macro, "metrics": metrics})
    best = max(
        sweep,
        key=lambda row: (
            row["macro_f1"],
            row["metrics"]["global"]["f1"],
            -abs(row["threshold"] - 0.5),
        ),
    )
    return {
        "selection_rule": "single fused threshold maximizing train-holdout macro(solo,comp) F1; no regime-specific runtime threshold",
        "best": best,
        "sweep": sweep,
    }


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    positives = [float(score) for score, label in zip(scores, labels) if int(label)]
    negatives = [float(score) for score, label in zip(scores, labels) if not int(label)]
    return _auc(positives, negatives)


def _candidate_auc(records, scores, arrangement: Optional[str]) -> Optional[float]:
    selected = [
        (float(score), int(record["birth"]))
        for record, score in zip(records, scores)
        if arrangement is None or record["arrangement"] == arrangement
    ]
    positives = [score for score, label in selected if label]
    negatives = [score for score, label in selected if not label]
    return _auc(positives, negatives)


def _cardinality_accuracy(probabilities: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.argmax(probabilities, axis=1) == targets)) if len(targets) else 0.0


def _class_summary(records: Sequence[dict], route: np.ndarray, cardinality: np.ndarray) -> dict:
    transition_counts = Counter((record["arrangement"], CLASS_NAMES[int(record["class_id"])]) for record in records)
    route_counts = Counter(int(value) for value in route[:, 0])
    cardinality_counts = Counter(int(value) for value in cardinality)
    return {
        "count": len(records),
        "tracks": len({record["member"] for record in records}),
        "transition_strata": {
            f"{arrangement}_{name}": transition_counts[(arrangement, name)]
            for arrangement in ("solo", "comp")
            for name in CLASS_NAMES
        },
        "regime_counts": {"isolated": route_counts[0], "cluster": route_counts[1]},
        "cardinality_counts": {str(index): cardinality_counts[index] for index in range(CARDINALITY_CLASSES)},
    }


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train V8.8 regime-gated onset mixture-of-experts.")
    parser.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--v86-weights", type=Path, required=True)
    parser.add_argument("--v86-report", type=Path, required=True)
    parser.add_argument("--v87-weights", type=Path, required=True)
    parser.add_argument("--v87-report", type=Path, required=True)
    parser.add_argument("--train-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-members", type=int, default=DEFAULT_TRAIN_MEMBERS)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED_V88)
    return parser


def run(args) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required") from exc
    tf.random.set_seed(args.seed)

    indexed = tuple(track for track in index_guitarset(args.dataset_dir) if track.player_id in ALLOWED_PLAYERS)
    by_member = {track.annotation_member: track for track in indexed}
    _, locked_validation = split_tracks_by_group(
        indexed, validation_fraction=DEFAULT_VALIDATION_FRACTION, seed=DEFAULT_SEED
    )
    locked12 = tuple(locked_validation[:12])
    locked_members = {track.annotation_member for track in locked_validation}

    train_audit = json.loads(args.train_audit.read_text(encoding="utf-8"))
    source_members = list(train_audit["scope"]["members"])[: args.train_members]
    if len(source_members) != args.train_members:
        raise V88Error("not enough source train members")
    if set(source_members) & locked_members:
        raise V88Error("train members overlap locked validation")
    fit_members, holdout_members = _split_members(source_members, args.seed)
    fit_set, holdout_set = set(fit_members), set(holdout_members)
    train_tracks = tuple(by_member[member] for member in source_members)
    fit_tracks = tuple(by_member[member] for member in fit_members)
    holdout_tracks = tuple(by_member[member] for member in holdout_members)

    v86_report = json.loads(args.v86_report.read_text(encoding="utf-8"))
    v87_report = json.loads(args.v87_report.read_text(encoding="utf-8"))
    floor = float(v86_report["configuration"]["candidate_floor"])
    if v86_report["configuration"].get("candidate_floor_selected_on_locked_validation") is not False:
        raise V88Error("V8.6 candidate floor was not train-only calibrated")
    if v87_report["configuration"].get("retain_threshold_selected_on_locked_validation") is not False:
        raise V88Error("V8.7 retention threshold was not train-only calibrated")

    _v86_model, v86_encoder = _load_v86_encoder(args.v86_weights)
    _v87_model, v87_encoder = _load_v87_encoder(args.v87_weights)

    print("mining frozen V8.4 scores and frozen V8.6/V8.7 candidate states")
    train_scores = _predict_score_tracks(args.base_model, train_tracks)
    all_records = _records(train_tracks, train_scores, floor)
    fit_records = [record for record in all_records if record["member"] in fit_set]
    holdout_records = [record for record in all_records if record["member"] in holdout_set]
    if not fit_records or not holdout_records:
        raise V88Error("empty fit or holdout records")

    fit_emb86, fit_prob86 = _encode_records(v86_encoder, fit_records)
    holdout_emb86, holdout_prob86 = _encode_records(v86_encoder, holdout_records)
    fit_seq, _ = _sequence_arrays(fit_records, fit_emb86, fit_prob86)
    holdout_seq, _ = _sequence_arrays(holdout_records, holdout_emb86, holdout_prob86)
    fit_hidden87, fit_prob87 = _encode_v87(v87_encoder, fit_seq)
    holdout_hidden87, holdout_prob87 = _encode_v87(v87_encoder, holdout_seq)

    x_fit = _feature_matrix(fit_records, fit_emb86, fit_prob86, fit_hidden87, fit_prob87)
    x_holdout = _feature_matrix(
        holdout_records, holdout_emb86, holdout_prob86, holdout_hidden87, holdout_prob87
    )
    route_fit, cardinality_fit = _regime_targets(fit_records, fit_tracks)
    route_holdout, cardinality_holdout = _regime_targets(holdout_records, holdout_tracks)

    model = _build_moe_head()
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)]
    history = model.fit(
        x_fit,
        _training_targets(fit_records, route_fit, cardinality_fit),
        sample_weight=_sample_weights(fit_records, route_fit),
        validation_data=(
            x_holdout,
            _training_targets(holdout_records, route_holdout, cardinality_holdout),
            _sample_weights(holdout_records, route_holdout),
        ),
        epochs=args.epochs,
        batch_size=128,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )

    holdout_predictions = _predictions(model, x_holdout)
    holdout_fused = _fused_scores(holdout_predictions)
    calibration = _calibrate(holdout_records, holdout_fused, holdout_tracks)
    retain_threshold = float(calibration["best"]["threshold"])

    print("evaluating once on frozen locked12")
    locked_scores = _predict_score_tracks(args.base_model, locked12)
    locked_ceiling = _candidate_ceiling(locked12, locked_scores, floor)
    locked_records = _records(locked12, locked_scores, floor)
    locked_emb86, locked_prob86 = _encode_records(v86_encoder, locked_records)
    locked_seq, _ = _sequence_arrays(locked_records, locked_emb86, locked_prob86)
    locked_hidden87, locked_prob87 = _encode_v87(v87_encoder, locked_seq)
    x_locked = _feature_matrix(
        locked_records, locked_emb86, locked_prob86, locked_hidden87, locked_prob87
    )
    route_locked, cardinality_locked = _regime_targets(locked_records, locked12)
    locked_outputs = _predictions(model, x_locked)
    locked_fused = _fused_scores(locked_outputs)
    locked_retained = _retained_predictions(locked_records, locked_fused, retain_threshold)
    locked_metrics = {
        key: _aggregate(locked12, locked_retained, arrangement)
        for key, arrangement in (("global", None), ("solo", "solo"), ("comp", "comp"))
    }
    candidate_aucs = {
        key: _candidate_auc(locked_records, locked_fused, arrangement)
        for key, arrangement in (("global", None), ("solo", "solo"), ("comp", "comp"))
    }
    router_scores = np.asarray(locked_outputs["cluster_router"]).reshape(-1)
    router_auc = _binary_auc(router_scores, route_locked[:, 0])
    cardinality_accuracy = _cardinality_accuracy(
        np.asarray(locked_outputs["local_cardinality"]), cardinality_locked
    )

    model.save_weights(args.output_dir / "v88-regime-moe.weights.h5")
    report = {
        "schema_version": 1,
        "architecture": {
            "name": "V8.8 causal regime-gated dual-expert birth decoder",
            "base": "frozen V8.4 onset + frozen V8.6 acoustic encoder + frozen V8.7 causal candidate-memory encoder",
            "base_trainable": False,
            "v86_encoder_trainable": False,
            "v87_encoder_trainable": False,
            "offset_stream_executed": False,
            "offset_weights_modified": False,
            "runtime_inputs_use_annotations": False,
            "future_candidate_context": False,
            "local_cluster_ms_training_target": LOCAL_CLUSTER_MS,
            "router": "learned isolated-vs-cluster acoustic regime",
            "experts": ["isolated_birth", "cluster_birth"],
            "auxiliary_cardinality_classes": ["0", "1", "2", "3+"],
            "maximum_verification_delay_samples": MAX_HORIZON,
            "maximum_verification_delay_ms": MAX_HORIZON * 1000.0 / SAMPLE_RATE,
            "moe_head_parameters": int(model.count_params()),
            "feature_dim": FEATURE_DIM,
        },
        "configuration": {
            "seed": args.seed,
            "candidate_floor": floor,
            "candidate_floor_reused_from_train_only_v86": True,
            "retain_threshold": retain_threshold,
            "retain_threshold_selected_on_locked_validation": False,
            "regime_specific_runtime_thresholds": False,
            "matching_tolerance_ms": TOLERANCE_MS,
            "requested_epochs": args.epochs,
            "epochs_ran": len(history.history["loss"]),
        },
        "data": {
            "source_train_members": source_members,
            "fit_members": list(fit_members),
            "holdout_members": list(holdout_members),
            "locked_validation_members": [track.annotation_member for track in locked12],
            "fit": _class_summary(fit_records, route_fit, cardinality_fit),
            "holdout": _class_summary(holdout_records, route_holdout, cardinality_holdout),
            "locked12": _class_summary(locked_records, route_locked, cardinality_locked),
        },
        "holdout_retain_calibration": calibration,
        "training_history": {
            key: [float(value) for value in values] for key, values in history.history.items()
        },
        "locked12": {
            "candidate_ceiling": locked_ceiling,
            "candidate_auc_birth_high": candidate_aucs,
            "router_cluster_auc": router_auc,
            "cardinality_accuracy": cardinality_accuracy,
            "metrics": locked_metrics,
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "candidate_floor": floor,
                "candidate_ceiling_recall": locked_ceiling["recall"],
                "retain_threshold": retain_threshold,
                "candidate_auc": candidate_aucs,
                "router_cluster_auc": router_auc,
                "cardinality_accuracy": cardinality_accuracy,
                "locked_metrics": locked_metrics,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = create_argument_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
