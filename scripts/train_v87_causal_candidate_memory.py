"""V8.7 causal candidate-memory onset refiner.

V8.6 proved that a low-floor V8.4 candidate pool exposes ~95% of locked12
references, but its independent per-candidate classifier still retains too many
false transitions.  V8.7 keeps the V8.4 onset stream and the trained V8.6
single-candidate acoustic encoder frozen, then adds a causal memory over recent
candidate embeddings.

For every current candidate the context head sees only the current candidate and
up to eight *previous* candidates.  No future candidate is consumed, so the
maximum verification delay remains the V8.6 acoustic horizon (1024 samples,
~23.22 ms).  The head learns the same anonymous transition classes:
continuation/retrigger, other false transient, isolated birth, cluster birth.

Runtime inputs are only audio, frozen V8.4 onset scores and previous candidate
embeddings.  GuitarSet annotations are used only for fit/evaluation.  Offset is
never executed or modified.
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
    BIRTH_CLASSES,
    CLASS_NAMES,
    HORIZONS,
    MAX_HORIZON,
    TOLERANCE_MS,
    TOLERANCE_SAMPLES,
    _arrays as _v86_arrays,
    _build_refiner as _build_v86_refiner,
    _candidate_ceiling,
    _predict_score_tracks,
    _records,
    _weights,
)

DEFAULT_SEED_V87 = 8731
DEFAULT_TRAIN_MEMBERS = 30
HISTORY_CANDIDATES = 8
SEQUENCE_LENGTH = HISTORY_CANDIDATES + 1
PAST_CONTEXT_MS = 120.0
PAST_CONTEXT_SAMPLES = milliseconds_to_samples(PAST_CONTEXT_MS)
V86_HIDDEN_LAYER = "transition_hidden2"
V86_EMBEDDING_DIM = 64
V86_CLASS_DIM = len(CLASS_NAMES)
# embedding + class probabilities + raw score + normalized multiplicity +
# baseline-edge source flag + signed/absolute relative time.
CANDIDATE_FEATURE_DIM = V86_EMBEDDING_DIM + V86_CLASS_DIM + 5


class V87Error(RuntimeError):
    pass


def _load_v86_encoder(weights_path: Path):
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for V8.7 training") from exc
    model = _build_v86_refiner()
    model.load_weights(weights_path)
    hidden = model.get_layer(V86_HIDDEN_LAYER).output
    encoder = keras.Model(model.inputs, [hidden, model.output], name="frozen_v86_candidate_encoder")
    encoder.trainable = False
    return model, encoder


def _encode_records(encoder, records: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray]:
    if not records:
        return (
            np.zeros((0, V86_EMBEDDING_DIM), dtype=np.float32),
            np.zeros((0, V86_CLASS_DIM), dtype=np.float32),
        )
    x, _ = _v86_arrays(records)
    embeddings, probabilities = encoder.predict(x, batch_size=256, verbose=0)
    return (
        np.asarray(embeddings, dtype=np.float32),
        np.asarray(probabilities, dtype=np.float32),
    )


def _candidate_feature(record: dict, embedding: np.ndarray, probabilities: np.ndarray, delta_samples: int) -> np.ndarray:
    delta_ms = float(delta_samples) * 1000.0 / SAMPLE_RATE
    values = np.concatenate(
        (
            np.asarray(embedding, dtype=np.float32),
            np.asarray(probabilities, dtype=np.float32),
            np.asarray(
                [
                    float(record["score"]),
                    (float(record["count"]) - 1.0) / 2.0,
                    float("baseline_edge" in record.get("sources", ())),
                    np.clip(delta_ms / PAST_CONTEXT_MS, -1.0, 0.0),
                    np.clip(abs(delta_ms) / PAST_CONTEXT_MS, 0.0, 1.0),
                ],
                dtype=np.float32,
            ),
        )
    )
    if values.shape != (CANDIDATE_FEATURE_DIM,):
        raise V87Error(f"unexpected candidate feature shape {values.shape}")
    return values


def _sequence_arrays(records: Sequence[dict], embeddings: np.ndarray, probabilities: np.ndarray):
    if len(records) != len(embeddings) or len(records) != len(probabilities):
        raise V87Error("record/embedding/probability length mismatch")
    sequences = np.zeros((len(records), SEQUENCE_LENGTH, CANDIDATE_FEATURE_DIM), dtype=np.float32)
    y = np.asarray([int(record["class_id"]) for record in records], dtype=np.int32)

    indices_by_member: Dict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(records):
        indices_by_member[record["member"]].append(index)

    for indices in indices_by_member.values():
        indices.sort(key=lambda index: (int(records[index]["sample"]), index))
        for position, center_index in enumerate(indices):
            center_sample = int(records[center_index]["sample"])
            valid: List[int] = []
            for previous_position in range(max(0, position - HISTORY_CANDIDATES), position):
                candidate_index = indices[previous_position]
                delta = int(records[candidate_index]["sample"]) - center_sample
                if delta < -PAST_CONTEXT_SAMPLES:
                    continue
                valid.append(candidate_index)
            valid = valid[-HISTORY_CANDIDATES:]
            start = HISTORY_CANDIDATES - len(valid)
            for slot, candidate_index in enumerate(valid, start=start):
                delta = int(records[candidate_index]["sample"]) - center_sample
                sequences[center_index, slot] = _candidate_feature(
                    records[candidate_index], embeddings[candidate_index], probabilities[candidate_index], delta
                )
            sequences[center_index, HISTORY_CANDIDATES] = _candidate_feature(
                records[center_index], embeddings[center_index], probabilities[center_index], 0
            )
    return sequences, y


def _build_memory_head():
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for V8.7 training") from exc
    sequence = keras.Input((SEQUENCE_LENGTH, CANDIDATE_FEATURE_DIM), name="candidate_history")
    masked = keras.layers.Masking(mask_value=0.0, name="history_mask")(sequence)
    projected = keras.layers.TimeDistributed(
        keras.layers.Dense(64, activation="relu"), name="candidate_projection"
    )(masked)
    memory = keras.layers.GRU(96, name="causal_candidate_memory")(projected)
    current = keras.layers.Lambda(lambda x: x[:, -1, :], name="current_candidate")(sequence)
    current = keras.layers.Dense(48, activation="relu", name="current_projection")(current)
    hidden = keras.layers.Concatenate(name="memory_plus_current")([memory, current])
    hidden = keras.layers.Dense(96, activation="relu", name="context_hidden1")(hidden)
    hidden = keras.layers.Dropout(0.15, name="context_dropout")(hidden)
    hidden = keras.layers.Dense(48, activation="relu", name="context_hidden2")(hidden)
    classes = keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="context_transition_class")(hidden)
    model = keras.Model(sequence, classes, name="v87_causal_candidate_memory")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4), loss="sparse_categorical_crossentropy")
    return model


def _birth_scores(model, sequences: np.ndarray) -> np.ndarray:
    probabilities = model.predict(sequences, batch_size=256, verbose=0)
    return probabilities[:, BIRTH_CLASSES[0]] + probabilities[:, BIRTH_CLASSES[1]]


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
        "selection_rule": "max train-holdout macro(solo,comp) track-level F1",
        "best": best,
        "sweep": sweep,
    }


def _candidate_auc(records, scores, arrangement: Optional[str]) -> Optional[float]:
    selected = [
        (float(score), int(record["birth"]))
        for record, score in zip(records, scores)
        if arrangement is None or record["arrangement"] == arrangement
    ]
    positives = [score for score, label in selected if label]
    negatives = [score for score, label in selected if not label]
    return _auc(positives, negatives)


def _class_summary(records: Sequence[dict]) -> dict:
    counts = Counter((record["arrangement"], CLASS_NAMES[int(record["class_id"])]) for record in records)
    return {
        "count": len(records),
        "tracks": len({record["member"] for record in records}),
        "strata": {
            f"{arrangement}_{name}": counts[(arrangement, name)]
            for arrangement in ("solo", "comp")
            for name in CLASS_NAMES
        },
    }


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train V8.7 causal candidate-memory onset refiner.")
    parser.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--v86-weights", type=Path, required=True)
    parser.add_argument("--v86-report", type=Path, required=True)
    parser.add_argument("--train-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-members", type=int, default=DEFAULT_TRAIN_MEMBERS)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED_V87)
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
        raise V87Error("not enough source train members")
    if set(source_members) & locked_members:
        raise V87Error("train members overlap locked validation")
    fit_members, holdout_members = _split_members(source_members, args.seed)
    fit_set, holdout_set = set(fit_members), set(holdout_members)
    train_tracks = tuple(by_member[member] for member in source_members)
    holdout_tracks = tuple(by_member[member] for member in holdout_members)

    v86_report = json.loads(args.v86_report.read_text(encoding="utf-8"))
    floor = float(v86_report["configuration"]["candidate_floor"])
    if v86_report["configuration"].get("candidate_floor_selected_on_locked_validation") is not False:
        raise V87Error("V8.6 candidate floor was not train-only calibrated")

    _v86_model, v86_encoder = _load_v86_encoder(args.v86_weights)
    print("mining frozen V8.4 raw onset scores with frozen V8.6 candidate floor", floor)
    train_scores = _predict_score_tracks(args.base_model, train_tracks)
    all_records = _records(train_tracks, train_scores, floor)
    fit_records = [record for record in all_records if record["member"] in fit_set]
    holdout_records = [record for record in all_records if record["member"] in holdout_set]
    if not fit_records or not holdout_records:
        raise V87Error("empty fit or holdout records")

    print("encoding candidates with frozen V8.6 acoustic refiner")
    fit_embeddings, fit_v86_probabilities = _encode_records(v86_encoder, fit_records)
    holdout_embeddings, holdout_v86_probabilities = _encode_records(v86_encoder, holdout_records)
    x_fit, y_fit = _sequence_arrays(fit_records, fit_embeddings, fit_v86_probabilities)
    x_holdout, y_holdout = _sequence_arrays(holdout_records, holdout_embeddings, holdout_v86_probabilities)

    model = _build_memory_head()
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)]
    history = model.fit(
        x_fit,
        y_fit,
        sample_weight=_weights(fit_records),
        validation_data=(x_holdout, y_holdout),
        epochs=args.epochs,
        batch_size=128,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )
    holdout_birth_scores = _birth_scores(model, x_holdout)
    calibration = _calibrate(holdout_records, holdout_birth_scores, holdout_tracks)
    retain_threshold = float(calibration["best"]["threshold"])

    print("evaluating once on frozen locked12")
    locked_scores = _predict_score_tracks(args.base_model, locked12)
    locked_ceiling = _candidate_ceiling(locked12, locked_scores, floor)
    locked_records = _records(locked12, locked_scores, floor)
    locked_embeddings, locked_v86_probabilities = _encode_records(v86_encoder, locked_records)
    x_locked, _ = _sequence_arrays(locked_records, locked_embeddings, locked_v86_probabilities)
    locked_birth_scores = _birth_scores(model, x_locked)
    locked_predictions = _retained_predictions(locked_records, locked_birth_scores, retain_threshold)
    locked_metrics = {
        key: _aggregate(locked12, locked_predictions, arrangement)
        for key, arrangement in (("global", None), ("solo", "solo"), ("comp", "comp"))
    }
    aucs = {
        key: _candidate_auc(locked_records, locked_birth_scores, arrangement)
        for key, arrangement in (("global", None), ("solo", "solo"), ("comp", "comp"))
    }

    model.save_weights(args.output_dir / "v87-causal-candidate-memory.weights.h5")
    report = {
        "schema_version": 1,
        "architecture": {
            "name": "V8.7 causal candidate-memory transition refiner",
            "base": "frozen V8.4 control epoch 01 onset stream + frozen V8.6 candidate encoder",
            "base_trainable": False,
            "v86_encoder_trainable": False,
            "offset_stream_executed": False,
            "offset_weights_modified": False,
            "runtime_inputs_use_annotations": False,
            "history_candidates": HISTORY_CANDIDATES,
            "past_context_ms": PAST_CONTEXT_MS,
            "future_candidate_context": False,
            "maximum_verification_delay_samples": MAX_HORIZON,
            "maximum_verification_delay_ms": MAX_HORIZON * 1000.0 / SAMPLE_RATE,
            "transition_classes": list(CLASS_NAMES),
            "memory_head_parameters": int(model.count_params()),
            "candidate_feature_dim": CANDIDATE_FEATURE_DIM,
        },
        "configuration": {
            "seed": args.seed,
            "candidate_floor": floor,
            "candidate_floor_reused_from_train_only_v86": True,
            "retain_threshold": retain_threshold,
            "retain_threshold_selected_on_locked_validation": False,
            "matching_tolerance_ms": TOLERANCE_MS,
            "requested_epochs": args.epochs,
            "epochs_ran": len(history.history["loss"]),
        },
        "data": {
            "source_train_members": source_members,
            "fit_members": list(fit_members),
            "holdout_members": list(holdout_members),
            "locked_validation_members": [track.annotation_member for track in locked12],
            "fit": _class_summary(fit_records),
            "holdout": _class_summary(holdout_records),
            "locked12": _class_summary(locked_records),
        },
        "holdout_retain_calibration": calibration,
        "training_history": {
            key: [float(value) for value in values] for key, values in history.history.items()
        },
        "locked12": {
            "candidate_ceiling": locked_ceiling,
            "candidate_auc_birth_high": aucs,
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
                "candidate_auc": aucs,
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
