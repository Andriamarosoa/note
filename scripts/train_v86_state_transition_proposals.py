"""V8.6 onset proposal recovery + multi-state transition pilot.

Scientific goal: address both residual V8.5 error stages without touching offset.
The frozen V8.4 control onset stream is used as an acoustic proposal prior, but
its hard 0.40 rising-edge decoder is replaced by a low-floor local-peak proposal
pool calibrated only on a train-only holdout. A learned multi-horizon refiner
classifies each anonymous candidate as one of:

  continuation/retrigger, other false transient, isolated birth, cluster birth.

At runtime only audio + frozen V8.4 onset scores are inputs. GuitarSet labels are
used only to train/evaluate the refiner. Offset is never executed or modified.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
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

from causal_note.guitarset import ALLOWED_PLAYERS, SAMPLE_RATE, index_guitarset
from causal_note.v84_onset_predictor import V84OnsetOnlyKerasPredictor
from scripts.audit_v84_solo_comp_onsets import _auc
from scripts.evaluate_boundaries import _count_metrics, match_boundaries, milliseconds_to_samples
from scripts.evaluate_v8_boundaries import DEFAULT_SEED, DEFAULT_VALIDATION_FRACTION, _arrangement, _reference_positions
from scripts.train_boundaries import decode_pcm16_mono_wav, split_tracks_by_group
from scripts.train_v85_transition_verifier import _split_members


RECEPTIVE_FIELD = 4093
CHUNK_SIZE = 512
TOLERANCE_MS = 50.0
TOLERANCE_SAMPLES = milliseconds_to_samples(TOLERANCE_MS)
BASELINE_THRESHOLD = 0.40
CANDIDATE_FLOORS = (0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
HORIZONS = (256, 512, 1024)
MAX_HORIZON = max(HORIZONS)
FFT_SIZE = 2048
BANDS = 96
MIN_HZ = 55.0
MAX_HZ = 8000.0
PEAK_MERGE_SAMPLES = 4
CLUSTER_MS = 20.0
CLUSTER_SAMPLES = milliseconds_to_samples(CLUSTER_MS)
CLASS_NAMES = ("continuation", "other_fp", "isolated_birth", "cluster_birth")
BIRTH_CLASSES = (2, 3)
DEFAULT_SEED_V86 = 8631
DEFAULT_TRAIN_MEMBERS = 30


class V86Error(RuntimeError):
    pass


def _pcm_window(samples: np.ndarray, start: int, length: int) -> np.ndarray:
    result = np.zeros(length, dtype=np.float64)
    source_start = max(0, start)
    source_end = min(len(samples), start + length)
    if source_end > source_start:
        destination = source_start - start
        result[destination:destination + source_end - source_start] = samples[source_start:source_end]
    return result


def _spectral_transition(samples: np.ndarray, sample: int, window: int) -> np.ndarray:
    pre_values = _pcm_window(samples, sample - window, window)
    post_values = _pcm_window(samples, sample, window)
    taper = np.hanning(window)
    pre_fft = np.abs(np.fft.rfft(pre_values * taper, n=FFT_SIZE)) ** 2
    post_fft = np.abs(np.fft.rfft(post_values * taper, n=FFT_SIZE)) ** 2
    freqs = np.fft.rfftfreq(FFT_SIZE, d=1.0 / SAMPLE_RATE)
    targets = np.geomspace(MIN_HZ, MAX_HZ, BANDS)
    pre = np.interp(targets, freqs, pre_fft)
    post = np.interp(targets, freqs, post_fft)
    eps = 1e-12
    pre_scale = float(np.mean(pre)) + eps
    post_scale = float(np.mean(post)) + eps
    pre_state = np.clip(np.log1p(pre / pre_scale), 0.0, 10.0)
    post_state = np.clip(np.log1p(post / post_scale), 0.0, 10.0)
    positive = np.clip(np.log1p(np.maximum(post - pre, 0.0) / pre_scale), 0.0, 10.0)
    negative = np.clip(np.log1p(np.maximum(pre - post, 0.0) / pre_scale), 0.0, 10.0)
    return np.concatenate((pre_state, post_state, positive, negative)).astype(np.float32)


def _score_context(presence: np.ndarray, multiplicity: np.ndarray, sample: int) -> np.ndarray:
    def value(index: int) -> float:
        if index < 0 or index >= len(presence):
            return 0.0
        return float(presence[index])

    offsets = (32, 128, 256, 512, 1024)
    peak = value(sample)
    features = [peak]
    for offset in offsets:
        features.append(peak - value(sample - offset))
        features.append(peak - value(sample + offset))
    left = presence[max(0, sample - 512):sample + 1]
    right = presence[sample:min(len(presence), sample + 513)]
    features.extend([
        float(np.mean(left)) if len(left) else 0.0,
        float(np.max(left)) if len(left) else 0.0,
        float(np.mean(right)) if len(right) else 0.0,
        float(np.max(right)) if len(right) else 0.0,
    ])
    row = multiplicity[sample] if 0 <= sample < len(multiplicity) else np.asarray([1.0, 0.0, 0.0])
    features.extend(float(x) for x in row)
    return np.asarray(features, dtype=np.float32)


def _predict_score_tracks(model_path: Path, tracks) -> Dict[str, dict]:
    predictor = V84OnsetOnlyKerasPredictor.from_path(str(model_path), receptive_field=RECEPTIVE_FIELD)
    predictor.warm_up(CHUNK_SIZE)
    output: Dict[str, dict] = {}
    for ordinal, track in enumerate(tracks, start=1):
        audio = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        predictor.reset()
        presence: List[float] = []
        multiplicity: List[Tuple[float, float, float]] = []
        position = 0
        while position < audio.frame_count:
            end = min(audio.frame_count, position + CHUNK_SIZE)
            values = tuple(sample / 32768.0 for sample in audio.samples[position:end])
            scores = predictor.predict_chunk(values, start_sample=position)
            presence.extend(scores.onset_presence)
            multiplicity.extend(scores.onset_multiplicity)
            position = end
        output[track.annotation_member] = {
            "presence": np.asarray(presence, dtype=np.float32),
            "multiplicity": np.asarray(multiplicity, dtype=np.float32),
        }
        print(f"scores {ordinal}/{len(tracks)}: {track.annotation_member}")
    return output


def _decode_count(row: np.ndarray) -> int:
    return int(np.argmax(row)) + 1


def _candidate_groups(score_record: dict, floor: float) -> List[dict]:
    presence = score_record["presence"]
    multiplicity = score_record["multiplicity"]
    raw: List[Tuple[int, float, int, str]] = []
    for sample in range(1, max(1, len(presence) - 1)):
        score = float(presence[sample])
        if score < floor:
            continue
        if score >= float(presence[sample - 1]) and score > float(presence[sample + 1]):
            raw.append((sample, score, _decode_count(multiplicity[sample]), "peak"))

    # Preserve the exact legacy V8.4 0.40 rising-edge proposal positions in the pool.
    high = False
    for sample, value in enumerate(presence):
        now = float(value) >= BASELINE_THRESHOLD
        if now and not high:
            raw.append((sample, float(value), _decode_count(multiplicity[sample]), "baseline_edge"))
        high = now

    raw.sort(key=lambda row: (row[0], -row[1], row[3]))
    merged: List[dict] = []
    for sample, score, count, source in raw:
        if merged and sample - int(merged[-1]["sample"]) <= PEAK_MERGE_SAMPLES:
            previous = merged[-1]
            if score > float(previous["score"]):
                previous.update(sample=int(sample), score=float(score), count=int(count))
            previous["sources"].add(source)
            continue
        merged.append({"sample": int(sample), "score": float(score), "count": int(count), "sources": {source}})
    for row in merged:
        row["sources"] = tuple(sorted(row["sources"]))
    return merged


def _expanded(groups: Sequence[dict]) -> Tuple[int, ...]:
    return tuple(int(group["sample"]) for group in groups for _ in range(int(group["count"])))


def _candidate_ceiling(tracks, score_by_member: Dict[str, dict], floor: float) -> dict:
    refs = preds = tp = 0
    by_arr = {"solo": [0, 0, 0], "comp": [0, 0, 0]}
    for track in tracks:
        audio = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        references, _ = _reference_positions(track, audio.frame_count)
        predictions = _expanded(_candidate_groups(score_by_member[track.annotation_member], floor))
        pairs = match_boundaries(references, predictions, TOLERANCE_SAMPLES)
        refs += len(references); preds += len(predictions); tp += len(pairs)
        arr = _arrangement(track.annotation_member)
        by_arr[arr][0] += len(references); by_arr[arr][1] += len(predictions); by_arr[arr][2] += len(pairs)
    result = asdict(_count_metrics(refs, preds, tp))
    result["prediction_reference_ratio"] = preds / refs if refs else None
    result["arrangement"] = {}
    for arr, (r, p, t) in by_arr.items():
        metrics = asdict(_count_metrics(r, p, t)); metrics["prediction_reference_ratio"] = p / r if r else None
        result["arrangement"][arr] = metrics
    return result


def _choose_floor(tracks, score_by_member: Dict[str, dict]) -> dict:
    sweep = [{"floor": float(floor), **_candidate_ceiling(tracks, score_by_member, floor)} for floor in CANDIDATE_FLOORS]
    best_recall = max(row["recall"] for row in sweep)
    target = 0.95 if best_recall >= 0.95 else max(0.0, best_recall - 0.005)
    eligible = [row for row in sweep if row["recall"] >= target]
    best = max(eligible, key=lambda row: (row["floor"], -row["prediction_reference_ratio"]))
    return {"selection_rule": "highest train-holdout floor retaining >=95% recall when attainable, otherwise within 0.5pt of best holdout proposal recall", "target_recall": target, "best": best, "sweep": sweep}


def _nearest_neighbor_distance(reference: int, references: Sequence[int]) -> Optional[int]:
    others = [abs(int(value) - int(reference)) for value in references if int(value) != int(reference)]
    return min(others) if others else None


def _previous_reference_age(sample: int, references: Sequence[int]) -> Optional[int]:
    previous = [sample - int(value) for value in references if int(value) <= sample]
    return min(previous) if previous else None


def _records(tracks, score_by_member: Dict[str, dict], floor: float) -> List[dict]:
    records: List[dict] = []
    for ordinal, track in enumerate(tracks, start=1):
        member = track.annotation_member
        audio = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        normalized = np.asarray(audio.samples, dtype=np.float64) / 32768.0
        references, _ = _reference_positions(track, audio.frame_count)
        groups = _candidate_groups(score_by_member[member], floor)
        expanded_samples = _expanded(groups)
        pairs = match_boundaries(references, expanded_samples, TOLERANCE_SAMPLES)
        matched_prediction_counts = Counter(pred for _, pred in pairs)
        matched_references_by_prediction: Dict[int, List[int]] = defaultdict(list)
        for reference, prediction in pairs:
            matched_references_by_prediction[int(prediction)].append(int(reference))

        presence = score_by_member[member]["presence"]
        multiplicity = score_by_member[member]["multiplicity"]
        for group in groups:
            sample = int(group["sample"])
            matched_refs = matched_references_by_prediction.get(sample, [])
            is_birth = bool(matched_refs)
            if is_birth:
                matched_ref = min(matched_refs, key=lambda ref: abs(ref - sample))
                nearest = _nearest_neighbor_distance(matched_ref, references)
                class_id = 3 if nearest is not None and nearest <= CLUSTER_SAMPLES else 2
            else:
                age = _previous_reference_age(sample, references)
                class_id = 0 if age is not None and age <= CLUSTER_SAMPLES else 1
            horizon_features = [_spectral_transition(normalized, sample, horizon) for horizon in HORIZONS]
            records.append({
                "member": member,
                "arrangement": _arrangement(member),
                "sample": sample,
                "count": int(group["count"]),
                "score": float(group["score"]),
                "sources": list(group["sources"]),
                "class_id": int(class_id),
                "birth": int(class_id in BIRTH_CLASSES),
                "matched_reference_count": int(matched_prediction_counts[sample]),
                "horizons": horizon_features,
                "score_context": _score_context(presence, multiplicity, sample),
            })
        print(f"features {ordinal}/{len(tracks)}: {member} candidates={len(groups)}")
    return records


def _arrays(records: Sequence[dict]):
    x = {f"h{horizon}": np.stack([r["horizons"][index] for r in records]) for index, horizon in enumerate(HORIZONS)}
    x["score_context"] = np.stack([r["score_context"] for r in records])
    y = np.asarray([r["class_id"] for r in records], dtype=np.int32)
    return x, y


def _weights(records: Sequence[dict]) -> np.ndarray:
    counts = Counter((r["arrangement"], int(r["class_id"])) for r in records)
    present = tuple(sorted(counts))
    total = len(records)
    return np.asarray([total / (len(present) * counts[(r["arrangement"], int(r["class_id"]))]) for r in records], dtype=np.float32)


def _build_refiner():
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for V8.6 training") from exc
    inputs = {}
    encoded = []
    for horizon in HORIZONS:
        inp = keras.Input((BANDS * 4,), name=f"h{horizon}")
        inputs[f"h{horizon}"] = inp
        x = keras.layers.LayerNormalization(name=f"h{horizon}_norm")(inp)
        x = keras.layers.Dense(64, activation="relu", name=f"h{horizon}_dense1")(x)
        x = keras.layers.Dense(32, activation="relu", name=f"h{horizon}_dense2")(x)
        encoded.append(x)
    score_context = keras.Input((18,), name="score_context")
    inputs["score_context"] = score_context
    score_z = keras.layers.LayerNormalization(name="score_context_norm")(score_context)
    score_z = keras.layers.Dense(24, activation="relu", name="score_context_dense")(score_z)
    encoded.append(score_z)
    hidden = keras.layers.Concatenate(name="multi_horizon_transition") (encoded)
    hidden = keras.layers.Dense(128, activation="relu", name="transition_hidden1")(hidden)
    hidden = keras.layers.Dropout(0.15, name="transition_dropout")(hidden)
    hidden = keras.layers.Dense(64, activation="relu", name="transition_hidden2")(hidden)
    classes = keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="transition_class")(hidden)
    model = keras.Model(inputs=inputs, outputs=classes, name="v86_state_transition_proposal_refiner")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4), loss="sparse_categorical_crossentropy")
    return model


def _birth_scores(model, records: Sequence[dict]) -> np.ndarray:
    x, _ = _arrays(records)
    probabilities = model.predict(x, batch_size=256, verbose=0)
    return probabilities[:, 2] + probabilities[:, 3]


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
        refs += len(references); preds += len(values); tp += len(pairs)
    result = asdict(_count_metrics(refs, preds, tp))
    result["prediction_reference_ratio"] = preds / refs if refs else None
    return result


def _evaluate_threshold(records, scores, tracks, threshold: float) -> dict:
    predictions = _retained_predictions(records, scores, threshold)
    return {key: _aggregate(tracks, predictions, arrangement) for key, arrangement in (("global", None), ("solo", "solo"), ("comp", "comp"))}


def _calibrate(records, scores, tracks) -> dict:
    sweep = []
    for threshold in np.arange(0.10, 0.901, 0.05):
        metrics = _evaluate_threshold(records, scores, tracks, float(threshold))
        macro = (metrics["solo"]["f1"] + metrics["comp"]["f1"]) / 2.0
        sweep.append({"threshold": float(round(threshold, 6)), "macro_f1": macro, "metrics": metrics})
    best = max(sweep, key=lambda row: (row["macro_f1"], row["metrics"]["global"]["f1"], -abs(row["threshold"] - 0.5)))
    return {"selection_rule": "max train-holdout macro(solo,comp) track-level F1", "best": best, "sweep": sweep}


def _class_summary(records: Sequence[dict]) -> dict:
    counts = Counter((r["arrangement"], CLASS_NAMES[int(r["class_id"])]) for r in records)
    return {"count": len(records), "tracks": len({r["member"] for r in records}), "strata": {f"{arr}_{name}": counts[(arr, name)] for arr in ("solo", "comp") for name in CLASS_NAMES}}


def _candidate_auc(records, scores, arrangement: Optional[str]) -> Optional[float]:
    selected = [(float(score), int(record["birth"])) for record, score in zip(records, scores) if arrangement is None or record["arrangement"] == arrangement]
    positives = [score for score, label in selected if label]
    negatives = [score for score, label in selected if not label]
    return _auc(positives, negatives)


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train V8.6 multi-state onset proposal refiner on frozen V8.4 onset scores.")
    parser.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--train-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-members", type=int, default=DEFAULT_TRAIN_MEMBERS)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED_V86)
    return parser


def run(args) -> dict:
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

    indexed = tuple(track for track in index_guitarset(args.dataset_dir) if track.player_id in ALLOWED_PLAYERS)
    by_member = {track.annotation_member: track for track in indexed}
    _, locked_validation = split_tracks_by_group(indexed, validation_fraction=DEFAULT_VALIDATION_FRACTION, seed=DEFAULT_SEED)
    locked12 = tuple(locked_validation[:12])
    locked_members = {track.annotation_member for track in locked_validation}

    audit = json.loads(args.train_audit.read_text(encoding="utf-8"))
    source_members = list(audit["scope"]["members"])[: args.train_members]
    if len(source_members) != args.train_members:
        raise V86Error("not enough source train members")
    if set(source_members) & locked_members:
        raise V86Error("train members overlap locked validation")
    fit_members, holdout_members = _split_members(source_members, args.seed)
    fit_set, holdout_set = set(fit_members), set(holdout_members)
    train_tracks = tuple(by_member[member] for member in source_members)
    holdout_tracks = tuple(by_member[member] for member in holdout_members)

    print("mining frozen V8.4 raw onset score streams on train-only members")
    train_scores = _predict_score_tracks(args.base_model, train_tracks)
    floor_calibration = _choose_floor(holdout_tracks, train_scores)
    floor = float(floor_calibration["best"]["floor"])
    print("selected proposal floor", floor, "holdout recall", floor_calibration["best"]["recall"], "pred/ref", floor_calibration["best"]["prediction_reference_ratio"])

    all_records = _records(train_tracks, train_scores, floor)
    fit_records = [record for record in all_records if record["member"] in fit_set]
    holdout_records = [record for record in all_records if record["member"] in holdout_set]
    if not fit_records or not holdout_records:
        raise V86Error("empty fit or holdout records")

    model = _build_refiner()
    x_fit, y_fit = _arrays(fit_records)
    x_holdout, y_holdout = _arrays(holdout_records)
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)]
    history = model.fit(
        x_fit, y_fit,
        sample_weight=_weights(fit_records),
        validation_data=(x_holdout, y_holdout),
        epochs=args.epochs,
        batch_size=64,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )
    holdout_birth_scores = _birth_scores(model, holdout_records)
    calibration = _calibrate(holdout_records, holdout_birth_scores, holdout_tracks)
    retain_threshold = float(calibration["best"]["threshold"])

    print("evaluating once on frozen locked12")
    locked_scores = _predict_score_tracks(args.base_model, locked12)
    locked_floor_ceiling = _candidate_ceiling(locked12, locked_scores, floor)
    locked_records = _records(locked12, locked_scores, floor)
    locked_birth_scores = _birth_scores(model, locked_records)
    locked_predictions = _retained_predictions(locked_records, locked_birth_scores, retain_threshold)
    locked_metrics = {key: _aggregate(locked12, locked_predictions, arrangement) for key, arrangement in (("global", None), ("solo", "solo"), ("comp", "comp"))}
    aucs = {key: _candidate_auc(locked_records, locked_birth_scores, arrangement) for key, arrangement in (("global", None), ("solo", "solo"), ("comp", "comp"))}

    model.save_weights(args.output_dir / "v86-state-transition-refiner.weights.h5")
    report = {
        "schema_version": 1,
        "architecture": {
            "name": "V8.6 anonymous multi-horizon state-transition proposal refiner",
            "base": "frozen V8.4 control epoch 01 onset stream",
            "base_trainable": False,
            "offset_stream_executed": False,
            "offset_weights_modified": False,
            "runtime_inputs_use_annotations": False,
            "transition_classes": list(CLASS_NAMES),
            "birth_classes": [CLASS_NAMES[index] for index in BIRTH_CLASSES],
            "horizons_samples": list(HORIZONS),
            "horizons_ms": [horizon * 1000.0 / SAMPLE_RATE for horizon in HORIZONS],
            "maximum_verification_delay_samples": MAX_HORIZON,
            "maximum_verification_delay_ms": MAX_HORIZON * 1000.0 / SAMPLE_RATE,
            "refiner_parameters": int(model.count_params()),
            "candidate_generator": "low-floor V8.4 local maxima + preserved 0.40 rising edges",
        },
        "configuration": {
            "seed": args.seed,
            "matching_tolerance_ms": TOLERANCE_MS,
            "cluster_ms": CLUSTER_MS,
            "candidate_floor_selected_on_locked_validation": False,
            "retain_threshold_selected_on_locked_validation": False,
            "candidate_floor": floor,
            "retain_threshold": retain_threshold,
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
        "proposal_floor_calibration": floor_calibration,
        "holdout_retain_calibration": calibration,
        "training_history": {key: [float(value) for value in values] for key, values in history.history.items()},
        "locked12": {
            "candidate_ceiling": locked_floor_ceiling,
            "candidate_auc_birth_high": aucs,
            "metrics": locked_metrics,
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "candidate_floor": floor,
        "holdout_candidate_recall": floor_calibration["best"]["recall"],
        "locked_candidate_recall": locked_floor_ceiling["recall"],
        "retain_threshold": retain_threshold,
        "candidate_auc": aucs,
        "locked_metrics": locked_metrics,
    }, indent=2, sort_keys=True))
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = create_argument_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
