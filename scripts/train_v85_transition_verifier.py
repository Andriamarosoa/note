"""Train a V8.5 onset state-transition verifier on frozen V8.4 proposals.

V8.4 control epoch 1 remains completely frozen.  This pilot learns a small
candidate-level verifier that decides whether an emitted onset proposal looks
like a genuine state birth or a continuation/retrigger of the current acoustic
state.  It uses audio only at inference time; annotation-derived active-note
masks from the novelty audit are not inputs.

The verifier compares anonymous pre/post spectral state.  A 1024-sample post
window makes the decision causal with a fixed 23.22 ms verification delay while
preserving the original candidate timestamp.  Offset is neither executed during
training nor modified.
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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_note.guitarset import ALLOWED_PLAYERS, SAMPLE_RATE, index_guitarset
from causal_note.v84_onset_predictor import V84OnsetOnlyKerasPredictor
from causal_note.v8_runtime import BoundaryKind, V8BoundaryDecoder
from scripts.audit_v84_solo_comp_onsets import _auc
from scripts.evaluate_boundaries import _count_metrics, match_boundaries, milliseconds_to_samples
from scripts.evaluate_v8_boundaries import DEFAULT_SEED, DEFAULT_VALIDATION_FRACTION, _arrangement, _reference_positions
from scripts.train_boundaries import decode_pcm16_mono_wav, split_tracks_by_group


PROPOSAL_THRESHOLD = 0.40
TOLERANCE_MS = 50.0
TOLERANCE_SAMPLES = milliseconds_to_samples(TOLERANCE_MS)
RECEPTIVE_FIELD = 4093
CHUNK_SIZE = 512
WINDOW = 1024
FFT_SIZE = 2048
BANDS = 96
MIN_HZ = 55.0
MAX_HZ = 8000.0
DEFAULT_TRAIN_MEMBERS = 30
DEFAULT_SEED_V85 = 8531


class V85Error(RuntimeError):
    pass


def _pcm_window(samples: np.ndarray, start: int, length: int) -> np.ndarray:
    result = np.zeros(length, dtype=np.float64)
    source_start = max(0, start)
    source_end = min(len(samples), start + length)
    if source_end > source_start:
        destination = source_start - start
        result[destination:destination + source_end - source_start] = samples[source_start:source_end]
    return result


def _spectral_transition(samples: np.ndarray, sample: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Anonymous state/change representation using audio only."""
    pre_values = _pcm_window(samples, sample - WINDOW, WINDOW)
    post_values = _pcm_window(samples, sample, WINDOW)
    taper = np.hanning(WINDOW)
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

    pre_rms = float(np.sqrt(np.mean(pre_values * pre_values)) + eps)
    post_rms = float(np.sqrt(np.mean(post_values * post_values)) + eps)
    rms_delta = float(np.clip(20.0 * math.log10(post_rms / pre_rms), -20.0, 20.0) / 10.0)
    flux_ratio = float(np.clip(np.log1p(np.maximum(post - pre, 0.0).sum() / (pre.sum() + eps)), 0.0, 10.0) / 5.0)
    scalars = np.asarray([rms_delta, flux_ratio], dtype=np.float32)
    return (
        pre_state.astype(np.float32),
        post_state.astype(np.float32),
        positive.astype(np.float32),
        negative.astype(np.float32),
        scalars,
    )


def _predict_tracks(model_path: Path, tracks) -> Dict[str, Tuple[int, ...]]:
    predictor = V84OnsetOnlyKerasPredictor.from_path(str(model_path), receptive_field=RECEPTIVE_FIELD)
    predictor.warm_up(CHUNK_SIZE)
    result = {}
    for ordinal, track in enumerate(tracks, start=1):
        audio = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        predictor.reset()
        decoder = V8BoundaryDecoder(onset_threshold=PROPOSAL_THRESHOLD, offset_threshold=PROPOSAL_THRESHOLD)
        predictions: List[int] = []
        position = 0
        while position < audio.frame_count:
            end = min(audio.frame_count, position + CHUNK_SIZE)
            values = tuple(sample / 32768.0 for sample in audio.samples[position:end])
            scores = predictor.predict_chunk(values, start_sample=position)
            for boundary in decoder.process_chunk(scores):
                if boundary.kind is BoundaryKind.ONSET:
                    predictions.extend([boundary.sample] * boundary.count)
            position = end
        result[track.annotation_member] = tuple(predictions)
        print(f"proposals {ordinal}/{len(tracks)}: {track.annotation_member} count={len(predictions)}")
    return result


def _candidate_records(tracks, predictions_by_member: Dict[str, Tuple[int, ...]]) -> List[dict]:
    records: List[dict] = []
    for ordinal, track in enumerate(tracks, start=1):
        member = track.annotation_member
        audio = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        references, _ = _reference_positions(track, audio.frame_count)
        predictions = predictions_by_member[member]
        pairs = match_boundaries(references, predictions, TOLERANCE_SAMPLES)
        matched = Counter(prediction for _, prediction in pairs)
        normalized = np.asarray(audio.samples, dtype=np.float64) / 32768.0
        cache = {}
        for sample in predictions:
            if sample not in cache:
                cache[sample] = _spectral_transition(normalized, sample)
            label = int(matched[sample] > 0)
            if label:
                matched[sample] -= 1
            pre, post, positive, negative, scalars = cache[sample]
            records.append({
                "member": member,
                "arrangement": _arrangement(member),
                "sample": int(sample),
                "label": label,
                "pre": pre,
                "post": post,
                "positive": positive,
                "negative": negative,
                "scalars": scalars,
            })
        print(f"features {ordinal}/{len(tracks)}: {member}")
    return records


def _split_members(members: Sequence[str], seed: int) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    rng = random.Random(seed)
    train: List[str] = []
    holdout: List[str] = []
    for arrangement in ("solo", "comp"):
        group = [member for member in members if _arrangement(member) == arrangement]
        rng.shuffle(group)
        if len(group) < 2:
            raise V85Error(f"not enough {arrangement} members for an internal holdout")
        n_holdout = max(1, round(len(group) * 0.20))
        holdout.extend(group[:n_holdout])
        train.extend(group[n_holdout:])
    return tuple(sorted(train)), tuple(sorted(holdout))


def _arrays(records: Sequence[dict]):
    return (
        {
            "pre_state": np.stack([r["pre"] for r in records]),
            "post_state": np.stack([r["post"] for r in records]),
            "positive_flux": np.stack([r["positive"] for r in records]),
            "negative_flux": np.stack([r["negative"] for r in records]),
            "scalars": np.stack([r["scalars"] for r in records]),
        },
        np.asarray([r["label"] for r in records], dtype=np.float32),
    )


def _stratified_weights(records: Sequence[dict]) -> np.ndarray:
    counts = Counter((r["arrangement"], int(r["label"])) for r in records)
    required = (("solo", 0), ("solo", 1), ("comp", 0), ("comp", 1))
    missing = [key for key in required if counts[key] == 0]
    if missing:
        raise V85Error(f"missing training strata: {missing}")
    total = len(records)
    return np.asarray([total / (4.0 * counts[(r["arrangement"], int(r["label"]))]) for r in records], dtype=np.float32)


def _build_verifier():
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for V8.5 training") from exc

    pre = keras.Input((BANDS,), name="pre_state")
    post = keras.Input((BANDS,), name="post_state")
    positive = keras.Input((BANDS,), name="positive_flux")
    negative = keras.Input((BANDS,), name="negative_flux")
    scalars = keras.Input((2,), name="scalars")

    state_norm = keras.layers.LayerNormalization(name="state_norm")
    state_projection = keras.layers.Dense(48, activation="relu", name="shared_state_projection")
    pre_z = state_projection(state_norm(pre))
    post_z = state_projection(state_norm(post))
    difference = keras.layers.Lambda(lambda values: tf.abs(values[0] - values[1]), name="state_abs_difference")([post_z, pre_z])
    product = keras.layers.Multiply(name="state_product")([pre_z, post_z])

    flux = keras.layers.Concatenate(name="flux_pair")([positive, negative])
    flux = keras.layers.LayerNormalization(name="flux_norm")(flux)
    flux = keras.layers.Dense(48, activation="relu", name="flux_projection")(flux)

    hidden = keras.layers.Concatenate(name="transition_evidence")([pre_z, post_z, difference, product, flux, scalars])
    hidden = keras.layers.Dense(64, activation="relu", name="transition_hidden_1")(hidden)
    hidden = keras.layers.Dense(32, activation="relu", name="transition_hidden_2")(hidden)
    retain = keras.layers.Dense(1, activation="sigmoid", name="retain_probability")(hidden)
    model = keras.Model(
        inputs={"pre_state": pre, "post_state": post, "positive_flux": positive, "negative_flux": negative, "scalars": scalars},
        outputs=retain,
        name="v85_state_transition_verifier",
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4), loss="binary_crossentropy")
    return model


def _classification_metrics(records: Sequence[dict], scores: np.ndarray, threshold: float, arrangement: Optional[str] = None) -> dict:
    selected = [(r, float(score)) for r, score in zip(records, scores) if arrangement is None or r["arrangement"] == arrangement]
    tp = fp = fn = tn = 0
    for record, score in selected:
        truth = bool(record["label"])
        pred = score >= threshold
        if truth and pred: tp += 1
        elif not truth and pred: fp += 1
        elif truth: fn += 1
        else: tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"count": len(selected), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def _calibrate(records: Sequence[dict], scores: np.ndarray) -> dict:
    rows = []
    for threshold in np.arange(0.10, 0.901, 0.05):
        solo = _classification_metrics(records, scores, float(threshold), "solo")
        comp = _classification_metrics(records, scores, float(threshold), "comp")
        rows.append({
            "threshold": float(round(threshold, 6)),
            "solo": solo,
            "comp": comp,
            "macro_f1": (solo["f1"] + comp["f1"]) / 2.0,
            "macro_recall": (solo["recall"] + comp["recall"]) / 2.0,
        })
    best = max(rows, key=lambda row: (row["macro_f1"], row["macro_recall"], -abs(row["threshold"] - 0.5)))
    return {"selection_rule": "max internal-holdout macro(solo,comp) candidate F1", "best": best, "sweep": rows}


def _score_records(model, records: Sequence[dict]) -> np.ndarray:
    x, _ = _arrays(records)
    return model.predict(x, batch_size=256, verbose=0).reshape(-1)


def _aggregate_track_metrics(tracks, predictions_by_member: Dict[str, Tuple[int, ...]], arrangement: Optional[str]) -> dict:
    refs = preds = tp = 0
    for track in tracks:
        arr = _arrangement(track.annotation_member)
        if arrangement is not None and arr != arrangement:
            continue
        audio = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        references, _ = _reference_positions(track, audio.frame_count)
        predictions = predictions_by_member.get(track.annotation_member, ())
        pairs = match_boundaries(references, predictions, TOLERANCE_SAMPLES)
        refs += len(references); preds += len(predictions); tp += len(pairs)
    metrics = asdict(_count_metrics(refs, preds, tp))
    metrics["prediction_reference_ratio"] = preds / refs if refs else None
    return metrics


def _evaluate_locked(model, records: Sequence[dict], tracks, baseline_predictions: Dict[str, Tuple[int, ...]], threshold: float) -> dict:
    scores = _score_records(model, records)
    retained: Dict[str, List[int]] = defaultdict(list)
    for record, score in zip(records, scores):
        if float(score) >= threshold:
            retained[record["member"]].append(int(record["sample"]))
    retained_frozen = {member: tuple(values) for member, values in retained.items()}

    result = {
        "candidate_auc_tp_high": {},
        "candidate_classification": {},
        "baseline": {},
        "gated": {},
    }
    for arrangement in (None, "solo", "comp"):
        key = "global" if arrangement is None else arrangement
        selected_scores = [float(score) for r, score in zip(records, scores) if arrangement is None or r["arrangement"] == arrangement]
        selected_labels = [int(r["label"]) for r in records if arrangement is None or r["arrangement"] == arrangement]
        positives = [s for s, y in zip(selected_scores, selected_labels) if y]
        negatives = [s for s, y in zip(selected_scores, selected_labels) if not y]
        result["candidate_auc_tp_high"][key] = _auc(positives, negatives)
        result["candidate_classification"][key] = _classification_metrics(records, scores, threshold, arrangement)
        result["baseline"][key] = _aggregate_track_metrics(tracks, baseline_predictions, arrangement)
        result["gated"][key] = _aggregate_track_metrics(tracks, retained_frozen, arrangement)
    return result


def _record_summary(records: Sequence[dict]) -> dict:
    counts = Counter((r["arrangement"], int(r["label"])) for r in records)
    return {
        "count": len(records),
        "tracks": len({r["member"] for r in records}),
        "strata": {f"{arrangement}_{'tp' if label else 'fp'}": counts[(arrangement, label)] for arrangement in ("solo", "comp") for label in (0, 1)},
    }


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train V8.5 state-transition verifier on frozen V8.4 onset proposals.")
    parser.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--train-audit", type=Path, required=True, help="Frozen audit used only to select its train-only member list")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-members", type=int, default=DEFAULT_TRAIN_MEMBERS)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED_V85)
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
    _, locked_validation = split_tracks_by_group(indexed, validation_fraction=DEFAULT_VALIDATION_FRACTION, seed=DEFAULT_SEED)
    locked12 = tuple(locked_validation[:12])
    locked_members = {track.annotation_member for track in locked_validation}

    audit = json.loads(args.train_audit.read_text(encoding="utf-8"))
    source_members = list(audit["scope"]["members"])
    if args.train_members > len(source_members):
        raise V85Error(f"requested {args.train_members} train members but audit has {len(source_members)}")
    source_members = source_members[: args.train_members]
    if set(source_members) & locked_members:
        raise V85Error("train audit member list overlaps locked validation")
    train_tracks_all = tuple(by_member[member] for member in source_members)
    train_members, holdout_members = _split_members(source_members, args.seed)

    print("mining frozen V8.4 control proposals on train-only members")
    train_predictions = _predict_tracks(args.base_model, train_tracks_all)
    all_train_records = _candidate_records(train_tracks_all, train_predictions)
    fit_records = [r for r in all_train_records if r["member"] in set(train_members)]
    holdout_records = [r for r in all_train_records if r["member"] in set(holdout_members)]
    if not fit_records or not holdout_records:
        raise V85Error("empty verifier fit or holdout candidate set")

    model = _build_verifier()
    x_train, y_train = _arrays(fit_records)
    x_holdout, y_holdout = _arrays(holdout_records)
    weights = _stratified_weights(fit_records)
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)]
    history = model.fit(
        x_train,
        y_train,
        sample_weight=weights,
        validation_data=(x_holdout, y_holdout),
        epochs=args.epochs,
        batch_size=64,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )
    holdout_scores = _score_records(model, holdout_records)
    calibration = _calibrate(holdout_records, holdout_scores)
    gate_threshold = float(calibration["best"]["threshold"])

    print("evaluating once on locked 12-track validation")
    locked_predictions = _predict_tracks(args.base_model, locked12)
    locked_records = _candidate_records(locked12, locked_predictions)
    locked_eval = _evaluate_locked(model, locked_records, locked12, locked_predictions, gate_threshold)

    model.save_weights(args.output_dir / "v85-transition-verifier.weights.h5")
    report = {
        "schema_version": 1,
        "architecture": {
            "name": "V8.5 anonymous spectral state-transition verifier",
            "base": "frozen V8.4 control epoch 01 onset stream",
            "base_trainable": False,
            "offset_stream_executed_during_verifier_training": False,
            "offset_weights_modified": False,
            "verifier_parameters": int(model.count_params()),
            "spectral_bands": BANDS,
            "pre_window_samples": WINDOW,
            "post_window_samples": WINDOW,
            "verification_delay_samples": WINDOW,
            "verification_delay_ms": WINDOW * 1000.0 / SAMPLE_RATE,
            "runtime_inputs_use_annotations": False,
        },
        "configuration": {
            "proposal_threshold": PROPOSAL_THRESHOLD,
            "matching_tolerance_ms": TOLERANCE_MS,
            "seed": args.seed,
            "requested_epochs": args.epochs,
            "epochs_ran": len(history.history["loss"]),
            "gate_threshold_selected_on_locked_validation": False,
        },
        "data": {
            "source_train_members": source_members,
            "fit_members": list(train_members),
            "holdout_members": list(holdout_members),
            "locked_validation_members": [track.annotation_member for track in locked12],
            "fit": _record_summary(fit_records),
            "holdout": _record_summary(holdout_records),
            "locked12": _record_summary(locked_records),
        },
        "training_history": {key: [float(v) for v in values] for key, values in history.history.items()},
        "holdout_calibration": calibration,
        "locked12": locked_eval,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = create_argument_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
