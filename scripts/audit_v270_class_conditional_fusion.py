"""V27 class-conditional fusion of frozen V10.4 and V26 count experts.

V10.4 remains the candidate-identity and event-ranking anchor.  The V26
uniform count head is allowed to arbitrate only rows where V10.4 predicts a
low cardinality.  No model is retrained and no outer label is used by either
fusion rule.

Two fixed rules are reported together:

``null_veto``
    Replace V10.4 K=1 by K=0 only when V26-uniform predicts K=0.

``low_k_fusion``
    Use V26-uniform whenever V10.4 predicts K in {0, 1}; retain V10.4 for
    every V10.4 prediction K>=2.

For a true polyphonic row (K>=2), either rule changes only a V10.4 prediction
that was already wrong.  Polyphonic exact-K therefore cannot decrease.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from causal_note.guitarset import SLOT_COUNT
from scripts import train_v104_class_conditional_fusion as v104
from scripts.train_v100_spectral_string_slots import _load_spectral_caches
from scripts.train_v91_ordinal_cardinality import _dataset_split


FOLD_COUNT = 5
V104_NESTED_RUN = 33647694565
V104_NESTED_SHA = "345dea1e92f6281583c58590a0b1286ce4df459b"
V111_AUDIT_RUN = 33664141756
V111_AUDIT_SHA = "91717bb8fc0381ce91d807aacfa910fb8a7b1355"
V260_RUN = 34201830555
V260_SHA = "5f41972a6e2f081d6f2ad652beae31dfff9f0076"
ARMS = ("v104", "v260_uniform", "null_veto", "low_k_fusion")


class V270Error(RuntimeError):
    pass


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _one(paths, label: str) -> Path:
    paths = list(paths)
    if len(paths) != 1:
        raise V270Error(f"expected one {label}, found {len(paths)}")
    return paths[0]


def _prediction(values, label: str) -> np.ndarray:
    pred = np.asarray(values, dtype=np.int32)
    if pred.ndim != 1 or np.any((pred < 0) | (pred > SLOT_COUNT)):
        raise V270Error(f"invalid {label} cardinality predictions")
    return pred


def fuse_counts(anchor, specialist, arm: str) -> np.ndarray:
    """Apply one fixed V27 rule without consulting labels."""
    anchor = _prediction(anchor, "anchor")
    specialist = _prediction(specialist, "specialist")
    if anchor.shape != specialist.shape:
        raise V270Error("anchor/specialist shape mismatch")
    if arm == "null_veto":
        return np.where((anchor == 1) & (specialist == 0), 0, anchor).astype(np.int32)
    if arm == "low_k_fusion":
        return np.where(anchor <= 1, specialist, anchor).astype(np.int32)
    raise ValueError(f"unknown V27 arm: {arm}")


def cardinality_report(k, pred) -> dict:
    k = _prediction(k, "truth")
    pred = _prediction(pred, "prediction")
    if k.shape != pred.shape:
        raise V270Error("truth/prediction shape mismatch")
    confusion = np.bincount(k * (SLOT_COUNT + 1) + pred, minlength=(SLOT_COUNT + 1) ** 2)
    confusion = confusion.reshape(SLOT_COUNT + 1, SLOT_COUNT + 1)
    poly = k >= 2
    by_true_k = {}
    for kk in range(SLOT_COUNT + 1):
        mask = k == kk
        by_true_k[str(kk)] = {
            "rows": int(mask.sum()),
            "correct": int(np.sum(pred[mask] == kk)),
            "exact": float(np.mean(pred[mask] == kk)) if mask.any() else None,
            "under": int(np.sum(pred[mask] < kk)),
            "over": int(np.sum(pred[mask] > kk)),
        }
    return {
        "rows": int(len(k)),
        "correct": int(np.sum(k == pred)),
        "exact": float(np.mean(k == pred)),
        "poly_rows": int(poly.sum()),
        "poly_correct": int(np.sum(k[poly] == pred[poly])),
        "poly_exact": float(np.mean(k[poly] == pred[poly])) if poly.any() else None,
        "under": int(np.sum(pred < k)),
        "over": int(np.sum(pred > k)),
        "k0_false_birth_rows": int(np.sum((k == 0) & (pred > 0))),
        "k1_omission_rows": int(np.sum((k == 1) & (pred == 0))),
        "confusion_true_by_predicted": confusion.tolist(),
        "by_true_k": by_true_k,
    }


def transition_report(k, anchor, treatment) -> dict:
    k = _prediction(k, "truth")
    anchor = _prediction(anchor, "anchor")
    treatment = _prediction(treatment, "treatment")
    if not (k.shape == anchor.shape == treatment.shape):
        raise V270Error("transition shape mismatch")
    changed = treatment != anchor
    corrected = changed & (anchor != k) & (treatment == k)
    regressed = changed & (anchor == k) & (treatment != k)
    return {
        "changed_rows": int(changed.sum()),
        "corrected_rows": int(corrected.sum()),
        "regressed_rows": int(regressed.sum()),
        "net_correct_rows": int(corrected.sum() - regressed.sum()),
        "changed_by_true_k": {
            str(kk): int(np.sum(changed & (k == kk))) for kk in range(SLOT_COUNT + 1)
        },
    }


def _load_v111(source_dir: Path):
    prediction_path = _one(source_dir.glob("**/predictions.npz"), "V11.1 predictions")
    report_path = _one(source_dir.glob("**/report.json"), "V11.1 report")
    report = json.loads(report_path.read_text())
    protocol = report.get("protocol", {})
    if protocol.get("train_only_nested_outer_holdout") is not True:
        raise V270Error("V11.1 source is not nested outer-clean")
    if protocol.get("historical_validation_or_locked12_indexed_or_evaluated") is not False:
        raise V270Error("V11.1 source touched locked validation")
    with np.load(prediction_path, allow_pickle=False) as z:
        required = {"global_index", "k", "member", "pred104"}
        if not required.issubset(z.files):
            raise V270Error("V11.1 prediction schema mismatch")
        data = {key: np.asarray(z[key]) for key in required}
    order = np.argsort(data["global_index"], kind="stable")
    data = {key: value[order] for key, value in data.items()}
    return data, report, prediction_path, report_path


def _load_v260(source_dir: Path):
    rows = []
    source_files = []
    for fold in range(FOLD_COUNT):
        prediction_path = _one(
            source_dir.glob(f"**/predictions-fold-{fold}.npz"), f"V26 fold {fold} predictions"
        )
        report_path = _one(
            source_dir.glob(f"**/report-fold-{fold}.json"), f"V26 fold {fold} report"
        )
        report = json.loads(report_path.read_text())
        if report.get("fold") != fold:
            raise V270Error(f"V26 report fold mismatch: {fold}")
        if report.get("protocol", {}).get("experiment") != "v260_count_weighting_ab":
            raise V270Error(f"V26 protocol mismatch: fold {fold}")
        with np.load(prediction_path, allow_pickle=False) as z:
            required = {"global_index", "member", "k", "uniform_probability"}
            if not required.issubset(z.files):
                raise V270Error(f"V26 prediction schema mismatch: fold {fold}")
            probability = np.asarray(z["uniform_probability"], dtype=np.float64)
            if probability.ndim != 2 or probability.shape[1] != SLOT_COUNT + 1:
                raise V270Error(f"V26 probability shape mismatch: fold {fold}")
            if not np.isfinite(probability).all() or not np.allclose(probability.sum(1), 1, atol=1e-5):
                raise V270Error(f"V26 probability normalization mismatch: fold {fold}")
            rows.append(
                {
                    "global_index": np.asarray(z["global_index"], dtype=np.int64),
                    "member": np.asarray(z["member"]).astype(str),
                    "k": np.asarray(z["k"], dtype=np.int32),
                    "uniform_probability": probability.astype(np.float32),
                    "fold": np.full(len(z["k"]), fold, dtype=np.int16),
                }
            )
        source_files.extend((prediction_path, report_path))
    merged = {key: np.concatenate([row[key] for row in rows]) for key in rows[0]}
    order = np.argsort(merged["global_index"], kind="stable")
    merged = {key: value[order] for key, value in merged.items()}
    return merged, source_files


def _same_event_counts(actual: dict, expected: dict) -> bool:
    return all(actual.get(key) == expected.get(key) for key in ("true_positive", "false_positive", "false_negative"))


def audit(args):
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    cache = _load_spectral_caches(args.cache_dir)
    _, train_split, validation = _dataset_split(args.dataset_dir)
    train_members = {track.annotation_member for track in train_split}
    validation_members = {track.annotation_member for track in validation}
    if set(cache["track_members"]) != train_members or train_members & validation_members:
        raise V270Error("dataset/cache split mismatch")

    v111, v111_report, v111_predictions, v111_report_path = _load_v111(args.v111_dir)
    v260, v260_files = _load_v260(args.v260_dir)
    expected_index = np.arange(len(cache["exact"]), dtype=np.int64)
    if not np.array_equal(v111["global_index"], expected_index):
        raise V270Error("V11.1 does not provide exact one-time global coverage")
    if not np.array_equal(v260["global_index"], expected_index):
        raise V270Error("V26 does not provide exact one-time global coverage")
    k = np.minimum(np.asarray(cache["exact"], dtype=np.int32), SLOT_COUNT)
    members = np.asarray(cache["members"]).astype(str)
    if not np.array_equal(v111["k"].astype(np.int32), k) or not np.array_equal(v260["k"], k):
        raise V270Error("source cardinality labels do not align with cache")
    if not np.array_equal(v111["member"].astype(str), members) or not np.array_equal(v260["member"], members):
        raise V270Error("source members do not align with cache")

    anchor = _prediction(v111["pred104"], "V10.4")
    uniform = np.argmax(v260["uniform_probability"], axis=1).astype(np.int32)
    predictions = {
        "v104": anchor,
        "v260_uniform": uniform,
        "null_veto": fuse_counts(anchor, uniform, "null_veto"),
        "low_k_fusion": fuse_counts(anchor, uniform, "low_k_fusion"),
    }

    cardinality = {name: cardinality_report(k, pred) for name, pred in predictions.items()}
    for name in ("null_veto", "low_k_fusion"):
        if cardinality[name]["poly_correct"] < cardinality["v104"]["poly_correct"]:
            raise V270Error(f"{name} violated the structural polyphonic guard")

    indices = np.arange(len(k), dtype=np.int64)
    event_metrics = {
        name: v104._metrics_for_indices(cache, train_split, indices, pred)
        for name, pred in predictions.items()
    }
    source_v104 = v111_report["strata"]["aggregate"]["v104"]["metrics"]
    if not _same_event_counts(event_metrics["v104"]["global"], source_v104["global"]):
        raise V270Error("frozen V10.4 event baseline does not reproduce V11.1")

    per_fold = {}
    for fold in range(FOLD_COUNT):
        mask = v260["fold"] == fold
        per_fold[str(fold)] = {
            name: cardinality_report(k[mask], pred[mask]) for name, pred in predictions.items()
        }

    transition = {
        name: transition_report(k, anchor, predictions[name])
        for name in ("null_veto", "low_k_fusion")
    }
    deltas = {}
    for name in ("v260_uniform", "null_veto", "low_k_fusion"):
        deltas[name] = {
            "exact_k_percentage_points_vs_v104": 100.0 * (
                cardinality[name]["exact"] - cardinality["v104"]["exact"]
            ),
            "poly_exact_k_percentage_points_vs_v104": 100.0 * (
                cardinality[name]["poly_exact"] - cardinality["v104"]["poly_exact"]
            ),
            "event_f1_percentage_points_vs_v104": 100.0 * (
                event_metrics[name]["global"]["f1"] - event_metrics["v104"]["global"]["f1"]
            ),
        }

    result = {
        "schema_version": 1,
        "protocol": {
            "experiment": "v270_class_conditional_count_fusion",
            "frozen_v104_anchor": True,
            "frozen_v260_uniform_specialist": True,
            "trainable_parameters": 0,
            "outer_labels_used_by_fusion_rules": False,
            "development_outer_predictions_previously_examined_to_define_rules": True,
            "historical_validation_or_locked12_indexed_or_evaluated": False,
            "candidate_identity_and_ranking": "frozen V10.4 cache/runtime ranking",
            "offset_model_untouched": True,
            "threshold_tuning": False,
            "rules": {
                "null_veto": "if V104_K == 1 and V260_uniform_K == 0: K=0; else V104_K",
                "low_k_fusion": "if V104_K <= 1: use V260_uniform_K; else use V104_K",
            },
            "polyphonic_guard": (
                "a treatment may alter only rows whose V10.4 prediction is <=1; "
                "therefore it cannot invalidate a correct prediction for true K>=2"
            ),
            "automatic_promotion": False,
        },
        "sources": {
            "v104_nested_run": V104_NESTED_RUN,
            "v104_nested_sha": V104_NESTED_SHA,
            "v111_audit_run": V111_AUDIT_RUN,
            "v111_audit_sha": V111_AUDIT_SHA,
            "v260_run": V260_RUN,
            "v260_sha": V260_SHA,
            "files": [
                {"path": str(v111_predictions), "sha256": digest(v111_predictions)},
                {"path": str(v111_report_path), "sha256": digest(v111_report_path)},
                *({"path": str(path), "sha256": digest(path)} for path in v260_files),
            ],
        },
        "data": {
            "rows": int(len(k)),
            "folds": FOLD_COUNT,
            "train_tracks": int(len(train_split)),
            "historical_validation_tracks_not_evaluated": int(len(validation)),
        },
        "cardinality": cardinality,
        "event_metrics": event_metrics,
        "transition_from_v104": transition,
        "deltas": deltas,
        "per_fold_cardinality": per_fold,
    }

    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        global_index=expected_index,
        fold=v260["fold"],
        member=members,
        k=k.astype(np.int16),
        pred104=anchor.astype(np.int16),
        pred260_uniform=uniform.astype(np.int16),
        pred270_null_veto=predictions["null_veto"].astype(np.int16),
        pred270_low_k_fusion=predictions["low_k_fusion"].astype(np.int16),
    )
    print(
        json.dumps(
            {
                name: {
                    "exact_k": cardinality[name]["exact"],
                    "poly_exact_k": cardinality[name]["poly_exact"],
                    "event_f1": event_metrics[name]["global"]["f1"],
                }
                for name in ARMS
            },
            indent=2,
            sort_keys=True,
        )
    )
    return result


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--v111-dir", type=Path, required=True)
    p.add_argument("--v260-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def main():
    audit(parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
