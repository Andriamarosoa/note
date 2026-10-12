"""Reproduce original S5 probabilities and bin signatures without labels or fitting.

Example: python -m scripts.verify_v273_s5_owned527_recovery --fetch
         --cache-root model/s5-recovery --output model/s5-recovery-proof
All original inputs are pinned and SHA256 checked. Output paths are immutable.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.ensemble._hist_gradient_boosting.binning import _find_binning_thresholds
from scripts.v273_s4_owned527 import build_owned527

FOLDS = (0, 1, 2, 4)
COMMIT = "7165e1ef54510aa9effc9f597b2783c832ca907a"
RELEASE = "v273-open-k0-k6-37837235723"
ORIGINAL_DESIGN_DIGEST = "fd4d4d32872e5d3856bf994db386e9f2976ce57a418a21b56b876bf6f6d349cd"
FEATURE_TAR_DIGEST = {
    0: "876cc5c1a32fd2377b20831b112de39ec54515a2de66541c303621443a9cbf7c",
    1: "d79af068795e2f9840f612510660088c2b69a6d4340c1aabaf115f60f42369f8",
    2: "060dc37e445b8d3ef49c1d4cb8f5cc9350908ae48cdabe7afff7bf4336c108e1",
    4: "639b810ec20aebd645e2026a5e820396f8e6fdbddd818ff90bc6f17a83e7ad51",
}
MODEL_DIGEST = {
    0: "64d6bfe1d3f5ba3367b22c8295e77409e3f33c2dd4fe390e03a8afad5974bae4",
    1: "1527955a9b39662ee888e562c3fd59524a6962128ac20e5bd6fc6e8ec268053c",
    2: "f4816f6919ba76fbcc5bea60576ea9ed1f0e183df54686c9caf0c48a8bcbfe5b",
    4: "12c1bf511a1b9c33c59a5ddb655c8094be9070f20a33bd6d0903197e8ebd6bed",
}
PREDICTION_DIGEST = {
    0: "16da408037d97da62831e519acd64a4739564b78dd2e1e36ff68fed872fbc93c",
    1: "1985371111eace0aee4f9bdf8fa1d7bcc1d61d678461c9dddb08f3c0ea42689a",
    2: "949cfaf396a796e0ae4b16855947e98ffa6a27d4a1ede8a10b7cdbae5198e709",
    4: "3736da74426a0633127aced925976e9bde7e1140ad32389c138242095b47ac3e",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def download(url, path, expected):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_name(path.name + ".partial")
        with urllib.request.urlopen(url, timeout=60) as source, temporary.open("wb") as dest:
            shutil.copyfileobj(source, dest, 1 << 20)
        require(digest(temporary) == expected, "download checksum mismatch: " + path.name)
        temporary.replace(path)
    require(digest(path) == expected, "cached source checksum mismatch: " + path.name)


def fetch_fold(root, fold):
    feature_name = f"v273-open-features-{fold}.tar"
    archive = root / feature_name
    download(f"https://github.com/Andriamarosoa/note/releases/download/{RELEASE}/{feature_name}",
             archive, FEATURE_TAR_DIGEST[fold])
    destination = root / "features" / f"fold{fold}"
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as source:
        source.extractall(destination, filter="data")
    base = (f"https://raw.githubusercontent.com/Andriamarosoa/note/{COMMIT}/"
            f"analysis/evidence/v273-open-k0-k6/series5/models/fold{fold}/")
    model_dir = root / "original-s5" / f"fold{fold}"
    download(base + "model.joblib", model_dir / "model.joblib", MODEL_DIGEST[fold])
    download(base + "predictions.npz", model_dir / "predictions.npz", PREDICTION_DIGEST[fold])


def verify(root, output):
    require(not output.exists(), "refuse to overwrite existing evidence output")
    designs, identities, fold_ids, states, references = [], [], [], [], []
    for fold in FOLDS:
        archive = root / f"v273-open-features-{fold}.tar"
        require(digest(archive) == FEATURE_TAR_DIGEST[fold], "feature archive checksum mismatch")
        feature_path = root / "features" / f"fold{fold}" / f"features-fold-{fold}.npz"
        # Check the local extraction against the immutable original archive.
        with tarfile.open(archive) as source:
            entries = [m for m in source.getmembers() if Path(m.name).name == feature_path.name]
            require(len(entries) == 1, "ambiguous original acoustic archive")
            stream = source.extractfile(entries[0])
            require(stream is not None, "missing original acoustic member")
            with stream:
                require(hashlib.file_digest(stream, "sha256").hexdigest() == digest(feature_path),
                        "extracted acoustic input differs from verified archive")
        with np.load(feature_path, allow_pickle=False) as source:
            ids = source["global_index"]
            require(np.all(source["fold"] == fold), "source fold identity mismatch")
            # Explicit allowlist: no k, truth, y, or MIDI arrays are accessed.
            design = build_owned527(source["summary"], source["geometry"],
                                    source["sequence"], source["baseline"])
        directory = root / "original-s5" / f"fold{fold}"
        require(digest(directory / "model.joblib") == MODEL_DIGEST[fold], "original model checksum")
        require(digest(directory / "predictions.npz") == PREDICTION_DIGEST[fold], "original predictions checksum")
        state = joblib.load(directory / "model.joblib")
        require(state["held_fold"] == fold and state["designs_sha256"] == ORIGINAL_DESIGN_DIGEST,
                "checkpoint training provenance mismatch")
        require(np.array_equal(ids, state["test_ids"]), "checkpoint held-event identity mismatch")
        with np.load(directory / "predictions.npz", allow_pickle=False) as source:
            require(np.array_equal(ids, source["global_index"]), "saved prediction identity mismatch")
            references.append((source["raw"], source["prior_corrected"]))
        states.append(state); designs.append(design); identities.append(ids)
        fold_ids.append(np.full(len(ids), fold, dtype=np.int8))
        print(json.dumps({"constructed_fold": fold, "rows": len(ids)}), flush=True)
    design = np.concatenate(designs); ids = np.concatenate(identities); folds = np.concatenate(fold_ids)
    require(design.shape == (59309, 527) and len(np.unique(ids)) == 59309, "full historical coverage mismatch")
    reports = []
    for index, fold in enumerate(FOLDS):
        state = states[index]; model = state["model"]
        mask = folds != fold
        require(np.array_equal(ids[mask], state["fit_ids"]), "training identities mismatch")
        require(not model.do_early_stopping_ and mask.sum() < model._bin_mapper.subsample,
                "unsupported original quantile sampling")
        mismatches = []; bins = model._bin_mapper.bin_thresholds_
        for column in range(527):
            reconstructed = _find_binning_thresholds(design[mask, column].astype(np.float64), model.max_bins)
            if not np.array_equal(reconstructed, bins[column]):
                mismatches.append(column)
        raw = model.predict_proba(designs[index])
        prior = raw / np.asarray(state["class_weights"])[None, :]
        prior /= prior.sum(axis=1, keepdims=True)
        expected_raw, expected_prior = references[index]
        error = np.max(abs(raw - expected_raw), axis=1)
        record = {
            "fold": fold, "rows": len(raw), "train_rows": int(mask.sum()),
            "bin_columns_checked": 527, "bin_thresholds_checked": sum(map(len, bins)),
            "bin_column_mismatches": mismatches,
            "raw_probability_max_abs": float(error.max()),
            "bit_exact_probability_rows": int((error == 0).sum()),
            "predicted_K_mismatches": int((raw.argmax(1) != expected_raw.argmax(1)).sum()),
            "prior_probability_max_abs": float(abs(prior - expected_prior).max()),
        }
        reports.append(record); print(json.dumps(record), flush=True)
    passed = all(not r["bin_column_mismatches"] and r["raw_probability_max_abs"] == 0
                 and r["prior_probability_max_abs"] == 0 for r in reports)
    report = {
        "status": "EXACT_HISTORICAL_S5_RECONSTRUCTION" if passed else "REPLAY_MISMATCH",
        "original_checkpoint_commit": COMMIT, "feature_release": RELEASE,
        "rows": len(ids), "columns": 527, "original_bin_columns_checked": 4 * 527,
        "original_bin_thresholds_checked": sum(r["bin_thresholds_checked"] for r in reports),
        "folds": reports, "full_historical_probability_equivalence": passed,
        "reconstructed_float32_matrix_sha256": hashlib.sha256(design.tobytes()).hexdigest(),
        "historical_designs_sha256": ORIGINAL_DESIGN_DIGEST,
        "historical_design_hash_serialization_known": False,
        "historical_matrix_byte_identity_proven": False,
        "hash_note": "The original designs_sha256 serialization is unavailable. Raw reconstructed matrix SHA is reported separately; no original-file hash identity is claimed.",
        "original_source_code_recovered": False, "truth_labels_read": False,
        "models_retrained": False, "independent_S80_accuracy_measured": False, "S80_promoted": False,
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                     "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
    }
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "report.md").write_text(
        "# S5 original numerical reconstruction\n\n"
        f"Status: **{report['status']}**. Original S80 remains unpromoted.\n\n"
        "| Fold | Events | Raw posterior max error | Prior posterior max error | Bin columns differing |\n"
        "|---|---:|---:|---:|---:|\n" +
        "".join(f"| {r['fold']} | {r['rows']} | {r['raw_probability_max_abs']} | {r['prior_probability_max_abs']} | {len(r['bin_column_mismatches'])} |\n" for r in reports) +
        "\nNo labels read, no retraining, no independent Exact-K accuracy measured. "
        "Original source text and original design-file serialization remain unavailable; "
        "the recovered runtime is verified by all original bin signatures and event probabilities.\n")
    require(passed, "original S5 replay differs; see output report")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    if args.fetch:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda fold: fetch_fold(args.cache_root, fold), FOLDS))
    verify(args.cache_root, args.output)
