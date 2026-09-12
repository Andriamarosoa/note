"""Read-only, post-hoc diagnosis of the frozen V28.0-F development result.

Requires the seven named GitHub artifact ZIPs below and NumPy. No TensorFlow,
audio, feature cache, new inference, fitting, or decoder search is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_v280_nested_outer as f

RUN_ID = 34462604665
HEAD_SHA = "50e615bfb6cbce6ac5bbc57843ef3d1378e1bf79"
ARTIFACTS = {
    "v280-f-outer-comparison": (10283052838, "67fbaf55e2e0181540a3e6fe694627279fd373594134dd72404ecf5e2ab26fc5"),
    "v280-f-preparation-audit": (10146299106, "30d7808b21ae7ca5e47994193c030e8bd52761ed643ba606c8f6a54f776e2bea"),
    "v280-f-fold-0": (10146433555, "6955bfa5253b2192aac23599463c956b7e2457684113073f32395834b1eb44f2"),
    "v280-f-fold-1": (10175019251, "4dbf741b6adc6d7902e23a2daf34f2cf1f0d0033eb41807a943e18baeeadc2be"),
    "v280-f-fold-2": (10186641717, "f9d78fa57079432f72fb414dee148111c7b7806e6ffc5c19f40e354b8fc322d4"),
    "v280-f-fold-3": (10265589564, "98fa6de75a9aa01773ed842af2c8260b35b651d895cf09dae10e97f723ef718b"),
    "v280-f-fold-4": (10282623236, "feaf8904bc85f2f626954ec6695b3ceade1ef208f0b09ade854713c1dc6287fb"),
}
CODE_SHA256 = {
    "scripts/train_v280_nested_outer.py": "f745d8c046d55bf099ed5774d3a4c5b64d0ea86d9dd01fdb233f2da506ca6c0a",
    "scripts/train_v280_internal_ablation.py": "2cb473d142829c83372bc43812c0cfdf8da1407a0a38cd08e42e4eefdc00aa94",
    "scripts/train_v280_harmonic_count.py": "963390d9779e38cbe2a43dde567274f2b8343e1e70e508a81bc4a76c287fdf55",
}


class AuditError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_archive(path, expected_sha):
    raw = Path(path).read_bytes()
    require(sha(raw) == expected_sha, f"archive SHA-256 mismatch: {Path(path).name}")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names = z.namelist()
        require(len(names) == len(set(names)), "duplicate ZIP entries")
        require(all(Path(name).name == name for name in names), "unexpected ZIP paths")
        return {name: z.read(name) for name in names}


def read_npz(raw):
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def verify_files(payload, records):
    for name, record in records.items():
        require(name in payload and sha(payload[name]) == record["sha256"], f"file mismatch: {name}")
        require(len(payload[name]) == record["bytes"], f"file length mismatch: {name}")


def count_stats(k, pred):
    k, pred = np.asarray(k), np.asarray(pred)
    n = len(k)
    correct = int(np.sum(pred == k))
    under = int(np.sum(pred < k))
    over = int(np.sum(pred > k))
    require(correct + under + over == n, "count outcomes do not partition rows")
    return {"rows": n, "correct": correct, "undercount": under, "overcount": over,
            "exact_percent": 100.0 * correct / n if n else None,
            "under_percent": 100.0 * under / n if n else None}


def paired_stats(k, pred, reference):
    new, old = pred == k, reference == k
    both, lost = int(np.sum(new & old)), int(np.sum(~new & old))
    gained, neither = int(np.sum(new & ~old)), int(np.sum(~new & ~old))
    require(both + lost + gained + neither == len(k), "paired outcomes do not partition rows")
    return {"both_correct": both, "v273_only_correct": lost, "v280_only_correct": gained,
            "neither_correct": neither, "net_correct_rows": gained - lost}


def comparison(k, pred, reference):
    return {"v273": count_stats(k, reference), "v280": count_stats(k, pred),
            "paired": paired_stats(k, pred, reference)}


def probability_diagnostics(k, p):
    p = np.asarray(p, dtype=np.float64)
    pred = p.argmax(axis=1)
    nll = -np.log(np.maximum(p[np.arange(len(k)), k], 1e-7))
    true_p = p[np.arange(len(k)), k]
    # Strict competition rank: ties are stated, not silently counted as top-2 wins.
    ranks = 1 + np.sum(p > true_p[:, None], axis=1)
    poly, under = k >= 2, (k >= 2) & (pred < k)
    result = {"poly_fraction_of_unweighted_oof_nll": float(nll[poly].sum() / nll.sum())}
    for label, mask in (("polyphonic", poly), ("polyphonic_undercount", under)):
        result[label] = {
            "rows": int(mask.sum()),
            "true_class_competition_rank_histogram_1_to_7": np.bincount(ranks[mask], minlength=8)[1:].tolist(),
            "rows_with_true_probability_ties": int(np.sum(np.sum(p[mask] == true_p[mask, None], axis=1) > 1)),
            "mean_max_probability": float(p[mask].max(axis=1).mean()),
            "mean_true_probability": float(true_p[mask].mean()),
            "wrong_with_max_probability_at_least_0p8": int(np.sum(mask & (pred != k) & (p.max(axis=1) >= .8))),
        }
    return result


def verify_inputs(input_dir):
    for name, digest in CODE_SHA256.items():
        require(sha((ROOT / name).read_bytes()) == digest, f"scientific implementation changed: {name}")
    archives = {name: read_archive(input_dir / (name + ".zip"), record[1]) for name, record in ARTIFACTS.items()}
    prepared = archives["v280-f-preparation-audit"]
    manifest = json.loads(prepared["manifest.json"])
    prepared_sha = sha(prepared["manifest.json"])
    require(prepared_sha == f.RECOVERY_SOURCE["prepared_manifest_sha256"], "prepared manifest changed")
    verify_files(prepared, {name: record for name, record in manifest["files"].items() if name != "features.npy"})
    arrays = read_npz(prepared["labels.npz"])
    summary = archives["v280-f-outer-comparison"]
    report = json.loads(summary["report.json"])
    require(report["status"] == "complete" and report["contract"] == f.CONTRACT, "unexpected final contract")
    require(report["source_head_sha"] == HEAD_SHA and report["source_run_id"] == str(RUN_ID), "unexpected final producer")
    require(report["prepared_manifest_sha256"] == prepared_sha, "summary data differs")
    require(sha(summary["predictions.npz"]) == report["predictions_sha256"], "summary predictions changed")
    predictions = read_npz(summary["predictions.npz"])
    parts, states = [], []
    for fold in range(5):
        payload = archives[f"v280-f-fold-{fold}"]
        outer = json.loads(payload["report.json"])
        require(outer["source_head_sha"] == HEAD_SHA and outer["source_run_id"] == str(RUN_ID), "unexpected fold producer")
        require(outer["fold"] == fold and outer["status"] == "complete" and outer["outer_inference_passes"] == 1, "invalid outer report")
        verify_files(payload, outer["files"])
        probe, refit = (json.loads(payload[name + "-state.json"]) for name in ("probe", "refit"))
        for state, phase in ((probe, "probe"), (refit, "refit")):
            require(state["contract"] == f.CONTRACT and state["sources"] == f.SOURCES, "training contract/source changed")
            require(state["status"] == "complete" and state["fold"] == fold and state["phase"] == phase, "wrong training state")
            require(state["prepared_manifest_sha256"] == prepared_sha, "training data differs")
            require(state["chunk"] == 2 and state["epochs_completed"] == state["epoch_budget"], "incomplete training")
            source = f.RECOVERY_SOURCE if fold == 0 else {"head_sha": HEAD_SHA, "run_id": RUN_ID}
            require(state["source_head_sha"] == source["head_sha"] and state["source_run_id"] == str(source["run_id"]), "wrong state producer")
            if fold == 0:
                require(sha(payload[phase + "-state.json"]) == source["state_sha256"][phase], "fold-0 recovery changed")
            f.validate_training_history(state, arrays)
        best = f.verify_probe_predictions(probe, read_npz(payload["probe-validation.npz"]), arrays)
        selection = {"epoch": best["epoch"], "validation": best["validation"],
                     "probe_state_sha256": sha(payload["probe-state.json"])}
        require(refit["selection"] == outer["selection"] == json.loads(payload["selection.json"]) == selection, "selected epoch differs")
        require(refit["epoch_budget"] == best["epoch"], "incorrect refit budget")
        require(probe["initial_shared_weights_sha256"] == refit["initial_shared_weights_sha256"], "initialization mismatch")
        for target, state, original in (("outer.weights.h5", refit, "last.weights.h5"),
                                         ("inner-best.weights.h5", probe, "best.weights.h5"),
                                         ("probe-validation.npz", probe, "validation-predictions.npz")):
            require(sha(payload[target]) == state["files"][original]["sha256"], f"checkpoint payload changed: {target}")
        part = read_npz(payload["predictions.npz"])
        require(f.same_count_metrics(f.e.count_metrics(part["k"], part["probability"]), outer["outer"]), "outer metrics differ")
        require(f.same_count_metrics(outer["outer"], report["per_fold"][str(fold)]["v280"]), "fold summary differs")
        parts.append(part)
        states.append((probe, refit, best))
    merged = f.validate_oof_parts(parts, arrays)
    for name, value in merged.items():
        require(np.array_equal(value, predictions[name]), f"summary and fold arrays differ: {name}")
    k, p = predictions["k"], predictions["probability"]
    require(f.same_count_metrics(f.e.count_metrics(k, p), report["v280"]), "V28 score mismatch")
    require(f.discrete_metrics(k, predictions["reference_prediction"]) == report["reference"], "V27.3 score mismatch")
    require(np.array_equal(arrays["target_poibin_cardinality"], k), "auxiliary count targets changed")
    require(np.all(arrays["weight_cardinality"] == 1), "main loss is not uniformly weighted")
    return predictions, arrays, manifest, read_npz(prepared["evaluation.npz"]), report, states


def diagnose(input_dir):
    a, labels, manifest, evaluation, original, states = verify_inputs(input_dir)
    k, p, reference = a["k"], a["probability"], a["reference_prediction"]
    pred, poly = p.argmax(axis=1), k >= 2
    by_k = {str(i): comparison(k[k == i], pred[k == i], reference[k == i]) for i in range(7)}
    learning, partitions = {}, {}
    for fold, (probe, refit, best) in enumerate(states):
        curve = []
        for h in probe["history"]:
            v = h["validation"]
            curve.append({"epoch": h["epoch"], "updates": h["updates"],
                          "train_count_loss": h["train_losses"]["cardinality_loss"],
                          "validation_nll": v["nll"], "validation_poly_nll": v["poly_nll"],
                          "validation_poly_exact_percent": 100 * v["poly_exact_k"],
                          "validation_poly_under_percent": 100 * sum(v["by_true_k"][str(i)]["undercount"] for i in range(2, 7)) / v["poly_rows"]})
        learning[str(fold)] = {
            "selected_epoch": best["epoch"], "probe_updates": probe["optimizer_updates"],
            "refit_epochs": refit["epochs_completed"], "refit_updates": refit["optimizer_updates"],
            "selected_inner_poly_exact_percent": 100 * best["validation"]["poly_exact_k"],
            "outer_poly_exact_percent": 100 * original["per_fold"][str(fold)]["v280"]["poly_exact_k"],
            "probe_curve": curve,
            "refit_train_count_loss_by_epoch": [h["train_losses"]["cardinality_loss"] for h in refit["history"]],
        }
        partitions[str(fold)] = {}
        for role, rows in f.partitions(a["row_fold"], a["member"], fold).items():
            counts = np.bincount(k[rows], minlength=7)
            partitions[str(fold)][role] = {"rows": len(rows), "class_counts": counts.tolist(),
                "poly_percent": 100.0 * int(counts[2:].sum()) / len(rows)}
    masked = {}
    for head in f.e.LOSS_WEIGHTS:
        weights = labels["weight_" + head]
        require(np.all((weights == 0) | (weights == 1)), "unexpected supervision weight")
        mask = weights == 0
        masked[head] = {"masked_rows": int(mask.sum()), "masked_poly_rows": int((mask & poly).sum()),
            "masked_rows_by_k": np.bincount(k[mask], minlength=7).tolist(),
            "usable_poly": comparison(k[poly & ~mask], pred[poly & ~mask], reference[poly & ~mask])}
    clipped = np.zeros(len(k), dtype=bool)
    for track in manifest["per_track"]:
        rows = np.flatnonzero(a["member"] == track["member"])
        clipped[rows[np.asarray(track["features"]["end_of_stream_clipped_local_indices"], dtype=int)]] = True
    alternatives = {}
    for name, mask in (("end_of_stream_clipped", clipped),
                       ("candidate_shortage", evaluation["candidate_count"] < k),
                       ("frozen_ranking_shortage", (evaluation["top_samples"] >= 0).sum(axis=1) < k)):
        alternatives[name] = {"rows": int(mask.sum()), "polyphonic": comparison(k[mask & poly], pred[mask & poly], reference[mask & poly])}
    subgroups = {}
    for name, groups in (("mode", [str(m).removesuffix(".jams").split("_")[-1] for m in a["member"]]),
                         ("performer", [str(m).split("_")[0] for m in a["member"]])):
        groups = np.asarray(groups)
        subgroups[name] = {}
        for group in np.unique(groups):
            mask = poly & (groups == group)
            subgroups[name][str(group)] = comparison(k[mask], pred[mask], reference[mask])
    return {"status": "complete", "experiment": "v280_f_posthoc_undercount_diagnostic",
        "source_run_id": RUN_ID, "source_head_sha": HEAD_SHA,
        "artifacts": {name: {"id": record[0], "sha256": record[1]} for name, record in ARTIFACTS.items()},
        "scientific_code_sha256": CODE_SHA256,
        "verification": {"zip_and_payload_hashes": True, "oof_coverage_and_identity": True,
            "original_metrics_reproduced": True, "inner_predictions_reproduced": True,
            "nested_partitions_epoch_orders_and_update_budgets": True, "selected_weights_preserved": True,
            "feature_cache_loaded": False, "new_inference_passes": 0, "training_launched": False,
            "outer_threshold_search": False, "reference_promoted": False},
        "population": {"rows": len(k), "tracks": len(set(a["member"])), "poly_rows": int(poly.sum()),
            "poly_percent": 100.0 * float(poly.mean()), "class_counts": np.bincount(k, minlength=7).tolist()},
        "overall": comparison(k, pred, reference), "polyphonic": comparison(k[poly], pred[poly], reference[poly]),
        "by_true_k": by_k, "probability_diagnostics": probability_diagnostics(k, p),
        "poly_error_histogram": {str(error): int(np.sum(poly & (pred - k == error))) for error in range(-6, 7)},
        "supervision": masked, "other_hypotheses": alternatives,
        "subgroups": subgroups, "partitions": partitions, "learning": learning,
        "original_event_metrics": original["event_metrics"], "original_bootstrap": original["track_bootstrap"],
        "limitations": ["post-hoc development analysis, not independent validation",
            "inner-selected probe and fresh-refit outer scores use different models and compositions",
            "class frequencies and loss curves do not establish a causal explanation",
            "saved predictions contain only final cardinality, not the auxiliary heads or residual logits",
            "no training-set evaluation predictions or alternate-epoch outer predictions are available",
            "OOF NLL share is descriptive, not the contribution to training gradients"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = diagnose(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "polyphonic": result["polyphonic"],
                      "training_launched": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
