"""S128: diagnose original S5 versus reconstructed 527D *count bias*.

Compare pinned ORIGINAL S5 fold-0 raw K0..K6 posterior probabilities with
the S123 reconstructed log-activation 527D model output on aligned events.
Do not read truth/MIDI even though the old archive contains truth.
The fixed grid is a counterfactual diagnostic, NEVER a deployable
calibration; its scores are fidelity to old predictions, not accuracy.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

BETAS = (0.0, 0.04, 0.08, 0.12, 0.16, 0.24, 0.32, 0.48, 0.64)
K = np.arange(7, dtype=np.float64)

def require(condition, reason):
    if not condition:
        raise ValueError(reason)

def tilted(probabilities, beta):
    """Deterministic ordinal logit tilt, not training or feature repair."""
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) + beta * K
    logits -= logits.max(axis=1, keepdims=True)
    ex = np.exp(logits)
    return ex / ex.sum(axis=1, keepdims=True)

def assess(old_prob, replay_prob):
    require(old_prob.shape == replay_prob.shape and old_prob.ndim == 2
            and old_prob.shape[1] == 7, "bad K0..K6 matrix")
    require(np.isfinite(old_prob).all() and np.isfinite(replay_prob).all(),
            "nonfinite posterior")
    require(np.allclose(old_prob.sum(1), 1, atol=1e-5)
            and np.allclose(replay_prob.sum(1), 1, atol=1e-5),
            "posterior not normalized")
    original_K = old_prob.argmax(1)
    replay_K = replay_prob.argmax(1)
    wrong0 = (original_K != replay_K)
    n = len(original_K)
    baseline_mismatches = int(wrong0.sum())
    rows = []
    for beta in BETAS:
        candidate_prob = tilted(replay_prob, beta)
        candidate = candidate_prob.argmax(1)
        wrong = (candidate != original_K)
        fixes = int(np.sum(wrong0 & ~wrong))
        breaks = int(np.sum(~wrong0 & wrong))
        no_change_wrong = int(np.sum(wrong0 & wrong))
        changed = int((candidate != replay_K).sum())
        require(fixes + no_change_wrong == baseline_mismatches,
                "conservation of existing source/replay mismatches failed")
        require(baseline_mismatches + breaks - fixes == int(wrong.sum()),
                "new mismatch conservation failed")
        rows.append({
            "fixed_beta": beta,
            "original_prediction_agreement": int(n - wrong.sum()),
            "original_prediction_disagreements": int(wrong.sum()),
            "replay_to_candidate_changes": changed,
            "original_replay_mismatches_reconciled": fixes,
            "original_replay_agreements_broken": breaks,
            "net_fidelity_gain_not_accuracy": fixes - breaks,
            "candidate_below_original_K": int(np.sum(candidate < original_K)),
            "candidate_above_original_K": int(np.sum(candidate > original_K)),
            "original_posterior_MAE": float(np.abs(candidate_prob-old_prob).mean()),
            "original_posterior_max_abs_error": float(np.max(np.abs(candidate_prob-old_prob)))
        })
    require(rows[0]["original_prediction_disagreements"] == baseline_mismatches
            and rows[0]["replay_to_candidate_changes"] == 0, "beta 0 invariant")
    # "Best" is solely a retrospective fidelity diagnostic; DO NOT deploy it.
    max_fidelity = max(x["original_prediction_agreement"] for x in rows)
    return {
        "status": "S128_FIXED_ORDINAL_TILT_FIDELITY_DIAGNOSTIC_ONLY",
        "rows": n, "historical_original_vs_replay_disagreements": baseline_mismatches,
        "baseline_replay_below_original_K": int((replay_K < original_K).sum()),
        "baseline_replay_above_original_K": int((replay_K > original_K).sum()),
        "fixed_grid": rows,
        "retrospective_max_agreement_count": max_fidelity,
        "retrospective_max_agreement_betas": [
            x["fixed_beta"] for x in rows if x["original_prediction_agreement"] == max_fidelity
        ],
        "best_fidelity_is_not_production_rule": True,
        "any_bias_grid_chosen_from_original_historical_outputs": False,
        "midi_or_truth_accessed": False, "accuracy_evaluated": False,
        "original_527D_feature_mapping_restored": False,
        "S80_external_exact_K": None, "S80_promoted": False
    }

def run(original, original_report, s123, out):
    require(not out.exists(), "immutable output exists")
    manifest = json.loads(original_report.read_text())
    require(hashlib.sha256(original.read_bytes()).hexdigest()
            == manifest["prediction_sha256"], "original S5 archive checksum mismatch")
    with np.load(original, allow_pickle=False) as z:
        # Only these two named arrays may be loaded. Never access z["truth"].
        ids = z["global_index"].copy()
        original_prob = z["raw"].astype(np.float64)
    with np.load(s123, allow_pickle=False) as z:
        require({"global_index", "probabilities", "predicted_K"}.issubset(z.files),
                "S123 original fixed replay contract")
        replay_ids = z["global_index"].copy()
        replay_prob = z["probabilities"].astype(np.float64)
        replay_K = z["predicted_K"].copy()
    require(np.array_equal(ids, replay_ids) and len(set(ids.tolist())) == 15952,
            "historical event IDs not identical")
    require(original_prob.shape == replay_prob.shape == (15952, 7),
            "original/replay posterior shape mismatch")
    require(np.array_equal(replay_prob.argmax(1), replay_K),
            "replay probability/K contradiction")
    report = assess(original_prob, replay_prob)
    require(report["historical_original_vs_replay_disagreements"] == 1882,
            "upstream frozen S127 diagnosis changed")
    out.mkdir(parents=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report

def selftest():
    orig = np.array([[.02,.88,.1,0,0,0,0],
                     [.88,.1,.02,0,0,0,0],
                     [.05,.1,.8,.05,0,0,0]], dtype=float)
    replica = np.array([[.55,.4,.05,0,0,0,0],
                        [.88,.1,.02,0,0,0,0],
                        [.05,.55,.38,.02,0,0,0]], dtype=float)
    a = assess(orig, replica)
    assert a["rows"] == 3 and a["historical_original_vs_replay_disagreements"] == 2
    assert a["fixed_grid"][0]["original_prediction_agreement"] == 1
    assert a["fixed_grid"][0]["original_replay_agreements_broken"] == 0
    assert a["accuracy_evaluated"] is False and a["S80_promoted"] is False
    assert np.allclose(tilted(replica, 0), replica)
    print("S128 PASS: fixed bias grid, mismatch accounting, no accuracy, no promotion")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for field in ("original", "original-report", "s123", "out"):
        ap.add_argument("--"+field, type=Path)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        selftest()
    else:
        require(all(v is not None for v in (args.original,args.original_report,
                                            args.s123,args.out)), "missing immutable inputs")
        run(args.original,args.original_report,args.s123,args.out)
