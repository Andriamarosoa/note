"""Exploratory outer-fold global measurement of fixed symmetric H2 TTA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.train_v273_group_gate_ab import build_model, metrics
from scripts.audit_v273_internal_b_like_boundary_corrector import nested_base, NEURONS, SEED
from scripts.v273_window_experiment import load_bundle, require
from scripts.audit_v273_semitone_disappearance_fold import shifted_map, model_probability

STEPS = (-2, -1, 0, 1, 2)


def load_npz(path: Path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def build_robust(uniform_weights: Path, freeze_weights: Path):
    uniform = build_model("learned_gate", SEED)
    uniform.load_weights(uniform_weights)
    robust = build_model("learned_gate", SEED)
    robust.load_weights(freeze_weights)

    bu = nested_base(uniform)
    br = nested_base(robust)
    ku, bu_bias = [np.asarray(x).copy() for x in bu.get_layer("candidate_hidden1").get_weights()]
    kr, br_bias = [np.asarray(x).copy() for x in br.get_layer("candidate_hidden1").get_weights()]
    ids = np.asarray(NEURONS, np.int64)
    kr[:, ids] = ku[:, ids]
    br_bias[ids] = bu_bias[ids]
    br.get_layer("candidate_hidden1").set_weights([kr, br_bias])
    return robust


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--uniform-weights", type=Path, required=True)
    p.add_argument("--freeze-weights", type=Path, required=True)
    p.add_argument("--outer-predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    require(not a.output.exists(), "refusing overwrite")
    a.output.mkdir(parents=True)

    cache, parts, _ = load_bundle(a.bundle, a.config)
    outer = np.asarray(parts["outer"], np.int64)
    require(len(outer) == 15279, "outer size drift")
    y_all = np.minimum(np.asarray(cache["exact"], np.int32), 6)
    y_outer = y_all[outer]

    frozen = load_npz(a.outer_predictions)
    require("B_low_mask" in frozen, "outer predictions lack B_low mask")
    np.testing.assert_array_equal(frozen["global_index"], outer)
    np.testing.assert_array_equal(frozen["k"], y_outer)

    base_pred = np.asarray(frozen["base_predicted"], np.int32)
    base_prob = np.asarray(frozen["base_probability"], np.float64)
    b_low = np.asarray(frozen["B_low_mask"], np.uint8).astype(bool)
    require(base_prob.shape == (len(outer), 7), "base probability shape drift")
    require(np.allclose(base_prob.sum(axis=1), 1.0, atol=1e-5), "bad base probabilities")

    target_pos = np.flatnonzero(b_low & (base_pred == 3))
    target_ids = outer[target_pos]
    require(len(target_ids) > 0, "empty B_low base-K3 target")

    # Reconstruct exactly the robust outer model used for the frozen base.
    model = build_robust(a.uniform_weights, a.freeze_weights)

    wanted = set(np.asarray(cache["members"][target_ids]).astype(str))
    tracks = {
        t.annotation_member: t
        for t in index_guitarset(a.dataset)
        if t.annotation_member in wanted
    }
    require(set(tracks) == wanted, "missing GuitarSet recordings")

    audio = {}
    for member, track in tracks.items():
        wav = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        audio[member] = np.asarray(wav.samples, np.float32) / 32768.0

    probs = np.empty((len(target_ids), len(STEPS), 7), np.float64)
    # Step zero is frozen exactly from the reference outer evaluation.
    probs[:, STEPS.index(0)] = base_prob[target_pos]

    for si, step in enumerate(STEPS):
        if step == 0:
            continue
        maps = np.empty((len(target_ids), 31, 64, 3), np.float32)
        for j, gid in enumerate(target_ids):
            member = str(cache["members"][gid])
            start = int(cache["cluster_start_samples"][gid])
            maps[j] = shifted_map(audio[member], start, step)
        probs[:, si] = model_probability(model, cache, target_ids, maps)
        print(json.dumps({
            "step": step,
            "rows": int(len(target_ids)),
            "k2": int(np.sum(probs[:, si].argmax(axis=1) == 2)),
            "k3": int(np.sum(probs[:, si].argmax(axis=1) == 3)),
        }), flush=True)

    margins = probs[:, :, 2] - probs[:, :, 3]
    mean_margin = margins.mean(axis=1)
    action = mean_margin > 0.0

    corrected = base_pred.copy()
    corrected[target_pos[action]] = 2

    # Strict invariant: only targeted B_low base-K3 rows may change, and only 3->2.
    changed = corrected != base_pred
    require(np.all(np.isin(np.flatnonzero(changed), target_pos)), "changed non-target rows")
    require(np.all(base_pred[changed] == 3) and np.all(corrected[changed] == 2),
            "non 3->2 change detected")

    base_m = metrics(y_outer, base_pred)
    corr_m = metrics(y_outer, corrected)

    changed_y = y_outer[changed]
    # Because the action is always 3->2:
    # - true K2 changes wrong->correct (correction),
    # - true K3 changes correct->wrong (regression),
    # - every other true K stays wrong->wrong (neutral action).
    corrections = int(np.sum(changed_y == 2))
    regressions = int(np.sum(changed_y == 3))
    neutral_other_k = int(np.sum(~np.isin(changed_y, (2, 3))))
    net = corrections - regressions

    by_k = {
        str(k): int(np.sum((corrected == y_outer) & (y_outer == k))
                    - np.sum((base_pred == y_outer) & (y_outer == k)))
        for k in range(7)
    }

    report = {
        "status": "completed",
        "experiment": "v273_outer_symmetric_h2_global",
        "outer_fold": 3,
        "outer_fold_previously_exposed": True,
        "automatic_promotion": False,
        "reference": {
            "model": "freeze_local_combo + candidate_hidden1 [42,52,61,64]",
            "rows": int(len(outer)),
            "base_metrics": base_m,
        },
        "h2": {
            "steps": list(STEPS),
            "rule": "mean(P(K2)-P(K3)) > 0 => 3->2",
            "B_low_baseK3_target_rows": int(len(target_ids)),
            "actions": int(action.sum()),
            "corrections": corrections,
            "regressions": regressions,
            "net_exact": net,
            "neutral_other_k_actions": neutral_other_k,
            "action_success_rate": float(corrections / max(1, corrections + regressions)),
            "corrected_metrics": corr_m,
            "by_true_k_net": by_k,
            "mean_margin": {
                "all_targets_mean": float(mean_margin.mean()),
                "all_targets_median": float(np.median(mean_margin)),
                "acted_mean": float(mean_margin[action].mean()) if action.any() else None,
            },
        },
        "delta": {
            "exact_global_points": float(100.0 * (corr_m["exact"] - base_m["exact"])),
            "poly_exact_points": float(100.0 * (corr_m["poly_exact"] - base_m["poly_exact"])),
        },
        "limitations": [
            "Outer fold 3 was already exposed in prior exploratory work.",
            "Only spectral maps are pitch-shifted; temporal candidate inputs remain frozen.",
        ],
    }

    (a.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        a.output / "predictions.npz",
        global_index=outer,
        k=y_outer,
        base_predicted=base_pred,
        corrected_predicted=corrected,
        B_low_mask=b_low.astype(np.uint8),
        h2_target_mask=(b_low & (base_pred == 3)).astype(np.uint8),
        h2_action_mask=np.isin(np.arange(len(outer)), target_pos[action]).astype(np.uint8),
        target_global_index=target_ids,
        target_probability=probs.astype(np.float32),
        target_mean_margin=mean_margin.astype(np.float32),
    )

    lines = [
        "# Exploratory outer global measurement — symmetric H2",
        "",
        "| Measure | Base robust | + symmetric H2 | Delta |",
        "|---|---:|---:|---:|",
        f"| Exact-K global | {100*base_m['exact']:.3f}% | {100*corr_m['exact']:.3f}% | {100*(corr_m['exact']-base_m['exact']):+.3f} pt |",
        f"| Poly Exact-K | {100*base_m['poly_exact']:.3f}% | {100*corr_m['poly_exact']:.3f}% | {100*(corr_m['poly_exact']-base_m['poly_exact']):+.3f} pt |",
        "",
        f"B_low + base K3 targets: **{len(target_ids)}**.",
        f"H2 actions: **{int(action.sum())}**.",
        f"Corrections/regressions: **{corrections}/{regressions}**.",
        f"Neutral actions on other true-K: **{neutral_other_k}**.",
        f"Net Exact-K: **{net:+d}**.",
        "By-K net: " + ", ".join(f"K{k} {by_k[str(k)]:+d}" for k in range(7)) + ".",
        "",
        "Outer fold 3 was already exposed historically; this is exploratory.",
        "No automatic promotion.",
    ]
    (a.output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
