"""Compare Librosa x4 (+24 st) disappearance with Rubber Band x4 on the same frozen K2/K3 cohort.

Source Librosa trajectories come from the completed semitone-disappearance audit.
For each held-out internal fold, all rows start at predicted K3 at +0. We then:
- read Librosa's +24-semitone prediction,
- recompute the +24-semitone spectral map with Rubber Band,
- compare whether the row becomes predicted K2.

This directly tests whether the previously observed K3->K2 disappearance at x4
persists under a higher-quality pitch-shift engine.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from causal_note.guitarset import index_guitarset
from scripts.audit_v273_semitone_disappearance_fold import (
    FOLDS,
    model_probability,
    robust_model,
)
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.v273_rubberband_pitch import rubberband_shifted_map
from scripts.v273_window_experiment import load_bundle, require

X4_STEP = 24


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def accounting(y, take):
    y = np.asarray(y, np.int32)
    take = np.asarray(take, bool)
    corr = int(np.sum(take & (y == 2)))
    reg = int(np.sum(take & (y == 3)))
    return {
        "actions": int(take.sum()),
        "corrections": corr,
        "regressions": reg,
        "net": corr - reg,
    }


def overlap_stats(y, lib_take, rub_take):
    y = np.asarray(y, np.int32)
    lib_take = np.asarray(lib_take, bool)
    rub_take = np.asarray(rub_take, bool)

    both = lib_take & rub_take
    lib_only = lib_take & ~rub_take
    rub_only = rub_take & ~lib_take
    neither = ~lib_take & ~rub_take

    def cell(mask):
        return {
            "rows": int(mask.sum()),
            "K2": int(np.sum(mask & (y == 2))),
            "K3": int(np.sum(mask & (y == 3))),
        }

    denom = int(lib_take.sum())
    return {
        "both_k2_at_x4": cell(both),
        "librosa_only_k2_at_x4": cell(lib_only),
        "rubberband_only_k2_at_x4": cell(rub_only),
        "neither_k2_at_x4": cell(neither),
        "librosa_disappearances": denom,
        "persist_with_rubberband": int(both.sum()),
        "persistence_rate": float(both.sum() / denom) if denom else None,
        "librosa_disappearances_lost_with_rubberband": int(lib_only.sum()),
        "new_rubberband_disappearances": int(rub_only.sum()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "dataset", "bundle", "config", "fold-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--val-fold", type=int, required=True)
    a = p.parse_args()

    require(a.val_fold in FOLDS and a.val_fold != 3, "bad fold")
    require(not a.output.exists(), "refusing overwrite")

    src = load_npz(a.source)
    steps = np.asarray(src["steps"], np.int32)
    require(np.array_equal(steps, np.arange(25)), "source steps drift")
    ids = np.asarray(src["row_id"], np.int64)
    y = np.asarray(src["true_k"], np.int32)
    lib_pred = np.asarray(src["predicted"], np.int32)
    lib_prob = np.asarray(src["probability"], np.float64)
    require(lib_pred.shape == (len(ids), 25), "Librosa prediction shape drift")
    require(lib_prob.shape == (len(ids), 25, 7), "Librosa probability shape drift")
    require(np.all(np.isin(y, (2, 3))), "cohort must be K2/K3")
    require(np.all(lib_pred[:, 0] == 3), "source cohort must start at K3")

    cache, _, _ = load_bundle(a.bundle, a.config)
    model = robust_model(a.fold_root, a.val_fold)

    wanted = set(np.asarray(cache["members"][ids]).astype(str))
    tracks = {t.annotation_member: t for t in index_guitarset(a.dataset) if t.annotation_member in wanted}
    require(set(tracks) == wanted, "missing tracks")

    audio = {}
    for member, track in tracks.items():
        wav = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        audio[member] = np.asarray(wav.samples, np.float32) / 32768.0

    maps = np.empty((len(ids), 31, 64, 3), np.float32)
    for j, rid in enumerate(ids):
        member = str(cache["members"][rid])
        start = int(cache["cluster_start_samples"][rid])
        maps[j] = rubberband_shifted_map(audio[member], start, X4_STEP)

    rub_prob = model_probability(model, cache, ids, maps)
    rub_pred = rub_prob.argmax(1).astype(np.int32)

    lib_x4 = lib_pred[:, X4_STEP]
    lib_take = lib_x4 == 2
    rub_take = rub_pred == 2

    report = {
        "status": "completed",
        "experiment": "v273_rubberband_x4_disappearance_fold",
        "validation_fold": int(a.val_fold),
        "rows": int(len(ids)),
        "K2_rows": int(np.sum(y == 2)),
        "K3_rows": int(np.sum(y == 3)),
        "starting_prediction": "K3 for all rows at x1/+0",
        "comparison_step": X4_STEP,
        "comparison_factor": 4.0,
        "librosa": {
            "pitch_shift": "+24 semitones, librosa source trajectory",
            "prediction_counts": {str(k): int(np.sum(lib_x4 == k)) for k in range(7)},
            "k3_to_k2_accounting": accounting(y, lib_take),
        },
        "rubberband": {
            "pitch_shift": "+24 semitones, pyrubberband/Rubber Band",
            "prediction_counts": {str(k): int(np.sum(rub_pred == k)) for k in range(7)},
            "k3_to_k2_accounting": accounting(y, rub_take),
        },
        "overlap": overlap_stats(y, lib_take, rub_take),
        "outer_fold_3_used": False,
        "automatic_promotion": False,
        "interpretation_guard": "A K2 prediction at x4 is treated only as a disappearance signal, not as a validated correction rule.",
    }

    a.output.mkdir(parents=True)
    (a.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        a.output / "comparison.npz",
        row_id=ids,
        true_k=y.astype(np.int8),
        librosa_x4_pred=lib_x4.astype(np.int8),
        rubberband_x4_pred=rub_pred.astype(np.int8),
        librosa_x4_probability=lib_prob[:, X4_STEP].astype(np.float32),
        rubberband_x4_probability=rub_prob.astype(np.float32),
        librosa_k2_action=lib_take.astype(np.uint8),
        rubberband_k2_action=rub_take.astype(np.uint8),
    )

    la = report["librosa"]["k3_to_k2_accounting"]
    ra = report["rubberband"]["k3_to_k2_accounting"]
    ov = report["overlap"]

    def pct(v):
        return "n/a" if v is None else f"{100*v:.1f}%"

    lines = [
        f"# Rubber Band x4 disappearance — fold {a.val_fold}",
        "",
        f"Rows: **{len(ids)}** (true K2={report['K2_rows']}, true K3={report['K3_rows']}).",
        "",
        "| engine | K3→K2 signals | true K2 corrections | true K3 regressions | net |",
        "|---|---:|---:|---:|---:|",
        f"| Librosa x4 | {la['actions']} | {la['corrections']} | {la['regressions']} | {la['net']:+d} |",
        f"| Rubber Band x4 | {ra['actions']} | {ra['corrections']} | {ra['regressions']} | {ra['net']:+d} |",
        "",
        f"Librosa x4 disappearances retained by Rubber Band x4: **{ov['persist_with_rubberband']}/{ov['librosa_disappearances']} = {pct(ov['persistence_rate'])}**.",
        f"Librosa-only disappearances: **{ov['librosa_disappearances_lost_with_rubberband']}**.",
        f"New Rubber-Band-only disappearances: **{ov['new_rubberband_disappearances']}**.",
        "",
        "No correction is promoted from this fold-level diagnostic.",
    ]
    (a.output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
