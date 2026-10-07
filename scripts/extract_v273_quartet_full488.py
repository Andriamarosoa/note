"""Extract the fixed quartet on the entire frozen 488-row K2/K3 cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.audit_v273_attack_novelty import raw_powers
from scripts.audit_v273_selected_f0_attack_exclusive import row_features as selected_f0_features
from scripts.v273_residual_audit import FOLDS, require

FEATURES = (
    "nov_pre_norm_range",
    "nov_onset_contrast_raw_range",
    "nov_post1_norm_median",
    "nov_positive_retained_fraction_range",
)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def category(lib, rub):
    if lib and rub:
        return "both"
    if lib:
        return "librosa_only"
    if rub:
        return "rubberband_only"
    return "neither"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cases", "dataset", "comparison", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--fold", type=int, required=True)
    a = p.parse_args()

    require(a.fold in FOLDS and a.fold != 3, "bad fold")
    require(not a.output.exists(), "refusing overwrite")

    cases = [json.loads(x) for x in a.cases.read_text().splitlines() if x.strip()]
    cmap = {int(r["row_id"]): r for r in cases}
    require(len(cmap) == 488, "case cohort drift")

    comp = load_npz(a.comparison)
    ids = np.asarray(comp["row_id"], np.int64)
    y = np.asarray(comp["true_k"], np.int32)
    lib = np.asarray(comp["librosa_k2_action"], bool)
    rub = np.asarray(comp["rubberband_k2_action"], bool)
    require(len(ids) == len(y) == len(lib) == len(rub), "comparison shape drift")

    rows = [cmap[int(rid)] for rid in ids]
    require(all(int(r["fold"]) == a.fold for r in rows), "fold identity drift")
    require(np.array_equal(np.asarray([r["true_K"] for r in rows], int), y), "truth drift")

    wanted = {r["recording_id"] for r in rows}
    tracks = {t.annotation_member: t for t in index_guitarset(a.dataset) if t.annotation_member in wanted}
    require(set(tracks) == wanted, "audio coverage incomplete")
    audio = {}
    for member, track in tracks.items():
        wav = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        audio[member] = np.asarray(wav.samples, np.float64) / 32768.0

    out = []
    for i, row in enumerate(rows):
        samples = audio[row["recording_id"]]
        freq, powers, _ = raw_powers(samples, int(row["start_sample"]))
        allf = selected_f0_features(row, freq, powers)
        feat = {name: float(allf[name]) for name in FEATURES}
        require(np.isfinite(np.asarray(list(feat.values()))).all(), "non-finite feature")
        out.append({
            "row_id": int(row["row_id"]),
            "fold": int(a.fold),
            "true_k": int(row["true_K"]),
            "category": category(bool(lib[i]), bool(rub[i])),
            "features": feat,
        })

    counts = {}
    for c in ("both", "librosa_only", "rubberband_only", "neither"):
        rr = [r for r in out if r["category"] == c]
        counts[c] = {
            "rows": len(rr),
            "K2": sum(r["true_k"] == 2 for r in rr),
            "K3": sum(r["true_k"] == 3 for r in rr),
        }

    a.output.mkdir(parents=True)
    (a.output / "rows.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in out))
    report = {
        "status": "completed",
        "experiment": "v273_quartet_full488_extract",
        "fold": int(a.fold),
        "rows": len(out),
        "categories": counts,
        "features": list(FEATURES),
        "outer_fold_3_used": False,
        "prediction_changes": False,
    }
    (a.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
