"""Prepare supervised harmonic-parent pair data from annotated audio.

Uses the SAME frozen causal S613 frontend and heads; looks at annotation
labels only to build supervised training targets, never during inference.
Do NOT include held-out S613 test or four demo WAVs as a reported test score.

Manifest JSON:
{"sources": [
  {"wav": "train_guitar.wav", "split": "train", "group": "guitar-player-A",
   "notes": [{"pitch": 40, "time_s": 1.20, "duration_s": 0.8},
             {"pitch": 52, "time_s": 1.22, "duration_s": 0.7}]},
  {"wav": "dev_guitar.wav", "split": "dev", "group": "guitar-player-B",
   "notes": [...]}]}

Split by players/songs before training, never randomize frames across splits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from causal_note.strict_live import (
    CausalPitchFeatures, CausalSelection, HOP, SR, Selection, load_heads,
)
from causal_note.harmonic_expert import harmonic_pair_features, FEATURE_NAMES
from scripts.live_strict_notes import DEFAULT_MODEL, blocks_wav


def make_dataset(sources, heads, *, min_onset=0.05, truth_tolerance_s=0.04):
    X_all, y_all, groups = [], [], []
    for src in sources:
        wav = Path(src["wav"])
        group = str(src["group"])
        raw_notes = src["notes"]
        if not wav.is_file():
            raise ValueError(f"Missing annotated audio {wav}")
        notes = [
            (int(n["pitch"]), float(n["time_s"]), float(n["duration_s"]))
            for n in raw_notes
        ]
        if not notes or any(not (40 <= p <= 88 and 0 <= start and duration > 0)
                            for p, start, duration in notes):
            raise ValueError(f"Invalid onsets/durations for {wav}")
        frontend = CausalPitchFeatures()
        native = CausalSelection(Selection(**heads.metadata["selection"]))
        start = 0
        for block in blocks_wav(wav, HOP):
            for end, features in frontend.push(block, start_sample=start):
                p, k = heads.predict(features)
                pairs, X = harmonic_pair_features(features, p, k, native.active)
                if len(pairs):
                    now = end / SR
                    child_onset = p[pairs[:, 1] - 40, 0]
                    # Genuine independent note-on at this pitch (including
                    # octave doubling and octave attacks on a held bass note).
                    labels = np.asarray([
                        any(pitch == int(child) and
                            abs(t - now) <= truth_tolerance_s
                            for pitch, t, _ in notes)
                        for _, child, _ in pairs
                    ], dtype=np.int8)
                    # Training candidates and hard negatives. This modest gate
                    # is below the original selector threshold 0.60.
                    keep = (child_onset >= min_onset) | (labels == 1)
                    X_all.extend(X[keep])
                    y_all.extend(labels[keep])
                    groups.extend([group] * int(keep.sum()))
                native.step(p, k, end, frontend.received)
            start += len(block)
    if not X_all:
        raise ValueError("No pair candidates; inspect annotation and microphone domain")
    return (np.asarray(X_all, np.float32), np.asarray(y_all, np.int8),
            np.asarray(groups))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sources = manifest["sources"]
    if not isinstance(sources, list) or not sources:
        parser.error("Manifest requires a nonempty sources array")
    if any(s["split"] not in ("train", "dev") for s in sources):
        parser.error("Only train/dev allowed: held-out test must remain untouched")
    train_groups = {s["group"] for s in sources if s["split"] == "train"}
    dev_groups = {s["group"] for s in sources if s["split"] == "dev"}
    if not train_groups or not dev_groups or train_groups & dev_groups:
        parser.error("Train and dev groups must be nonempty and disjoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {part: args.output_dir / f"{part}_harmonic_pairs.npz"
             for part in ("train", "dev")}
    if any(path.exists() for path in paths.values()):
        parser.error("Pair output exists; use a new output directory")
    heads = load_heads(args.model)
    with threadpool_limits(limits=1):
        for part in ("train", "dev"):
            items = [s for s in sources if s["split"] == part]
            X, y, groups = make_dataset(items, heads)
            np.savez_compressed(paths[part], X=X, y=y, groups=groups,
                                feature_names=np.asarray(FEATURE_NAMES))
            print(json.dumps(dict(split=part, output=str(paths[part]),
                                  examples=len(y), genuine_onset=int(y.sum()),
                                  not_genuine=int((1 - y).sum()),
                                  groups=len(set(groups.tolist())))))


if __name__ == "__main__":
    main()
