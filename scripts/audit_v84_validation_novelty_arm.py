"""Audit one frozen V8.4 onset arm on the locked 12-track validation subset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_note.guitarset import ALLOWED_PLAYERS, index_guitarset
from causal_note.v84_onset_predictor import V84OnsetOnlyKerasPredictor
from scripts import audit_v84_validation_novelty as _base
from scripts.evaluate_boundaries import milliseconds_to_samples
from scripts.train_boundaries import split_tracks_by_group

_base.V84KerasPredictor = V84OnsetOnlyKerasPredictor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    indexed = tuple(track for track in index_guitarset(args.dataset_dir) if track.player_id in ALLOWED_PLAYERS)
    _, locked = split_tracks_by_group(
        indexed,
        validation_fraction=_base.DEFAULT_VALIDATION_FRACTION,
        seed=_base.DEFAULT_SEED,
    )
    tracks = locked[: _base.DEFAULT_LIMIT_TRACKS]
    predictions = _base._predict_onsets(
        args.model,
        tracks,
        chunk_size=_base.DEFAULT_CHUNK_SIZE,
        receptive_field=_base.DEFAULT_RECEPTIVE_FIELD,
        threshold=_base.DEFAULT_THRESHOLD,
    )
    arm = _base._audit_arm(
        args.name,
        args.model,
        tracks,
        predictions,
        tolerance_samples=milliseconds_to_samples(_base.DEFAULT_TOLERANCE_MS),
    )
    result = {
        "schema_version": 1,
        "name": args.name,
        "split": {
            "seed": _base.DEFAULT_SEED,
            "evaluated_tracks": len(tracks),
            "validation_members": [track.annotation_member for track in tracks],
        },
        "configuration": {
            "threshold": _base.DEFAULT_THRESHOLD,
            "tolerance_ms": _base.DEFAULT_TOLERANCE_MS,
            "transfer_rules_frozen_from_train": True,
            "validation_threshold_refit": False,
            "offset_stream_executed": False,
            "offset_training_or_mutation": False,
        },
        "arm": arm,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
