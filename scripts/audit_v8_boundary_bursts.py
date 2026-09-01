"""Audit anonymous boundary burst cardinality over causal time horizons."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_note.guitarset import ALLOWED_PLAYERS, SAMPLE_RATE, index_guitarset, load_boundary_slots
from scripts.train_boundaries import split_tracks_by_group


def _positions(track):
    slots = load_boundary_slots(track.annotation_zip, track.annotation_member)
    onset = []
    offset = []
    for notes in slots:
        for note in notes:
            onset.append(note.onset_sample)
            offset.append(note.offset_sample)
    return tuple(sorted(onset)), tuple(sorted(offset))


def _cluster_fixed_span(positions, horizon_samples):
    """Cluster positions whose total cluster span is <= horizon."""
    clusters = []
    index = 0
    values = tuple(positions)
    while index < len(values):
        start = values[index]
        end = index + 1
        while end < len(values) and values[end] - start <= horizon_samples:
            end += 1
        clusters.append(values[index:end])
        index = end
    return clusters


def _report(tracks, horizon_ms):
    horizon_samples = int(round(horizon_ms * SAMPLE_RATE / 1000.0))
    result = {}
    for kind_index, kind in enumerate(("onset", "offset")):
        size_counts = Counter()
        total_boundaries = 0
        multi_boundaries = 0
        spans = []
        clusters_total = 0
        for track in tracks:
            positions = _positions(track)[kind_index]
            total_boundaries += len(positions)
            for cluster in _cluster_fixed_span(positions, horizon_samples):
                size = len(cluster)
                clusters_total += 1
                size_counts[min(size, 6)] += 1
                if size > 1:
                    multi_boundaries += size
                spans.append(cluster[-1] - cluster[0] if size > 1 else 0)
        result[kind] = {
            "boundaries": total_boundaries,
            "clusters": clusters_total,
            "cluster_size_counts": {
                "1": size_counts[1], "2": size_counts[2], "3": size_counts[3],
                "4": size_counts[4], "5": size_counts[5], "6+": size_counts[6],
            },
            "fraction_boundaries_in_multi_cluster": multi_boundaries / total_boundaries if total_boundaries else 0.0,
            "mean_boundaries_per_cluster": total_boundaries / clusters_total if clusters_total else 0.0,
            "max_span_ms": (max(spans) * 1000.0 / SAMPLE_RATE) if spans else 0.0,
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizons-ms", nargs="+", type=float, default=[5, 10, 20, 30, 50])
    args = parser.parse_args(argv)

    indexed = tuple(track for track in index_guitarset(args.dataset_dir) if track.player_id in ALLOWED_PLAYERS)
    train, validation = split_tracks_by_group(indexed, validation_fraction=0.2, seed=1337)
    report = {
        "schema_version": 1,
        "train_tracks": len(train),
        "validation_tracks": len(validation),
        "horizons": {},
    }
    for horizon in args.horizons_ms:
        report["horizons"][f"{horizon:g}ms"] = {
            "train": _report(train, horizon),
            "validation": _report(validation, horizon),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for horizon, split_payload in report["horizons"].items():
        print(horizon)
        for split in ("train", "validation"):
            for kind in ("onset", "offset"):
                item = split_payload[split][kind]
                print(
                    split, kind,
                    f"mean={item['mean_boundaries_per_cluster']:.3f}",
                    f"multi_boundary_fraction={item['fraction_boundaries_in_multi_cluster']:.3f}",
                    f"sizes={item['cluster_size_counts']}",
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
