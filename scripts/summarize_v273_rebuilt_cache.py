"""Report rebuilt native inputs and compare only preserved historical arrays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.rebuild_v273_sources import SOURCE_KIND, write_json


SHARED = ("sequence", "mask", "exact", "members", "top_samples", "slot_targets")


def summarize(cache_dir, archived_metadata, output_dir):
    paths = sorted(Path(cache_dir).rglob("v100-spectral-shard-*.npz"))
    expected = {f"v100-spectral-shard-{i:02d}.npz" for i in range(8)}
    if len(paths) != 8 or {p.name for p in paths} != expected:
        raise RuntimeError("expected exactly eight native input shards")
    tracks = []
    rows = 0
    poly = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            required = {*SHARED, "spectral", "stats", "target", "track_members", "schema_version"}
            if not required.issubset(z.files) or int(z["schema_version"][0]) != 1:
                raise RuntimeError("incomplete native cache; distilled metadata is insufficient")
            k = z["exact"]
            n = len(k)
            spectral, stats = z["spectral"], z["stats"]
            if spectral.shape != (n, 23, 64, 3) or stats.shape != (n, 8):
                raise RuntimeError("native input shapes changed")
            if not np.isfinite(spectral).all() or not np.isfinite(stats).all():
                raise RuntimeError("nonfinite reconstructed inputs")
            if any(len(z[key]) != n for key in SHARED):
                raise RuntimeError("native cache row alignment mismatch")
            shard_tracks = z["track_members"].astype(str).tolist()
            if not set(z["members"].astype(str)).issubset(set(shard_tracks)):
                raise RuntimeError("cluster references an unknown track")
            tracks.extend(shard_tracks)
            rows += n
            poly += int(np.sum(k >= 2))
    if len(tracks) != 240 or len(set(tracks)) != 240:
        raise RuntimeError("expected 240 unique train tracks across the eight shards")
    comparison = {}
    with np.load(archived_metadata, allow_pickle=False) as old:
        archived_rows = len(old["exact"])
        if sorted(tracks) != sorted(old["track_members"].astype(str).tolist()):
            raise RuntimeError("rebuild changed the historical train track set")
        for key in SHARED:
            chunks = []
            for path in paths:
                with np.load(path, allow_pickle=False) as z:
                    chunks.append(z[key])
            new = np.concatenate(chunks, axis=0)
            original = old[key]
            comparison[key] = {
                "same_shape": new.shape == original.shape,
                "identical_values": bool(np.array_equal(new, original)),
            }
            del chunks, new, original
    result = {
        "source_kind": SOURCE_KIND,
        "status": "native_input_cache_rebuilt_historical_parity_unverified",
        "rows": rows, "polyphonic_rows": poly, "tracks": len(tracks),
        "archived_rows": archived_rows, "shared_metadata_comparison": comparison,
        "shared_metadata_values_identical": all(v["identical_values"] for v in comparison.values()),
        "historical_stats_and_spectral_available_for_comparison": False,
        "historical_v273_reproduced": False,
        "native_decay_comparison_started": False,
        "next_required_step": "rebuild strict inner experts and a paired V27.3 control before testing decay",
    }
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    write_json(output_dir / "summary.json", result)
    (output_dir / "summary.md").write_text(
        "# Reconstruction des entrées natives\n\n"
        f"{rows} clusters, dont {poly} polyphoniques, sur {len(tracks)} pistes.\n\n"
        f"Valeurs des métadonnées conservées identiques : {result['shared_metadata_values_identical']}.\n\n"
        "Les anciens tableaux stats et spectral restent indisponibles pour comparaison. "
        "L'identité de V27.3 n'est donc pas établie.\n\n"
        "Le test natif avec/sans décroissance n'a pas commencé. "
        "La référence archivée à 42,6019 % reste inchangée.\n"
    )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--archived-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.cache_dir, args.archived_metadata, args.output_dir)
