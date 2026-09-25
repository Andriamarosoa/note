"""Re-mine and audit exact timestamps on the frozen outer fold 3 only.

Frozen proposal inference, no training or output correction. Compare serialized
cache targets/windows against runtime groups made directly from raw records.
"""
import argparse
import gc
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

from scripts import candidate_timing as timing
from scripts import train_v91_ordinal_cardinality as v91
from scripts import train_v92_string_factorized_cardinality as v92
from scripts import train_v100_spectral_string_slots as v100
from scripts import train_v102_source_time_assignment as v102
from scripts.rebuild_v273_sources import validate_sources, verify_dataset, digest, write_json


def equal(actual, expected, name):
    matches = np.array_equal(actual, expected, equal_nan=True) if np.asarray(actual).dtype.kind in "fc" else np.array_equal(actual, expected)
    if not matches:
        raise RuntimeError(f"exact timing audit failed: {name}")


def run(args):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    cfg = json.loads(args.config.read_text())
    allowed = {m for m, fold in cfg["member_folds"].items() if fold == 3}
    if len(allowed) != 50:
        raise RuntimeError("unexpected frozen fold 3 track count")
    source_state = validate_sources(args.source_dir)
    verify_dataset(args.dataset_dir)
    _, train, locked = v91._dataset_split(args.dataset_dir)
    tracks = tuple(sorted((t for t in train if t.annotation_member in allowed), key=lambda t: t.annotation_member))
    if {t.annotation_member for t in tracks} != allowed or allowed & {t.annotation_member for t in locked}:
        raise RuntimeError("fold 3 scope mismatch or locked validation overlap")
    with np.load(args.previous_metadata, allow_pickle=False) as z:
        old = {k: np.asarray(z[k]) for k in z.files}
    if set(old["members"].astype(str)) != allowed:
        raise RuntimeError("previous metadata is not the same outer fold")
    model_args = SimpleNamespace(base_model=args.source_dir / "v84/control.epoch-01.keras")
    for version, name in (("v86", "v86-state-transition-refiner"),
                          ("v87", "v87-causal-candidate-memory"), ("v88", "v88-regime-moe")):
        setattr(model_args, version + "_weights", args.source_dir / version / (name + ".weights.h5"))
        setattr(model_args, version + "_report", args.source_dir / version / "report.json")
    floor, _, enc86, enc87, model88 = v91._load_frozen_stack(model_args)
    if not 0 <= args.part < args.parts or args.parts != 5:
        raise ValueError("expected one of five parts of fold 3")
    args.output_dir.mkdir(parents=True)
    reports = []
    for i, track in enumerate(tracks):
        if i % args.parts != args.part:
            continue
        member = track.annotation_member
        print(f"AUDIT TRACK {i + 1}/{len(tracks)} {member}", flush=True)
        streams, records, x88, out88 = v91._represent_full((track,), model_args.base_model, floor, enc86, enc87, model88)
        clusters, fused, assignment, sequence, mask, stats, target, exact, truncated = v91._cluster_data((track,), records, x88, out88)
        shard = args.output_dir / f"track-{i:02d}"
        shard.mkdir()
        captured = timing.capture_timing(clusters, records, fused, v91.MAX_CANDIDATES)
        v91._save_cache(shard / f"v91-cache-shard-{i:02d}.npz", sequence=sequence, mask=mask, stats=stats,
            target=target, exact=exact, truncated=truncated, members=[member] * len(clusters),
            top_samples=v91._top_samples(clusters, records, fused), track_members=[member], timing=captured)
        cache = v91._load_caches(shard)
        for key in timing.TIMING_KEYS:
            equal(cache[key], captured[key], f"V91 round trip {key}")
        retained, _ = v92._reconstruct_candidates(cache)
        full = timing.full_samples(cache)
        raw = [np.sort(np.asarray([records[j]["sample"] for j in c["indices"]], np.int64)) for c in clusters]
        for row, samples in enumerate(raw):
            equal(full[row], samples, "full raw candidate times")
        slots, _, slot_diag = v92._derive_cache_slot_targets(cache, args.dataset_dir)
        runtime_slots, runtime_slot_diag = v92._slot_targets_for_runtime_clusters((track,), clusters, records)
        equal(slots, runtime_slots, "cached vs runtime slots")
        spectral, _ = v100._spectral_maps_for_cache(cache, args.dataset_dir)
        live = v100._spectral_maps_for_runtime((track,), clusters, records)
        equal(spectral, live.astype(np.float16), "cached vs runtime spectral map")
        v100._save_spectral_cache(shard / f"v100-spectral-shard-{i:02d}.npz", cache, spectral, slots)
        spectral_cache = v100._load_spectral_caches(shard)
        for key in timing.TIMING_KEYS:
            equal(spectral_cache[key], captured[key], f"V100 round trip {key}")
        equal(spectral_cache["spectral"], spectral, "V100 spectral round trip")
        actual = v102._derive_supervision(cache["members"], retained, args.dataset_dir,
            assignment_cache=spectral_cache, expected_slot_targets=slots)
        runtime = v102._derive_supervision(cache["members"], raw, args.dataset_dir,
            expected_slot_targets=runtime_slots)
        for name, a, b in zip(("pitch", "time_mask", "time_distribution", "relative_time"), actual[:4], runtime[:4]):
            equal(a, b, f"cached vs runtime {name}")
        # Recompute event counts independently of the cache slot routine, retaining
        # multiplicity. Occupancy can legitimately differ for same-string births.
        counts = np.zeros(len(clusters), np.int32)
        outside = 0
        for slot, onset, midi in v102.v101._pitch_events(track):
            distances = [int(np.min(np.abs(samples - onset))) for samples in raw]
            distance, row = min((d, row) for row, d in enumerate(distances))
            if distance <= v92.LOCAL_RADIUS_SAMPLES:
                counts[row] += 1
                relative = onset - int(captured["cluster_start_samples"][row])
                outside += int(relative < -v100.PRE_SAMPLES or relative >= v100.POST_SAMPLES)
        equal(counts, exact, "independent full-group onset counts")
        # Compare to prior data only if the regenerated row population really matches.
        previous_ids = np.flatnonzero(old["members"].astype(str) == member)
        identity_keys = ("sequence", "mask", "stats", "exact", "top_samples", "truncated")
        identical = len(previous_ids) == len(exact) and all(
            np.array_equal(cache[k], old[k][previous_ids]) for k in identity_keys)
        legacy = {key: value for key, value in cache.items() if key not in timing.TIMING_KEYS}
        legacy_samples, _ = v92._reconstruct_candidates(legacy)
        legacy_starts = np.asarray([min(samples) for samples in legacy_samples])
        changed_windows = legacy_starts != captured["cluster_start_samples"]
        legacy_slots, _, _ = v92._derive_cache_slot_targets(legacy, args.dataset_dir)
        wrong_slots = np.any(legacy_slots != slots, axis=1)
        mismatch = slots.sum(1).astype(int) != np.minimum(exact, 6)
        if np.any(mismatch) and slot_diag["same_slot_collisions"] == 0:
            raise RuntimeError("new unexplained occupancy/count discrepancy")
        row_report = {"member": member, "rows": len(exact), "poly_rows": int(np.sum(exact >= 2)),
            "raw_candidates": len(records), "truncated_groups": int(np.sum(truncated > 0)),
            "dropped_candidates": int(truncated.sum()), "historical_cache_identical": identical,
            "changed_window_origins": int(changed_windows.sum()), "changed_slot_rows": int(wrong_slots.sum()),
            "changed_origin_old_global_indices": old["global_index"][previous_ids][changed_windows].tolist() if identical else None,
            "changed_slot_old_global_indices": old["global_index"][previous_ids][wrong_slots].tolist() if identical else None,
            "new_occupancy_count_mismatches": int(mismatch.sum()), "same_string_collisions": slot_diag["same_slot_collisions"],
            "outside_spectral_window_events": outside,
            "assignment": assignment, "time_supervision": actual[4], "all_runtime_comparisons_equal": True}
        np.savez_compressed(shard / "audit.npz", cluster_start=captured["cluster_start_samples"],
            legacy_window_start=legacy_starts, exact=exact, new_slots=slots, legacy_slots=legacy_slots)
        write_json(shard / "audit.json", row_report)
        reports.append(row_report)
        write_json(args.output_dir / "progress.json", {"completed_tracks": len(reports), "expected_tracks": 10, "tracks": reports})
        del streams, records, x88, out88, clusters, sequence, spectral, live, cache, spectral_cache
        gc.collect()
    total_keys = ("rows", "poly_rows", "raw_candidates", "truncated_groups", "dropped_candidates", "changed_window_origins",
                  "changed_slot_rows", "new_occupancy_count_mismatches", "same_string_collisions", "outside_spectral_window_events")
    report = {"status": "passed", "outer_fold": 3, "part": args.part, "parts": args.parts, "tracks": reports,
        "totals": {key: sum(r[key] for r in reports) for key in total_keys},
        "all_historical_cache_features_identical": all(r["historical_cache_identical"] for r in reports),
        "all_runtime_comparisons_equal": True, "training_performed": False, "model_score_improvement_measured": False,
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config_sha256": digest(args.config), "previous_metadata_sha256": digest(args.previous_metadata),
        "source_state_sha256": digest(args.source_dir / "rebuild-state.json"),
        "proposal_source_kind": source_state["source_kind"],
        "scope": "10 of the 50 frozen fold-3 tracks; no other fold inference, training or calibration",
        "limitation": "Data preparation equivalence does not establish better Exact K. Old weights were not retrained or selected here."}
    write_json(args.output_dir / "report.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "tracks"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("source-dir", "dataset-dir", "previous-metadata", "config", "output-dir"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--part", type=int, required=True)
    parser.add_argument("--parts", type=int, default=5)
    run(parser.parse_args())
