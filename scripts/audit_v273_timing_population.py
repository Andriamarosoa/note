"""Audit numerical drift revealed by the exact-timing re-mining experiment.

No audio inference, label fitting, model training or score comparison. Read the
saved data and distinguish changed floating features from changed row geometry.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from scripts.candidate_timing import timing_fields, full_samples
from scripts.rebuild_v273_sources import digest, write_json, DATA_MD5
from scripts.audit_v273_control_inputs import nearest_assignment
from causal_note.guitarset import load_boundary_slots
from scripts.train_v100_spectral_string_slots import PRE_SAMPLES, POST_SAMPLES


def differences(new, old):
    new, old = np.asarray(new), np.asarray(old)
    if new.shape != old.shape:
        return {"same_shape": False, "new_shape": list(new.shape), "old_shape": list(old.shape)}
    changed = new != old
    result = {"same_shape": True, "changed_values": int(changed.sum()),
              "changed_rows": int(changed.reshape(len(new), -1).any(1).sum()),
              "max_absolute_difference": float(np.max(np.abs(new.astype(float) - old.astype(float)), initial=0))}
    if new.dtype == np.float16 and old.dtype == np.float16:
        def ordered(a):
            bits = a.view(np.uint16).astype(np.int32)
            return np.where(bits & 0x8000, 0x8000 - (bits & 0x7fff), 0x8000 + bits)
        ulps = np.abs(ordered(new) - ordered(old))
        result["max_float16_steps"] = int(ulps.max(initial=0))
        result["changed_by_one_float16_step"] = int((ulps == 1).sum())
    return result


def audit(root, metadata, config, output, annotation_zip):
    if output.exists():
        raise FileExistsError(output)
    cfg = json.loads(config.read_text())
    if digest(annotation_zip, "md5") != DATA_MD5["annotation.zip"]:
        raise RuntimeError("annotation checksum mismatch")
    allowed = {m for m, fold in cfg["member_folds"].items() if fold == 3}
    with np.load(metadata, allow_pickle=False) as z:
        old = {key: np.asarray(z[key]) for key in z.files}
    if set(old["members"].astype(str)) != allowed:
        raise RuntimeError("previous metadata is not fold 3")
    paths = sorted(root.rglob("v91-cache-shard-*.npz"))
    reports, seen, outside_events = [], set(), []
    total_annotations = unassigned_annotations = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            timing = timing_fields(z, required=True)
            names = set(z["members"].astype(str))
            if len(names) != 1 or names & seen or not names <= allowed:
                raise RuntimeError("unexpected or duplicate track in corrected data")
            seen |= names
            member = next(iter(names))
            ids = np.flatnonzero(old["members"].astype(str) == member)
            if len(ids) != len(z["exact"]):
                raise RuntimeError(f"candidate row population changed: {member}")
            comparisons = {key: differences(z[key], old[key][ids])
                           for key in ("sequence", "mask", "stats", "exact", "top_samples", "truncated")}
            structural = all(np.array_equal(z[key], old[key][ids]) for key in ("mask", "exact", "top_samples", "truncated"))
            structural &= np.array_equal(z["sequence"][..., -2:], old["sequence"][ids, :, -2:])
            structural &= np.array_equal(z["stats"][:, :2], old["stats"][ids, :2])
            if not structural:
                raise RuntimeError(f"retained geometry, original group size/width or count labels changed: {member}")
            full = full_samples(z)
            flat = np.concatenate(full)
            row_ids = np.repeat(np.arange(len(full)), [len(x) for x in full])
            order = np.argsort(flat, kind="stable")
            flat, row_ids = flat[order], row_ids[order]
            counts = np.zeros(len(full), np.int32)
            for slot, boundaries in enumerate(load_boundary_slots(annotation_zip, member)):
                for boundary in boundaries:
                    total_annotations += 1
                    onset = int(boundary.onset_sample)
                    nearest = nearest_assignment(onset, flat, row_ids)
                    if nearest is None:
                        unassigned_annotations += 1
                        continue
                    row, distance = nearest
                    counts[row] += 1
                    relative = onset - int(timing["cluster_start_samples"][row])
                    if relative < -PRE_SAMPLES or relative >= POST_SAMPLES:
                        outside_events.append({"member": member, "global_index": int(old["global_index"][ids[row]]),
                            "k": int(z["exact"][row]), "slot": slot, "onset_sample": onset,
                            "relative_sample": relative, "nearest_distance": distance,
                            "samples_after_last_candidate": onset - int(full[row][-1])})
            if not np.array_equal(counts, z["exact"]):
                raise RuntimeError(f"independent annotation count mismatch: {member}")
            row_audit = json.loads((path.parent / "audit.json").read_text())
            with np.load(path.parent / "audit.npz", allow_pickle=False) as a:
                changed = a["legacy_window_start"] != a["cluster_start"]
                changed_slots = np.any(a["legacy_slots"] != a["new_slots"], axis=1)
            # These IDs are aligned by the verified geometry/anchors/counts,
            # not by claiming equality of all floating candidate features.
            reports.append({"member": member, "rows": len(ids), "comparisons": comparisons,
                "retained_geometry_and_count_labels_equal": True,
                "changed_window_old_global_indices": old["global_index"][ids][changed].tolist(),
                "changed_slot_old_global_indices": old["global_index"][ids][changed_slots].tolist(),
                "new_occupancy_count_mismatches": row_audit["new_occupancy_count_mismatches"],
                "source_sha256": digest(path)})
    if seen != allowed or len(paths) != 50:
        raise RuntimeError("incomplete fold 3 cache coverage")
    totals = {}
    for key in ("sequence", "mask", "stats", "exact", "top_samples", "truncated"):
        values = [r["comparisons"][key] for r in reports]
        totals[key] = {"changed_values": sum(v["changed_values"] for v in values),
                       "changed_rows": sum(v["changed_rows"] for v in values),
                       "max_absolute_difference": max(v["max_absolute_difference"] for v in values)}
        if "max_float16_steps" in values[0]:
            totals[key].update(max_float16_steps=max(v["max_float16_steps"] for v in values),
                              changed_by_one_float16_step=sum(v["changed_by_one_float16_step"] for v in values))
    result = {"status": "passed_geometry_with_measured_numeric_drift", "outer_fold": 3,
        "tracks": reports, "track_count": len(reports), "rows": sum(r["rows"] for r in reports),
        "all_retained_geometry_and_count_labels_equal": True, "feature_differences": totals,
        "previous_metadata_sha256": digest(metadata), "config_sha256": digest(config),
        "audit_source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "remaining_spectral_coverage": {
            "total_annotations": total_annotations, "unassigned_annotations": unassigned_annotations,
            "assigned_annotations": total_annotations - unassigned_annotations,
            "outside_events": outside_events,
            "outside_event_count": len(outside_events),
            "outside_row_count": len({e["global_index"] for e in outside_events}),
            "outside_poly_row_count": len({e["global_index"] for e in outside_events if e["k"] >= 2}),
            "before_window_count": sum(e["relative_sample"] < -PRE_SAMPLES for e in outside_events),
            "after_window_count": sum(e["relative_sample"] >= POST_SAMPLES for e in outside_events),
            "max_relative_sample": max((e["relative_sample"] for e in outside_events), default=None),
            "post_window_samples": POST_SAMPLES, "assignment_radius_samples": 882,
            "cause": "Onsets can be assigned up to 882 samples after a candidate, but the spectral window ends 1764 samples after the group origin.",
            "causal_score_effect_measured": False},
        "interpretation": "Same retained geometry, top anchors, full group counts/widths and K targets. Some floating proposal features differ after re-mining.",
        "limitation": "The mechanism causing numeric drift and its effect on trained predictions are not established. These caches do not isolate a timestamps-only training treatment against old features."}
    write_json(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "tracks"}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "metadata", "config", "output", "annotation-zip"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    audit(a.root, a.metadata, a.config, a.output, a.annotation_zip)
