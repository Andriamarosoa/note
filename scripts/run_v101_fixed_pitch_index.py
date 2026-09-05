"""V10.1 retry with corrected pitch-supervision row indexing.

The first V10.1 run correctly failed its supervision integrity guard because
training-only MIDI pitch labels were written with a per-track local row index
into global cache arrays. This wrapper replaces only that target-construction
function; architecture, losses, calibration, latency and locked12 protocol stay
identical to V10.1.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_v101_string_query_attention as v


def _derive_pitch_targets_fixed(cache, dataset_dir: Path):
    indexed = tuple(t for t in v.index_guitarset(dataset_dir) if t.player_id in v.ALLOWED_PLAYERS)
    by_member = {t.annotation_member: t for t in indexed}
    candidate_samples, reconstruction = v._reconstruct_candidates(cache)
    pitch = np.zeros((len(cache["target"]), v.SLOT_COUNT), dtype=np.float32)
    mask = np.zeros_like(pitch)
    by_member_rows: Dict[str, List[int]] = defaultdict(list)
    for i, member in enumerate(cache["members"]):
        by_member_rows[str(member)].append(i)

    assigned = 0
    unassigned = 0
    collisions = 0
    values: List[float] = []
    for member, ids in by_member_rows.items():
        track = by_member.get(member)
        if track is None:
            raise v.V101Error(f"cache member missing from GuitarSet index: {member}")
        for slot, onset, midi in v._pitch_events(track):
            choices = []
            for cid in ids:
                samples = candidate_samples[cid]
                if not len(samples):
                    continue
                dist = int(np.min(np.abs(samples - onset)))
                if dist <= v.LOCAL_RADIUS_SAMPLES:
                    choices.append((dist, cid))
            if not choices:
                unassigned += 1
                continue
            _, cid = min(choices)
            # cid is already the global cache-row index. The failed first run
            # incorrectly converted it to a per-track local row before writing.
            if mask[cid, slot] > 0.5:
                collisions += 1
            pitch[cid, slot] = float(midi / v.PITCH_SCALE)
            mask[cid, slot] = 1.0
            values.append(float(midi))
            assigned += 1

    slot_truth = np.asarray(cache["slot_targets"], dtype=np.float32) > 0.5
    pitch_truth = mask > 0.5
    agreement = float(np.mean(slot_truth == pitch_truth))
    active_agreement = float(np.mean(pitch_truth[slot_truth])) if np.any(slot_truth) else 1.0
    if agreement < 0.999 or active_agreement < 0.995:
        raise v.V101Error(
            f"pitch assignment does not reproduce slot supervision: agreement={agreement} active={active_agreement}"
        )
    return pitch, mask, {
        **reconstruction,
        "assigned_pitch_events": assigned,
        "unassigned_pitch_events": unassigned,
        "same_slot_collisions": collisions,
        "slot_pitch_mask_agreement": agreement,
        "active_slot_pitch_coverage": active_agreement,
        "midi_min": min(values) if values else None,
        "midi_max": max(values) if values else None,
        "midi_mean": float(np.mean(values)) if values else None,
        "indexing_fix": "global_cluster_row",
    }


v._derive_pitch_targets = _derive_pitch_targets_fixed


if __name__ == "__main__":
    raise SystemExit(v.main())
