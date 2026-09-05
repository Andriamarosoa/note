"""Run the V10.1 data audit with exact indexed event-to-cluster assignment."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from causal_note.guitarset import ALLOWED_PLAYERS, index_guitarset
from scripts.train_v92_string_factorized_cardinality import LOCAL_RADIUS_SAMPLES
from scripts import audit_v101_poly_data as audit


def _assign_events_fast(dataset_dir: Path, members, candidate_samples):
    indexed = tuple(t for t in index_guitarset(dataset_dir) if t.player_id in ALLOWED_PLAYERS)
    by_member_track = {t.annotation_member: t for t in indexed}
    by_member_rows = defaultdict(list)
    for cid, member in enumerate(members):
        by_member_rows[str(member)].append(cid)

    events = [[] for _ in members]
    assigned = 0
    unassigned = 0
    same_slot_collisions = 0

    for member, ids in by_member_rows.items():
        track = by_member_track.get(member)
        if track is None:
            raise audit.AuditError(f"missing track {member}")

        sample_parts = []
        cid_parts = []
        for cid in ids:
            samples = np.asarray(candidate_samples[cid], dtype=np.int64)
            if len(samples):
                sample_parts.append(samples)
                cid_parts.append(np.full(len(samples), cid, dtype=np.int64))
        if sample_parts:
            flat_samples = np.concatenate(sample_parts)
            flat_cids = np.concatenate(cid_parts)
            order = np.argsort(flat_samples, kind="stable")
            flat_samples = flat_samples[order]
            flat_cids = flat_cids[order]
        else:
            flat_samples = np.empty(0, dtype=np.int64)
            flat_cids = np.empty(0, dtype=np.int64)

        occupied = set()
        for slot, onset, midi in audit.v101._pitch_events(track):
            lo = int(np.searchsorted(flat_samples, onset - LOCAL_RADIUS_SAMPLES, side="left"))
            hi = int(np.searchsorted(flat_samples, onset + LOCAL_RADIUS_SAMPLES, side="right"))
            if lo >= hi:
                unassigned += 1
                continue
            local_samples = flat_samples[lo:hi]
            local_cids = flat_cids[lo:hi]
            distances = np.abs(local_samples - int(onset))
            min_dist = int(np.min(distances))
            tied_cids = local_cids[distances == min_dist]
            cid = int(np.min(tied_cids))
            if (cid, int(slot)) in occupied:
                same_slot_collisions += 1
            occupied.add((cid, int(slot)))
            events[cid].append((int(slot), int(onset), float(midi), min_dist))
            assigned += 1

    return events, {
        "assigned": assigned,
        "unassigned": unassigned,
        "assigned_fraction": assigned / (assigned + unassigned) if assigned + unassigned else None,
        "same_slot_collisions": same_slot_collisions,
        "assignment_implementation": "exact_sorted_candidate_index",
    }


audit._assign_events = _assign_events_fast

if __name__ == "__main__":
    raise SystemExit(audit.main())
