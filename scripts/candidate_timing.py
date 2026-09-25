"""Lossless candidate timestamps, separate from the 48 model input rows.

Schema 2 caches carry original starts, retained samples in model-row order,
and every pre-truncation sample as a flat array with CSR-style offsets.
No annotation or model prediction is used to recover these integers.
"""
import numpy as np


TIMING_KEYS = (
    "cluster_start_samples", "candidate_samples",
    "full_candidate_samples", "full_candidate_offsets",
)


def retained_indices(cluster, records, scores, limit):
    indices = list(cluster["indices"])
    if len(indices) > limit:
        indices = sorted(indices, key=lambda i: (-float(scores[i]), int(records[i]["sample"]), i))[:limit]
        indices = sorted(indices, key=lambda i: (int(records[i]["sample"]), i))
    return indices


def capture_timing(clusters, records, scores, limit):
    starts = np.empty(len(clusters), np.int64)
    retained = np.full((len(clusters), limit), -1, np.int64)
    parts, offsets = [], [0]
    for row, cluster in enumerate(clusters):
        full = np.sort(np.asarray([records[i]["sample"] for i in cluster["indices"]], np.int64))
        if not len(full):
            raise ValueError("empty candidate cluster")
        starts[row] = full[0]
        indices = retained_indices(cluster, records, scores, limit)
        retained[row, :len(indices)] = [records[i]["sample"] for i in indices]
        parts.append(full)
        offsets.append(offsets[-1] + len(full))
    return dict(zip(TIMING_KEYS, (starts, retained,
        np.concatenate(parts) if parts else np.empty(0, np.int64), np.asarray(offsets, np.int64))))


def timing_fields(cache, *, required=False):
    present = [key in cache for key in TIMING_KEYS]
    if not any(present):
        if required:
            raise ValueError("exact timing missing; re-mine from original candidates")
        return {}
    if not all(present):
        raise ValueError("incomplete exact candidate timing")
    result = {key: np.asarray(cache[key]) for key in TIMING_KEYS}
    if any(a.dtype.kind not in "iu" for a in result.values()):
        raise ValueError("candidate timestamps and offsets must be integers")
    starts, retained, full, offsets = (result[key] for key in TIMING_KEYS)
    mask = np.asarray(cache["mask"])
    n = len(mask)
    if (starts.shape != (n,) or retained.shape != mask.shape or full.ndim != 1
            or offsets.shape != (n + 1,) or offsets[0] != 0 or offsets[-1] != len(full)
            or np.any(np.diff(offsets.astype(np.int64)) <= 0)):
        raise ValueError("invalid exact timing shapes or offsets")
    if np.any((mask != 0) & (mask != 1)) or np.any(retained[mask == 0] != -1):
        raise ValueError("exact candidate padding/mask mismatch")
    for row in range(n):
        values = full[int(offsets[row]):int(offsets[row + 1])]
        selected = retained[row, mask[row] > 0]
        if (not len(selected) or values[0] < 0 or np.any(np.diff(values.astype(np.int64)) < 0)
                or starts[row] != values[0] or not np.isin(selected, values).all()):
            raise ValueError(f"invalid exact candidate timing at row {row}")
        if "truncated" in cache and len(values) - len(selected) != int(cache["truncated"][row]):
            raise ValueError(f"candidate truncation/timing mismatch at row {row}")
    # The start-relative feature must remain aligned with each retained model row.
    if "sequence" in cache:
        from scripts.train_v90_structured_cluster_cardinality import CLUSTER_WINDOW_SAMPLES
        rel = np.rint(np.asarray(cache["sequence"])[..., -2].astype(np.float64)
                      * CLUSTER_WINDOW_SAMPLES).astype(np.int64)
        if np.any((retained - starts[:, None])[mask > 0] != rel[mask > 0]):
            raise ValueError("exact timestamps disagree with retained feature rows")
    return result


def full_samples(cache):
    fields = timing_fields(cache, required=True)
    offsets = fields["full_candidate_offsets"]
    samples = fields["full_candidate_samples"]
    return [samples[int(a):int(b)] for a, b in zip(offsets[:-1], offsets[1:])]


def merge_timing(shards):
    """Validate each shard and rebase ragged offsets; reject legacy/exact mixing."""
    fields = [timing_fields(shard) for shard in shards]
    if not any(fields):
        return {}
    if not all(fields):
        raise ValueError("cannot mix legacy and exact-timing caches")
    offsets, size = [0], 0
    for field in fields:
        offsets.extend((field["full_candidate_offsets"][1:] + size).tolist())
        size += len(field["full_candidate_samples"])
    result = {key: np.concatenate([field[key] for field in fields], axis=0)
              for key in TIMING_KEYS if key != "full_candidate_offsets"}
    result["full_candidate_offsets"] = np.asarray(offsets, np.int64)
    return result


def check_schema(cache, version):
    if version not in (1, 2):
        raise ValueError(f"unsupported candidate cache schema {version}")
    fields = timing_fields(cache, required=version == 2)
    if version == 1 and fields:
        raise ValueError("legacy cache contains unversioned exact timing")
