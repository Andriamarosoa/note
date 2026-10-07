"""Fixed, annotation-independent bridge from note onsets to native Exact-K rows."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

FOLDS = (0, 1, 2, 4)
SAMPLE_RATE = 44100
CLUSTER_SAMPLES = 1764
RADIUS_SAMPLES = 882
SPACE_REVISION = "5e66c1ea173a8186e0d20432b841d3180cc015b5"
EXPERIMENT = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops"
CHECKPOINT_PATH = f"amt/logs/2024/{EXPERIMENT}/checkpoints/last.ckpt"
CHECKPOINT_SHA256 = "ae38e415c79efd5592dcb9b658cdb99ddb11d4c4e1eaa364cab04a052473fc25"
MODEL_ARGS = [EXPERIMENT + "@last.ckpt", "-p", "2024", "-tk", "mc13_full_plus_256",
              "-dec", "multi-t5", "-nl", "26", "-enc", "perceiver-tf", "-sqr", "1",
              "-ff", "moe", "-wf", "4", "-nmoe", "8", "-kmoe", "2", "-act", "silu",
              "-epe", "rope", "-rp", "1", "-ac", "spec", "-hop", "300", "-atc", "1",
              "-pr", "32", "-wb", "disabled"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def cluster_ends(sequence, mask, starts, stats):
    """Recover original endpoints even when candidate rows were truncated.

    Last two features are (sample-start)/1764 and (sample-center)/1764.
    Their difference gives (end-start)/3528 for EVERY retained candidate.
    No labels, model predictions, or rounded time windows enter this recovery.
    """
    sequence = np.asarray(sequence)
    mask = np.asarray(mask) > 0
    starts = np.asarray(starts, np.int64)
    require(mask.ndim == 2 and np.all(mask.any(axis=1)), "empty candidate group")
    widths_float = 2 * CLUSTER_SAMPLES * (sequence[..., -2].astype(np.float64)
                                         - sequence[..., -1].astype(np.float64))
    widths = np.rint(widths_float).astype(np.int64)
    require(np.max(np.abs(widths_float[mask] - widths[mask])) < 0.01,
            "candidate timing cannot be recovered sample-accurately")
    width = widths[np.arange(len(starts)), mask.argmax(axis=1)]
    require(np.all((widths == width[:, None]) | ~mask), "inconsistent group centers")
    require(np.all((width >= 0) & (width <= CLUSTER_SAMPLES)), "invalid group widths")
    stats_width = np.rint(np.asarray(stats)[:, 1].astype(np.float64) * CLUSTER_SAMPLES)
    require(np.array_equal(stats_width, np.maximum(1, width)), "width metadata drift")
    return starts + width


def assign_onsets(onsets, starts, ends):
    """Assign each onset once, with the native 20 ms radius and earliest-row tie.

    Inside a <=40 ms interval an onset is always within 20 ms of an endpoint.
    No neighboring, disjoint cluster can be closer. Outside an interval its
    nearest original candidate is an endpoint. Thus endpoints preserve the
    native nearest-candidate GROUP decision, including for truncated groups.
    """
    onsets = np.asarray(onsets, np.int64)
    starts, ends = np.asarray(starts, np.int64), np.asarray(ends, np.int64)
    require(starts.ndim == ends.ndim == onsets.ndim == 1, "invalid timing dimensions")
    require(len(starts) == len(ends) and len(starts) > 0, "missing intervals")
    require(np.all(ends >= starts) and np.all(ends - starts <= CLUSTER_SAMPLES),
            "invalid intervals")
    require(np.all(starts[1:] > ends[:-1]), "groups overlap or are not ordered")
    counts = np.zeros(len(starts), np.int32)
    assignments = np.full(len(onsets), -1, np.int64)
    for i, sample in enumerate(onsets):
        distances = np.maximum(np.maximum(starts - sample, sample - ends), 0)
        j = int(np.argmin(distances))
        if distances[j] <= RADIUS_SAMPLES:
            counts[j] += 1
            assignments[i] = j
    return counts, assignments


def metrics(y, p):
    y, p = np.asarray(y, np.int32), np.asarray(p, np.int32)
    require(y.shape == p.shape and y.ndim == 1, "metric alignment error")
    require(np.all((0 <= y) & (y <= 6) & (0 <= p) & (p <= 6)), "bad K class")

    def subset(which):
        a, b = y[which], p[which]
        n = len(a)
        correct = int(np.sum(a == b))
        return {"rows": n, "correct": correct, "exact": correct / n if n else None,
                "under": int(np.sum(b < a)), "over": int(np.sum(b > a))}

    confusion = np.zeros((7, 7), np.int64)
    np.add.at(confusion, (y, p), 1)
    return {**subset(np.ones(len(y), bool)), "poly": subset(y >= 2),
            "by_k": {str(k): subset(y == k) for k in range(7)},
            "confusion_true_by_predicted": confusion.tolist()}


def paired(y, base, pred):
    y, base, pred = map(np.asarray, (y, base, pred))

    def subset(mask):
        fixed = int(np.sum(mask & (base != y) & (pred == y)))
        broken = int(np.sum(mask & (base == y) & (pred != y)))
        return {"corrections": fixed, "regressions": broken, "net": fixed - broken,
                "changed": int(np.sum(mask & (base != pred)))}

    return {"global": subset(np.ones(len(y), bool)), "poly": subset(y >= 2),
            "by_k": {str(k): subset(y == k) for k in range(7)}}


def report_markdown(report):
    b, m = report["freeze_local_combo"], report["yourmt3_plus"]
    lines = ["# YourMT3+ versus freeze_local_combo", "",
             "Exploratory pretrained offline comparison; GuitarSet training overlap is not excluded.",
             "This is not a validation under the baseline's causal latency constraints.", "",
             "| True K | Rows | freeze exact | YourMT3+ exact | Under | Over | Net exact |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for k in range(7):
        a, c = b["by_k"][str(k)], m["by_k"][str(k)]
        def score(x):
            return "n/a" if x["exact"] is None else f"{x['correct']} ({100*x['exact']:.4f}%)"
        lines.append(f"| {k} | {c['rows']} | {score(a)} | {score(c)} | {c['under']} | {c['over']} | {c['correct']-a['correct']:+d} |")
    for label, a, c, d in [("Global", b, m, report["paired"]["global"]),
                            ("Poly K2–K6", b["poly"], m["poly"], report["paired"]["poly"])]:
        lines += ["", f"{label}: freeze {a['correct']}/{a['rows']} ({100*a['exact']:.4f}%), "
                  f"YourMT3+ {c['correct']}/{c['rows']} ({100*c['exact']:.4f}%).",
                  f"Corrections {d['corrections']}; regressions {d['regressions']}; net {d['net']:+d}."]
    lines += ["", "K0 means no new assigned attack, not necessarily silence. Class 6 is capped at 6, as in the native target.",
              "All decoded note events are counted, with no label-dependent instrument/pitch filter or offset tuning.",
              "Raw notes, unmatched events, overflow counts, and row-level predictions are retained. No automatic promotion."]
    return "\n".join(lines) + "\n"
