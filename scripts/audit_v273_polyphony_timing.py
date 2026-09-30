"""Read-only timing audit of the fixed epoch-12 residual/native validation pair.

Replays the original nearest-candidate assignment from full candidate timing.
Annotations describe outcomes/context only; no model, decoder or labels change.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

FS = 44100
RADIUS = 882
ARMS = ("observed_only", "with_residual")
ANNOTATION_SHA = "8daa02e6417ccca1685feb44b135e95928ad7037e5032ecb326b5791856fda99"
PREDICTION_SHAS = {
    "observed_only": "81ebb2b97450fcfa1721392f62459df5784e8a68497982f20ffa01a349e2e4af",
    "with_residual": "61816fa55cb0b3f9784e5939d511f55016c0e3b08420bd3e9a0035cb673e5124",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(a):
    h = hashlib.sha256(str((a.shape, str(a.dtype))).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def npy_hash(a):
    f = io.BytesIO()
    np.save(f, a, allow_pickle=False)
    return hashlib.sha256(f.getvalue()).hexdigest()


def assign(onsets, candidates, owners):
    """Inclusive +/-20 ms, closest candidate, then lowest group index."""
    result = np.full(len(onsets), -1, np.int32)
    for i, onset in enumerate(onsets):
        left = np.searchsorted(candidates, onset - RADIUS, side="left")
        right = np.searchsorted(candidates, onset + RADIUS, side="right")
        if left < right:
            distances = np.abs(candidates[left:right] - onset)
            result[i] = owners[left:right][distances == distances.min()].min()
        # Independently check search boundaries and tie resolution on every onset.
        distances = np.abs(candidates - onset)
        nearest = distances.min()
        expected = -1 if nearest > RADIUS else owners[distances == nearest].min()
        require(result[i] == expected, "nearest assignment disagrees with full search")
    return result


def features(on, off, pitch, strings, selected):
    """Intervals are [on, off): a note ending exactly at an onset has ended."""
    chosen = np.flatnonzero(selected)
    chosen = chosen[np.argsort(on[chosen], kind="stable")]
    count = len(chosen)
    result = dict(k=count, span_samples=-1, common_overlap_samples=-1,
                  overlapping_pairs=0, peak_current_notes=0, carried=0,
                  carried_20ms=0, carried_50ms=0, carried_100ms=0,
                  foreign_during_any_attack=0, previous_attack_gap_samples=-1,
                  bass=0, same_string_extra_attacks=0)
    if count == 0:
        return result
    a, b = on[chosen], off[chosen]
    first = a[0]
    result["span_samples"] = int(a[-1] - first)
    result["common_overlap_samples"] = int(max(0, b.min() - a.max()))
    overlap = np.maximum(a[:, None], a) < np.minimum(b[:, None], b)
    result["overlapping_pairs"] = int(np.triu(overlap, 1).sum())
    result["peak_current_notes"] = int(((a[:, None] <= a) & (b[:, None] > a)).sum(0).max())
    foreign = ~selected
    previous = foreign & (on < first)
    alive = previous & (off > first)
    result["carried"] = int(alive.sum())
    for ms in (20, 50, 100):
        result[f"carried_{ms}ms"] = int((alive & (on <= first - round(ms * FS / 1000))).sum())
    result["foreign_during_any_attack"] = int((foreign & ((on[:, None] < a) & (off[:, None] > a)).any(1)).sum())
    if previous.any():
        result["previous_attack_gap_samples"] = int(first - on[previous].max())
    result["bass"] = int(np.any(pitch[chosen] < 48))
    result["same_string_extra_attacks"] = count - len(set(strings[chosen]))
    return result


def composition(member):
    return str(member)[3:].rsplit("_", 1)[0]


def metrics(mask, k, predictions):
    n = int(mask.sum())
    out = {"n": n, "k_histogram": np.bincount(k[mask], minlength=7).tolist()}
    for arm, p in predictions.items():
        correct = int(np.sum(mask & (p == k)))
        out[arm] = dict(correct=correct, errors=n-correct,
                        accuracy=correct / n if n else None,
                        under=int(np.sum(mask & (p < k))),
                        over=int(np.sum(mask & (p > k))))
    a, b = (predictions[arm] for arm in ARMS)
    out["corrected"] = int(np.sum(mask & (a != k) & (b == k)))
    out["regressed"] = int(np.sum(mask & (a == k) & (b != k)))
    return out


def run(args):
    geometry = json.loads((args.geometry_dir / "report.json").read_text())
    config = json.loads(args.config.read_text())
    sources = {arm: json.loads((args.prediction_dir / arm / "report.json").read_text()) for arm in ARMS}
    data = {}
    for arm, report in sources.items():
        path = args.prediction_dir / arm / "epoch-12-validation.npz"
        require(digest(path) == PREDICTION_SHAS[arm] == report["checkpoints"]["12"]["predictions_sha256"], "prediction archive changed")
        require(report["status"] == "completed" and report["primary_epoch"] == 12, "not the completed fixed epoch")
        require(digest(args.config) == report["config_sha256"], "config changed")
        require(digest(args.geometry_dir / "report.json") == report["geometry_report_sha256"], "geometry report changed")
        with np.load(path, allow_pickle=False) as z:
            data[arm] = {name: z[name] for name in z.files}
    a, b = (data[arm] for arm in ARMS)
    for key in ("global_index", "member", "k"):
        np.testing.assert_array_equal(a[key], b[key])
    with np.load(args.geometry_dir / "rows.npz", allow_pickle=False) as z:
        members, starts = z["member"], z["start"]
    for key, arr in (("members", members), ("cluster_start_samples", starts)):
        require(npy_hash(arr) == geometry["bundle_fields"][key]["sha256"], "geometry rows changed: " + key)
    folds = np.array([config["member_folds"][str(m)] for m in members])
    ids, k = a["global_index"], a["k"]
    np.testing.assert_array_equal(ids, np.flatnonzero(folds == 0))
    np.testing.assert_array_equal(members[ids], a["member"])
    require(array_hash(ids) == sources[ARMS[0]]["val_indices_sha256"], "validation indices changed")
    require(digest(args.annotations) == ANNOTATION_SHA, "annotation archive changed")
    predictions = {arm: data[arm]["predicted"] for arm in ARMS}
    for arm in ARMS:
        np.testing.assert_array_equal(predictions[arm], np.argmax(data[arm]["probability"], axis=1))
    positions = {int(g): i for i, g in enumerate(ids)}
    allowed = set(a["member"].astype(str))
    rows = [None] * len(ids)
    details = [None] * len(ids)
    seen = set()
    total_events = unassigned = 0
    with zipfile.ZipFile(args.annotations) as archive:
        for record in geometry["tracks"]:
            member = record["member"]
            if member not in allowed:
                continue
            require(member not in seen, "duplicate track")
            seen.add(member)
            path = args.geometry_dir / record["timing_file"]
            require(digest(path) == record["timing_sha256"], "candidate timing changed")
            with np.load(path, allow_pickle=False) as z:
                group_ids, flat, offsets = z["global_index"], z["full_candidate_samples"], z["full_candidate_offsets"]
            np.testing.assert_array_equal(members[group_ids], np.repeat(member, len(group_ids)))
            np.testing.assert_array_equal(starts[group_ids], flat[offsets[:-1]])
            owners = np.repeat(np.arange(len(group_ids)), np.diff(offsets))
            order = np.argsort(flat, kind="stable")
            doc = json.loads(archive.read(member))
            events = []
            for annotation in doc["annotations"]:
                if annotation["namespace"] != "note_midi":
                    continue
                string = int(annotation["annotation_metadata"]["data_source"])
                for d in annotation["data"]:
                    events.append((round(d["time"] * FS), round((d["time"] + d["duration"]) * FS), int(np.rint(d["value"])), string))
            ev = np.asarray(events, np.int64)
            on, off, pitch, strings = ev.T
            require(np.all(off > on), "nonpositive annotated duration")
            assignment = assign(on, flat[order], owners[order])
            total_events += len(on)
            unassigned += int((assignment < 0).sum())
            for local, global_id in enumerate(group_ids):
                pos = positions[int(global_id)]
                chosen = assignment == local
                rows[pos] = features(on, off, pitch, strings, chosen)
                rows[pos]["active_at_group_start"] = int(((on <= starts[global_id]) & (off > starts[global_id])).sum())
                selected_indices = np.flatnonzero(chosen)
                first = on[chosen].min() if chosen.any() else starts[global_id]
                old_indices = np.flatnonzero(~chosen & (on < first) & (off > first))
                details[pos] = dict(assigned=ev[selected_indices].tolist(), carried=ev[old_indices].tolist())
    require(seen == allowed and len(seen) == 50, "incomplete validation population")
    require(all(row is not None for row in rows), "missing rows")
    f = {key: np.array([row[key] for row in rows], np.int64) for key in rows[0]}
    np.testing.assert_array_equal(f.pop("k"), k)
    require(total_events - unassigned == int(k.sum()), "assignment mass differs")
    poly = k >= 2
    positive = k >= 1
    comps = np.array([composition(m) for m in a["member"]])
    styles = np.array([str(m).rsplit("_", 1)[-1].split(".")[0] for m in a["member"]])
    gap_bins = np.digitize(f["previous_attack_gap_samples"], np.rint(np.array([0, 20, 50, 100, 200, 500])*FS/1000))
    onset_bins = np.digitize(f["span_samples"], np.rint(np.array([5, 10, 20, 40])*FS/1000))
    measure = lambda mask: metrics(mask, k, predictions)
    # Primary onset threshold 10 ms; sensitivity is always reported, not selected.
    predicates = {
        "all_current_notes_overlap": poly & (f["common_overlap_samples"] > 0),
        "some_current_notes_overlap": poly & (f["overlapping_pairs"] > 0) & (f["common_overlap_samples"] == 0),
        "no_current_notes_overlap": poly & (f["overlapping_pairs"] == 0),
        "onset_span_le_10ms": (f["span_samples"] >= 0) & (f["span_samples"] <= 441),
        "onset_span_gt_10ms": f["span_samples"] > 441,
        "carried_any": f["carried"] > 0,
        "carried_none": f["carried"] == 0,
        "carried_at_least_20ms_old": f["carried_20ms"] > 0,
        "no_carried_at_least_20ms_old": f["carried_20ms"] == 0,
        "same_string_multiple_attacks": f["same_string_extra_attacks"] > 0,
    }
    strata = {name: {"positive": measure(positive & mask), "poly": measure(poly & mask),
                     "by_k": {str(i): measure((k == i) & mask) for i in range(1, 7)}}
              for name, mask in predicates.items()}
    span_bins = {}
    for low, high in ((0, 5), (5, 10), (10, 20), (20, 40), (40, 80)):
        mask = poly & (f["span_samples"] > (round(low * FS / 1000) if low else -1)) & (f["span_samples"] <= round(high * FS / 1000))
        span_bins[f"{low}_to_{high}ms"] = {"all": measure(mask), "by_k": {str(i): measure(mask & (k == i)) for i in range(2, 7)}}
    require(sum(s["all"]["n"] for s in span_bins.values()) == int(poly.sum()), "onset span outside grouping range")

    def contrast(left, right, population, stratifiers):
        """Both rates get identical weights from their pooled supported strata."""
        groups = {}
        for i in np.flatnonzero(population & (left | right)):
            key = tuple(str(v[i]) for v in stratifiers)
            groups.setdefault(key, []).append(i)
        supported, excluded = [], []
        for key, indices in sorted(groups.items()):
            mask = np.zeros(len(k), bool)
            mask[indices] = True
            l, r = mask & left, mask & right
            record = dict(stratum=list(key), left=measure(l), right=measure(r))
            (supported if l.sum() >= 5 and r.sum() >= 5 else excluded).append(record)
        total = sum(s["left"]["n"] + s["right"]["n"] for s in supported)
        rates = {}
        for arm in ARMS:
            if total:
                lrate = sum((s["left"]["n"] + s["right"]["n"]) * s["left"][arm]["accuracy"] for s in supported) / total
                rrate = sum((s["left"]["n"] + s["right"]["n"]) * s["right"][arm]["accuracy"] for s in supported) / total
                rates[arm] = dict(left_accuracy=lrate, right_accuracy=rrate, right_minus_left_pp=100*(rrate-lrate))
        compact = lambda s: dict(stratum=s["stratum"], left_n=s["left"]["n"], right_n=s["right"]["n"],
                                 left_correct={arm: s["left"][arm]["correct"] for arm in ARMS},
                                 right_correct={arm: s["right"][arm]["correct"] for arm in ARMS})
        return dict(min_per_side_per_stratum=5, supported_rows=total, excluded_rows=sum(s["left"]["n"] + s["right"]["n"] for s in excluded),
                    supported_strata=[compact(s) for s in supported],
                    excluded_strata=[dict(stratum=s["stratum"], left_n=s["left"]["n"], right_n=s["right"]["n"]) for s in excluded], standardized=rates)

    contrasts = {}
    for name, left, right, population in (
        ("poly_spread_vs_close", predicates["onset_span_le_10ms"], predicates["onset_span_gt_10ms"], poly),
        ("poly_carried_vs_none", predicates["carried_none"], predicates["carried_any"], poly),
        ("k1_carried_vs_none", predicates["carried_none"], predicates["carried_any"], k == 1),
        ("poly_old_carried_vs_none", predicates["no_carried_at_least_20ms_old"], predicates["carried_at_least_20ms_old"], poly),
    ):
        contrasts[name] = {
            "unadjusted": {"left": measure(population & left), "right": measure(population & right)},
            "equal_k_mix": contrast(left, right, population, [k]),
            "equal_k_composition_arrangement_mix": contrast(left, right, population, [k, comps, styles]),
            "equal_k_track_mix": contrast(left, right, population, [k, a["member"]]),
            "equal_k_composition_arrangement_bass_mix": contrast(left, right, population, [k, comps, styles, f["bass"]]),
        }
        if "carried" in name:
            contrasts[name]["equal_k_composition_arrangement_previous_gap_mix"] = contrast(left, right, population, [k, comps, styles, gap_bins])
            contrasts[name]["equal_k_composition_arrangement_onset_span_mix"] = contrast(left, right, population, [k, comps, styles, onset_bins])

    sensitivity = {}
    for ms in (5, 10, 20, 30, 40):
        close = f["span_samples"] <= round(ms * FS / 1000)
        sensitivity[f"span_{ms}ms"] = dict(close=measure(poly & close), spread=measure(poly & ~close),
                                           equal_k_mix=contrast(close, ~close, poly, [k]))
    for age in (0, 20, 50, 100):
        carried = f["carried" if age == 0 else f"carried_{age}ms"] > 0
        sensitivity[f"carried_age_{age}ms"] = dict(poly_none=measure(poly & ~carried), poly_yes=measure(poly & carried),
            k1_none=measure((k == 1) & ~carried), k1_yes=measure((k == 1) & carried),
            poly_equal_k_mix=contrast(~carried, carried, poly, [k]))
    cross = {}
    for close in (True, False):
        for carried in (True, False):
            mask = poly & (predicates["onset_span_le_10ms"] == close) & (predicates["carried_any"] == carried)
            cross[f"{'close' if close else 'spread'}_{'carried' if carried else 'no_carried'}"] = {
                "all": measure(mask), "by_k": {str(i): measure(mask & (k == i)) for i in range(2, 7)}}
    previous_gap = {}
    gap = f["previous_attack_gap_samples"]
    for low, high in ((0, 20), (20, 50), (50, 100), (100, 200), (200, 100000)):
        mask = positive & (gap > round(low*FS/1000)) & (gap <= round(high*FS/1000))
        previous_gap[f"{low}_to_{high}ms"] = dict(positive=measure(mask), poly=measure(mask & poly), k1=measure(mask & (k == 1)))
    by_composition = {c: {"all": measure(comps == c), "poly": measure((comps == c) & poly),
        "strata": {name: measure((comps == c) & poly & mask) for name, mask in predicates.items()}} for c in sorted(set(comps))}
    by_arrangement = {style: {"all": measure(styles == style), "poly": measure((styles == style) & poly),
        "by_k_and_carried": {str(i): {"carried": measure((styles == style) & (k == i) & predicates["carried_any"]),
                                     "none": measure((styles == style) & (k == i) & predicates["carried_none"])} for i in range(1, 7)}} for style in sorted(set(styles))}

    examples = []
    # First by global row ID, rather than handpicking dramatic cases.
    selections = {"poly_overlap_failure": poly & predicates["all_current_notes_overlap"],
                  "poly_no_overlap": poly & predicates["no_current_notes_overlap"],
                  "poly_close_failure": poly & predicates["onset_span_le_10ms"],
                  "poly_spread_failure": poly & predicates["onset_span_gt_10ms"],
                  "poly_close_success": poly & predicates["onset_span_le_10ms"],
                  "poly_spread_success": poly & predicates["onset_span_gt_10ms"],
                  "single_on_old_note_failure": (k == 1) & predicates["carried_any"],
                  "single_without_old_note_failure": (k == 1) & predicates["carried_none"],
                  "single_on_old_note_success": (k == 1) & predicates["carried_any"],
                  "single_without_old_note_success": (k == 1) & predicates["carried_none"]}
    for name, mask in selections.items():
        failure = predictions[ARMS[0]] != k
        outcome = failure if "failure" in name else (~failure if "success" in name else np.ones(len(k), bool))
        selected = np.flatnonzero(mask & outcome)[:2]
        for pos in selected:
            examples.append(dict(category=name, global_index=int(ids[pos]), member=str(a["member"][pos]),
                group_start_seconds=float(starts[ids[pos]]/FS), true_k=int(k[pos]),
                predictions={arm: int(p[pos]) for arm, p in predictions.items()},
                features={key: int(val[pos]) for key, val in f.items()}, events=details[pos]))
    report = dict(
        audit="v273-polyphony-timing", run_id=36688979041, epoch=12,
        source_commit=sources[ARMS[0]]["source_sha"], training_performed=False, models_changed=False,
        scope="Existing internal validation only; 50 tracks, 5 compositions; no outer/Locked12 evaluation. Observational, not causal.",
        provenance=dict(annotation_sha256=ANNOTATION_SHA, geometry_report_sha256=digest(args.geometry_dir/"report.json"),
                        config_sha256=digest(args.config), prediction_sha256=PREDICTION_SHAS,
                        script_sha256=digest(__file__), validation_indices_sha256=array_hash(ids),
                        prediction_report_sha256={arm: digest(args.prediction_dir/arm/"report.json") for arm in ARMS}),
        checks=dict(validation_labels_recomputed=True, independent_nearest_assignment_checks=total_events,
                    assigned_notes=total_events-unassigned, unassigned_notes=unassigned, prediction_argmax_verified=True,
                    tracks=len(seen), compositions=len(set(comps)), outer_rows_evaluated=0),
        definitions=dict(sample_rate=FS, assignment_radius_samples=RADIUS,
            k="Number of assigned new annotated attacks; not the number of sounding notes. Original 0..6 labels unchanged.",
            intervals="Half-open [onset, offset), in rounded original-audio samples. Offsets and strings from note_midi annotations.",
            near_simultaneous="All assigned onsets span <=10 ms; timing sensitivity also at 5,20,30,40 ms.",
            overlap="Assigned notes have overlapping duration intervals; all-current overlap means positive common interval across all K.",
            carried="A note outside the assigned group starts strictly before the earliest assigned onset and ends strictly after it; includes unassigned annotations.",
            old_carried="Same, additionally started at least 20 ms earlier; age sensitivity 0,20,50,100 ms.",
            standardization="Identical pooled row weights for strata containing at least 5 examples on each side; unsupported rows excluded and enumerated. This does not establish causality.",
            previous_gap_bins="Previous foreign onset gap: no previous, [0,20),[20,50),[50,100),[100,200),[200,500),>=500 ms. Rounded sample boundaries.",
            onset_span_bins="Span: <5,[5,10),[10,20),[20,40),>=40 ms. Rounded sample boundaries.",
            event_example_columns=["onset_sample", "offset_sample", "rounded_midi", "string_data_source"]),
        all=measure(np.ones(len(k), bool)), positive=measure(positive), poly=measure(poly),
        by_k={str(i): measure(k == i) for i in range(7)}, strata=strata, span_bins=span_bins,
        contrasts=contrasts, sensitivity=sensitivity, span_by_carried=cross, previous_attack_gap=previous_gap,
        by_composition=by_composition, by_arrangement=by_arrangement, examples=examples,
        distribution=dict(span_samples_quantiles=np.quantile(f["span_samples"][poly], [0,.1,.25,.5,.75,.9,1]).tolist(),
            common_overlap_samples_quantiles=np.quantile(f["common_overlap_samples"][poly], [0,.1,.25,.5,.75,.9,1]).tolist(),
            carried_histogram_poly=dict(sorted(Counter(f["carried"][poly].astype(str)).items())),
            no_current_overlap_groups=int(np.sum(poly & predicates["no_current_notes_overlap"])),
            k0_with_annotated_activity_at_group_start=int(np.sum((k == 0) & (f["active_at_group_start"] > 0)))),
        limitations=["The original candidate-conditioned validation is not a balanced overlap/non-overlap benchmark.",
            "K is attack count. A correct K does not demonstrate correct pitch identity or note duration detection.",
            "Annotated offsets are a proxy for sounding duration; actual resonances may extend beyond annotations.",
            "This is one completed model pair, one seed, an already inspected internal split, and only five compositions.",
            "Event rows in one track are correlated; no row-independent statistical significance is asserted.",
            "Annotations outside any candidate assignment are not counted as missed notes in this Exact-K score."])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    if args.rows_output:
        np.savez_compressed(args.rows_output, global_index=ids, member=a["member"], k=k,
                            **predictions, **f, event_json=np.array([json.dumps(d) for d in details]))
    print(json.dumps({"all": report["all"], "poly": report["poly"], "checks": report["checks"],
                      "distribution": report["distribution"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("analysis/v273-native-paired-config.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path)
    run(parser.parse_args())
