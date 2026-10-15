"""S615: post-freeze error taxonomy for S614 causal NOTE Live.

This is DESCRIPTIVE on an opened test. It does not train, tune thresholds,
replace a checkpoint, alter S80, or claim new held-out validation.
Run: python -m scripts.s615_postfreeze_live_diagnostics --out /tmp/s615.json
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


EVIDENCE = Path("analysis/evidence/v273-s613-strict-live")
SAME_NOTE_WINDOW = 0.080
NEAR_WINDOW = 0.180
HARMONIC_OFFSETS = {12, 19, 24}


def archive_json(root: Path, index: dict, name: str):
    meta = index[name]
    compressed = (root / meta["archive"]).read_bytes()
    assert hashlib.sha256(compressed).hexdigest() == meta["archive_sha256"], name
    raw = gzip.decompress(compressed)
    assert len(raw) == meta["original_bytes"], name
    assert hashlib.sha256(raw).hexdigest() == meta["original_sha256"], name
    return json.loads(raw)


def noteons(events):
    return [event for event in events if event["type"] == "note_on"]


def intervals(events):
    """Reconstruct emitted MIDI lifetimes; do not confuse panic with offsets."""
    active, result = {}, []
    for e in events:
        pitch = e["pitch"]
        if e["type"] == "note_on":
            if pitch in active:
                raise AssertionError("Unbalanced same-pitch NoteOn")
            active[pitch] = len(result)
            result.append({"pitch": pitch, "on": e["time"], "off": None, "off_reason": None})
        elif e["type"] == "note_off":
            if pitch not in active:
                raise AssertionError("Unbalanced NoteOff")
            row = result[active.pop(pitch)]
            row["off"] = e["time"]
            row["off_reason"] = e.get("reason")
    if active:
        raise AssertionError("Unclosed active notes")
    return result


def classify_fp(event, truth, matched_onsets):
    pitch, when = event["pitch"], event["time"]
    if any(e["pitch"] == pitch and abs(e["time"] - when) <= SAME_NOTE_WINDOW for e in matched_onsets):
        return "same_pitch_duplicate_near_matched_onset"
    nearby = [abs(t["time"] - when) for t in truth if t["pitch"] == pitch]
    if nearby and min(nearby) <= SAME_NOTE_WINDOW:
        return "same_pitch_multiple_emissions_near_truth"
    if nearby and min(nearby) <= NEAR_WINDOW:
        return "same_pitch_onset_outside_80ms"
    if any(abs(pitch - t["pitch"]) in HARMONIC_OFFSETS and
           t["time"] - SAME_NOTE_WINDOW <= when <= t["time"] + t["duration"] + SAME_NOTE_WINDOW
           for t in truth):
        return "harmonic_interval_coincidence_not_causation"
    return "other_unmatched_onset"


def classify_fn(note, predicted, held_intervals):
    pitch, when = note["pitch"], note["time"]
    distance = [abs(e["time"] - when) for e in predicted if e["pitch"] == pitch]
    if distance and min(distance) <= SAME_NOTE_WINDOW:
        return "same_pitch_competing_assignment"
    if distance and min(distance) <= NEAR_WINDOW:
        return "same_pitch_emitted_outside_80ms"
    if any(row["pitch"] == pitch and row["on"] < when and
           (row["off"] is None or row["off"] > when)
           for row in held_intervals):
        return "already_active_at_truth_rearticulation"
    return "no_same_pitch_onset_within_180ms"


def unmatched_novelty(selected_fps, baseline_fps, tolerance=0.035):
    """Descriptive same-pitch nearest-time pairing of false emissions."""
    available = set(range(len(baseline_fps)))
    retained = 0
    for p in selected_fps:
        options = [(abs(p["time"] - baseline_fps[i]["time"]), i)
                   for i in available if p["pitch"] == baseline_fps[i]["pitch"]
                   and abs(p["time"] - baseline_fps[i]["time"]) <= tolerance]
        if options:
            available.remove(min(options)[1])
            retained += 1
    return {"common_fp_approx": retained,
            "new_fp_approx": len(selected_fps) - retained,
            "removed_fp_approx": len(available)}


def analyze(root: Path):
    index = json.loads((root / "archived_ledgers.json").read_text())
    sel_audit = archive_json(root, index, "selected_individual_audit.json")
    sel_events = archive_json(root, index, "selected_predictions_before_labels.json")["predictions"]
    base_audit = archive_json(root, index, "baseline_individual_audit.json")
    base_events = archive_json(root, index, "baseline_predictions_before_labels.json")["predictions"]
    post = json.loads((root / "regression_audit.json").read_text())
    expected = json.loads((root / "selected_test_report.json").read_text())["results"]["80ms"]["overall"]
    protocol = json.loads((root / "protocol.json").read_text())
    testing = [s for s in protocol["sources"] if s["split"] == "test"]
    ids = {s["id"] for s in testing}
    assert ids == set(sel_audit) == set(base_audit) == set(sel_events) == set(base_events)
    fp_totals, fn_totals = Counter(), Counter()
    by_corpus = {}
    sources = []
    total = Counter()
    novelty_totals = Counter()
    gains = losses = 0
    for spec in testing:
        sid = spec["id"]
        original, current = base_audit[sid], sel_audit[sid]
        truth = current["truth"]
        assert truth == original["truth"]
        op, npred = noteons(base_events[sid]), noteons(sel_events[sid])
        old_pairs = original["80ms"]["pairs"]
        new_pairs = current["80ms"]["pairs"]
        old_pi, old_ti = {i for i, j in old_pairs}, {j for i, j in old_pairs}
        new_pi, new_ti = {i for i, j in new_pairs}, {j for i, j in new_pairs}
        assert len(new_pi) == len(new_ti) == len(new_pairs)
        assert len(old_pi) == len(old_ti) == len(old_pairs)
        gains += len(new_ti - old_ti)
        losses += len(old_ti - new_ti)
        selected_fp = [v for i, v in enumerate(npred) if i not in new_pi]
        old_fp = [v for i, v in enumerate(op) if i not in old_pi]
        current_matched = [npred[i] for i in new_pi]
        fp = Counter(classify_fp(e, truth, current_matched) for e in selected_fp)
        held = intervals(sel_events[sid])
        fn = Counter(classify_fn(n, npred, held) for i, n in enumerate(truth) if i not in new_ti)
        novelty = unmatched_novelty(selected_fp, old_fp)
        score = current["80ms"]["score"]
        assert score["TP"] == len(new_pairs)
        assert score["FP"] == len(selected_fp) and score["FN"] == len(truth) - len(new_pairs)
        assert sum(fp.values()) == score["FP"] and sum(fn.values()) == score["FN"]
        fp_totals.update(fp)
        fn_totals.update(fn)
        novelty_totals.update(novelty)
        total.update({k: score[k] for k in ("TP", "FP", "FN")})
        corpus = by_corpus.setdefault(spec["corpus"], {"FP": Counter(), "FN": Counter()})
        corpus["FP"].update(fp)
        corpus["FN"].update(fn)
        before = original["80ms"]["score"]
        sources.append({"source": spec["name"], "corpus": spec["corpus"],
                        "baseline_f1": before["F1"], "selected_f1": score["F1"],
                        "delta_tp": score["TP"] - before["TP"],
                        "delta_fp": score["FP"] - before["FP"],
                        "gained_true_notes": len(new_ti - old_ti),
                        "lost_true_notes": len(old_ti - new_ti),
                        "fp_taxonomy": dict(sorted(fp.items())),
                        "fn_taxonomy": dict(sorted(fn.items())), **novelty})
    assert all(total[k] == expected[k] for k in ("TP", "FP", "FN"))
    assert gains == post["gained_true_noteons"] == 150
    assert losses == post["lost_true_noteons"] == 28
    return {"status": "S615_POSTFREEZE_DESCRIPTIVE_ONLY_NO_MODEL_CHANGE",
            "test_is_open_not_new_independent_validation": True,
            "heuristics_not_proofs_of_acoustic_causation": True,
            "tolerances_seconds": {"onset_match": SAME_NOTE_WINDOW,
                                   "near": NEAR_WINDOW, "fp_similarity": 0.035},
            "selected": dict(total), "gained": gains, "lost": losses,
            "fp_classification": dict(sorted(fp_totals.items())),
            "fn_classification": dict(sorted(fn_totals.items())),
            "fp_difference_from_baseline": dict(novelty_totals),
            "by_corpus": {k: {side: dict(sorted(v[side].items())) for side in ("FP", "FN")}
                          for k, v in sorted(by_corpus.items())},
            "worst_fp_increase": sorted(sources, key=lambda x: x["delta_fp"], reverse=True)[:8],
            "sources": sources}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=EVIDENCE)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = analyze(args.root)
    if args.out.exists():
        raise FileExistsError("Do not overwrite prior diagnostics")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: result[k] for k in
                     ("status", "selected", "gained", "lost", "fp_classification",
                      "fn_classification", "fp_difference_from_baseline", "worst_fp_increase")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
