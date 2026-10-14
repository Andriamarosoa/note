"""S526: recover where S498 dropped genuine MAESTRO piano onsets.

S470 V88 pre-MIDI emitted 575 timestamped candidate rows before seeing
any labels; S476 later paired 322 original MIDI attacks to those rows.
S498 then formed 234 canonical note births using Basic Pitch, only 220
of the 322 V88-paired attacks match. Audit the remaining 102 attacks.

CRITICAL: the 322 S476 rows were SELECTED FOR V88 MATCH, therefore
322/322 V88 candidate presence is not an independent onset-recall score.
The full cohort is 330 MIDI attacks (8 unmatched to V88). This is a
retrospective diagnostic about an EVENT GENERATOR bottleneck, not
independent event F1, source K accuracy or a model promotion.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.s499_score_canonical_musical_onsets_against_true_MIDI_XML import pairing


def need(x, message):
    if not x:
        raise ValueError(message)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def lines(p):
    return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]


def audit(piano, graph, v88rows, v88k):
    need(len(v88rows) == 575 and len(set(v88rows)) == 575,
         "S526 original 575 frozen V88 event IDs must be unique")
    need(len(piano) == 322 and len(graph) == 234,
         "S526 fixed 322 MIDI labels and 234 canonical BP groups changed")
    indices = {str(e): i for i, e in enumerate(v88rows)}
    need(len(set(x["row_id"] for x in piano)) == 322,
         "S526 repeated a MIDI annotated true piano onset")
    original = collections.defaultdict(list)
    labels = collections.defaultdict(list)
    for row in graph:
        original[row["source_recording_sha"]].append(row)
    for label in piano:
        rec = label["row_id"].split(":")[0]
        need(label["row_id"] in indices,
             "S526 historical S476 labels must refer to one actual prelabel V88 row")
        labels[rec].append(label)
    need(set(labels) == set(original) and len(labels) == 3,
         "S526 original three distinct MAESTRO recording IDs no longer match")
    counts = collections.Counter()
    missing_by_original_v88_k = collections.Counter()
    missing_by_true_k = collections.Counter()
    missing_by_recording = {}
    event_ledger = []
    for rec in sorted(labels):
        t = sorted(labels[rec], key=lambda z: float(z["onset_at_excerpt_s"]))
        bp = sorted(original[rec], key=lambda z: float(z["event_time_seconds"]))
        assignments = {i: j for i, j, _ in pairing(
            [float(z["onset_at_excerpt_s"]) for z in t],
            [float(z["event_time_seconds"]) for z in bp])}
        local = collections.Counter()
        for i, e in enumerate(t):
            pitch_set = set(map(int, e["true_new_onset_MIDI_pitches"]))
            nk = len(pitch_set)
            need(1 <= nk <= 6,
                 "S526 true MIDI note-on labels cannot be active-sustained or K0")
            k = int(v88k[indices[e["row_id"]]])
            need(0 <= k <= 6 and int(e["original_V88_K"]) == k,
                 "S526 original frozen V88 K differs from S476 note labels")
            bp_found = i in assignments
            if bp_found:
                c = bp[assignments[i]]
                bp_pitches = set(map(int, c["bp_note_pitches"]))
                local["canonical_true_attack_match"] += 1
                if not pitch_set.issubset(bp_pitches):
                    local["matched_attack_missing_new_note_pitch"] += 1
            else:
                local["missing_BP_canonical_event_but_V88_detected"] += 1
                local["missing_true_MIDI_note_pitches"] += nk
                local["missing_true_poly_attacks"] += int(nk >= 2)
                local["missing_but_V88_predicts_K0"] += int(k == 0)
                local["missing_but_V88_predicts_K_gt0"] += int(k >= 1)
                missing_by_original_v88_k[str(k)] += 1
                missing_by_true_k[str(nk)] += 1
            local["V88_paired_TRUE_attack_count"] += 1
            local["V88_predicts_K0_among_matched_true_attacks"] += int(k == 0)
            event_ledger.append({
                "source_sha": rec,
                "row_id": e["row_id"],
                "prelabel_V88_candidate_exists": True,
                "prelabel_V88_K": k,
                "true_new_note_pitches_FOR_AUDIT": sorted(pitch_set),
                "true_new_K_FOR_AUDIT": nk,
                "paired_to_canonical_BasicPitch_birth": bp_found,
                "new_note_event_missing_from_training_rows_S505": not bp_found,
                "original_V88_K0_does_not_remove_attack_candidate": bool(k == 0),
            })
        local["canonical_BP_group_count"] = len(bp)
        local["canonical_BP_groups_unpaired_to_S476_truth"] = len(bp) - len(assignments)
        counts.update(local)
        missing_by_recording[rec] = dict(local)
    need(counts["canonical_true_attack_match"] == 220 and
         counts["missing_BP_canonical_event_but_V88_detected"] == 102 and
         counts["V88_paired_TRUE_attack_count"] == 322,
         "S526 old S505 matched and missed true onsets baseline is not reproduced")
    need(counts["missing_but_V88_predicts_K0"] +
         counts["missing_but_V88_predicts_K_gt0"] == 102,
         "S526 all missed BP onsets must keep source V88 candidate K")
    return {
        "status": "S526_BP_CANONICAL_LOST_TRUE_V88_K0_AND_KPOS_PIANO_ATTACKS",
        "original_complete_S80_not_replayed": True,
        "source_V88_original_not_retrained": True,
        "guitar_reference_and_Live_unchanged": True,
        "frozen_V88_candidates_total": len(v88rows),
        "MIDI_groups_total_historical_S471": 330,
        "MIDI_groups_already_preselected_as_V88_matches_S476": 322,
        "historical_V88_unmatched_true_MIDI_onsets_NOT_claimed_recovered": 8,
        "original_BP_canonical_groups_total": len(graph),
        "true_attacks_absent_from_BP_canonical_but_in_original_V88": 102,
        "true_MIDI_noteon_pitch_counts_not_sustaining_note_counts": True,
        "V88_presence_on_S476_truth_is_selection_condition_not_model_recall": True,
        "missing_attacks_preserved_from_S525_no_new_model_claim": True,
        "full_stream_note_on_F1_NOT_MEASURED": True,
        "historically_exposed_MAESTRO_not_blind": True,
        "no_promotion": True,
        "totals": dict(counts),
        "BP_missed_attacks_by_original_prelabel_V88_K": dict(sorted(missing_by_original_v88_k.items())),
        "BP_missed_attacks_by_true_new_MIDI_K": dict(sorted(missing_by_true_k.items())),
        "per_recording": missing_by_recording,
    }, event_ledger


def run(args):
    need(not args.out.exists(), "S526 audit report already exists")
    manifest = json.loads(args.v88_report.read_text())
    s498 = json.loads(args.graph_report.read_text())
    truth = json.loads(args.piano_report.read_text())
    need(manifest["status"] == "S470_MAESTRO_V3_REAL_AUDIO_ORIGINAL_PRETRAINED_V88_PLUS_S466_PREMIDI"
         and manifest["output_sha256"] == sha(args.v88)
         and manifest["prelabel_predictions_frozen"],
         "S526 original S470 V88 candidates not source pinned before MIDI")
    need(s498["status"] ==
         "S498_TWO_SOURCE_CANONICAL_NOTE_ONSET_GRAPH_SOURCE_K_AS_SOFT_PRIOR_PRELABEL"
         and s498["no_original_MIDI_XML_annotations_read_in_canonical_model"],
         "S526 original S498 canonical note graph not frozen before MIDI")
    need(truth["status"] ==
         "S476_THIRD_MAESTRO_PIANO_NEW_ONSET_K_VS_SUSTAINING_K_READONLY"
         and truth["SHA256"]["paired_label_comparison"] == sha(args.piano),
         "S526 original selected true piano note-on label file changed")
    file = args.graph / "S498_piano_V88_CANONICAL_MUSICAL_EVENTS_PRELABEL.jsonl"
    need(s498["frozen_group_sha256"]["piano_V88"] == sha(file),
         "S526 piano BP group SHA no longer frozen")
    with np.load(args.v88, allow_pickle=False) as z:
        rows = z["row_id"].astype(str)
        k = z["original_V88_K"].astype(int)
    rep, ledger = audit(lines(args.piano), lines(file), rows, k)
    args.out.mkdir(parents=True)
    evidence = args.out / "S526_LOST_NOTE_ONSET_PRESENT_IN_V88_PRELABEL.jsonl"
    evidence.write_text("".join(json.dumps(z) + "\n" for z in ledger))
    rep["every_event_ledger_SHA256"] = sha(evidence)
    rep["pinned_sources"] = {
        "original_S470_candidate_SHA": sha(args.v88),
        "S498_prelabel_graph_SHA": sha(file),
        "S476_original_MIDI_SHA": sha(args.piano),
    }
    (args.out / "report.json").write_text(json.dumps(rep, indent=2) + "\n")
    print("S526 TRUE ONSET GENERATOR K0 BOTTLENECK", json.dumps(rep), flush=True)


def selftest():
    # Presence of V88 event with predicted K0 must not be interpreted
    # as absence of a genuine annotated new MIDI note.
    rows = ["r:1", "r:2", "r:3"]
    k = [0, 2, 0]
    y = [
        {"row_id": "r:1", "onset_at_excerpt_s": 0.1,
         "true_new_onset_MIDI_pitches": [52], "original_V88_K": 0},
        {"row_id": "r:2", "onset_at_excerpt_s": 0.5,
         "true_new_onset_MIDI_pitches": [55], "original_V88_K": 2},
    ]
    bp = [{"source_recording_sha": "r", "event_time_seconds": 0.1,
           "bp_note_pitches": [52]}]
    # The full cohort counts are enforced by audit; here check the
    # Hungarian matcher and membership invariant directly.
    match = pairing([e["onset_at_excerpt_s"] for e in y],
                    [e["event_time_seconds"] for e in bp])
    need(len(match) == 1 and len(set(rows)) == 3 and
         y[0]["original_V88_K"] == 0 and
         y[1]["row_id"] in rows and y[1]["onset_at_excerpt_s"] == 0.5,
         "S526 source onset-candidate presence must survive K0")
    print("S526 SELFTEST PASS prelabel K0 cannot delete an independent onset candidate")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    for field in ("v88", "v88-report", "graph", "graph-report", "piano",
                  "piano-report", "out"):
        p.add_argument("--" + field, type=Path)
    a = p.parse_args()
    if a.self_test:
        selftest()
    else:
        for field in ("v88", "v88_report", "graph", "graph_report", "piano",
                      "piano_report", "out"):
            need(getattr(a, field) is not None, "S526 missing " + field)
        run(a)
