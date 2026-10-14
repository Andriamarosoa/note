"""S525 retrospective physical-onset rescue audit (no model refit).

Match the PRE-ANNOTATION independent S513 raw-acoustic peaks against the
same 45 guitar XML + 322 piano new-note MIDI attack labels used for S505.
Compare their one-to-one event recall to the frozen S498 Basic Pitch
canonical birth graph. In particular, investigate the 102 piano attacks
without a S498 event, and whether top-6/top-12 raw pitches COULD cover
their actual note identities (oracle coverage, never deployment score).

Both evaluated corpora were historically inspected. Physical groups not
paired to an annotation are UNKNOWN, not confirmed false positives.
Full-stream note-on F1, promotion, and online latency are NOT measured.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.s499_score_canonical_musical_onsets_against_true_MIDI_XML import pairing


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def score_source(truth, canonical, acoustic, domain):
    """Independent one-to-one matching; no true pitch used to choose any peak."""
    time_key = "original_XML_anchor" if domain == "guitar_S80" else "onset_at_excerpt_s"
    pitch_key = "true_active" if domain == "guitar_S80" else "true_new_onset_MIDI_pitches"
    rec_key = "source_wav" if domain == "guitar_S80" else "row_id"
    selected = collections.defaultdict(list)
    old = collections.defaultdict(list)
    phys = collections.defaultdict(list)
    for t in truth:
        rec = str(t[rec_key]) if domain == "guitar_S80" else str(t[rec_key]).split(":")[0]
        selected[rec].append(t)
    for c in canonical:
        old[c["source_recording_sha"]].append(c)
    pairs = collections.defaultdict(dict)
    for row in acoustic:
        ident = row["WAV_filename_SHA256"]
        if row["policy"] in ("top6", "top12"):
            pairs[(ident, row["physical_event_id"])][row["policy"]] = row
    for (rec, identifier), choices in pairs.items():
        need(set(choices) == {"top6", "top12"},
             "S525 physical peak must have immutable top6 and top12")
        a, b = choices["top6"], choices["top12"]
        need(abs(a["independent_physical_onset_s"] -
                 b["independent_physical_onset_s"]) < 1e-9,
             "S525 mixed physical times from different audio peaks")
        need(set(map(int, a["candidate_MIDI"])).issubset(
             set(map(int, b["candidate_MIDI"]))),
             "S525 top6 should be nested in top12")
        phys[rec].append(a | {"top12_MIDI": b["candidate_MIDI"]})
    # A source with ZERO physical peaks is a detector miss, not an invalid
    # recording: retain it as zero onset recall. The BP graph and annotations
    # must still cover every original recording identity.
    need(set(selected) == set(old) and set(phys).issubset(set(selected)),
         "S525 frozen raw event identities must not introduce unrelated sources")
    audit = collections.Counter()
    per_rec = {}
    per_k = collections.defaultdict(collections.Counter)
    ledger = []
    for rec in sorted(selected):
        t = sorted(selected[rec], key=lambda z: float(z[time_key]))
        bp = sorted(old[rec], key=lambda z: float(z["event_time_seconds"]))
        raw = sorted(phys[rec], key=lambda z: float(z["independent_physical_onset_s"]))
        bpm = {i: j for i, j, _ in pairing(
            [float(z[time_key]) for z in t],
            [float(z["event_time_seconds"]) for z in bp])}
        pcm = {i: j for i, j, _ in pairing(
            [float(z[time_key]) for z in t],
            [float(z["independent_physical_onset_s"]) for z in raw])}
        tally = collections.Counter()
        for i, truthrow in enumerate(t):
            true = set(map(int, truthrow[pitch_key]))
            k = len(true)
            bp_hit = i in bpm
            physical_hit = i in pcm
            cover6 = cover12 = top1 = False
            any6 = any12 = 0
            if physical_hit:
                p = raw[pcm[i]]
                options6 = set(map(int, p["candidate_MIDI"]))
                options12 = set(map(int, p["top12_MIDI"]))
                cover6 = true.issubset(options6)
                cover12 = true.issubset(options12)
                top1 = (bool(p["candidate_MIDI"]) and
                        int(p["candidate_MIDI"][0]) in true)
                any6 = len(true & options6)
                any12 = len(true & options12)
            metrics = {
                "true_attacks": 1,
                "canonical_matched": int(bp_hit),
                "physical_matched": int(physical_hit),
                "both_matched": int(bp_hit and physical_hit),
                "neither_matched": int(not bp_hit and not physical_hit),
                "raw_rescues_BP_missing_onset": int(not bp_hit and physical_hit),
                "BP_matches_raw_misses": int(bp_hit and not physical_hit),
                "full_true_pitch_set_in_top6_ORACLE": int(physical_hit and cover6),
                "full_true_pitch_set_in_top12_ORACLE": int(physical_hit and cover12),
                "physical_first_pitch_correct": int(physical_hit and top1),
                "true_pitch_candidates_top6": any6,
                "true_pitch_candidates_top12": any12,
                "total_true_note_pitches": k,
                "rescued_true_pitches_top6": any6 if not bp_hit else 0,
                "rescued_true_pitches_top12": any12 if not bp_hit else 0,
                "rescued_full_true_chord_top6_ORACLE": int(not bp_hit and physical_hit and cover6),
                "rescued_full_true_chord_top12_ORACLE": int(not bp_hit and physical_hit and cover12),
            }
            for key, value in metrics.items():
                tally[key] += value
                per_k[str(k)][key] += value
            ledger.append({
                "source_recording_sha": rec,
                "corpus": domain,
                "original_annotation_id": str(truthrow.get("row_id", i)),
                "true_attack_time_FOR_AUDIT": float(truthrow[time_key]),
                "true_note_pitches_FOR_AUDIT": sorted(true),
                "canonical_matched": bp_hit,
                "independent_physical_matched": physical_hit,
                "independent_top6_pitches_PREANNOTATION":
                    raw[pcm[i]]["candidate_MIDI"] if physical_hit else [],
                "independent_top12_pitches_PREANNOTATION":
                    raw[pcm[i]]["top12_MIDI"] if physical_hit else [],
                "BP_missing_but_raw_physical_matched": not bp_hit and physical_hit,
                "top6_complete_pitch_ORACLE_not_model_output": bool(cover6),
                "top12_complete_pitch_ORACLE_not_model_output": bool(cover12),
            })
        tally["BP_original_canonical_candidates"] = len(bp)
        tally["physical_candidate_groups"] = len(raw)
        tally["unknown_unmatched_physical_groups"] = len(raw) - len(pcm)
        tally["unknown_unmatched_canonical_groups"] = len(bp) - len(bpm)
        need(tally["unknown_unmatched_physical_groups"] >= 0,
             "S525 physical onset matcher repeated one candidate")
        per_rec[rec] = dict(tally)
        audit.update(tally)
    return dict(audit), per_rec, {k: dict(v) for k, v in sorted(per_k.items())}, ledger


def run(a):
    need(not a.out.exists(), "S525 immutable audit already exists")
    g = json.loads(a.graph_report.read_text())
    raw = json.loads(a.generator_report.read_text())
    guitar = json.loads(a.guitar_report.read_text())
    piano = json.loads(a.piano_report.read_text())
    need(g["status"] ==
         "S498_TWO_SOURCE_CANONICAL_NOTE_ONSET_GRAPH_SOURCE_K_AS_SOFT_PRIOR_PRELABEL"
         and g["no_original_MIDI_XML_annotations_read_in_canonical_model"],
         "S525 wrong frozen Basic Pitch canonical graph")
    need(raw["status"] ==
         "S511_NEW_INDEPENDENT_RAW_HARMONIC_PITCH_ONSET_CANDIDATES_PREMIDI"
         and raw["original_WAVs"] == 8
         and raw["no_XML_or_MIDI_ground_truth_read_in_training_or_inference"]
         and raw["candidates_sha256"] == sha(a.generator),
         "S525 physical S513 generator not frozen before MIDI/XML")
    need(guitar["status"] == "S488_PAIRED_IDMT_45_TRUE_GUITAR_ATTACKS_MULTISOURCE_BASIC_PITCH"
         and guitar["matched_rows_SHA256"] == sha(a.guitar)
         and piano["status"] ==
         "S476_THIRD_MAESTRO_PIANO_NEW_ONSET_K_VS_SUSTAINING_K_READONLY"
         and piano["SHA256"]["paired_label_comparison"] == sha(a.piano),
         "S525 original known-positive annotations not verified")
    independent = records(a.generator)
    result = {}
    everything = []
    for name, file, n, canonical_count, matched_count in (
        ("guitar_S80", a.guitar, 45, 60, 45),
        ("piano_V88", a.piano, 322, 234, 220),
    ):
        path = a.graph / ("S498_" + name + "_CANONICAL_MUSICAL_EVENTS_PRELABEL.jsonl")
        need(g["frozen_group_sha256"][name] == sha(path),
             "S525 changed original pre-MIDI canonical event groups")
        t = records(file)
        groups = records(path)
        r = [z for z in independent if z["source"] == name]
        need(len(t) == n and len(groups) == canonical_count,
             "S525 expected original 45+322 historical true onsets")
        overall, rec, per_k, ledger = score_source(t, groups, r, name)
        need(overall["canonical_matched"] == matched_count,
             "S525 original S505 canonical 45/220 matched-onset baseline changed")
        need(overall["unknown_unmatched_canonical_groups"] == canonical_count - matched_count,
             "S525 frozen S498 matching drifted")
        result[name] = dict(overall=overall, per_recording=rec, by_true_new_or_active_K=per_k)
        everything.extend(ledger)
        print("S525 REPLAY ORIGINAL UNMODIFIED TRUE ONSET RESCUE", json.dumps({
            "domain": name, "overall": overall}), flush=True)
    a.out.mkdir(parents=True)
    ledgerfile = a.out / "S525_EVERY_PHYSICAL_ONSET_RESCUE_VS_FROZEN_BP.jsonl"
    ledgerfile.write_text("".join(json.dumps(z) + "\n" for z in everything))
    report = {
        "status": "S525_INDEPENDENT_PHYSICAL_ONSET_RESCUE_AUDIT_PRELABEL_S513_VS_BP_S498",
        "evaluated_guitar_XML_positive_attacks": 45,
        "evaluated_piano_MIDI_new_note_attacks": 322,
        "other_MAESTRO_true_attacks_outside_S476_matched_322_not_scored": 8,
        "piano_102_BP_missing_known_true_onsets_preserved": True,
        "unmatched_physical_onsets_not_verified_false_positives": True,
        "top6_top12_true_note_coverage_oracle_not_model_accuracy": True,
        "retrospective_historically_exposed_recordings_NOT_blind": True,
        "cannot_claim_full_stream_note_onset_precision_or_F1": True,
        "no_retraining_no_threshold_reselection": True,
        "original_S80_ExactK_and_Live_unchanged": True,
        "no_promotion": True,
        "by_source": result,
        "every_attack_SHA256": sha(ledgerfile),
        "source_SHA256": {
            "S513_audio_only_predictions": sha(a.generator),
            "S498_graph_prelabel": sha(a.graph_report),
            "S488_original_IDMT_XML": sha(a.guitar),
            "S476_original_MAESTRO_MIDI": sha(a.piano),
        },
    }
    (a.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def selftest():
    truth = [
        dict(row_id="x:1", onset_at_excerpt_s=.10, true_new_onset_MIDI_pitches=[52, 55]),
        dict(row_id="x:2", onset_at_excerpt_s=.50, true_new_onset_MIDI_pitches=[59]),
    ]
    baseline = [dict(source_recording_sha="x", event_time_seconds=.10)]
    oldphys = dict(WAV_filename_SHA256="x", physical_event_id="x:50",
                   independent_physical_onset_s=.50, candidate_MIDI=[59, 71])
    phys = [dict(oldphys, policy="top6"),
            dict(oldphys, policy="top12", candidate_MIDI=[59, 71, 83])]
    s, _, k, ledger = score_source(truth, baseline, phys, "piano_V88")
    need(s["canonical_matched"] == 1 and
         s["physical_matched"] == 1 and
         s["raw_rescues_BP_missing_onset"] == 1 and
         s["rescued_full_true_chord_top6_ORACLE"] == 1 and
         len(ledger) == 2 and k["1"]["true_attacks"] == 1,
         "S525 rescue must be physical NEW onset, not an old BP event")
    print("S525 SELFTEST PASS original event recall distinct from rescued note pitch oracle")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    for k in ("graph", "graph-report", "generator", "generator-report",
              "guitar", "guitar-report", "piano", "piano-report", "out"):
        ap.add_argument("--" + k, type=Path)
    args = ap.parse_args()
    if args.self_test:
        selftest()
    else:
        for name in ("graph", "graph_report", "generator", "generator_report",
                     "guitar", "guitar_report", "piano", "piano_report", "out"):
            need(getattr(args, name) is not None, "S525 missing argument " + name)
        run(args)
