"""S632 fixed-protocol per-register causal rescue evaluation on real audio.

This audit never loads held-out TEST, never changes production heads or MIDI
and distinguishes recovered annotated NoteOn from extra false NoteOn.
All proposals require independently rising fundamental at the CURRENT frame.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np

from causal_note.register_rescue import propose, variants
from causal_note.strict_live import SR, Selection, CausalSelection, load_heads
from scripts.s616_live_dev_ablation import ROOT, MODEL
from scripts.s613_prepare_strict_live import download, digest
from scripts.s613_train_strict_live import decode, event_score, summary
from scripts.s626_train_sharp_native_onset import get_source
from scripts.s631_high_overlap_augmentation import category

CONTROL = {"GuitarSet": dict(TP=998, FP=585, FN=582),
           "URMP": dict(TP=243, FP=149, FN=256)}
REGISTER_BOUNDS = {
    "bass_E2_to_Dsharp4": lambda pitch: pitch <= 63,
    "mid_E4_to_Dsharp5": lambda pitch: 64 <= pitch <= 75,
    "high_E5_plus": lambda pitch: pitch >= 76,
}


def replay(row, config, policy):
    state = CausalSelection(config)
    events = []
    proposals = dict(bass=0, high=0, bass_emitted=0, high_emitted=0)
    for t, features, native, count in zip(row["ends"], row["x"], row["p"], row["k"]):
        corrected, choices = propose(features, native, state, policy, count)
        proposals["bass"] += int(choices["bass"].sum())
        proposals["high"] += int(choices["high"].sum())
        emitted = state.step(corrected, count, int(t), int(t))
        on = [e["pitch"] for e in emitted if e["type"] == "note_on"]
        proposals["bass_emitted"] += sum(bool(choices["bass"][p-40]) for p in on)
        proposals["high_emitted"] += sum(bool(choices["high"][p-40]) for p in on)
        events.extend(emitted)
    events.extend(state.panic(row["n"]))
    return events, proposals


def register_counts(row, events):
    score, pairs = event_score(events, row["truth"])
    matched_true = {j for _, j in pairs}
    matched_pred = {i for i, _ in pairs}
    predicted = [e for e in events if e["type"] == "note_on"]
    by = {}
    for name, bounds in REGISTER_BOUNDS.items():
        matched_truth = [j for j, note in enumerate(row["truth"]) if bounds(note["pitch"])]
        estimated = [i for i, event in enumerate(predicted) if bounds(event["pitch"])]
        by[name] = dict(
            true_notes=len(matched_truth),
            recovered=sum(j in matched_true for j in matched_truth),
            predicted_notes=len(estimated),
            false_positive=sum(i not in matched_pred for i in estimated),
        )
    overlaps = [j for j, note in enumerate(row["truth"])
                if category(note, row["truth"], j) == ("high_E5_plus", "overlap")]
    by["high_overlap"] = dict(
        true_notes=len(overlaps),
        recovered=sum(j in matched_true for j in overlaps),
        predicted_notes=0, false_positive=0,
    )
    return score, by


def merge_register(rows):
    by = {}
    for row in rows:
        for name, counts in row.items():
            dst = by.setdefault(name, dict(true_notes=0, recovered=0,
                                         predicted_notes=0, false_positive=0))
            for k, val in counts.items():
                dst[k] += val
    for v in by.values():
        v["recall"] = v["recovered"]/v["true_notes"] if v["true_notes"] else None
    return by


def run(out):
    if out.exists():
        raise FileExistsError("S632 output must be immutable")
    start = time.perf_counter()
    frozen = json.loads((ROOT/"selection_frozen.json").read_text())
    assert digest(MODEL.read_bytes()) == frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes()) == frozen["protocol_sha256"]
    protocol = json.loads((ROOT/"protocol.json").read_text())
    train_groups = {(s["corpus"], s["composition"]) for s in protocol["sources"]
                    if s["split"] == "train"}
    test_groups = {(s["corpus"], s["composition"]) for s in protocol["sources"]
                   if s["split"] == "test"}
    dev = [s for s in protocol["sources"] if s["split"] == "dev"
           and s["corpus"] in ("GuitarSet", "URMP")]
    assert len(dev) == 17 and len([s for s in dev if s["corpus"]=="URMP"]) == 5
    dev_groups = {(s["corpus"], s["composition"]) for s in dev}
    assert not dev_groups & train_groups and not dev_groups & test_groups
    heads = load_heads(MODEL, expected_sha256=frozen["model_sha256"])
    config = Selection(**heads.metadata["selection"])
    records = []
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root/"data").mkdir()
        z = protocol["guitar_archives"]
        for key, filename in (("GuitarSet_audio", "GuitarSet_audio.zip"),
                              ("GuitarSet_annotations", "GuitarSet_annotations.zip")):
            item = z[key]
            download(item["url"], root/"data"/filename, item["size"], item["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as archive:
            for spec in dev:
                row = get_source(spec, root, heads, archive if spec["corpus"]=="GuitarSet" else None)
                original_events = decode(row["ends"], row["p"], row["k"], config, row["n"])
                original_score, original_reg = register_counts(row, original_events)
                entries = {}
                for name, policy in variants():
                    events, opportunities = replay(row, config, policy)
                    score, regs = register_counts(row, events)
                    if name == "S614_control":
                        assert score["TP"]==original_score["TP"]
                        assert score["FP"]==original_score["FP"]
                        assert score["FN"]==original_score["FN"]
                        assert regs == original_reg
                        assert opportunities["bass"]==opportunities["high"]==0
                    entries[name] = dict(score=score, registers=regs, proposals=opportunities)
                records.append(dict(source=row["name"], corpus=row["corpus"],
                                    composition=row["composition"],
                                    source_sha256=row["source_sha256"], variants=entries))
                print("S632 DEV",row["corpus"],row["name"],len(records),"/",len(dev),flush=True)
    by_policy = {}
    for name, policy in variants():
        results = {}
        for corpus, control in CONTROL.items():
            these = [r["variants"][name] for r in records if r["corpus"]==corpus]
            score = summary([r["score"] for r in these])
            if name == "S614_control":
                assert all(score[k] == v for k,v in control.items()), (corpus,score)
            regs = merge_register([r["registers"] for r in these])
            opportunities = {key:sum(r["proposals"][key] for r in these)
                             for key in ("bass","high","bass_emitted","high_emitted")}
            results[corpus] = dict(score=score, registers=regs,
                                   proposals=opportunities)
        by_policy[name] = dict(policy=asdict(policy), corpus=results)
        for corpus in CONTROL:
            original = by_policy["S614_control"]["corpus"][corpus]["score"]
            current = results[corpus]["score"]
            results[corpus]["delta_F1_points"] = 100*(current["F1"]-original["F1"])
            results[corpus]["delta_TP"] = current["TP"] - original["TP"]
            results[corpus]["delta_FP"] = current["FP"] - original["FP"]
        print("S632 POLICY", name,
              "G", round(results["GuitarSet"]["delta_F1_points"],3),
              "U", round(results["URMP"]["delta_F1_points"],3), flush=True)
    # This is an audit, not automatic promotion based on historical DEV.
    report = dict(
        status="S632_DUAL_REGISTER_RESCUE_CAUSAL_FROZEN_DEV_ONLY",
        model_sha256=frozen["model_sha256"],
        original_S80_and_S614_preserved=True,
        no_future_audio=True, no_inference_latency_added_by_design=True,
        dev_historically_exposed=True, test_never_loaded=True,
        candidates_not_promoted=True,
        source_count=len(records), candidate_count=len(by_policy),
        policies=by_policy, sources=records,
        elapsed_seconds=round(time.perf_counter()-start,2),
    )
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S632 SUMMARY", json.dumps({k:report[k] for k in
        ("status","source_count","candidate_count","policies","elapsed_seconds")},
        indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    run(a.out)
