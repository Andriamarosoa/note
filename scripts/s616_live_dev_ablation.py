"""S616-LIVE genuine URMP development ablation, **not** a test promotion.

Uses the exact S614 checkpoint, its native PCM frontend, the original held
development titles and true reference score. Neither train nor test annotations
are used for choosing a selection variant. No S80, S601 or original S614 code,
head, checkpoint or threshold is modified. Opened test data is never loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from causal_note.strict_live import SR, feature_track, load_heads
from causal_note.s616_live_guard import GuardParameters, candidate_variants, replay
from scripts.s613_prepare_strict_live import urmp_blob, labels, digest
from scripts.s613_train_strict_live import real_notes, event_score, offset_audit, summary, decode

ROOT = Path("analysis/evidence/v273-s613-strict-live")
MODEL = Path("model/live/strict_note/NOTE_strict_live.npz")
BASELINE_URMP = {"TP":243, "FP":149, "FN":256}
TEST_DISALLOWED = {"IDMT", "GuitarSet"}


def prepare_dev_source(spec, cache_root):
    # Pinned original URMP audio; old source manifests already contain exact SHA.
    blob = urmp_blob(spec["audio"], cache_root)
    rate, pcm = wavfile.read(io.BytesIO(blob))
    if pcm.dtype not in (np.dtype("int16"), np.dtype("int32")):
        raise ValueError("Unexpected pinned URMP audio dtype")
    audio = pcm.astype(np.float32) / (32768 if pcm.dtype == np.int16 else 2147483648)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if rate != SR:
        g = np.gcd(rate, SR)
        audio = resample_poly(audio, SR // g, rate // g).astype(np.float32)
    audio = audio[:int(spec["seconds"] * SR)].copy()
    ends, features = feature_track(audio)
    annotated = labels(spec, cache_root)
    truth = real_notes(annotated["notes"], len(audio)/SR)
    return dict(ends=ends, features=features, samples=len(audio), truth=truth,
                source_sha256=digest(blob), label_sha256=annotated["sha256"])


def aggregate(rows):
    return summary([r["score"] for r in rows])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError("S616 evidence is immutable")
    args.out.mkdir(parents=True)
    source_manifest = json.loads((ROOT/"protocol.json").read_text())
    chosen = [s for s in source_manifest["sources"]
              if s["split"] == "dev" and s["corpus"] == "URMP"]
    assert len(chosen) == 5 and len({s["composition"] for s in chosen}) == 5
    assert all(s["corpus"] not in TEST_DISALLOWED for s in chosen)
    frozen = json.loads((ROOT/"selection_frozen.json").read_text())
    checkpoint = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    assert checkpoint == frozen["model_sha256"], "S614 original weights were altered"
    assert digest((ROOT/"protocol.json").read_bytes()) == frozen["protocol_sha256"]
    heads = load_heads(MODEL, expected_sha256=checkpoint)
    cfg = heads.metadata["selection"]
    from causal_note.strict_live import Selection
    selection = Selection(**cfg)
    variants = candidate_variants()
    score_lists = {name: [] for name, _, _ in variants}
    per_source = []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        cache_root = Path(td)
        (cache_root/"data").mkdir()
        for spec in chosen:
            pack = prepare_dev_source(spec, cache_root)
            ends, x = pack["ends"], pack["features"]
            probabilities, counts = heads.predict(x)
            assert probabilities.shape == (len(ends), 49, 3)
            assert counts.shape == (len(ends), 7)
            # Comparison against the canonical S614 selector by events, not
            # just against an F1 number, guards against a subtle ABI difference.
            baseline_events = decode(ends, probabilities, counts, selection, pack["samples"])
            neutral_events = replay(ends, probabilities, counts, x, selection,
                                    GuardParameters(), pack["samples"])
            assert baseline_events == neutral_events, "The no-op ablation changed original S614 output"
            for name, params, release_frames in variants:
                individual_selection = replace(selection, release_frames=release_frames)
                events = replay(ends, probabilities, counts, x, individual_selection,
                                params, pack["samples"])
                scored, pairs = event_score(events, pack["truth"])
                offset = offset_audit(events, pack["truth"], pairs, pack["samples"])
                score_lists[name].append(dict(source=spec["name"], score=scored, offset=offset))
            per_source.append(dict(source=spec["name"], composition=spec["composition"],
                                   source_sha256=pack["source_sha256"],
                                   annotation_sha256=pack["label_sha256"],
                                   frames=len(ends), notes=len(pack["truth"])))
            print("S616 SOURCE", spec["name"], "notes",len(pack["truth"]),
                  "frames",len(ends), "elapsed_s",round(time.perf_counter()-started,1),flush=True)
    baseline_score = aggregate(score_lists["S614_baseline"])
    assert all(baseline_score[key] == expected for key, expected in BASELINE_URMP.items()), (
        "Baseline must reproduce pre-existing S614 URMP development results, got",
        baseline_score)
    rows = []
    for name, params, release_frames in variants:
        scores = score_lists[name]
        measure = aggregate(scores)
        offsets = Counter()
        for record in scores:
            offsets.update({k:record["offset"][k] for k in
                            ("matched_noteons_with_observed_reference_offset",
                             "offsets_within_max_50ms_or_20pct_duration",
                             "forced_or_missing_offsets")})
        denom = offsets["matched_noteons_with_observed_reference_offset"]
        rows.append(dict(name=name, params=dict(harmonic_penalty=params.harmonic_penalty,
                          attack_rescue=params.attack_rescue, release_frames=release_frames),
                         overall={k:measure[k] for k in ("TP","FP","FN","F1")},
                         offsets=dict(offsets, conditional_fraction=(
                             offsets["offsets_within_max_50ms_or_20pct_duration"] / denom if denom else 0)),
                         per_source={r["source"]:{"TP":r["score"]["TP"],"FP":r["score"]["FP"],
                                                  "FN":r["score"]["FN"],"F1":r["score"]["F1"]}
                                     for r in scores}))
    winner = max(rows,key=lambda a:a["overall"]["F1"])
    changed = winner["name"] != "S614_baseline"
    for r in rows:
        r["delta_F1_points"] = round(100*(r["overall"]["F1"] - baseline_score["F1"]), 5)
        r["delta_TP"] = r["overall"]["TP"] - baseline_score["TP"]
        r["delta_FP"] = r["overall"]["FP"] - baseline_score["FP"]
    report = dict(status="S616_LIVE_URMP_DEVELOPMENT_ONLY_NO_PROMOTION",
                  experiment="causal harmonic competition, rising-onset rescue, offset release loops",
                  original_S80_unchanged=True, original_S614_unchanged=True,
                  original_checkpoint_sha256=checkpoint,
                  protocol_sha256=frozen["protocol_sha256"],
                  test_annotations_or_events_read=False,
                  sources_previously_in_S614_development=True,
                  no_new_held_out_F1=True,
                  strict_causal_no_lookahead=True,
                  reference_baseline=baseline_score,
                  dev_sources=per_source, variants=rows,
                  best_dev_variant=winner["name"],
                  best_dev_F1=winner["overall"]["F1"],
                  dev_variant_better_than_baseline=changed,
                  official_promotion=False,
                  note="Selecting on these sources requires a different, unopened held-out group before promotion",
                  elapsed_seconds=round(time.perf_counter()-started,2))
    (args.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S616 VERIFIED DEV SUMMARY",json.dumps({
        "baseline":baseline_score,
        "best_dev":winner["name"],
        "best_dev_F1":winner["overall"]["F1"],
        "variants":[{k:r[k] for k in ("name","delta_F1_points","delta_TP","delta_FP")} for r in rows]
    },indent=2),flush=True)


if __name__ == "__main__":
    main()
