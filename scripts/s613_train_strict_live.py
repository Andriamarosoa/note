"""Development loops for new native causal NOTE heads, then sealed evaluation.

All model/threshold selection uses train/dev compositions only. The test
command writes emissions before the score command is allowed to open labels.
Actual emission samples, not retrospectively shifted attacks, are scored.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from causal_note.strict_live import (
    SR, HOP, PITCHES, SCHEMA, count_features, NativeLiveHeads, load_heads,
    CausalSelection, Selection, StrictLiveStream,
)
from scripts.s613_prepare_strict_live import save, labels, digest


def targets(ends, notes):
    t = ends / SR
    y = np.zeros((len(ends), 49, 3), np.float32)
    for note in notes:
        if note["pitch"] not in PITCHES:
            continue
        start, stop = note["time"], note["time"] + note["duration"]
        p = note["pitch"] - 40
        y[(t >= start) & (t < start + .060), p, 0] = 1
        # A labelled past state (20ms) accommodates acoustic build-up. It is
        # a training target, NOT future audio or a runtime buffering timer.
        y[(t >= start + .020) & (t < stop + .020), p, 1] = 1
        y[(t >= stop) & (t < stop + .060), p, 2] = 1
    k = np.minimum(y[..., 1].sum(-1), 6).astype(np.int64)
    return y, k


def load_source(root, spec):
    stem = root / "cache" / spec["id"]
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as z:
        ends, x = z["ends"], z["features"]
    proof = json.loads(stem.with_suffix(".json").read_text())
    return ends, x, proof


def real_notes(notes, seconds):
    # Exact same-time/same-pitch unisons collapse, matching a MIDI channel.
    found = {}
    for n in notes:
        if 0 <= n["time"] < seconds and 40 <= n["pitch"] <= 88:
            key = (n["time"], n["pitch"])
            found.setdefault(key, dict(n))
            found[key]["duration"] = max(found[key]["duration"], n["duration"])
    return sorted(found.values(), key=lambda n: (n["time"], n["pitch"]))


def event_score(events, notes, tolerance=.080):
    predicted = [e for e in events if e["type"] == "note_on"]
    matched = []
    for pitch in PITCHES:
        pi = [i for i, n in enumerate(predicted) if n["pitch"] == pitch]
        ti = [i for i, n in enumerate(notes) if n["pitch"] == pitch]
        if not pi or not ti:
            continue
        distance = np.abs(np.asarray([predicted[i]["time"] for i in pi])[:, None] - np.asarray([notes[i]["time"] for i in ti]))
        # Cost penalty dominates all valid costs, yielding maximum cardinality
        # among admissible one-to-one pairs, then minimum timing error.
        cost = np.where(distance <= tolerance, distance, 1e6)
        a, b = linear_sum_assignment(cost)
        matched.extend((pi[i], ti[j]) for i, j in zip(a, b) if distance[i, j] <= tolerance)
    tp = len(matched)
    delays = [(predicted[i]["time"] - notes[j]["time"]) * 1000 for i, j in matched]
    return dict(TP=tp, FP=len(predicted) - tp, FN=len(notes) - tp,
                F1=2 * tp / max(1, len(predicted) + len(notes)),
                latency_ms=delays), matched


def summary(scores):
    count = Counter()
    delays = []
    for score in scores:
        count.update({k: score[k] for k in ("TP", "FP", "FN")})
        delays.extend(score["latency_ms"])
    out = dict(count)
    out["F1"] = 2 * count["TP"] / max(1, 2 * count["TP"] + count["FP"] + count["FN"])
    out["matched_emission_delay_ms"] = dict(zip(("p10", "median", "p95"), np.percentile(delays, [10, 50, 95]).tolist())) if delays else None
    return out


def offset_audit(events, truth, pairs, seconds):
    """Offsets conditional on correct NoteOn; EOF panic is never a success."""
    active, intervals = {}, []
    for event in events:
        pitch = event["pitch"]
        if event["type"] == "note_on":
            active[pitch] = len(intervals)
            intervals.append(dict(on=event))
        elif pitch in active:
            intervals[active.pop(pitch)]["off"] = event
    eligible, correct, forced, errors = 0, 0, 0, []
    for pred, reference in pairs:
        target = truth[reference]
        expected = target["time"] + target["duration"]
        if expected >= seconds:
            continue  # right-censored reference; offset not observed in clip
        eligible += 1
        off = intervals[pred].get("off")
        if off is None or off["reason"] in ("stream_end", "shutdown", "stream_error"):
            forced += 1
            continue
        error = off["time"] - expected
        errors.append(error * 1000)
        correct += abs(error) <= max(.05, .2 * target["duration"])
    return dict(matched_noteons_with_observed_reference_offset=eligible,
                offsets_within_max_50ms_or_20pct_duration=int(correct),
                forced_or_missing_offsets=forced,
                conditional_offset_accuracy=correct / max(1, eligible),
                acoustic_offset_error_ms=errors)


def decode(ends, p, k, selection, total_samples=None):
    stream = CausalSelection(selection)
    events = []
    for end, probability, count in zip(ends, p, k):
        events.extend(stream.step(probability, count, int(end), int(end)))
    events.extend(stream.panic(int(total_samples if total_samples is not None else ends[-1])))
    return events


def build_training(root, protocol):
    rng = np.random.default_rng(613)
    xs, ys, ws, cx, cy = [], [], [], [], []
    proof = []
    groups = {split: {(s["corpus"], s["composition"]) for s in protocol["sources"] if s["split"] == split}
              for split in ("train", "dev", "test")}
    assert not (groups["train"] & groups["dev"] or groups["train"] & groups["test"] or groups["dev"] & groups["test"])
    for spec in protocol["sources"]:
        if spec["split"] != "train":
            continue
        assert not (spec["corpus"] == "GuitarSet" and spec["name"].startswith("05_"))
        ends, x, meta = load_source(root, spec)
        y, k = targets(ends, meta["labels"]["notes"])
        cx.append(count_features(x[::4])); cy.append(k[::4])
        # Keep all attacks/offsets, half of sustains, 8% of background; inverse
        # sampling weights preserve the full-stream negative distribution.
        rate = np.where(np.maximum(y[..., 0], y[..., 2]) > 0, 1., np.where(y[..., 1] > 0, .5, .08))
        chosen = rng.random(rate.shape) < rate
        xs.append(x[chosen]); ys.append(y[chosen]); ws.append((1 / rate[chosen]).astype(np.float32))
        proof.append(dict(id=spec["id"], name=spec["name"], composition=spec["composition"],
                          corpus=spec["corpus"], frames=len(x), selected_pitch_rows=int(chosen.sum()),
                          audio_sha256=meta["audio_sha256"], label_sha256=meta["labels"]["sha256"]))
        print("TRAIN SOURCE", spec["name"], int(chosen.sum()), flush=True)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ws), np.concatenate(cx), np.concatenate(cy), proof


def train(root, out):
    import torch
    from torch import nn
    torch.set_num_threads(3)
    torch.manual_seed(613)
    protocol = json.loads((root / "protocol.json").read_text())
    if out.exists():
        raise FileExistsError("Refusing to replace a training experiment")
    out.mkdir(parents=True)
    x, y, weight, cx, cy, proof = build_training(root, protocol)
    save(out / "training_sources.json", proof)
    print("TRAIN ROWS", x.shape, cx.shape, flush=True)

    def model(dim, output):
        return nn.Sequential(nn.Linear(dim, 96), nn.ReLU(), nn.Linear(96, 48), nn.ReLU(), nn.Linear(48, output))

    def normal(values):
        mean = values.mean(0, dtype=np.float64).astype(np.float32)
        scale = np.maximum(values.std(0, dtype=np.float64), .05).astype(np.float32)
        return mean, scale

    cm, cs = normal(cx)
    count = model(cx.shape[1], 7)
    cxt, cyt = torch.from_numpy((cx - cm) / cs), torch.from_numpy(cy)
    optimizer = torch.optim.Adam(count.parameters(), lr=.002)
    for epoch in range(24):
        order = torch.randperm(len(cx))
        for ids in order.split(2048):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(count(cxt[ids]), cyt[ids])
            loss.backward(); optimizer.step()
        if epoch % 6 == 5:
            print("COUNT EPOCH", epoch + 1, float(loss), flush=True)
    # The pitch heads are explicitly conditioned on predicted K probabilities,
    # never on true K. Half the training K inputs are softened for robustness.
    cp = []
    # Replay the source sampling without retaining all feature matrices.
    rng = np.random.default_rng(613)
    for spec in protocol["sources"]:
        if spec["split"] != "train":
            continue
        ends, features, meta = load_source(root, spec)
        target, _ = targets(ends, meta["labels"]["notes"])
        rate = np.where(np.maximum(target[..., 0], target[..., 2]) > 0, 1., np.where(target[..., 1] > 0, .5, .08))
        chosen = rng.random(rate.shape) < rate
        with torch.no_grad():
            prob = torch.softmax(count(torch.from_numpy((count_features(features) - cm) / cs)), -1).numpy()
        cp.append(np.broadcast_to(prob[:, None], (len(features), 49, 7))[chosen].copy())
    cp = np.concatenate(cp)
    softened = np.random.default_rng(614).random(len(cp)) < .5
    cp[softened] = .7 * cp[softened] + .3 / 7
    x = np.concatenate((x, cp), axis=1)
    del cp, features, cx
    pm, ps = normal(x)
    xt, yt, wt = torch.from_numpy((x - pm) / ps), torch.from_numpy(y), torch.from_numpy(weight / weight.mean())
    del x, y, weight
    pitch = model(xt.shape[1], 3)
    optimizer = torch.optim.Adam(pitch.parameters(), lr=.0015)
    pos_weight = torch.tensor([6., 1.5, 6.])
    versions = []

    def export(epoch):
        arrays = dict(schema=np.asarray(SCHEMA), metadata=np.asarray(json.dumps(dict(epoch=epoch, selection=asdict(Selection()),
                     protocol_sha256=digest((root / "protocol.json").read_bytes()), training_source_count=len(proof),
                     heads=["native active K0..6", "per-pitch attack", "per-pitch sustain", "per-pitch release"],
                     no_BP_or_V88_inputs=True, no_postattack_buffer=True))))
        for name, network, mean, scale in (("count", count, cm, cs), ("pitch", pitch, pm, ps)):
            arrays[name + ".mean"] = mean; arrays[name + ".scale"] = scale
            for i, layer in enumerate((network[0], network[2], network[4])):
                arrays[f"{name}.w{i}"] = layer.weight.detach().numpy().T.copy()
                arrays[f"{name}.b{i}"] = layer.bias.detach().numpy().copy()
        path = out / f"heads_epoch{epoch:02d}.npz"
        np.savez_compressed(path, **arrays)
        versions.append(str(path))

    for epoch in range(28):
        before = time.perf_counter()
        pitch.train()
        order = torch.randperm(len(xt))
        total = 0.
        for ids in order.split(4096):
            optimizer.zero_grad(set_to_none=True)
            logits = pitch(xt[ids])
            loss = (nn.functional.binary_cross_entropy_with_logits(logits, yt[ids], reduction="none", pos_weight=pos_weight) * wt[ids, None]).mean()
            loss.backward(); optimizer.step(); total += float(loss) * len(ids)
        print("PITCH EPOCH", epoch + 1, "loss", total / len(xt), "seconds", time.perf_counter() - before, flush=True)
        if epoch + 1 in (3, 8, 16, 28):
            export(epoch + 1)
    save(out / "trained_checkpoints.json", versions)
    tune(root, out, protocol, versions)


def tune(root, out, protocol, versions, refinement=False):
    assert not (out / "test_predictions_before_labels.json").exists(), "No tuning after test inference"
    configs = [Selection(onset=on, active_on=active, active_off=.25, onset_reset=min(.2, on * .5),
                         release_frames=5, refractory_frames=10, count_confidence=kc)
               for on, active, kc in ((.3, .3, 1.1), (.5, .3, 1.1), (.7, .3, 1.1),
                                     (.3, .5, 1.1), (.5, .5, 1.1), (.7, .5, 1.1),
                                     (.3, .3, .8), (.5, .3, .8))]
    if refinement:
        configs = [Selection(onset=on, active_on=active, confirmation_frames=confirm,
                             restrike_onset=restrike, refractory_frames=refractory, release_frames=release)
                   for on, active, confirm, restrike, refractory, release in (
                       (.7,.7,1,0.,10,5), (.85,.5,1,0.,10,5), (.85,.7,1,0.,10,5), (.95,.5,1,0.,10,5),
                       (.95,.7,1,0.,10,5), (.7,.85,1,0.,10,5), (.85,.85,1,0.,10,5), (.95,.85,1,0.,10,5),
                       (.5,.5,2,.9,10,8), (.7,.5,2,.9,10,8), (.85,.5,2,.95,10,8), (.7,.7,2,.95,10,8),
                       (.5,.5,1,.9,18,8), (.7,.5,1,.9,18,8), (.85,.5,1,.95,18,8), (.7,.7,1,.95,18,8))]
    history = json.loads((out / "development_loops.json").read_text()) if refinement else []
    best = max(history, key=lambda row: row["overall"]["F1"]) if history else None
    for version in versions:
        heads = NativeLiveHeads(version)
        scores = [[] for _ in configs]
        for spec in protocol["sources"]:
            if spec["split"] != "dev":
                continue
            ends, x, meta = load_source(root, spec)
            p, k = heads.predict(x)
            truth = real_notes(meta["labels"]["notes"], meta["prepared_samples"] / SR)
            for i, config in enumerate(configs):
                events = decode(ends, p, k, config, meta["prepared_samples"])
                score, _ = event_score(events, truth)
                scores[i].append(dict(source=spec["name"], corpus=spec["corpus"], **score))
        for config, items in zip(configs, scores):
            result = dict(checkpoint=Path(version).name, selection=asdict(config), overall=summary(items),
                          by_corpus={c: summary([s for s in items if s["corpus"] == c]) for c in ("GuitarSet", "URMP")})
            history.append(result)
            print("DEV LOOP", json.dumps(result), flush=True)
            if best is None or result["overall"]["F1"] > best["overall"]["F1"]:
                best = result
        save(out / "development_loops.json", history)
    with np.load(out / best["checkpoint"], allow_pickle=False) as z:
        arrays = {key: z[key].copy() for key in z.files}
    meta = json.loads(str(arrays["metadata"]))
    meta["selection"] = best["selection"]
    meta["development_F1"] = best["overall"]["F1"]
    arrays["metadata"] = np.asarray(json.dumps(meta))
    np.savez_compressed(out / "selected_heads.npz", **arrays)
    save(out / "selection_frozen.json", dict(best=best, model_sha256=digest((out / "selected_heads.npz").read_bytes()),
                                             protocol_sha256=digest((root / "protocol.json").read_bytes()),
                                             test_labels_read=False, total_development_loops=len(history)))


def freeze_test(root, experiment):
    path = experiment / "test_predictions_before_labels.json"
    if path.exists():
        raise FileExistsError("Held-out emissions already frozen")
    protocol = json.loads((root / "protocol.json").read_text())
    frozen = json.loads((experiment / "selection_frozen.json").read_text())
    assert frozen["protocol_sha256"] == digest((root / "protocol.json").read_bytes())
    filename = frozen.get("model_filename", "selected_heads.npz")
    heads = load_heads(experiment / filename, expected_sha256=frozen["model_sha256"])
    predictions, proofs, times = {}, [], []
    for spec in protocol["sources"]:
        if spec["split"] != "test":
            continue
        stem = root / "cache" / spec["id"]
        meta = json.loads(stem.with_suffix(".json").read_text())
        assert not meta["annotations_read"] and "labels" not in meta
        audio = np.load(stem.with_suffix(".audio.npy"), allow_pickle=False)
        stream = StrictLiveStream(heads)
        output = []
        for start in range(0, len(audio), HOP):
            output.extend(stream.push(audio[start:start + HOP], start_sample=start))
        output.extend(stream.finish())
        predictions[spec["id"]] = output
        times.extend(stream.timings_ms)
        proofs.append(dict(source_id=spec["id"], name=spec["name"], corpus=spec["corpus"],
                           seconds=len(audio) / SR, audio_sha256=meta["audio_sha256"]))
        print("TEST AUDIO ONLY", spec["name"], len(output), flush=True)
    save(path, dict(predictions=predictions, audio_proof=proofs, model_sha256=heads.sha256,
                    protocol_sha256=frozen["protocol_sha256"], labels_read=False,
                    runtime_sha256={name: digest(Path(name).read_bytes()) for name in (
                        "src/causal_note/strict_live.py", "scripts/live_strict_notes.py")},
                    timing_ms=dict(zip(("median", "p95", "p99", "max"), np.percentile(times, [50, 95, 99, 100]).tolist()))))


def score_test(root, experiment):
    path = experiment / "test_predictions_before_labels.json"
    frozen = json.loads(path.read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    assert frozen["protocol_sha256"] == digest((root / "protocol.json").read_bytes())
    scores, individual, offsets, scopes = {"50ms": [], "80ms": []}, {}, [], []
    for spec in protocol["sources"]:
        if spec["split"] != "test":
            continue
        note_data = labels(spec, root)
        proof = next(p for p in frozen["audio_proof"] if p["source_id"] == spec["id"])
        truth = real_notes(note_data["notes"], proof["seconds"])
        original_in_clip = [n for n in note_data["notes"] if 0 <= n["time"] < proof["seconds"]]
        scopes.append(dict(source=spec["name"], corpus=spec["corpus"], original_noteons=len(original_in_clip),
                           out_of_MIDI_40_88=sum(not 40 <= n["pitch"] <= 88 for n in original_in_clip),
                           scored_unique_noteons=len(truth)))
        events = frozen["predictions"][spec["id"]]
        individual[spec["id"]] = dict(truth=truth, labels_sha256=note_data["sha256"])
        for name, tolerance in (("50ms", .05), ("80ms", .08)):
            score, pairs = event_score(events, truth, tolerance)
            scores[name].append(dict(source=spec["name"], corpus=spec["corpus"], **score))
            individual[spec["id"]][name] = dict(score=score, pairs=pairs)
            if name == "80ms":
                audit = offset_audit(events, truth, pairs, proof["seconds"])
                individual[spec["id"]]["offsets"] = audit
                offsets.append(audit)
    report = dict(status="S613_STRICT_CAUSAL_NATIVE_AUDIO_TO_NOTES_HELDOUT", predictions_sha256=digest(path.read_bytes()),
                  model_sha256=frozen["model_sha256"], source_count=len(frozen["audio_proof"]),
                  seconds=sum(p["seconds"] for p in frozen["audio_proof"]), timing_ms=frozen["timing_ms"],
                  held_out_from_new_training_and_tuning=True, sources_previously_used_in_historical_experiments=True,
                  score_uses_audio_availability_time_without_backdating=True,
                  compute_and_device_latency_not_added_to_note_timestamps=True,
                  lookahead_samples=0, input_sidecars=False,
                  original_S80_promotion=False, hardware_microphone_latency_measured=False,
                  pitch_scope=[40, 88], exact_same_time_pitch_unisons_collapsed=True, source_scope=scopes,
                  results={name: dict(overall=summary(items), by_corpus={c: summary([s for s in items if s["corpus"] == c])
                                  for c in ("GuitarSet", "URMP", "IDMT")}) for name, items in scores.items()})
    report["offsets_conditional_on_matched_80ms_noteon"] = {
        key: sum(row[key] for row in offsets) for key in (
            "matched_noteons_with_observed_reference_offset", "offsets_within_max_50ms_or_20pct_duration", "forced_or_missing_offsets")}
    offset_errors = [e for row in offsets for e in row["acoustic_offset_error_ms"]]
    report["offsets_conditional_on_matched_80ms_noteon"]["acoustic_error_ms_p10_median_p95"] = np.percentile(offset_errors, [10, 50, 95]).tolist() if offset_errors else []
    save(experiment / "test_individual_audit.json", individual)
    save(experiment / "test_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("train", "tune", "refine", "freeze-test", "score-test"), required=True)
    args = parser.parse_args()
    if args.stage == "train":
        train(args.root, args.out)
    elif args.stage == "tune":
        tune(args.root, args.out, json.loads((args.root / "protocol.json").read_text()), json.loads((args.out / "trained_checkpoints.json").read_text()))
    elif args.stage == "refine":
        tune(args.root, args.out, json.loads((args.root / "protocol.json").read_text()), [str(args.out / "heads_epoch28.npz")], refinement=True)
    elif args.stage == "freeze-test":
        freeze_test(args.root, args.out)
    else:
        score_test(args.root, args.out)


if __name__ == "__main__":
    main()
