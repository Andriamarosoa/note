"""Paired causal audio audit, CAL-only learned selection, fresh audio replay.

The old NoteOff bug is reproduced only in LegacyRecovery below. No label is
available to the stream. Archived events are valid for CAL rematching because
the final event head never feeds back into the candidate generators.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path
import time
from types import SimpleNamespace
import zipfile

import joblib
import numpy as np
from scipy.io import wavfile
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from onset256.acoustics import HOP, SR
from onset256.error_evidence69 import hard_matrix, load_hard_model
from onset256.event_confidence import SCHEMA, load_event_head
from onset256.fn_recovery import FNRecoverySelector, load_fn_recovery
from onset256.fp_discriminator import load_fp_discriminator
from onset256.model import load
from onset256.runtime import Stream
from scripts.replay_native256 import match_onsets

COMPS = ("BN1", "Funk2", "Jazz1", "Rock3", "SS1")
METRICS = ("tp", "fp", "fn")
EXPECTED = {
    "native": (8099, 4600, 3168),
    "risk": (8054, 4411, 3213),
    "old_four": (8160, 4605, 3107),
    "control69_original": (8127, 4519, 3140),
}
SEED = 20261023
CUTS = tuple(float(x) for x in np.linspace(0., 1., 51))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fd:
        for block in iter(lambda: fd.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def one(root, pattern):
    paths = sorted(Path(root).rglob(pattern))
    if len(paths) != 1:
        raise ValueError(f"expected one {pattern} under {root}, got {len(paths)}")
    return paths[0]


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def counts(events, truth):
    obj = [SimpleNamespace(pitch=int(e["pitch"]), sample=int(e["sample"]))
           for e in events]
    tp = len(match_onsets(obj, truth))
    return metric(tp, len(events) - tp, len(truth) - tp)


def metric(tp, fp, fn):
    return dict(tp=int(tp), fp=int(fp), fn=int(fn),
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
                precision=tp / (tp + fp) if tp + fp else 0.,
                recall=tp / (tp + fn) if tp + fn else 0.)


def sum_scores(items):
    return metric(*(sum(x[k] for x in items) for k in METRICS))


def compact(events):
    return [dict(pitch=int(e.pitch), sample=int(e.sample)) for e in events]


def proposal_onsets(events):
    return [dict(pitch=int(e["pitch"]), sample=int(e["decision_sample"]))
            for e in events if e.get("type", "note_on") == "note_on"]


class LegacyRecovery(FNRecoverySelector):
    """Emulate the parent commit's missing event-type check, for audit only."""
    def select(self, **kw):
        original = list(kw["proposals"])
        masked = [dict(e, type="note_on") for e in original]
        restore = {id(m): e for m, e in zip(masked, original)}
        selected = super().select(**dict(kw, proposals=masked))
        return tuple(restore[id(e)] for e in selected)


class PairedRecovery:
    """Two states, identical current observations; neither changes Main."""
    def __init__(self, model):
        self.old = LegacyRecovery(model.calibrated_cut, model.enabled)
        self.fixed = FNRecoverySelector(model.calibrated_cut, model.enabled)
        self.old_events = self.fixed_events = ()
        self.audit = Counter()

    def select(self, **kw):
        self.old_events = self.old.select(**kw)
        self.fixed_events = self.fixed.select(**kw)
        self.audit["note_off_proposals"] += sum(
            e.get("type") == "note_off" for e in kw["proposals"])
        self.audit["note_off_accepted_by_old_selector"] += sum(
            e.get("type") == "note_off" for e in self.old_events)
        return self.old_events


class RecordingRisk:
    requires_selector_state = True
    trained_hop_samples = 256

    def __init__(self, original):
        self.original = original
        self.cut = original.cut
        self.evidence = None

    @property
    def calls(self):
        return self.original.calls

    def predict_correct_probability(self, feat, p, k, rescue, history, state):
        # Snapshot is taken before the selectors. Copy is used by the
        # experimental head on the SAME hop, never at the next boundary.
        self.evidence = hard_matrix(feat, p, k, rescue, history, state)
        return self.original.predict_correct_probability(
            feat, p, k, rescue, history, state)


def original_models(root, comp):
    def load_signed(subdir, stem, loader):
        md = root / subdir / "trained" / "checkpoints" / (stem + ".json")
        meta = json.loads(md.read_text())
        if (meta["outer_composition"] != comp or comp in meta["train_compositions"]
                or comp == meta["cal_composition"]):
            raise ValueError("original model fold contamination")
        return loader(md.with_suffix(".joblib"), md,
                      expected_sha256=meta["checkpoint_sha256"])
    risk = load_signed("risk", f"onset_error_outer_{comp}", load_hard_model)
    recovery = load_signed("four", f"fn_recovery_outer_{comp}", load_fn_recovery)
    fp = load_signed("fp", f"fp_control69_outer_{comp}", load_fp_discriminator)
    return risk, recovery, fp


def verify_inputs(root):
    root = Path(root)
    spec = json.loads(Path("analysis/evidence/v273-s613-strict-live/protocol.json")
                      .read_text())["guitar_archives"]["GuitarSet_audio"]
    audio = root / "GuitarSet_audio.zip"
    if audio.stat().st_size != spec["size"] or digest(audio) != spec["sha256"]:
        raise ValueError("original WAV archive SHA256 mismatch")
    manifest = json.loads(one(root / "manifest", "report.json").read_text())
    if (manifest.get("schema") != "S800_FIT80_OOF_PER_FRAGMENT_MIDI256_V1"
            or not manifest.get("complete") or len(manifest["fit_files"]) != 80
            or any(manifest.get(k) for k in ("CAL_opened", "HELD_opened", "TEST_opened"))):
        raise ValueError("signed FIT80 manifest mismatch")
    return manifest


def audio_fold(task):
    root, output, comp, selected_dir = task
    root, output = Path(root), Path(output)
    with threadpool_limits(limits=1):
        manifest = json.loads(one(root / "manifest", "report.json").read_text())
        traces = one(root / "traces", "fit_000.json").parent
        original, specialist = load("model/live/strict_note/NOTE_strict_live.npz",
                                    one(root / "s701", "s701_onset_shadow.joblib"))
        risk, recovery, control = original_models(root, comp)
        items = sorted((r for r in manifest["fit_files"] if r["composition"] == comp),
                       key=lambda r: r["index"])
        if len(items) != 16:
            raise ValueError("incomplete composition")
        summaries = []
        with zipfile.ZipFile(root / "GuitarSet_audio.zip") as z, gzip.open(
                output / f"events_{comp}.jsonl.gz", "wt") as archive:
            for item in items:
                annotation = traces / f"fit_{item['index']:03d}.json"
                if digest(annotation) != item["json_sha256"]:
                    raise ValueError("signed annotation changed")
                meta = json.loads(annotation.read_text())
                if meta["name"] != item["name"]:
                    raise ValueError("track identity mismatch")
                sr, pcm = wavfile.read(io.BytesIO(z.read(meta["name"] + "_mic.wav")))
                if sr != SR or pcm.dtype != np.int16 or pcm.ndim != 1:
                    raise ValueError("original 44100 mono int16 required")
                wav = pcm[:meta["samples"]].astype(np.float32) / 32768.
                observed = RecordingRisk(risk)
                risk_calls_before = risk.calls
                stream = Stream(original, specialist, risk_model=observed,
                                recovery_model=recovery)
                paired = PairedRecovery(recovery)
                stream.fn_recovery_arbiter = paired
                selected = None
                if selected_dir:
                    ck = Path(selected_dir) / "checkpoints" / f"selected_{comp}.joblib"
                    selected = load_event_head(ck, ck.with_suffix(".json"), outer=comp)
                native = []; main = []; old_recovery = []; fixed_recovery = []
                control_events = []; candidate_events = []; timings = []; head_times = []
                frames = 0
                for start in range(0, len(wav), HOP):
                    t0 = time.perf_counter_ns()
                    decisions = stream.push(wav[start:start+HOP], start_sample=start)
                    if len(decisions) != int(start + HOP <= len(wav)):
                        raise AssertionError("not exactly one decision per complete PCM256")
                    for d in decisions:
                        frames += 1
                        if d.sample != frames * HOP or stream.received != d.sample:
                            raise AssertionError("future audio or output grid drift")
                        matrix = np.zeros((49, 74), np.float32)
                        matrix[:, :69] = observed.evidence
                        q_control = control.predict_correct(matrix)
                        native.extend(compact(d.intents))
                        old_recovery.extend(proposal_onsets(paired.old_events))
                        fixed = proposal_onsets(paired.fixed_events)
                        fixed_recovery.extend(fixed)
                        for e in d.predecision_intents:
                            keep = bool(q_control[e.pitch - 40] >= control.cut)
                            main.append(dict(pitch=e.pitch, sample=e.sample,
                                             x=observed.evidence[e.pitch - 40].astype(float).tolist(),
                                             control_keep=keep))
                            if keep:
                                control_events.append(dict(pitch=e.pitch, sample=e.sample))
                        if selected:
                            th = time.perf_counter_ns()
                            q = selected.predict(observed.evidence, sample=d.sample)
                            kept = selected.select(d.predecision_intents, q, sample=d.sample)
                            head_times.append((time.perf_counter_ns() - th) / 1e6)
                            candidate_events.extend(compact(kept))
                            candidate_events.extend(fixed)
                        timings.append((time.perf_counter_ns() - t0) / 1e6)
                stream.finish()
                if (frames != len(wav) // HOP or stream.calls != frames
                        or observed.calls - risk_calls_before != frames or stream.fn_recovery_calls != frames
                        or paired.old.calls != frames or paired.fixed.calls != frames):
                    raise AssertionError("missing native256 inference")
                # First use of ground truth: after this track's entire audio pass.
                truth = meta["fit_truth"]
                matched = {i for i, _ in match_onsets(
                    [SimpleNamespace(pitch=e["pitch"], sample=e["sample"]) for e in main], truth)}
                for i, e in enumerate(main):
                    e["y"] = int(i in matched)
                scores = {
                    "native": counts(native, truth),
                    "risk": counts(main, truth),
                    "old_four": counts(main + old_recovery, truth),
                    "fixed_four": counts(main + fixed_recovery, truth),
                    "control69_original": counts(control_events + old_recovery, truth),
                    "control69_fixed": counts(control_events + fixed_recovery, truth),
                }
                if selected:
                    scores["selected_hgb_or_control69"] = counts(candidate_events, truth)
                original_keys = {(e["pitch"], e["sample"]) for e in old_recovery}
                fixed_keys = {(e["pitch"], e["sample"]) for e in fixed_recovery}
                row = dict(track=meta["name"], composition=comp, index=item["index"],
                           frames=frames, main=main, old_recovery=old_recovery,
                           fixed_recovery=fixed_recovery, truth=truth,
                           scores=scores, bug_audit=dict(paired.audit),
                           recovered_event_additions=len(fixed_keys - original_keys),
                           recovered_event_removals=len(original_keys - fixed_keys),
                           selected_events=candidate_events if selected else None)
                archive.write(json.dumps(row) + "\n")
                summary = {k: row[k] for k in (
                    "track", "composition", "frames", "scores", "bug_audit",
                    "recovered_event_additions", "recovered_event_removals")}
                summary["instrumented_replay_ms_p95"] = float(np.quantile(timings, .95))
                summary["new_head_ms_p95"] = float(np.quantile(head_times, .95)) if head_times else None
                summaries.append(summary)
                print("ONSET_EVENT_INTEGRITY_TRACK", comp, len(summaries),
                      json.dumps(scores.get("selected_hgb_or_control69", scores["control69_fixed"])),
                      flush=True)
        dump(output / f"report_{comp}.json", summaries)
        return summaries


def audio_pass(root, out, workers, selected=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError("immutable experiment output already exists")
    out.mkdir(parents=True)
    tasks = [(str(root), str(out), c, str(selected) if selected else None) for c in COMPS]
    if workers == 1:
        groups = list(map(audio_fold, tasks))
    else:
        with ProcessPoolExecutor(max_workers=min(5, workers)) as pool:
            groups = list(pool.map(audio_fold, tasks))
    tracks = [x for group in groups for x in group]
    if len(tracks) != 80 or sum(x["frames"] for x in tracks) != 306536:
        raise AssertionError("incomplete signed full audio replay")
    totals = {key: sum_scores([r["scores"][key] for r in tracks])
              for key in tracks[0]["scores"]}
    for key, expected in EXPECTED.items():
        if tuple(totals[key][k] for k in METRICS) != expected:
            raise AssertionError(f"frozen {key} changed: {totals[key]}")
    audit = Counter()
    for row in tracks:
        audit.update(row["bug_audit"])
    report = dict(complete=True, phase="fresh_audio_replay" if selected else "paired_audio_audit",
                  tracks=80, frames=306536, totals=totals, bug_audit=dict(audit),
                  per_track=tracks, hop_samples=HOP, sample_rate=SR,
                  confirmation_hops=1, future_audio_samples=0, added_hold_samples=0,
                  historical_teacher_overlap=True, external_generalization_proven=False,
                  hardware_latency_measured=False, promoted=False)
    if selected:
        expected = json.loads((Path(selected) / "report.json").read_text())["event_cache_oof"]
        actual = totals["selected_hgb_or_control69"]
        if any(actual[k] != expected[k] for k in METRICS):
            raise AssertionError("fresh audio result differs from sealed event-cache predictions")
        expected_rows = {r["track"]: r for r in read_archive(Path(selected).parent / "collected")}
        frozen = json.loads((Path(selected) / "predictions.json").read_text())
        for row in read_archive(out):
            if row["main"] != expected_rows[row["track"]]["main"]:
                raise AssertionError("main event/evidence not identical in fresh audio pass")
            keys = sorted((e["sample"], e["pitch"]) for e in row["selected_events"])
            if keys != [tuple(e) for e in frozen[row["track"]]]:
                raise AssertionError("fresh replay events differ despite aggregate score parity")
        report["exact_event_parity_with_sealed_cache"] = True
        report["f1_improved_vs_control69"] = actual["f1"] > totals["control69_original"]["f1"]
    dump(out / "report.json", report)
    print("ONSET_EVENT_INTEGRITY_AUDIO_FINAL", json.dumps(totals), flush=True)
    return report


def read_archive(root):
    rows = []
    for comp in COMPS:
        with gzip.open(Path(root) / f"events_{comp}.jsonl.gz", "rt") as fd:
            rows.extend(json.loads(line) for line in fd if line.strip())
    if len(rows) != 80 or len({r["track"] for r in rows}) != 80:
        raise ValueError("archive missing signed tracks")
    return rows


def selected_events(row, probabilities=None, cut=0., *, baseline=False):
    main = row["main"]
    if baseline:
        selected = [e for e in main if e["control_keep"]]
    else:
        selected = [e for e, q in zip(main, probabilities) if q >= cut]
    return selected + row["fixed_recovery"]


def train(root, collected, out):
    out, root = Path(out), Path(root)
    if out.exists():
        raise FileExistsError("immutable learned models already exist")
    out.mkdir(parents=True)
    (out / "checkpoints").mkdir()
    rows = read_archive(collected)
    if sum(len(r["main"]) for r in rows) != 12465:
        raise AssertionError("changed frozen candidate pool")
    folds = []; sealed = {}; outer_results = []
    with threadpool_limits(limits=1):
        for i, outer in enumerate(COMPS):
            cal = COMPS[(i + 1) % len(COMPS)]
            train_comps = [c for c in COMPS if c not in (cal, outer)]
            fit = [e for r in rows if r["composition"] in train_comps for e in r["main"]]
            x = np.asarray([e["x"] for e in fit], np.float32)
            y = np.asarray([e["y"] for e in fit], np.int8)
            cal_rows = [r for r in rows if r["composition"] == cal]
            outer_rows = [r for r in rows if r["composition"] == outer]
            _, _, control = original_models(root, outer)
            baseline_score = sum_scores([counts(selected_events(r, baseline=True), r["truth"])
                                         for r in cal_rows])
            choice = dict(name="control69", cut=control.cut, cal=baseline_score)
            winner = control.model
            candidates = []
            # Two predeclared capacities; no OUTER-driven parameter changes.
            for leaves in (7, 15):
                model = HistGradientBoostingClassifier(
                    max_iter=160, learning_rate=.05, max_leaf_nodes=leaves,
                    min_samples_leaf=25, l2_regularization=10.,
                    early_stopping=False, random_state=SEED + i)
                model.fit(x, y)
                name = f"hgb69_leaves{leaves}"
                probs = {r["track"]: np.asarray(model.predict_proba(
                    np.asarray([e["x"] for e in r["main"]], np.float32))[:, 1], np.float32)
                         for r in cal_rows}
                choices = []
                for cut in CUTS:
                    sc = sum_scores([counts(selected_events(r, probs[r["track"]], cut), r["truth"])
                                     for r in cal_rows])
                    choices.append(dict(name=name, cut=cut, cal=sc))
                best = max(choices, key=lambda c: (c["cal"]["f1"], -c["cut"]))
                candidates.append(best)
                joblib.dump(model, out / "checkpoints" / f"{name}_{outer}.joblib", compress=3)
                if best["cal"]["f1"] > choice["cal"]["f1"] + 1e-12:
                    winner, choice = model, best
            checkpoint = out / "checkpoints" / f"selected_{outer}.joblib"
            joblib.dump(winner, checkpoint, compress=3)
            meta = dict(schema=SCHEMA, trained_hop_samples=256, feature_dim=69,
                        outer=outer, cal=cal, train=train_comps,
                        selected_model=choice["name"], cut=choice["cut"],
                        checkpoint_sha256=digest(checkpoint), frozen_before_outer=True,
                        calibration="exact_one_to_one_event_rematching_on_CAL_only",
                        historical_upstream_overlap=True, no_state_feedback=True)
            dump(checkpoint.with_suffix(".json"), meta)
            # OUTER confidence is accessed only AFTER model and threshold are sealed.
            results = []
            for row in outer_rows:
                q = np.asarray(winner.predict_proba(
                    np.asarray([e["x"] for e in row["main"]], np.float32))[:, 1], np.float32)
                events = selected_events(row, q, choice["cut"])
                sc = counts(events, row["truth"])
                results.append(sc)
                sealed[row["track"]] = sorted((e["sample"], e["pitch"]) for e in events)
                outer_results.append(dict(track=row["track"], composition=outer, score=sc))
            fold = dict(outer=outer, train=train_comps, cal=cal,
                        train_rows=len(fit), selected=choice, candidates=candidates,
                        outer_event_cache=sum_scores(results))
            folds.append(fold)
            print("ONSET_HGB_CAL_SEALED", json.dumps(fold), flush=True)
    report = dict(complete=True, schema=SCHEMA, folds=folds,
                  event_cache_oof=sum_scores([r["score"] for r in outer_results]),
                  per_track=outer_results, new_models_trained=10, hop_samples=256,
                  real_new_audio_replay_required=True, promoted=False)
    dump(out / "report.json", report)
    dump(out / "predictions.json", sealed)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=5, choices=range(1, 6))
    p.add_argument("--phase", choices=("all", "collect", "train", "replay"), default="all")
    a = p.parse_args()
    if a.phase in ("all", "collect", "replay"):
        verify_inputs(a.root)
    if a.phase in ("all", "collect"):
        audio_pass(a.root, a.out / "collected", a.workers)
    if a.phase in ("all", "train"):
        train(a.root, a.out / "collected", a.out / "trained")
    if a.phase in ("all", "replay"):
        audio_pass(a.root, a.out / "replayed", a.workers, a.out / "trained")


if __name__ == "__main__":
    main()
