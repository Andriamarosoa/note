"""S701: learn causal onset *novelty* while freezing NOTE K, pitches and S675.

A new binary per-(frame,pitch) head learns from TRAIN JAMS onsets only.
At inference every feature is current/past native NOTE audio/posteriors.
Native Onsets are preserved; only subthreshold candidates can be rescued.
TRAIN 5 composition families, HELD 4 other TRAIN composition families, DEV
historically exposed (diagnostic only), TEST unopened. No LIVE deployment.
"""
from __future__ import annotations
import argparse
import json
import tempfile
import time
import zipfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from causal_note.strict_live import HOP, SR, Selection, feature_track, load_heads
from scripts.s613_prepare_strict_live import digest, download, labels
from scripts.s613_train_strict_live import decode, event_score, real_notes
from scripts.s672_causal_offset_timing_grid import (
    MODEL, ROOT, native_posteriors, raw_guitar_audio,
    note_instances, native_rhythm_clock,
)
from scripts.s682_onset_four_lock_guard import (
    aggregate, full_aggregate, native_metrics, safe_full, score_release,
)

# Predeclared strategies; held data select once; DEV never tunes thresholds.
POLICIES = ("native", "head_0.60_floor_0.10", "head_0.75_floor_0.10",
            "head_0.85_floor_0.10", "head_0.92_floor_0.10",
            "head_0.75_floor_0.20", "head_0.85_floor_0.20",
            "head_0.92_floor_0.20")
RANDOM_SEED = 701
TRAIN_CAP_POS = 2400
TRAIN_CAP_NEG = 3600


def lag(x, steps):
    """Look backward only; stable for the first frame."""
    t = np.maximum(0, np.arange(len(x)) - steps)
    return x[t]


def onset_features(feat, post, k):
    """Per-note 3D feature tensor from NOTE's source 126D ABI, current/past only."""
    feat = np.asarray(feat, np.float32)
    post = np.asarray(post, np.float32)
    k = np.asarray(k, np.float32)
    n = len(feat)
    if feat.shape != (n, 49, 126) or post.shape != (n, 49, 3) or k.shape != (n, 7):
        raise ValueError("S701 frozen NOTE ABI mismatch")
    if n == 0:
        return np.empty((0, 49, 39), np.float32)
    onset, sus, release = post.transpose(2, 0, 1)
    fundamental = np.maximum(feat[:, :, [0, 8, 16]], 0).mean(axis=2)
    overtone2 = np.maximum(feat[:, :, [1, 9, 17]], 0).mean(axis=2)
    overtone3 = np.maximum(feat[:, :, [2, 10, 18]], 0).mean(axis=2)
    near = np.maximum(feat[:, :, 54:57], 0).mean(axis=2)
    distant = np.maximum(feat[:, :, 66:69], 0).mean(axis=2)
    f_rise = (.65 * near + .35 * distant) / (.2 + fundamental)
    # Own fundamental vs evidence copied from lower-octave / fifth harmonics.
    cols = np.arange(49)
    lower12 = np.maximum(cols - 12, 0)
    lower19 = np.maximum(cols - 19, 0)
    low12_ok = np.broadcast_to((cols >= 12)[None, :], (n, 49))
    low19_ok = np.broadcast_to((cols >= 19)[None, :], (n, 49))
    sus12 = np.where(low12_ok, sus[:, lower12], 0)
    ons12 = np.where(low12_ok, onset[:, lower12], 0)
    sus19 = np.where(low19_ok, sus[:, lower19], 0)
    ons19 = np.where(low19_ok, onset[:, lower19], 0)
    pitch = np.broadcast_to((cols[None, :] + 40.) / 127., (n, 49))
    # This list is deliberately fixed; every column is inferable from past audio.
    values = [
        onset, sus, release, lag(onset, 2), lag(onset, 8),
        lag(sus, 2), lag(sus, 8), onset - lag(onset, 2),
        onset - lag(onset, 8), sus - lag(sus, 8),
        fundamental, overtone2, overtone3, near, distant, f_rise,
        fundamental - lag(fundamental, 2),
        fundamental - lag(fundamental, 8),
        overtone2 - lag(overtone2, 8),
        overtone3 - lag(overtone3, 8),
        overtone2 / (.2 + fundamental),
        overtone3 / (.2 + fundamental),
        sus12, ons12, sus19, ons19,
        f_rise - f_rise[:, lower12],
        f_rise - f_rise[:, lower19],
        np.maximum(0, onset - ons12), np.maximum(0, onset - ons19),
        pitch,
        *[np.broadcast_to(k[:, i, None], (n, 49)) for i in range(7)],
        np.broadcast_to(np.max(k[:, 2:], axis=1)[:, None], (n, 49)),
    ]
    out = np.stack(values, axis=2).astype(np.float32, copy=False)
    if out.shape[-1] != 39 or not np.isfinite(out).all():
        raise AssertionError("S701 feature layout error")
    return out


def true_onset_rows(ends, notes, n):
    """Supervised TRAIN targets only. Never passed into inference."""
    y = np.zeros((n, 49), dtype=np.uint8)
    for note in notes:
        pitch = int(note["pitch"])
        if not 40 <= pitch <= 88:
            continue
        start = int(np.searchsorted(ends, int(round(note["time"] * SR)), side="left"))
        if start >= n:
            continue
        stop = min(n, start + 11)  # 0..58ms after annotated onset.
        y[start:stop, pitch - 40] = 1
    return y


def training_rows(x, post, ends, truth, seed):
    """Subsample both positives and onset-like hard negatives by TRAIN only."""
    rng = np.random.default_rng(seed)
    y = true_onset_rows(ends, truth, len(x))
    # Spectral and native near-attack examples; don't train on irrelevant silence.
    possible = ((post[:, :, 0] >= .09) |
                (post[:, :, 1] >= .37) |
                (x[:, :, 15] >= .12))
    positives = np.flatnonzero((y & possible).ravel())
    hard = np.flatnonzero((possible & ~y.astype(bool)).ravel())
    if len(positives) > TRAIN_CAP_POS:
        positives = rng.choice(positives, TRAIN_CAP_POS, replace=False)
    if len(hard) > TRAIN_CAP_NEG:
        # Hard negatives: prioritize scores which would pass a naive lowering.
        hscore = post[:, :, 0].ravel()[hard]
        high = hard[np.flatnonzero(hscore >= .20)]
        other = hard[np.flatnonzero(hscore < .20)]
        cap_high = min(len(high), 2700)
        cap_other = min(len(other), TRAIN_CAP_NEG - cap_high)
        high = rng.choice(high, cap_high, replace=False)
        other = rng.choice(other, cap_other, replace=False)
        hard = np.concatenate((high, other))
    selected = np.concatenate((positives, hard))
    if not len(positives) or not len(hard):
        raise AssertionError("S701 missing positive or negative TRAIN class")
    return (x.reshape(-1, x.shape[-1])[selected].copy(),
            y.ravel()[selected].copy(),
            dict(positive=int(len(positives)), negative=int(len(hard))))


def fit(rows):
    x = np.vstack([r[0] for r in rows])
    y = np.concatenate([r[1] for r in rows])
    if len(x) < 10000 or int(y.sum()) < 1000:
        raise AssertionError("S701 insufficient authentic GuitarSet training events")
    model = HistGradientBoostingClassifier(
        max_iter=145, learning_rate=.055, max_leaf_nodes=23,
        min_samples_leaf=110, l2_regularization=25.,
        early_stopping=False, random_state=RANDOM_SEED)
    # A false new pitch can create both a false Onset and a false note.
    model.fit(x, y, sample_weight=np.where(y == 1, 1.8, 1.0))
    return model, dict(rows=len(y), positive=int(y.sum()),
                       negative=int((y == 0).sum()),
                       feature_dims=x.shape[1])


def predict_onsets(model, x):
    flat = x.reshape(-1, x.shape[-1])
    score = np.empty(len(flat), np.float32)
    for a in range(0, len(flat), 8192):
        b = min(len(flat), a + 8192)
        score[a:b] = model.predict_proba(flat[a:b])[:, 1]
    return score.reshape(x.shape[:2])


def rescue(post, scores, policy, sel):
    """Shadow-only rescue; never suppress native note attacks or modify K/off."""
    if policy == "native":
        return post
    _, threshold, _, floor = policy.split("_")
    threshold = float(threshold)
    floor = float(floor)
    if scores.shape != post.shape[:2]:
        raise ValueError("S701 score ABI mismatch")
    # Existing NOTE selector still enforces active note and retrigger logic.
    allowed = ((post[:, :, 0] >= floor) &
               (post[:, :, 0] < sel.onset) &
               (post[:, :, 1] >= sel.active_on) &
               (scores >= threshold))
    result = post.copy()
    result[:, :, 0] = np.where(allowed, min(.999, sel.onset + .025), post[:, :, 0])
    assert np.array_equal(result[:, :, 1:], post[:, :, 1:])
    return result


def score_source(spec, root, archive, heads, sel, model, policies, offmodel=None):
    audio, sha = raw_guitar_audio(spec, archive)
    truth = real_notes(labels(spec, root)["notes"], len(audio) / SR)
    ends, feat = feature_track(audio, block_size=4096)
    post, k = native_posteriors(feat, heads)
    protected = digest(post[:, :, 1:].tobytes() + k.tobytes())
    x = onset_features(feat, post, k)
    probabilities = predict_onsets(model, x)
    scores, release = {}, {}
    for name in policies:
        selected = rescue(post, probabilities, name, sel)
        events = decode(ends, selected, k, sel, len(audio))
        scores[name] = native_metrics(events, truth, ends)
        if offmodel is not None:
            onset, pairs = event_score(events, truth, tolerance=.080)
            item = dict(truth=truth, ends=ends, feats=feat, post=post, k=k,
                        events=events, onset=onset, pairs=pairs,
                        notes=note_instances(events, ends),
                        clock=native_rhythm_clock(events, ends))
            release[name] = score_release(item, offmodel)
        if digest(selected[:, :, 1:].tobytes() + k.tobytes()) != protected:
            raise AssertionError("S701 modified frozen native K/sustain/release")
    return dict(name=spec["name"], composition=spec["composition"],
                sha256=sha, scores=scores, release=release)


def summary(rows, policies):
    scores = {name: aggregate([r["scores"] for r in rows], name)
              for name in policies}
    by_composition = {
        comp: {name: aggregate([r["scores"] for r in rows
                                  if r["composition"] == comp], name)
               for name in policies}
        for comp in sorted({r["composition"] for r in rows})
    }
    by_texture = {
        kind: {name: aggregate([r["scores"] for r in rows
                                 if r["name"].endswith("_" + kind)], name)
               for name in policies}
        for kind in ("solo", "comp")
    }
    return scores, by_composition, by_texture


def select_on_held(scores, groups):
    original = scores["native"]
    approved, notes = [], {}
    for name in POLICIES[1:]:
        trial = scores[name]
        good = sum(groups[c][name]["onset_F1"] >=
                   groups[c]["native"]["onset_F1"] for c in groups)
        passed = (trial["onset_F1"] > original["onset_F1"] + .002
                  and trial["K_frames_correct"] >= original["K_frames_correct"]
                  and trial["tp"] > original["tp"]
                  and trial["fp"] - original["fp"] <=
                  1.7 * (trial["tp"] - original["tp"])
                  and good >= 3)
        notes[name] = dict(eligible=bool(passed), nonreg_compositions=good)
        if passed:
            approved.append(name)
    best = max(POLICIES[1:], key=lambda n: scores[n]["onset_F1"])
    selected = max(approved, key=lambda n: scores[n]["onset_F1"]) if approved else "native"
    return selected, best, notes


def run(out, checkpoint):
    if out.exists():
        raise FileExistsError("S701 refuses to overwrite benchmark")
    st = time.perf_counter()
    frozen = json.loads((ROOT / "selection_frozen.json").read_text())
    proto = json.loads((ROOT / "protocol.json").read_text())
    assert digest(MODEL.read_bytes()) == frozen["model_sha256"]
    assert digest((ROOT / "protocol.json").read_bytes()) == frozen["protocol_sha256"]
    tr = [s for s in proto["sources"]
          if s["corpus"] == "GuitarSet" and s["split"] == "train"]
    dv = [s for s in proto["sources"]
          if s["corpus"] == "GuitarSet" and s["split"] == "dev"]
    comps = sorted({s["composition"] for s in tr})
    assert len(tr) == 36 and len(comps) == 9 and len(dv) == 12
    train_comp, held_comp = set(comps[:5]), set(comps[5:])
    assert train_comp.isdisjoint(held_comp)
    heads = load_heads(MODEL, expected_sha256=frozen["model_sha256"])
    sel = Selection(**heads.metadata["selection"])
    from causal_note.s672_live_offset import S672TrialNoteOff
    offset = S672TrialNoteOff.load(checkpoint, heads)
    assert offset.gate == .60 and offset.persistence == 1
    report = dict(schema="S701_TRAINED_SOURCE_ONLY_CAUSAL_ONSET",
                  frozen_native_sha256=frozen["model_sha256"],
                  frozen_offset_sha256=digest(checkpoint.read_bytes()),
                  training_compositions=sorted(train_comp),
                  held_compositions=sorted(held_comp),
                  original_K_pitch_sustain_Offset_untouched=True,
                  no_future_PCM_or_GT_at_inference=True,
                  shadow_only_no_LIVE_changes=True,
                  DEV_previously_exposed=True, TEST_labels_unopened=True)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "data").mkdir()
        for key, file in (("GuitarSet_audio", "GuitarSet_audio.zip"),
                          ("GuitarSet_annotations", "GuitarSet_annotations.zip")):
            src = proto["guitar_archives"][key]
            download(src["url"], root / "data" / file, src["size"], src["sha256"])
        with zipfile.ZipFile(root / "data/GuitarSet_audio.zip") as z:
            # Four authentic players per composition; group separation prevents
            # leakage of the same song between training and evaluation.
            from scripts.s699_player_held_full_train_K import expanded_train_specs
            all_train = expanded_train_specs(tr, z,
                zipfile.ZipFile(root / "data/GuitarSet_annotations.zip"))
            learn = [s for s in all_train if s["composition"] in train_comp]
            hold = [s for s in all_train if s["composition"] in held_comp]
            assert len(learn) == 80 and len(hold) == 64
            examples = []
            for i, spec in enumerate(learn, 1):
                audio, _ = raw_guitar_audio(spec, z)
                truth = real_notes(labels(spec, root)["notes"], len(audio) / SR)
                ends, feat = feature_track(audio, block_size=4096)
                p, k = native_posteriors(feat, heads)
                x = onset_features(feat, p, k)
                examples.append(training_rows(x, p, ends, truth,
                                              RANDOM_SEED + i))
                print("S701 TRAIN", i, "/", len(learn), spec["name"], flush=True)
                del x, p, k, feat, ends, audio
            model, stats = fit(examples)
            report["TRAIN_fit"] = stats
            del examples
            holdrows = []
            for i, spec in enumerate(hold, 1):
                holdrows.append(score_source(spec, root, z, heads, sel,
                                             model, POLICIES))
                print("S701 HELD", i, "/", len(hold), spec["name"], flush=True)
            hold_scores, groups, texture = summary(holdrows, POLICIES)
            selected, best, eligibility = select_on_held(hold_scores, groups)
            report["HELD_scores"] = hold_scores
            report["HELD_texture"] = texture
            report["HELD_gate"] = eligibility
            report["HELD_selected"] = selected
            report["HELD_best_exploratory"] = best
            print("S701 HELD FROZEN", selected, best,
                  json.dumps(hold_scores[best]), flush=True)
            # Always inspect the best TRAIN-held proposal as shadow, even
            # if its K or false-positive guard vetoed selection.
            candidate = selected if selected != "native" else best
            variants = ("native", candidate)
            held_release = []
            for i, spec in enumerate(hold, 1):
                held_release.append(score_source(spec, root, z, heads, sel,
                                     model, variants, offset.model))
                print("S701 HELD OFFSET", i, "/", len(hold), flush=True)
            hrel = {v: full_aggregate([r["release"] for r in held_release], v)
                    for v in variants}
            offset_held_safe = safe_full(hrel["native"], hrel[candidate])
            report["HELD_release"] = hrel
            report["HELD_release_safe"] = bool(offset_held_safe)
            devrows = []
            for i, spec in enumerate(dv, 1):
                devrows.append(score_source(spec, root, z, heads, sel,
                                            model, variants, offset.model))
                print("S701 DEV", i, "/", len(dv), spec["name"], flush=True)
    dscore, dgroup, dtexture = summary(devrows, variants)
    drel = {v: full_aggregate([r["release"] for r in devrows], v)
            for v in variants}
    report["DEV_onset"] = dscore
    report["DEV_by_composition"] = dgroup
    report["DEV_by_texture"] = dtexture
    report["DEV_release"] = drel
    report["DEV_detailed"] = devrows
    safe = bool(selected != "native" and candidate == selected
                and offset_held_safe
                and dscore[candidate]["onset_F1"] > dscore["native"]["onset_F1"]
                and dscore[candidate]["K_frames_correct"] >=
                    dscore["native"]["K_frames_correct"]
                and safe_full(drel["native"], drel[candidate]))
    report["four_locks_passed"] = safe
    report["promoted"] = False
    report["elapsed_s"] = round(time.perf_counter() - st, 2)
    out.mkdir(parents=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    joblib.dump(model, out / "s701_onset_shadow.joblib", compress=3)
    print("S701 FINAL", json.dumps(dict(
        TRAIN_selected=selected, candidate=candidate,
        HELD=hold_scores[candidate], DEV_original=dscore["native"],
        DEV_candidate=dscore[candidate],
        DEV_comp_native=dtexture["comp"]["native"],
        DEV_comp_candidate=dtexture["comp"][candidate],
        S675_base=drel["native"], S675_candidate=drel[candidate],
        held_release_safe=offset_held_safe,
        four_locks=safe, promoted=False)), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    args = ap.parse_args()
    with threadpool_limits(limits=1):
        run(args.out, args.checkpoint)
