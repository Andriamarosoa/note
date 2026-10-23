"""Train a causal HGB gate only on cross-fitted reference candidates.

Uses the immutable 80-track direct-acoustic replay and original reference events.
The original acoustic probabilities for each composition were produced by a
model that excluded that composition. Training labels are NOT inference inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

from onset256.acoustic_attack import AttackPolicy
from onset256.acoustic_calibration import decode_cache
from onset256.feature_cache import load_features
from onset256.learned_hgb_arbiter import (
    DIM, SCHEMA, GatePolicy, candidate_rows, replay_precomputed)
from scripts.onset_acoustic_direct256 import baseline_mask, mask_events
from scripts.onset_event_integrity256 import COMPS, counts, sum_scores
from scripts.replay_native256 import match_onsets

EXPECTED = (7585, 3068, 3682)
SEED = 2026102311
CUTS = (.05, .10, .15, .20, .25, .30, .35, .40, .45, .50,
        .55, .60, .65, .70, .75, .80, .85, .90, .95)
GAPS = (1, 7, 14)


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fd:
        for block in iter(lambda: fd.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def inputs(experiment):
    fresh = json.loads((experiment / "replayed/report.json").read_text())
    totals = fresh["totals"]
    if (not fresh["complete"] or fresh["tracks"] != 80 or
            fresh["frames"] != 306536 or
            not all(fresh[k] for k in
                ("exact_event_parity", "exact_feature_parity",
                 "exact_reference_parity")) or
            fresh["hop_samples"] != 256 or
            fresh["future_audio_samples"] != 0 or
            fresh["added_hold_samples"] != 0 or
            totals["tp"] + totals["fn"] != 11267 or
            any(totals[k] < 0 for k in ("tp", "fp", "fn"))):
        raise AssertionError("unverified signed 80-WAV causal audio replay")
    actual = tuple(totals[k] for k in ("tp","fp","fn"))
    if actual != EXPECTED:
        print("CONTROL_RETRAINING_DRIFT_DISTINCT_FROM_ARCHIVE",
              dict(archived=EXPECTED, measured=actual),
              "Do not inherit the historic 825-loss attribution.", flush=True)
    source = Path("series/ONSET_SCORE_RECOVERY_HGB256/evidence/sealed_predictions.json")
    reference = json.loads(source.read_text())
    collection = json.loads((experiment / "collection_report.json").read_text())
    rows = sorted(collection["per_track"], key=lambda x: x["index"])
    if len(rows) != 80 or sum(x["frames"] for x in rows) != 306536:
        raise AssertionError("incomplete signed 80-track evidence")
    return rows, reference


def track_inputs(experiment, row, reference, *, need_features):
    comp = row["composition"]
    trained = experiment / "trained" / comp
    meta = json.loads((trained / "frozen_model_and_policy.json").read_text())
    if (meta["outer_composition"] != comp or
            comp in meta["train_compositions"] or
            comp == meta["cal_composition"] or
            not meta["frozen_before_outer"] or
            meta["future_audio_samples"] or meta["added_hold_samples"]):
        raise AssertionError("invalid direct-acoustic OOF/causal lineage")
    q_path = trained / ("probability_%03d.npy" % row["index"])
    frozen = json.loads((trained / "outer_prediction_freeze.json").read_text())
    if sha(q_path) != frozen["probabilities"][q_path.name]:
        raise AssertionError("changed acoustic probabilities")
    q = np.load(q_path, allow_pickle=False)
    baseline = baseline_mask(row, reference)
    direct = decode_cache(q, baseline, AttackPolicy(**meta["selected"]["policy"]))
    x = load_features(experiment / "features", row["index"]) if need_features else None
    if q.shape != baseline.shape or q.shape != (row["frames"], 49):
        raise AssertionError("wrong cached hop dimensions")
    return q, baseline, direct, x


def event_pairs(events, truth):
    return match_onsets(
        [SimpleNamespace(sample=int(s), pitch=int(p)) for s, p in events], truth)


def truth_ids(mask, truth):
    events = sorted((e["sample"], e["pitch"]) for e in mask_events(mask))
    return {j for _, j in event_pairs(events, truth)}


def prepare(experiment, out):
    rows, reference = inputs(experiment)
    d = out / "candidates"
    d.mkdir(parents=True, exist_ok=False)
    seen = []
    for row in rows:
        q, baseline, direct, x = track_inputs(experiment, row, reference, need_features=True)
        candidates, loc = candidate_rows(q, x, baseline, direct)
        old_events = sorted(tuple(e) for e in reference[row["track"]])
        actual = [((f + 1) * 256, p) for f, p in loc]
        if actual != old_events:
            raise AssertionError("HGB candidate ordering mismatch")
        before_ids = truth_ids(direct, row["truth"])
        original = dict(event_pairs(old_events, row["truth"]))
        y = np.array([int(i in original and original[i] not in before_ids)
                      for i in range(len(old_events))], np.int8)
        np.savez_compressed(d / ("track_%03d.npz" % row["index"]),
                            X=candidates, y=y)
        seen.append(dict(index=row["index"], track=row["track"],
                         comp=row["composition"], candidates=len(y),
                         incremental_positive=int(y.sum()), feature_dim=DIM))
        print("HGB_GATE_CANDIDATES", row["composition"], row["track"],
              len(y), int(y.sum()), flush=True)
    actual_control = json.loads(
        (experiment / "replayed/report.json").read_text())["totals"]
    dump(out / "prepared.json", dict(schema=SCHEMA, tracks=80,
         rows=seen, future_audio_samples=0, added_hold_samples=0,
         archived_control=dict(zip(("tp","fp","fn"), EXPECTED)),
         actual_control=actual_control,
         exact_archived_control_reproduced=tuple(
             actual_control[k] for k in ("tp","fp","fn")) == EXPECTED,
         historic_825_losses_not_transferred_to_retraining=True,
         reference_candidate_labels_only_for_training=True,
         direct_replay_sha256=sha(experiment / "replayed/report.json"),
         reference_events_sha256=sha(Path(
             "series/ONSET_SCORE_RECOVERY_HGB256/evidence/sealed_predictions.json"))))


def score_gate(experiment, rows, reference, model, policy, *, gate_scores=None):
    scores = []
    records = []
    for row in rows:
        q, baseline, direct, _ = track_inputs(
            experiment, row, reference, need_features=False)
        if model is not None:
            if gate_scores is not None:
                probability = gate_scores[row["track"]]
            else:
                raise ValueError("model probabilities must be frozen before scoring")
            merged = replay_precomputed(direct, baseline, probability, policy)
        else:
            merged = direct
        score = counts(mask_events(merged), row["truth"])
        old_truth = truth_ids(direct, row["truth"])
        new_truth = truth_ids(merged, row["truth"])
        records.append(dict(track=row["track"], composition=row["composition"],
                            before=counts(mask_events(direct), row["truth"]),
                            after=score, recovered_truth=len(new_truth - old_truth),
                            lost_truth=len(old_truth - new_truth),
                            extra_emissions=int(merged.sum()-direct.sum())))
        scores.append(score)
    return sum_scores(scores), records


def fold_candidate_scores(out, rows, model):
    result = {}
    for row in rows:
        with np.load(out / "candidates" /
                     ("track_%03d.npz" % row["index"]),
                     allow_pickle=False) as data:
            result[row["track"]] = np.asarray(
                model.predict_proba(data["X"])[:, 1], np.float32)
    return result


def fit(experiment, out):
    rows, reference = inputs(experiment)
    prepared = json.loads((out / "prepared.json").read_text())
    if prepared["direct_replay_sha256"] != sha(experiment / "replayed/report.json"):
        raise AssertionError("training input mutated")
    from sklearn.ensemble import HistGradientBoostingClassifier

    all_records = []
    for outer in COMPS:
        cal = COMPS[(COMPS.index(outer) + 1) % len(COMPS)]
        train = [c for c in COMPS if c not in (outer, cal)]
        frozen = out / "trained" / outer
        frozen.mkdir(parents=True, exist_ok=False)
        xx, yy = [], []
        for row in rows:
            if row["composition"] not in train:
                continue
            with np.load(out / "candidates" /
                         ("track_%03d.npz" % row["index"]), allow_pickle=False) as data:
                xx.append(data["X"]); yy.append(data["y"])
        X, y = np.concatenate(xx), np.concatenate(yy)
        if X.shape[1] != DIM or len(np.unique(y)) != 2:
            raise AssertionError("invalid train features or missing class")
        model = HistGradientBoostingClassifier(
            max_leaf_nodes=15, max_iter=200, learning_rate=.05,
            min_samples_leaf=35, max_bins=127, l2_regularization=30.,
            class_weight="balanced", early_stopping=False,
            random_state=SEED + COMPS.index(outer))
        model.fit(X, y)
        checkpoint = frozen / "hgb_arbiter.joblib"
        joblib.dump(model, checkpoint, compress=3)
        del X, y, xx, yy
        cal_rows = [r for r in rows if r["composition"] == cal]
        cal_scores = fold_candidate_scores(out, cal_rows, model)
        control, _ = score_gate(experiment, cal_rows, reference, None, None)
        selected = dict(enabled=False, policy=None, score=control)
        policies = []
        # All threshold/gap combinations are chosen on CAL; no OUTER score
        # or OUTER labels are used in selecting this policy.
        for gap in GAPS:
            for cut in CUTS:
                policy = GatePolicy(cut=cut, gap_hops=gap)
                metric, _ = score_gate(experiment, cal_rows, reference,
                                       model, policy, gate_scores=cal_scores)
                policies.append(dict(policy=dict(cut=cut, gap_hops=gap),
                                     score=metric))
                if (metric["f1"], metric["tp"], -metric["fp"]) > (
                        selected["score"]["f1"], selected["score"]["tp"],
                        -selected["score"]["fp"]):
                    selected = dict(enabled=True,
                        policy=dict(cut=cut, gap_hops=gap), score=metric)
        meta = dict(schema=SCHEMA, trained_hop_samples=256,
                    future_audio_samples=0, added_hold_samples=0,
                    outer_composition=outer, cal_composition=cal,
                    train_compositions=train, feature_dim=DIM,
                    training_candidates=sum(r["candidates"] for r in
                       prepared["rows"] if r["comp"] in train),
                    selected=selected, checkpoint_sha256=sha(checkpoint),
                    frozen_before_outer=True,
                    upstream_acoustic_oof=True,
                    current_branch_unchanged_axes=True,
                    past_fitted_reference_overlap=True,
                    independent_external_validation=False,
                    promoted=False)
        dump(frozen / "frozen_gate.json", meta)
        dump(frozen / "calibration.json", dict(control=control,
                                              all_policies=policies))
        outer_rows = [r for r in rows if r["composition"] == outer]
        chosen_model = model if selected["enabled"] else None
        chosen_policy = (GatePolicy(**selected["policy"])
                         if selected["enabled"] else None)
        outer_scores = (fold_candidate_scores(out, outer_rows, chosen_model)
                        if chosen_model is not None else None)
        actual, per_track = score_gate(experiment, outer_rows, reference,
                                       chosen_model, chosen_policy,
                                       gate_scores=outer_scores)
        record = dict(outer=outer, train=train, cal=cal,
                      selected=selected, actual=actual, per_track=per_track)
        dump(frozen / "outer_result.json", record)
        all_records.append(record)
        print("HGB_GATE_OUTER", outer, json.dumps(actual), flush=True)
    total = sum_scores([r["actual"] for r in all_records])
    raw = [r for g in all_records for r in g["per_track"]]
    base = sum_scores([r["before"] for r in raw])
    fresh = json.loads((experiment / "replayed/report.json").read_text())
    if any(base[k] != fresh["totals"][k] for k in ("tp","fp","fn")):
        raise AssertionError("direct acoustic control changed from signed fresh replay")
    archival_parity = tuple(base[k] for k in ("tp","fp","fn")) == EXPECTED
    report = dict(schema=SCHEMA, complete=True, hop_samples=256,
                  archival_reference=dict(zip(("tp","fp","fn"), EXPECTED)),
                  original_69_206_exact_reproduction=archival_parity,
                  historically_reported_825_losses_apply=archival_parity,
                  independent_retraining_if_different=not archival_parity,
                  future_audio_samples=0, added_hold_samples=0,
                  control=base, candidate=total, folds=all_records,
                  recovered_truth=sum(r["recovered_truth"] for r in raw),
                  lost_truth=sum(r["lost_truth"] for r in raw),
                  extra_emissions=sum(r["extra_emissions"] for r in raw),
                  model_count=5, calibration_on_cal_only=True,
                  independent_external_validation=False,
                  fresh_audio_replay_required=True, promoted=False)
    dump(out / "trained" / "report.json", report)
    print("HGB_GATE_TOTAL", json.dumps(total), flush=True)



def replay_audio(root, experiment, out):
    """Revalidate the gate from signed 80 original WAVs, not just cached scores."""
    import io
    import zipfile
    from scipy.io import wavfile
    from onset256.learned_hgb_live import load_arbitrated_stream

    from scripts.onset_event_integrity256 import verify_inputs
    verify_inputs(root)
    rows, reference = inputs(experiment)
    expected = json.loads((out / "trained" / "report.json").read_text())
    manifest = json.loads((root / "manifest/report.json").read_text())
    signed = {x["name"]: x for x in manifest["fit_files"]}
    recs = []
    with zipfile.ZipFile(root / "GuitarSet_audio.zip") as audio:
        for row in rows:
            item = signed[row["track"]]
            signed_truth = root / "traces" / ("fit_%03d.json" % row["index"])
            if sha(signed_truth) != item["json_sha256"]:
                raise AssertionError("signed annotations changed")
            annotation = json.loads(signed_truth.read_text())
            sr, pcm = wavfile.read(io.BytesIO(
                audio.read(annotation["name"] + "_mic.wav")))
            if sr != 44100 or pcm.dtype != np.int16 or pcm.ndim != 1:
                raise AssertionError("signed WAV ABI changed")
            wav = pcm[:annotation["samples"]].astype(np.float32) / 32768.
            comp = row["composition"]
            fold_meta = json.loads(
                (out / "trained" / comp / "frozen_gate.json").read_text())
            gate_model = (joblib.load(out / "trained" / comp / "hgb_arbiter.joblib")
                if fold_meta["selected"]["enabled"] else None)
            gate_policy = (GatePolicy(**fold_meta["selected"]["policy"])
                if gate_model is not None else None)
            with np.load(out / "candidates" /
                         ("track_%03d.npz" % row["index"]),
                         allow_pickle=False) as candidate:
                scores = (np.asarray(gate_model.predict_proba(
                          candidate["X"])[:,1], np.float32)
                          if gate_model is not None else None)
            q, baseline, direct, cached = track_inputs(
                experiment, row, reference, need_features=True)
            expected_mask = (replay_precomputed(direct, baseline, scores, gate_policy)
                             if gate_model is not None else direct)
            expected_events = sorted(
                (e["sample"],e["pitch"]) for e in mask_events(expected_mask))
            stream = load_arbitrated_stream(
                experiment, out / "trained" / comp, fold=comp,
                base_bundle="model/onset256_research_39915038301",
                recovery_bundle="model/onset256_recovery_20261023",
                native_path="model/live/strict_note/NOTE_strict_live.npz")
            emitted = []; original = []
            for frame in range(row["frames"]):
                start = frame * 256
                got = stream.push(wav[start:start+256], start_sample=start)
                if got.sample != start+256:
                    raise AssertionError("Onset output delayed")
                if not np.array_equal(
                    cached[frame], stream.acoustic.reference.risk.acoustic_evidence):
                    raise AssertionError("acoustic features differ from signed cache")
                emitted.extend((e.sample,e.pitch) for e in got.intents)
                original.extend((e.sample,e.pitch)
                                for e in stream.last_reference.intents)
            if stream.finish() or stream.calls != row["frames"]:
                raise AssertionError("stream buffered a decision")
            if sorted(original) != [tuple(e) for e in reference[row["track"]]]:
                raise AssertionError("HGB reference stream differs")
            if sorted(emitted) != expected_events:
                raise AssertionError("LIVE and cached gated events differ")
            recs.append(dict(track=row["track"], composition=comp,
                        frames=row["frames"], score=counts(
                            mask_events(expected_mask), row["truth"]),
                        exact_gate_parity=True, exact_feature_parity=True,
                        exact_hgb_parity=True, no_added_holds=True))
            print("HGB_GATE_FRESH_WAV", comp, len(recs), flush=True)
    actual = sum_scores([r["score"] for r in recs])
    if (len(recs) != 80 or sum(r["frames"] for r in recs) != 306536 or
            any(actual[k] != expected["candidate"][k]
                for k in ("tp", "fp", "fn"))):
        raise AssertionError("full audio does not reproduce selected gate")
    dump(out / "replayed" / "report.json",
         dict(complete=True, tracks=80, frames=306536, hop_samples=256,
              candidate=actual, per_track=recs, exact_gate_parity=True,
              exact_feature_parity=True, exact_hgb_parity=True,
              future_audio_samples=0, added_hold_samples=0,
              external_generalization_proven=False, promoted=False))
    print("HGB_GATE_FRESH_AUDIO_TOTAL", json.dumps(actual), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "train", "replay"), required=True)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.experiment, args.out)
    elif args.phase == "train":
        fit(args.experiment, args.out)
    else:
        if args.root is None:
            parser.error("--root required for full WAV replay")
        replay_audio(args.root, args.experiment, args.out)


if __name__ == "__main__":
    main()
