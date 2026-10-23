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
    DIM, SCHEMA, GatePolicy, candidate_rows, replay_gate)
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
            (totals["tp"], totals["fp"], totals["fn"]) != EXPECTED):
        raise AssertionError("refuse unverified complete 69.206% audio replay")
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
    dump(out / "prepared.json", dict(schema=SCHEMA, tracks=80,
         rows=seen, future_audio_samples=0, added_hold_samples=0,
         reference_candidate_labels_only_for_training=True,
         direct_replay_sha256=sha(experiment / "replayed/report.json"),
         reference_events_sha256=sha(Path(
             "series/ONSET_SCORE_RECOVERY_HGB256/evidence/sealed_predictions.json"))))


def score_gate(experiment, rows, reference, model, policy):
    scores = []
    records = []
    for row in rows:
        q, baseline, direct, x = track_inputs(experiment, row, reference,
                                               need_features=model is not None)
        merged = (replay_gate(q, x, baseline, direct, model, policy)
                  if model is not None else direct)
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
        control, _ = score_gate(experiment, cal_rows, reference, None, None)
        selected = dict(enabled=False, policy=None, score=control)
        policies = []
        # All threshold/gap combinations are chosen on CAL; no OUTER score
        # or OUTER labels are used in selecting this policy.
        for gap in GAPS:
            for cut in CUTS:
                policy = GatePolicy(cut=cut, gap_hops=gap)
                metric, _ = score_gate(experiment, cal_rows, reference,
                                       model, policy)
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
        actual, per_track = score_gate(experiment, outer_rows, reference,
                                       chosen_model, chosen_policy)
        record = dict(outer=outer, train=train, cal=cal,
                      selected=selected, actual=actual, per_track=per_track)
        dump(frozen / "outer_result.json", record)
        all_records.append(record)
        print("HGB_GATE_OUTER", outer, json.dumps(actual), flush=True)
    total = sum_scores([r["actual"] for r in all_records])
    raw = [r for g in all_records for r in g["per_track"]]
    base = sum_scores([r["before"] for r in raw])
    if (base["tp"], base["fp"], base["fn"]) != EXPECTED:
        raise AssertionError("direct acoustic control changed")
    report = dict(schema=SCHEMA, complete=True, hop_samples=256,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "train"), required=True)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.experiment, args.out)
    else:
        fit(args.experiment, args.out)


if __name__ == "__main__":
    main()
