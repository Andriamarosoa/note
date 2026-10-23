"""Score-focused Onset: four inner compositions calibrate each frozen outer fold.

No CPU timing criterion, TP-loss cap, filename feature or future audio.
All historical upstream checkpoints stay frozen; this is exploratory FIT80.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from onset256.event_confidence import SCHEMA
from scripts.onset_event_integrity256 import (
    COMPS, counts, digest, dump, read_archive, selected_events, sum_scores)

SEED = 2026102302
CUTS = tuple(float(c) for c in np.r_[np.linspace(0., .8, 41), 1.])
SPECS = (
    dict(name="hgb7_320", max_leaf_nodes=7, max_iter=320,
         l2_regularization=20., min_samples_leaf=30),
    dict(name="hgb15_320", max_leaf_nodes=15, max_iter=320,
         l2_regularization=30., min_samples_leaf=30),
    dict(name="hgb31_320", max_leaf_nodes=31, max_iter=320,
         l2_regularization=40., min_samples_leaf=40),
    dict(name="hgb63_240", max_leaf_nodes=63, max_iter=240,
         l2_regularization=60., min_samples_leaf=40),
)


def model_for(spec, seed):
    return HistGradientBoostingClassifier(
        **{k: v for k, v in spec.items() if k != "name"},
        learning_rate=.035, early_stopping=False, random_state=seed)


def features(rows):
    events = [e for r in rows for e in r["main"]]
    return (np.asarray([e["x"] for e in events], np.float32),
            np.asarray([e["y"] for e in events], np.int8))


def probabilities(model, rows):
    return {r["track"]: np.asarray(model.predict_proba(
        np.asarray([e["x"] for e in r["main"]], np.float32))[:, 1], np.float32)
        for r in rows}


def train_fold(task):
    collected, out, outer = task
    out = Path(out)
    rows = read_archive(collected)
    eligible = [c for c in COMPS if c != outer]
    inner_rows = [r for r in rows if r["composition"] in eligible]
    candidates = []
    with threadpool_limits(limits=1):
        for spec in SPECS:
            oof = {}
            splits = []
            for j, cal in enumerate(eligible):
                train_comps = [c for c in eligible if c != cal]
                fit = [r for r in inner_rows if r["composition"] in train_comps]
                held = [r for r in inner_rows if r["composition"] == cal]
                x, y = features(fit)
                model = model_for(spec, SEED + COMPS.index(outer) * 10 + j)
                model.fit(x, y)
                oof.update(probabilities(model, held))
                splits.append(dict(train=train_comps, cal=cal, train_rows=len(y)))
            if len(oof) != 64:
                raise AssertionError("incomplete composition-wise inner calibration")
            choices = []
            for cut in CUTS:
                score = sum_scores([
                    counts(selected_events(r, oof[r["track"]], cut), r["truth"])
                    for r in inner_rows])
                choices.append(dict(cut=cut, score=score))
            best = max(choices, key=lambda v: (v["score"]["f1"], -v["cut"]))
            candidates.append(dict(spec=spec, splits=splits, **best))
            print("ROBUST_INNER_CAL", outer, spec["name"], json.dumps(best), flush=True)

        # No OUTER confidence or score is consulted before this selection.
        choice = max(candidates, key=lambda v: (v["score"]["f1"], -v["cut"]))
        x, y = features(inner_rows)
        final = model_for(choice["spec"], SEED + COMPS.index(outer))
        final.fit(x, y)
        cp = out / "checkpoints" / f"selected_{outer}.joblib"
        joblib.dump(final, cp, compress=3)
        meta = dict(schema=SCHEMA, trained_hop_samples=256, feature_dim=69,
                    outer=outer, train=eligible, cal="four_inner_compositions",
                    cal_compositions=eligible, selected_model=choice["spec"]["name"],
                    cut=choice["cut"], checkpoint_sha256=digest(cp),
                    frozen_before_outer=True, historical_upstream_overlap=True,
                    calibration="4x_grouped_inner_OOF_exact_event_micro_F1",
                    no_state_feedback=True, score_is_only_objective=True)
        dump(cp.with_suffix(".json"), meta)
        outer_rows = [r for r in rows if r["composition"] == outer]
        q = probabilities(final, outer_rows)
        records = []; sealed = {}
        for r in outer_rows:
            events = selected_events(r, q[r["track"]], choice["cut"])
            records.append(dict(track=r["track"], composition=outer,
                                score=counts(events, r["truth"])))
            sealed[r["track"]] = sorted((e["sample"], e["pitch"]) for e in events)
        result = dict(outer=outer, candidates=candidates, selected=choice,
                      final_train_compositions=eligible,
                      outer_event_cache=sum_scores([r["score"] for r in records]))
        dump(out / f"fold_{outer}.json", result)
        print("ROBUST_OUTER_SEALED", outer, json.dumps(result["outer_event_cache"]), flush=True)
        return result, records, sealed


def train(args):
    out = args.out / "trained"
    if out.exists():
        raise FileExistsError("immutable research experiment")
    (out / "checkpoints").mkdir(parents=True)
    dump(out / "preregistration.json", dict(
        seed=SEED, specs=SPECS, thresholds=CUTS,
        objective="maximum inner OOF micro event F1; low cut breaks ties",
        calibration="four compositions, each held out of its inner model",
        historical_upstream_models_not_nested=True, hop_samples=256,
        added_hold_samples=0, performance_optimization_deferred=True))
    tasks = [(str(args.collected), str(out), c) for c in COMPS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(train_fold, tasks))
    folds = []; records = []; sealed = {}
    for fold, rows, predictions in results:
        folds.append(fold); records.extend(rows); sealed.update(predictions)
    score = sum_scores([r["score"] for r in records])
    report = dict(complete=True, schema=SCHEMA, folds=folds,
                  event_cache_oof=score, per_track=records,
                  new_models_trained=85, hop_samples=256,
                  real_new_audio_replay_required=True,
                  historical_upstream_overlap=True,
                  external_generalization_proven=False, promoted=False)
    dump(out / "report.json", report); dump(out / "predictions.json", sealed)
    print("ROBUST_EVENT_CACHE_FINAL", json.dumps(score), flush=True)


def replay(args):
    from scripts.onset_event_integrity256 import verify_inputs
    from scripts.validate_compiled_onset256 import fold
    verify_inputs(args.root)
    out = args.out / "replayed"
    if out.exists():
        raise FileExistsError("immutable fresh audio replay")
    out.mkdir(parents=True)
    tasks = [(str(args.root), str(args.out), str(out), c) for c in COMPS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        groups = list(pool.map(fold, tasks))
    rows = [r for group in groups for r in group]
    total = sum_scores([r["score"] for r in rows])
    expected = json.loads((args.out / "trained" / "report.json").read_text())["event_cache_oof"]
    if len(rows) != 80 or sum(r["frames"] for r in rows) != 306536:
        raise AssertionError("incomplete fresh audio replay")
    if any(total[k] != expected[k] for k in ("tp", "fp", "fn")):
        raise AssertionError("fresh audio score differs from sealed predictions")
    report = dict(complete=True, tracks=80, frames=306536, totals=total,
                  per_track=rows, hop_samples=256, added_hold_samples=0,
                  future_audio_samples=0, exact_event_parity=True,
                  historical_upstream_overlap=True,
                  external_generalization_proven=False, promoted=False)
    dump(out / "report.json", report)
    print("ROBUST_FRESH_AUDIO_FINAL", json.dumps(total), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--collected", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--phase", choices=("train", "replay", "all"), default="all")
    p.add_argument("--workers", type=int, choices=range(1, 6), default=5)
    args = p.parse_args()
    if args.phase in ("train", "all"): train(args)
    if args.phase in ("replay", "all"): replay(args)


if __name__ == "__main__":
    main()
