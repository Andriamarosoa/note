"""Learn missed attacks from old signed PCM256 proposals; score is the objective.

Frozen best main-event head from run 39915038301. Recovery is calibrated with
four inner compositions per outer fold, then frozen before outer scoring.
The event-only proxy is checked against the old real replay before training.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import zipfile

import joblib
import numpy as np
from scipy.io import wavfile
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from onset256.compiled_inference import accelerated
from onset256.event_confidence import load_event_head
from onset256.fn_recovery import (
    CANDIDATE_POLICIES, EVAL_DEDUP_SAMPLES, SCHEMA, load_fn_recovery)
from onset256.live_event_stream import LowLatencyOnsetStream
from onset256.model import load
from scripts.onset_event_integrity256 import (
    COMPS, compact, counts, digest, dump, metric, one, original_models,
    read_archive, sum_scores, verify_inputs)
from scripts.replay_native256 import match_onsets

SEED = 2026102303
CUTS = tuple(float(c) for c in np.linspace(0., 1., 41))
SPECS = (
    dict(name="recovery_hgb15", max_leaf_nodes=15, max_iter=320,
         learning_rate=.04, l2_regularization=15., min_samples_leaf=25),
    dict(name="recovery_hgb31", max_leaf_nodes=31, max_iter=400,
         learning_rate=.04, l2_regularization=30., min_samples_leaf=30),
)


def eligibility(samples, pitches, native):
    """Exact past-native exclusion; never inspect future emitted attacks."""
    result = np.ones(len(samples), bool)
    for pitch in range(40, 89):
        indices = np.flatnonzero(pitches == pitch)
        times = np.asarray([s for s, p in native if p == pitch], np.int64)
        if len(times) and len(indices):
            previous = np.searchsorted(times, samples[indices], side="right") - 1
            last = np.where(previous >= 0, times[np.maximum(previous, 0)], -10**12)
            result[indices] = samples[indices] - last > EVAL_DEDUP_SAMPLES
    return result


def selected_mask(proposals, q, cut):
    """Event-only equivalent of FNRecoverySelector on sorted NoteOn proposals."""
    keep = np.zeros(len(q), bool)
    last_added = np.full(49, -10**12, np.int64)
    # Match the live selector's float32 scalar versus Python-float comparison.
    # Array comparison would round e.g. cut=.9 down to float32 before testing.
    for i in np.flatnonzero(proposals["eligible"] & (np.asarray(q, np.float64) >= cut)):
        pitch = int(proposals["pitches"][i]) - 40
        sample = int(proposals["samples"][i])
        if sample - last_added[pitch] > EVAL_DEDUP_SAMPLES:
            keep[i] = True; last_added[pitch] = sample
    return keep


def extra_events(proposals, mask):
    return [dict(sample=int(s), pitch=int(p)) for s, p in zip(
        proposals["samples"][mask], proposals["pitches"][mask])]


class EventMatcher:
    """Cache pair distances; retain the exact original Hungarian assignment."""
    def __init__(self, row, proposals):
        self.n_main = len(row["main"])
        samples = np.r_[[e["sample"] for e in row["main"]], proposals["samples"]]
        pitches = np.r_[[e["pitch"] for e in row["main"]], proposals["pitches"]]
        self.n_truth = len(row["truth"])
        self.groups = []
        for pitch in range(40, 89):
            pi = np.flatnonzero(pitches == pitch)
            gt = np.asarray([n["time"] for n in row["truth"] if int(n["pitch"]) == pitch])
            if len(pi) and len(gt):
                delta = np.abs(samples[pi, None] / 44100. - gt[None, :])
                self.groups.append((pi, delta, np.where(delta <= .080, delta, 1e6)))

    def score(self, mask):
        selected = np.r_[np.ones(self.n_main, bool), mask]
        tp = 0
        for pi, distance, cost in self.groups:
            sub = selected[pi]
            if np.any(sub):
                i, j = linear_sum_assignment(cost[sub])
                tp += int(np.sum(distance[sub][i, j] <= .080))
        n = self.n_main + int(mask.sum())
        return metric(tp, n - tp, self.n_truth - tp)


def prepare(args):
    out = args.out
    if out.exists(): raise FileExistsError("immutable recovery experiment")
    out.mkdir(parents=True)
    original = read_archive(args.previous / "collected")
    predictions = json.loads((args.previous / "trained" / "predictions.json").read_text())
    by_track = {}
    for r in original:
        old_keys = {(e["sample"], e["pitch"]) for e in r["fixed_recovery"]}
        main = [dict(sample=s, pitch=p) for s, p in predictions[r["track"]]
                if (s, p) not in old_keys]
        by_track[r["track"]] = dict(track=r["track"], composition=r["composition"],
            truth=r["truth"], main=main, old_recovery=r["fixed_recovery"], native=[],
            proposals={p: [] for p in CANDIDATE_POLICIES})
    for line in (args.root / "fn_audit" / "main_events.jsonl").open():
        r = json.loads(line)
        by_track[r["track"]]["native"].append((r["sample"], r["pitch"]))
    for line in (args.root / "fn_audit" / "candidate_proposals.jsonl").open():
        r = json.loads(line)
        if r["sample"] % 256 or r["composition"] != by_track[r["track"]]["composition"]:
            raise ValueError("candidate identity or PCM256 grid mismatch")
        by_track[r["track"]]["proposals"][r["policy"]].append(
            (r["sample"], r["pitch"], r["hard_predecision_vector"]))
    stats = {p: dict(candidates=0, eligible=0, positive=0) for p in CANDIDATE_POLICIES}
    models = {c: original_models(args.root, c)[1] for c in COMPS}
    with threadpool_limits(limits=1):
        for row in by_track.values():
            row["native"].sort()
            matched = {j for _, j in match_onsets(
                [SimpleNamespace(**e) for e in row["main"]], row["truth"])}
            missing = [n for j, n in enumerate(row["truth"]) if j not in matched]
            old = models[row["composition"]]
            for policy, candidates in row["proposals"].items():
                candidates.sort(key=lambda e: (e[0], e[1]))
                samples = np.asarray([e[0] for e in candidates], np.int64)
                pitches = np.asarray([e[1] for e in candidates], np.int16)
                x = np.asarray([e[2] for e in candidates], np.float32)
                if x.shape != (len(samples), 69) or not np.isfinite(x).all():
                    raise ValueError("invalid signed 69D recovery evidence")
                eligible = eligibility(samples, pitches, row["native"])
                indices = np.flatnonzero(eligible)
                predicted = [SimpleNamespace(sample=int(samples[i]), pitch=int(pitches[i]))
                             for i in indices]
                positives = {indices[i] for i, _ in match_onsets(predicted, missing)}
                y = np.asarray([int(i in positives) for i in range(len(samples))], np.int8)
                group = dict(samples=samples, pitches=pitches, X=x, y=y, eligible=eligible)
                row["proposals"][policy] = group
                for key, value in (('candidates', len(samples)), ('eligible', int(eligible.sum())),
                                   ('positive', int(y.sum()))): stats[policy][key] += value
                if policy == old.candidate_policy:
                    q = np.asarray(old.model.predict_proba(x)[:, 1], np.float32)
                    mask = selected_mask(group, q, old.calibrated_cut) if old.enabled else np.zeros(len(q), bool)
                    got = sorted((e["sample"], e["pitch"]) for e in extra_events(group, mask))
                    wanted = sorted((e["sample"], e["pitch"]) for e in row["old_recovery"])
                    if got != wanted:
                        raise AssertionError("old recovery proxy differs from verified audio: " + row["track"])
                    a = EventMatcher(row, group).score(mask)
                    b = counts(row["main"] + row["old_recovery"], row["truth"])
                    if any(a[k] != b[k] for k in ('tp', 'fp', 'fn')):
                        raise AssertionError("cached assignment changed original scorer")
    rows = list(by_track.values())
    score = sum_scores([counts(r["main"] + r["old_recovery"], r["truth"]) for r in rows])
    if (score["tp"], score["fp"], score["fn"]) != (8012, 4264, 3255):
        raise AssertionError("best frozen reference changed")
    joblib.dump(rows, out / "prepared.joblib", compress=3)
    dump(out / "prepared_report.json", dict(complete=True, tracks=80, baseline=score,
        candidates=stats, old_recovery_exact_event_parity=True,
        labels_after_audio=True, preparation_sha256=digest(out / "prepared.joblib")))
    dump(out / "preregistration.json", dict(seed=SEED, specs=SPECS, cuts=CUTS,
        policies=CANDIDATE_POLICIES, objective="inner composition OOF micro event F1",
        train_only_causally_eligible_proposals=True, relabel_to_frozen_filtered_main=True,
        hop_samples=256, added_hold_samples=0, performance_optimization_deferred=True,
        historical_upstream_models_not_nested=True))
    print("RECOVERY_PREPARED", json.dumps(stats), flush=True)


def fit_model(rows, policy, spec, seed):
    x = np.concatenate([r["proposals"][policy]["X"][r["proposals"][policy]["eligible"]] for r in rows])
    y = np.concatenate([r["proposals"][policy]["y"][r["proposals"][policy]["eligible"]] for r in rows])
    model = HistGradientBoostingClassifier(**{k: v for k, v in spec.items() if k != "name"},
        early_stopping=False, random_state=seed)
    model.fit(x, y)
    return model


def fold_train(task):
    root, directory, outer = map(str, task)
    root, directory = Path(root), Path(directory)
    rows = joblib.load(directory / "prepared.joblib")
    inner = [r for r in rows if r["composition"] != outer]
    eligible = [c for c in COMPS if c != outer]
    baseline = sum_scores([counts(r["main"] + r["old_recovery"], r["truth"]) for r in inner])
    best = dict(name="legacy_recovery", score=baseline)
    candidates = []
    with threadpool_limits(limits=1):
        for policy in CANDIDATE_POLICIES:
            scorers = {r["track"]: EventMatcher(r, r["proposals"][policy]) for r in inner}
            for spec in SPECS:
                oof = {}
                for j, cal in enumerate(eligible):
                    model = fit_model([r for r in inner if r["composition"] != cal],
                                      policy, spec, SEED + j)
                    for r in inner:
                        if r["composition"] == cal:
                            oof[r["track"]] = np.asarray(model.predict_proba(
                                r["proposals"][policy]["X"])[:, 1], np.float32)
                cuts = []
                for cut in CUTS:
                    score = sum_scores([scorers[r["track"]].score(selected_mask(
                        r["proposals"][policy], oof[r["track"]], cut)) for r in inner])
                    cuts.append(dict(cut=cut, score=score))
                choice = max(cuts, key=lambda v: (v["score"]["f1"], -v["cut"]))
                candidate = dict(name=spec["name"], policy=policy, spec=spec, **choice)
                candidates.append(candidate)
                print("RECOVERY_INNER_CAL", outer, policy, spec["name"], json.dumps(choice), flush=True)
                if choice["score"]["f1"] > best["score"]["f1"] + 1e-12: best = candidate
        cp = directory / "trained" / "checkpoints" / f"recovery_{outer}.joblib"
        if best["name"] == "legacy_recovery":
            source = root / "four" / "trained" / "checkpoints" / f"fn_recovery_outer_{outer}.joblib"
            shutil.copyfile(source, cp); shutil.copyfile(source.with_suffix('.json'), cp.with_suffix('.json'))
            meta = json.loads(cp.with_suffix('.json').read_text())
            model = joblib.load(cp); cut = meta["calibrated_cut"]; policy = meta["candidate_policy"]
            enabled = meta["enabled"]
        else:
            policy, cut, enabled = best["policy"], best["cut"], True
            model = fit_model(inner, policy, best["spec"], SEED + COMPS.index(outer))
            joblib.dump(model, cp, compress=3)
            meta = dict(schema=SCHEMA, trained_hop_samples=256, dim=69,
                outer_composition=outer, train_compositions=eligible,
                cal_composition="four_inner_compositions", cal_compositions=eligible,
                candidate_policy=policy, calibrated_cut=cut, enabled=True,
                checkpoint_sha256=digest(cp), frozen_before_outer=True,
                calibrated_only_on_cal=True, calibration="4x_grouped_inner_OOF_micro_F1",
                historical_upstream_overlap=True)
            dump(cp.with_suffix('.json'), meta)
        # Selected weights, policy and cut exist on disk before any OUTER scoring.
        records = []; sealed = {}
        for r in rows:
            if r["composition"] != outer: continue
            proposals = r["proposals"][policy]
            q = np.asarray(model.predict_proba(proposals["X"])[:, 1], np.float32)
            keep = selected_mask(proposals, q, cut) if enabled else np.zeros(len(q), bool)
            events = r["main"] + extra_events(proposals, keep)
            records.append(dict(track=r["track"], composition=outer, score=counts(events, r["truth"])))
            sealed[r["track"]] = sorted((e["sample"], e["pitch"]) for e in events)
        result = dict(outer=outer, selected=best, candidates=candidates,
                      outer_event_cache=sum_scores([r["score"] for r in records]))
        dump(directory / "trained" / f"fold_{outer}.json", result)
        print("RECOVERY_OUTER_SEALED", outer, json.dumps(result["outer_event_cache"]), flush=True)
        return result, records, sealed


def train(args):
    out = args.out / "trained"
    if out.exists(): raise FileExistsError("immutable trained recovery")
    (out / "checkpoints").mkdir(parents=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(fold_train, [(args.root, args.out, c) for c in COMPS]))
    folds = []; records = []; sealed = {}
    for fold, rows, predictions in results:
        folds.append(fold); records.extend(rows); sealed.update(predictions)
    score = sum_scores([r["score"] for r in records])
    dump(out / "report.json", dict(complete=True, folds=folds, event_cache_oof=score,
        per_track=records, inner_models_trained=120,
        final_models_trained=sum(f['selected']['name']!='legacy_recovery' for f in folds),
        hop_samples=256, added_hold_samples=0, real_new_audio_replay_required=True,
        historical_upstream_overlap=True, external_generalization_proven=False, promoted=False))
    dump(out / "predictions.json", sealed)
    print("RECOVERY_EVENT_CACHE_FINAL", json.dumps(score), flush=True)


def fold_replay(task):
    root, previous, directory, comp = task
    root, previous, directory = Path(root), Path(previous), Path(directory)
    with threadpool_limits(limits=1):
        manifest = json.loads(one(root / "manifest", "report.json").read_text())
        traces = one(root / "traces", "fit_000.json").parent
        expected = json.loads((directory / "trained" / "predictions.json").read_text())
        native, s701 = load("model/live/strict_note/NOTE_strict_live.npz", one(root / "s701", "s701_onset_shadow.joblib"))
        risk, _, _ = original_models(root, comp)
        cp = directory / "trained" / "checkpoints" / f"recovery_{comp}.joblib"
        meta = json.loads(cp.with_suffix('.json').read_text())
        recovery = load_fn_recovery(cp, cp.with_suffix('.json'), expected_sha256=meta['checkpoint_sha256'])
        head_cp = previous / "trained" / "checkpoints" / f"selected_{comp}.joblib"
        head = load_event_head(head_cp, head_cp.with_suffix('.json'), outer=comp)
        s701 = accelerated(s701); risk.model = accelerated(risk.model)
        recovery.model = accelerated(recovery.model); head.model = accelerated(head.model)
        records = []
        with zipfile.ZipFile(root / "GuitarSet_audio.zip") as audio:
            for item in sorted((r for r in manifest['fit_files'] if r['composition']==comp), key=lambda r:r['index']):
                annotation = traces / f"fit_{item['index']:03d}.json"
                if digest(annotation) != item['json_sha256']: raise ValueError('annotation SHA changed')
                metadata = json.loads(annotation.read_text())
                sr, pcm = wavfile.read(io.BytesIO(audio.read(metadata['name']+'_mic.wav')))
                if sr != 44100 or pcm.dtype != np.int16 or pcm.ndim != 1: raise ValueError('audio ABI changed')
                wav = pcm[:metadata['samples']].astype(np.float32)/32768.
                stream = LowLatencyOnsetStream(native,s701,risk,recovery,head)
                events = []; frames = 0
                for start in range(0,len(wav)-255,256):
                    decision = stream.push(wav[start:start+256],start_sample=start)
                    events.extend(compact(decision.intents)); frames += 1
                stream.finish()
                keys = sorted((e['sample'],e['pitch']) for e in events)
                if keys != [tuple(e) for e in expected[metadata['name']]]:
                    raise AssertionError('fresh recovery events differ: '+metadata['name'])
                records.append(dict(track=metadata['name'],composition=comp,frames=frames,
                    score=counts(events,metadata['fit_truth']),exact_event_parity=True))
                print('RECOVERY_FRESH_AUDIO_TRACK',comp,len(records),flush=True)
        dump(directory/'replayed'/f'{comp}.json',records)
        return records


def replay(args):
    verify_inputs(args.root)
    out = args.out/'replayed'
    if out.exists(): raise FileExistsError('immutable replay')
    out.mkdir()
    tasks=[(str(args.root),str(args.previous),str(args.out),c) for c in COMPS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool: groups=list(pool.map(fold_replay,tasks))
    rows=[r for group in groups for r in group]
    total=sum_scores([r['score'] for r in rows])
    expected=json.loads((args.out/'trained/report.json').read_text())['event_cache_oof']
    if len(rows)!=80 or sum(r['frames'] for r in rows)!=306536: raise AssertionError('incomplete audio')
    if any(total[k]!=expected[k] for k in ('tp','fp','fn')): raise AssertionError('audio score differs')
    dump(out/'report.json',dict(complete=True,tracks=80,frames=306536,totals=total,
        per_track=rows,hop_samples=256,added_hold_samples=0,future_audio_samples=0,
        exact_event_parity=True,historical_upstream_overlap=True,
        external_generalization_proven=False,promoted=False))
    print('RECOVERY_FRESH_AUDIO_FINAL',json.dumps(total),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,choices=range(1,6),default=5)
    p.add_argument('--phase',choices=('prepare','train','replay','all'),default='all')
    args=p.parse_args()
    if args.phase in ('prepare','all'): prepare(args)
    if args.phase in ('train','all'): train(args)
    if args.phase in ('replay','all'): replay(args)


if __name__=='__main__': main()
