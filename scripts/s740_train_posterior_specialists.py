"""S740: real GuitarSet FIT/CAL training for independent Sustain or Release.

S708 signed K+post, S734 H5 and S707 hazard stay frozen; each target
gets its own (fresh) Exact-K-architecture FragmentMaskRouter. All 32
subsets are learned per fragment, evaluated using actual MIDI NoteOn/Off
and ALL six official metrics. No automatic Live changes or promotion.
The remaining S740 adapters (Offset, identity, K) require separate
event/action-native supervised pipelines and are not passed off as trained.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile
import zipfile
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from causal_note.s736_fragment_mask import (
    MASKS, FragmentMaskRouter, block_features, choose_masks, all_mask_offers)
from causal_note.s740_multitarget_dynamic import (
    posterior_target, target_descriptors, apply_domain, PromotionGate)
from scripts.s737_train_fragment_onset_loops import (
    frozen_H5, training_audit, audit_all_fit, train_one, structural_mask,
    REFERENCE_NAME, STEPS, SEEDS, CAL_CONFIDENCE, MAX_FIT_BLOCKS, save)
from scripts.s733_train_onset_only import baseline_input
from scripts.s735_audit_32_onset_combinations import load_s734
from scripts.s613_prepare_strict_live import download, labels, digest
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s704_train_joint_loops import targets
from scripts.s704_joint_metrics import SCORES, metric, aggregate
from scripts.s706_train_sharp_temporal_loops import (
    FIT_COMPS, CAL_COMPS, HELD_COMPS, load_model)
from scripts.s705_train_temporal_loops import setup
from scripts.s731_auditory_scene_calibration import emit
from causal_note.s708_cardinality_selection import GuidedPolicy
from causal_note.strict_live import SR

THRESHOLD = .45
CHANNELS = {"sustain": 1, "release": 2}


def one_track(spec, audio, root, heads, selection, offset, network,
              policy, release, h5, target):
    item, post, k, baseline, onset_head5, context, onset_eligible = baseline_input(
        spec, audio, root, heads, selection, offset, network, policy, release)
    from causal_note.s736_fragment_mask import validate_decision_grid
    validate_decision_grid(item["decision_ends"])
    onset = frozen_H5(h5, onset_head5, context, onset_eligible, k)
    feat = block_features(onset, context, onset_eligible, k)
    base, offers = posterior_target(
        target, post, item["native_post"], context, k)
    eligible = (offers.max(axis=-1) > .05) | (base > .05) \
               | (context[:, :, 4] > .05)
    descriptors = target_descriptors(
        base, offers, eligible, novelty=context[:, :, 4])
    return dict(item=item, post=post, k=k, baseline=baseline, base=base,
                head=offers, eligible=eligible, features=feat,
                descriptors=descriptors, context=context)


def supervision(track, truth, seed, target):
    y, ktrue = targets(track["item"]["decision_ends"], truth)
    wanted = y[:, :, CHANNELS[target]].astype(bool)
    base = track["base"] >= THRESHOLD
    options = all_mask_offers(
        # Reuse the same 32 masks and threshold for H0-max proposals,
        # placing the target in Onset slot ONLY inside shadow evaluation.
        np.stack((track["base"], track["base"], track["base"]), axis=-1),
        track["head"], track["eligible"])
    new = (options >= THRESHOLD) & ~base[:, None, :]
    correct = (new & wanted[:, None, :]).sum(axis=-1)
    wrong = (new & ~wanted[:, None, :]).sum(axis=-1)
    activated = new.sum(axis=-1)
    bits = np.array([int(mask).bit_count() for mask in MASKS])
    gain = 2.0 * correct - 2.5 * wrong - .025 * (bits[None, :] - 1)
    best = gain.argmax(-1)
    target_actions = np.where(
        gain[np.arange(len(gain)), best] > 0, best + 1, 0).astype(np.int64)
    rng = np.random.default_rng(seed)
    priority = (target_actions > 0) | wanted.any(-1)
    essential = np.flatnonzero(priority)
    other = np.flatnonzero(~priority)
    if len(essential) > MAX_FIT_BLOCKS // 2:
        essential = np.sort(rng.choice(
            essential, MAX_FIT_BLOCKS // 2, replace=False))
    available = MAX_FIT_BLOCKS - len(essential)
    if len(other) > available:
        other = np.sort(rng.choice(other, available, replace=False))
    ix = np.sort(np.r_[essential, other])
    if not len(ix):
        raise AssertionError("Empty FIT subset")
    return dict(
        features=track["features"][ix],
        descriptors=track["descriptors"][ix],
        k=track["k"][ix],
        source=track["item"]["spec"]["composition"],
        target=target_actions[ix],
        audit_true_k=ktrue[ix],
        risk=np.stack((correct[ix] > 0, wrong[ix] > 0), axis=-1).astype(np.float32),
        coverage=(activated[ix] > 0).astype(np.float32),
        total=int(len(y)), rescuable=int((target_actions > 0).sum()),
        sampled=int(len(ix)))


def evaluate(tracks, target, model, mean, std, audit, confidence):
    old_scores, new_scores = [], []
    losses = Counter()
    histogram = Counter()
    for track in tracks:
        n = len(track["k"])
        masks, _, _ = choose_masks(
            model, track["features"], track["descriptors"],
            np.broadcast_to(audit, (n, 32, 7, 3)), track["k"],
            structural_mask(n).numpy(), mean, std,
            min_confidence=confidence)
        updated = apply_domain(target, track["base"], track["head"],
                               track["eligible"], masks)
        changed = track["post"].copy()
        changed[:, :, CHANNELS[target]] = updated
        emitted = emit(track["item"], changed, track["k"], track["policy"],
                       track["offset"], track["release"])
        truth = track["truth"]
        old = metric(track["baseline"], truth, track["item"]["ends"])
        actual = metric(emitted, truth, track["item"]["ends"])
        losses["correct_offsets"] += len(
            set(old["correct_offset_truth_ids"]) -
            set(actual["correct_offset_truth_ids"]))
        losses["long_onsets"] += len(
            set(old["long_onset_truth_ids"]) -
            set(actual["long_onset_truth_ids"]))
        histogram.update(str(int(m)) for m in masks)
        old_scores.append(old)
        new_scores.append(actual)
    original, actual = aggregate(old_scores), aggregate(new_scores)
    gate = PromotionGate(
        original, actual, losses["correct_offsets"], losses["long_onsets"],
        evaluated_complete_decoder=True, signed_fit_checkpoint=True,
        fit_composition_oof_audit=True)
    return dict(reference={m:float(original[m]) for m in SCORES},
                actual={m:float(actual[m]) for m in SCORES},
                delta={m:float(actual[m]-original[m]) for m in SCORES},
                lost_correct_offsets=losses["correct_offsets"],
                lost_long_onsets=losses["long_onsets"],
                safe=gate.safe() and actual["tp"] >= original["tp"],
                all_gt80=gate.above_80(),
                mask_histogram=dict(histogram))


def run(args):
    if args.target not in CHANNELS:
        raise ValueError("Only Sustain/Release have valid event posterior adapters")
    if args.out.exists():
        raise FileExistsError("Research output is immutable")
    args.out.mkdir(parents=True)
    frozen = json.loads((args.s708_dir / "report.json").read_text())
    sha = digest((args.s708_dir / "s708_selected.pt").read_bytes())
    if sha != frozen["checkpoint_sha256"]:
        raise AssertionError("S708 signed source changed")
    h5report = json.loads(args.s734_report.read_text())
    if h5report["base_sha256"] != sha or not h5report["complete"]:
        raise AssertionError("S734 source mismatch")
    torch.set_num_threads(2)
    network = load_model(args.s708_dir / "s708_selected.pt")[0]
    h5 = load_s734(args.s734_checkpoint, sha)
    policy = GuidedPolicy(**frozen["frozen_candidate"]["choice"]["policy"])
    release = frozen["frozen_candidate"]["choice"]["release"]
    if release != dict(gate=.8, persistence=1, grace=24):
        raise AssertionError("Do not silently change S707 offset")
    report = dict(
        schema="S740_REAL_AUDIO_DOMAIN_32_MASK_FIT_CAL",
        target=args.target, base_signed_sha256=sha,
        status="FIT_CAL_RUNNING", completed=False,
        selected=REFERENCE_NAME, promoted=False, LIVE_changed=False,
        TEST_opened=False, training="FIT80 only",
        FIT={}, CAL_rounds=[], achieved_all_six_gt80=False,
        objective="each of six MIDI scores strictly > 80 percent",
        test_leakage_policy="NO HELD/DEV until CAL-selected safe checkpoint",
        experimental=True)
    save(args.out / "report.json", report)
    with tempfile.TemporaryDirectory() as work:
        root = Path(work)
        (root / "data").mkdir()
        proto, heads, selection, offset = setup(root, args.offset_checkpoint)
        for category, name in (
                ("GuitarSet_audio", "GuitarSet_audio.zip"),
                ("GuitarSet_annotations", "GuitarSet_annotations.zip")):
            source = proto["guitar_archives"][category]
            download(source["url"], root / "data" / name,
                     source["size"], source["sha256"])
        train = [s for s in proto["sources"] if
                 s["corpus"] == "GuitarSet" and s["split"] == "train"]
        dev = [s for s in proto["sources"] if
               s["corpus"] == "GuitarSet" and s["split"] == "dev"]
        with zipfile.ZipFile(root / "data/GuitarSet_audio.zip") as audio, \
             zipfile.ZipFile(root / "data/GuitarSet_annotations.zip") as annotations:
            expanded = expanded_train_specs(train, audio, annotations)
            groups = dict(
                FIT=[s for s in expanded if s["composition"] in FIT_COMPS],
                CAL=[s for s in expanded if s["composition"] in CAL_COMPS],
                HELD=[s for s in expanded if s["composition"] in HELD_COMPS],
                DEV=dev)
            if tuple(len(groups[x]) for x in ("FIT", "CAL", "HELD", "DEV")) \
               != (80, 32, 32, 12):
                raise AssertionError("Unexpected GuitarSet composition split")
            fits = []
            for i, spec in enumerate(groups["FIT"]):
                track = one_track(spec, audio, root, heads, selection, offset,
                                  network, policy, release, h5, args.target)
                truth = real_notes(labels(spec, root)["notes"],
                                   track["item"]["samples"] / SR)
                fits.append(supervision(track, truth, seed=740+i,
                                        target=args.target))
                print("S740 FIT", args.target, i+1, 80,
                      fits[-1]["rescuable"], flush=True)
            combined = np.concatenate([x["features"] for x in fits])
            mean = combined.mean(axis=0)
            std = np.maximum(combined.std(axis=0), .02)
            oof = [training_audit(fits, i) for i in range(80)]
            global_audit = audit_all_fit(fits)
            report["FIT"] = dict(
                eligible_fragments=sum(x["sampled"] for x in fits),
                potentially_correctable=sum(x["rescuable"] for x in fits),
                compositions=sorted(set(x["source"] for x in fits)))
            save(args.out / "report.json", report)

            def load_split(phase):
                result = []
                for i, spec in enumerate(groups[phase]):
                    track = one_track(spec, audio, root, heads, selection, offset,
                                      network, policy, release, h5, args.target)
                    track["truth"] = real_notes(
                        labels(spec, root)["notes"], track["item"]["samples"]/SR)
                    track["policy"] = policy
                    track["offset"] = offset
                    track["release"] = release
                    result.append(track)
                    print("S740 EVAL LOAD", args.target, phase,
                          i+1, len(groups[phase]), flush=True)
                return result

            cal = load_split("CAL")
            baseline = aggregate([
                metric(x["baseline"], x["truth"], x["item"]["ends"])
                for x in cal])
            for metric_name in SCORES:
                if abs(baseline[metric_name] -
                       frozen["frozen_candidate"]["CAL"][metric_name]) > 1e-8:
                    raise AssertionError("Frozen S708 CAL mismatch " + metric_name)
            passing = []
            for seed in SEEDS:
                def callback(model, run_seed, epoch, fit_loss):
                    rows = []
                    for confidence in CAL_CONFIDENCE:
                        row = evaluate(cal, args.target, model, mean, std,
                                       global_audit, confidence)
                        row.update(seed=run_seed, epoch=epoch,
                                   confidence=confidence, FIT_loss=fit_loss)
                        rows.append(row)
                        report["CAL_rounds"].append(row)
                        print("S740 REAL CAL", json.dumps({
                            "target":args.target, "seed":run_seed, "epoch":epoch,
                            "confidence":confidence, "actual":row["actual"],
                            "safe":row["safe"], "all_gt80":row["all_gt80"]
                        }), flush=True)
                        if row["safe"] and any(
                                row["delta"][m] > 1e-10 for m in SCORES):
                            filename = (f"s740_{args.target}_{seed}_{epoch}.pt")
                            torch.save(dict(model=model.state_dict(),
                                 base_sha256=sha, target=args.target,
                                 mean=mean.tolist(), std=std.tolist()),
                                 args.out / filename)
                            passing.append((row, filename))
                    save(args.out / "report.json", report)
                    return rows
                train_one(seed, fits, oof, mean, std, max(STEPS), callback)

            if passing:
                best, filename = max(passing, key=lambda pair:(
                    pair[0]["all_gt80"],
                    min(pair[0]["delta"].values()),
                    sum(pair[0]["actual"].values())))
                model = FragmentMaskRouter()
                checkpoint = torch.load(
                    args.out / filename, map_location="cpu", weights_only=True)
                if checkpoint["base_sha256"] != sha \
                   or checkpoint["target"] != args.target:
                    raise AssertionError("Wrong FIT checkpoint")
                model.load_state_dict(checkpoint["model"], strict=True)
                report["selected"] = filename
                report["frozen_on_CAL"] = dict(
                    seed=best["seed"], epoch=best["epoch"],
                    confidence=best["confidence"])
                for phase in ("HELD", "DEV"):
                    tested = load_split(phase)
                    value = evaluate(tested, args.target, model, mean, std,
                                     global_audit, best["confidence"])
                    old = frozen[phase]["candidate"]
                    for name in SCORES:
                        if abs(value["reference"][name] - old[name]) > 1e-8:
                            raise AssertionError("Frozen S708 replay mismatch " + name)
                    report[phase] = value
                    save(args.out / "report.json", report)
            else:
                report["CAL_no_safe_improvement"] = True
                report["selected"] = REFERENCE_NAME

            report["status"] = "FIT_CAL_COMPLETED"
            report["completed"] = True
            report["achieved_all_six_gt80"] = bool(
                passing and any(row["safe"] and row["all_gt80"]
                                for row, _ in passing)
                and all(report.get(phase, {}).get("safe", False)
                        for phase in ("HELD", "DEV")))
            save(args.out / "report.json", report)
            print("S740_FINAL", json.dumps(dict(
                target=args.target, selected=report["selected"],
                rounds=len(report["CAL_rounds"]),
                achieved_all_six_gt80=report["achieved_all_six_gt80"],
                no_live_change=True)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=tuple(CHANNELS), required=True)
    parser.add_argument("--s708-dir", type=Path, required=True)
    parser.add_argument("--s734-checkpoint", type=Path, required=True)
    parser.add_argument("--s734-report", type=Path, required=True)
    parser.add_argument("--offset-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args)
