"""CAL-only experiment on verified S710 cached probabilities; no model loading."""
import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
from causal_note.s704_joint_heads import JointPolicy, JointSelector
from causal_note.s708_cardinality_selection import GuidedPolicy, GuidedSelector
from causal_note.s712_attack_memory import MemoryPolicy, decode_memory
from scripts.s704_joint_metrics import metric, aggregate, target_rank, SCORES
from scripts.s704_verify_emitted_artifact import recount, total


def base_selector(family):
    common = dict(onset=.6, sustain_on=.5, release=.85, count_confidence=1.1, blend=.7)
    if family == "706":
        return JointSelector(JointPolicy(**common, sustain_off=.3))
    if family == "709":
        return GuidedSelector(GuidedPolicy(**common, sustain_off=.15,
                              guide_confidence=.5, weak_onset=.2, weak_sustain=.35))
    raise ValueError(family)


def choices():
    return {f"{family}_on{on}_presence{presence}": dict(
        family=family, memory=asdict(MemoryPolicy(on, presence)))
        for family, on, presence in itertools.product(("706", "709"), range(4), range(3))}


def decode_choice(item, choice):
    post, count = item["models"][choice["family"]]
    return decode_memory(item["decision_ends"], post, count, base_selector(choice["family"]),
                         MemoryPolicy(**choice["memory"]), item["samples"])


def calibrate(cache, out):
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((cache / "manifest.json").read_text())
    diagnosis = json.loads((cache / "diagnosis.json").read_text())
    assert manifest["TEST_opened"] is False
    policies = choices()
    rows = {name: [] for name in policies}
    paths = sorted(cache.glob("cal_*.joblib"))
    assert len(paths) == 32
    for i, path in enumerate(paths):
        item = joblib.load(path)
        assert item["spec"]["composition"] in manifest["cal_compositions"]
        for name, choice in policies.items():
            events = decode_choice(item, choice)
            rows[name].append(metric(events, item["truth"], item["ends"]))
        print("S712 CAL", i + 1, len(paths), flush=True)
    scores = {name: aggregate(values) for name, values in rows.items()}
    for family in ("706", "709"):
        expected = diagnosis[family]["all"]["emitted"]
        observed = scores[family + "_on0_presence0"]
        for key in (*SCORES, "tp", "fp", "fn", "offset_correct", "k_correct"):
            assert abs(expected[key] - observed[key]) < 1e-12, (family, key)
    best = max(scores, key=lambda name: target_rank(scores[name]))
    independent = []
    with (out / "cal_selected_events.jsonl").open("w") as fh:
        for path in paths:
            item = joblib.load(path)
            events = decode_choice(item, policies[best])
            independent.append(recount(events, item["truth"], item["samples"]))
            fh.write(json.dumps(dict(name=item["spec"]["name"], samples=item["samples"],
                                     sha256=item["sha256"], truth=item["truth"], events=events)) + "\n")
    verified = total(independent)
    for key in verified:
        if key in scores[best]:
            assert abs(verified[key] - scores[best][key]) < 1e-12, key
    control = max((family + "_on0_presence0" for family in ("706", "709")),
                  key=lambda name: target_rank(scores[name]))
    report = dict(schema="S712_CAUSAL_ATTACK_PRESENCE_MEMORY_CAL", TEST_opened=False,
                  promoted=False, official_LIVE_changed=False, HELD_evaluated=False, DEV_evaluated=False,
                  source_checkpoints=manifest["checkpoint_sha256"], policies=policies,
                  rounds=[dict(family=family, scores={n: s for n, s in scores.items()
                         if policies[n]["family"] == family}) for family in ("706", "709")],
                  policy_evaluations=len(policies), frozen_candidate=dict(name=best, choice=policies[best], CAL=scores[best]),
                  controls_reproduce_S710_CAL=True, selected_CAL_independent_recount=verified,
                  improves_best_control=target_rank(scores[best]) > target_rank(scores[control]),
                  best_control=control, calibration_only=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "selection_frozen.json").write_text(json.dumps(report["frozen_candidate"], indent=2) + "\n")
    print("S712 CAL FINAL " + json.dumps({k: report[k] for k in (
          "frozen_candidate", "improves_best_control", "policy_evaluations")}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cache", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    calibrate(args.cache, args.out)
