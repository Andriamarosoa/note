"""Post-freeze, independent matching and per-note regressions; never tune."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
from scripts.s613_prepare_strict_live import digest, save


def maximum_matching(predicted, truth, tolerance):
    """Independent augmenting-path cardinality check, without scipy/Hungarian."""
    adjacency = [[j for j, note in enumerate(truth) if note["pitch"] == event["pitch"]
                  and abs(note["time"] - event["time"]) <= tolerance] for event in predicted]
    owner = {}
    def augment(i, visited):
        for j in adjacency[i]:
            if j in visited:
                continue
            visited.add(j)
            if j not in owner or augment(owner[j], visited):
                owner[j] = i
                return True
        return False
    return sum(augment(i, set()) for i in range(len(predicted)))


def audit(root, baseline, selected, out):
    protocol = json.loads((root / "protocol.json").read_text())
    old = json.loads((baseline / "test_individual_audit.json").read_text())
    new = json.loads((selected / "test_individual_audit.json").read_text())
    op = json.loads((baseline / "test_predictions_before_labels.json").read_text())
    np = json.loads((selected / "test_predictions_before_labels.json").read_text())
    assert set(old) == set(new) == set(op["predictions"]) == set(np["predictions"])
    checks, sources, gains, losses = 0, [], [], []
    for spec in protocol["sources"]:
        sid = spec["id"]
        if spec["split"] != "test":
            continue
        truth = new[sid]["truth"]
        assert truth == old[sid]["truth"]
        for events, ledger in ((op["predictions"][sid], old[sid]), (np["predictions"][sid], new[sid])):
            predicted = [e for e in events if e["type"] == "note_on"]
            for name, tolerance in (("50ms", .05), ("80ms", .08)):
                assert maximum_matching(predicted, truth, tolerance) == ledger[name]["score"]["TP"]
                checks += 1
        a = {j for i, j in old[sid]["80ms"]["pairs"]}
        b = {j for i, j in new[sid]["80ms"]["pairs"]}
        gains.extend(dict(source=spec["name"], corpus=spec["corpus"], **truth[j]) for j in sorted(b - a))
        losses.extend(dict(source=spec["name"], corpus=spec["corpus"], **truth[j]) for j in sorted(a - b))
        os, ns = old[sid]["80ms"]["score"], new[sid]["80ms"]["score"]
        sources.append(dict(source=spec["name"], corpus=spec["corpus"], truth=len(truth),
                            baseline={k: os[k] for k in ("TP", "FP", "FN", "F1")},
                            selected={k: ns[k] for k in ("TP", "FP", "FN", "F1")},
                            gained=len(b - a), lost=len(a - b)))
    result = dict(status="S614_RESERVED_POSTFROZEN_INDIVIDUAL_AUDIT", independent_matching_checks=checks,
                  independent_matching_disagreements=0, gained_true_noteons=len(gains), lost_true_noteons=len(losses),
                  source_F1_improved=sum(s["selected"]["F1"] > s["baseline"]["F1"] for s in sources),
                  source_F1_regressed=sum(s["selected"]["F1"] < s["baseline"]["F1"] for s in sources),
                  source_F1_equal=sum(s["selected"]["F1"] == s["baseline"]["F1"] for s in sources),
                  by_source=sources, gained_notes=gains, lost_notes=losses,
                  baseline_predictions_sha256=digest((baseline / "test_predictions_before_labels.json").read_bytes()),
                  selected_predictions_sha256=digest((selected / "test_predictions_before_labels.json").read_bytes()),
                  no_model_or_threshold_change=True)
    save(out, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("by_source", "gained_notes", "lost_notes")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "baseline", "selected", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.baseline, args.selected, args.out)
