"""Inventory EVERY tracked historical hypothesis/experiment from all Git branches.

No source file is silently discarded: every experiment-bearing file in every
available branch receives a stable registry identifier. Registry entries are
CANDIDATES, not runnable trained heads. Exact-K adapters must provide aligned
out-of-fold K0..K6 logits and training-only audits before activation.

Usage: python scripts/build_v273_head_history_registry.py --output DIR
For complete history, run on a git clone with all remote branches fetched.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

MARKERS = (
    "train_", "audit_", "evaluate_", "extract_", "summarize_", "calibrate_",
    "prepare_", "probe_", "replay_", "predict_", "run_", "cluster_",
    "restore_", "recover_", "diagnose_", "sweep_", "mine_", "benchmark_",
    "detect_", "test_", "bootstrap_", "check_",
)
GROUPS = {
    "energy_flow":r"energ|flux|birth|death|continuity|navier|damp|decay|mass|source.*time",
    "pitch_transform":r"pitch|compression|transpos|rubber|semitone|high[_-]",
    "harmonics":r"harmonic|f0|spectral|residual|reconstruction|cqt|frequency",
    "temporal":r"temporal|onset|attack|hysteresis|window|persistence|stability|trajectory",
    "candidate_selection":r"candidate|rank|group|subset|selection|slot|set[_-]|exclusiv|matching",
    "learning_architecture":r"expert|moe|fusi|calib|classifier|decoder|model|neuron|head",
    "correctors_fixes":r"correct|fix|guard|veto|rescue|repair|transplant|ablation|intervention",
    "audit_metrics":r"audit|score|evaluate|nested|oof|confus|metric|validation",
}
def git(*args):
    p=subprocess.run(["git",*args],capture_output=True,text=True,check=True)
    return p.stdout.strip()

def branches():
    refs=git("for-each-ref","--format=%(refname:short)","refs/remotes/origin","refs/heads")
    refs=[r for r in refs.splitlines() if r and r!="origin/HEAD"]
    if not refs:refs=["HEAD"]
    return sorted(set(refs))

def family(path):
    name=Path(path).name.lower()
    matches=[k for k,rx in GROUPS.items() if re.search(rx,name)]
    return matches if matches else ["misc_archived"]

def role(path):
    name=Path(path).name.lower()
    if re.search(r"correct|fix|guard|veto|rescue|transplant|patch|repair|intervention",name):
        return "correction_or_fix_candidate"
    if re.search(r"(train_|decode|_moe|_expert|_network|_model)",name):
        return "predictor_or_neural_candidate"
    if re.search(r"extract|spectral|cqt|morphol|flow|feature|embedding",name):
        return "feature_candidate"
    return "evidence_or_audit"

def tracked(path):
    if path.startswith("scripts/") and path.endswith(".py"):
        return any(Path(path).name.startswith(m) for m in MARKERS)
    if path.startswith("analysis/") and Path(path).suffix.lower() in {".md",".json"}:
        return True  # preserve every historical hypothesis/protocol and its provenance
    if path.startswith("src/") and path.endswith(".py") and "test" not in path:
        return True
    return False

def collect(refs):
    registry={}
    inaccessible=[]
    for branch in refs:
        try:
            revision=git("rev-parse",branch)
            files=git("ls-tree","-r","--name-only",branch).splitlines()
        except subprocess.CalledProcessError:
            inaccessible.append(branch)
            continue
        for path in files:
            if not tracked(path):continue
            key=hashlib.sha256((branch+"\0"+path).encode()).hexdigest()[:16]
            registry[key]=dict(
                head_id="HIST-"+key,source_branch=branch,
                commit=revision,source_path=path,source_type=role(path),
                research_families=family(path),
                implementation_status="historical_candidate_not_activated",
                aligned_oof_predictions=False,train_only_k_audit=False,
                adapter=None,selection_eligible=False,
                provenance="tracked branch file; no automatic performance claim",
            )
    return list(registry.values()),inaccessible

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError("refuse overwriting existing registry")
    refs=branches()
    entries,missing=collect(refs)
    if len(entries)<100:raise RuntimeError("historical inventory suspiciously small")
    by_role=dict(Counter(x["source_type"] for x in entries))
    by_family=dict(Counter(f for x in entries for f in x["research_families"]))
    report=dict(
        scope="all fetched local and remote git refs; NOT proof all remote history has been fetched",
        refs=refs,unreadable_refs=missing,
        entries=len(entries),counts_by_role=by_role,
        counts_by_family=by_family,
        note=(
            "Each entry is an EXPERIMENT or EVIDENCE source, not a runnable neural head. "
            "An actual prediction, feature, correction or fix becomes selectable only "
            "once it exposes aligned OOF K0-K6 action logits and train-only audit."
        ),
    )
    a.output.mkdir(parents=True)
    (a.output/"history-head-registry.json").write_text(
        json.dumps(entries,indent=2,sort_keys=True)+"\n"
    )
    (a.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("entries","counts_by_role","counts_by_family","unreadable_refs")},
                     sort_keys=True),flush=True)

if __name__=="__main__":
    main()
