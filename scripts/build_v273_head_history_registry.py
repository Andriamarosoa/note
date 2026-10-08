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
    if re.search(r"(train_|learn_|decode|_moe|_expert|_network|_model)",name):
        return "predictor_or_neural_candidate"
    if re.search(r"extract|spectral|cqt|morphol|flow|feature|embedding",name):
        return "feature_candidate"
    return "evidence_or_audit"

def tracked(path):
    if path.startswith("scripts/") and path.endswith(".py"):
        return True  # helpers/learn_ modules also implement hypotheses and corrections
    if path.startswith("analysis/"):
        return True  # retain numerical evidence (CSV/NPZ/etc.), not only prose
    if path.startswith(("src/", "test/", "tests/")) and path.endswith(".py"):
        return True
    if path.startswith(".github/workflows/") and Path(path).suffix in {".yml", ".yaml"}:
        return True
    if path.startswith("docs/") or Path(path).name.lower().startswith("readme"):
        return True
    return False

def historical_fix_commits():
    """Every reachable Git commit explicitly labeled as a correction or fix.

    A historical *patch* can later become an action/head candidate. A patch
    alone is not a runnable neural head, so it remains inactive pending an
    adapted, aligned prediction and independent training-only audit.
    """
    command=("log","--all","--no-merges","--extended-regexp",
             "--regexp-ignore-case",
             "--grep=(fix|correct|rescue|guard|transplant|repair|patch|veto|rollback)",
             "--name-only","--pretty=format:COMMIT\\t%H\\t%s")
    lines=git(*command).splitlines()
    commit=None
    records=[]
    seen=set()
    for line in lines:
        if line.startswith("COMMIT\\t"):
            parts=line.split("\\t",2)
            commit=(parts[1],parts[2]) if len(parts)==3 else None
        elif commit and line.startswith(("scripts/","src/")) and line.endswith(".py"):
            revision,subject=commit
            key=(revision,line)
            if key in seen:continue
            seen.add(key)
            id_=hashlib.sha256(("PATCH\\0"+revision+"\\0"+line).encode()).hexdigest()[:16]
            records.append(dict(
                head_id="PATCH-"+id_,
                source_branch="historical_git_commit",
                commit=revision,source_path=line,
                commit_subject=subject,source_type="correction_or_fix_candidate",
                research_families=family(line),
                implementation_status="historical_patch_not_activated",
                aligned_oof_predictions=False,train_only_k_audit=False,
                adapter=None,selection_eligible=False,
                provenance="commit-level fix/correction patch; adapter and alignment required",
            ))
    return records

def collect(refs):
    registry={}
    inaccessible=[]
    for branch in refs:
        try:
            revision=git("rev-parse",branch)
            entries=git("ls-tree","-r","--format=%(objectname) %(path)",branch).splitlines()
            files={line.split(" ",1)[1]:line.split(" ",1)[0] for line in entries}
        except subprocess.CalledProcessError:
            inaccessible.append(branch)
            continue
        for path in files:
            if not tracked(path):continue
            key=hashlib.sha256((branch+"\0"+path).encode()).hexdigest()[:16]
            registry[key]=dict(
                head_id="HIST-"+key,source_branch=branch,
                commit=revision,source_path=path,source_type=role(path),
                source_blob_sha=files[path],
                research_families=family(path),
                implementation_status="historical_candidate_not_activated",
                aligned_oof_predictions=False,train_only_k_audit=False,
                adapter=None,selection_eligible=False,
                provenance="tracked branch file; no automatic performance claim",
            )
    # Preserve fixes/repair patches in their OWN right, even when they
    # modify a model file that otherwise appears only as one head source.
    patches=historical_fix_commits()
    return list(registry.values())+patches,inaccessible,len(patches)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError("refuse overwriting existing registry")
    refs=branches()
    entries,missing,patch_count=collect(refs)
    if len(entries)<100:raise RuntimeError("historical inventory suspiciously small")
    by_role=dict(Counter(x["source_type"] for x in entries))
    by_family=dict(Counter(f for x in entries for f in x["research_families"]))
    report=dict(
        scope="all fetched local and remote git refs; NOT proof all remote history has been fetched",
        refs=refs,unreadable_refs=missing,
        entries=len(entries),fix_patch_entries=patch_count,counts_by_role=by_role,
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
    print(json.dumps({k:report[k] for k in ("entries","fix_patch_entries","counts_by_role","counts_by_family","unreadable_refs")},
                     sort_keys=True),flush=True)

if __name__=="__main__":
    main()

