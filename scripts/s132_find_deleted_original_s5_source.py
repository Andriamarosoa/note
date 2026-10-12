"""S132 forensic search for deleted ORIGINAL S4-owned/S5-527D source.

Walk real Git commit *history*, not merely current branch tips, before the
original S5 checkpoint preservation date. A previously deleted script may
exist in an ancestor commit even if it is missing on every branch HEAD.
Never treats textual matches as executable reconstruction proof.
"""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path

BEFORE = "2026-10-08T21:10:00+00:00"
PIN = "7165e1ef54510aa9effc9f597b2783c832ca907a"
SIGS = [
    "owned",
    "527",
    "positive",
    "negative",
    "shift",
    "concentration",
    "ownership",
    "S5",
]

def cmd(*args, allowed=(0,)):
    p=subprocess.run(["git",*args],capture_output=True,text=True)
    if p.returncode not in allowed:
        raise RuntimeError(str(args)+": "+p.stderr[-1600:])
    return p.stdout

def scan(out):
    if out.exists():raise ValueError("immutable output required")
    reachable = cmd("rev-list","--count","--all","--before="+BEFORE).strip()
    pin_files=cmd("diff-tree","--no-commit-id","--name-only","-r",PIN).splitlines()
    # A protected historical chronological boundary: no S118+ or new work
    # may masquerade as original S4/S5 source.
    patterns = [
       (r"\bowned\b|\bowned527\b|\bS4\b", "owned_S4"),
       (r"\b527\b|\b416\b|extra416", "dimension_527_416"),
       (r"concentration|negative_mass|shift_frames|temporal.*flux", "protocol_operations"),
    ]
    hits={}
    for expression, name in patterns:
        # --pickaxe-regex option is absent: -G is inherently regex matching.
        rows=cmd("log","--all","--before="+BEFORE,"--format=%H%x09%ad%x09%s",
                 "--date=iso-strict","-G",expression,"--",
                 "scripts/","src/",".github/workflows/",":(glob)**/*.ipynb").splitlines()
        detail=[]
        for row in rows[:45]:
            if "\t" not in row:continue
            sha, date, title=row.split("\t",2)
            names=cmd("diff-tree","--no-commit-id","--name-only","-r",sha,
                      "--","scripts/","src/",".github/workflows/",":(glob)**/*.ipynb").splitlines()
            # Some commit messages contain matches only in imported context.
            detail.append(dict(commit=sha,date=date,title=title,
                               path_candidates=names[:35]))
        hits[name]={"match_count":len(rows),"displayed":detail,
                    "truncated":len(rows)>45}
    proto=cmd("show","--format=%s","--no-patch","8a9410a4232c653615eb32db401410202831377f").strip()
    old=cmd("ls-tree","-r","--name-only",PIN,"scripts/").splitlines()
    direct=[x for x in old if
            ("owned" in x.lower() or "series5" in x.lower() or "series4" in x.lower())]
    diff_nondata=[f for f in pin_files if f.startswith(("scripts/",".github/workflows/"))]
    # Look for historical actual source additions by exact checkpoint ancestry
    log=cmd("log","--oneline","8a9410a4232c653615eb32db401410202831377f.."+PIN).splitlines()
    report={
       "status":"S132_ORIGINAL_GIT_HISTORY_DELETED_SOURCE_SEARCH",
       "original_S5_checkpoint_commit":PIN,
       "cutoff_before_checkpoint_date":BEFORE,
       "commits_accessible_before_cutoff":int(reachable),
       "source_signatures_checked":[n for _,n in patterns],
       "source_history_hits":hits,
       "original_s4_s5_training_files_named_at_pin":direct,
       "checkpoint_commit_source_or_workflow_files":diff_nondata,
       "protocol_to_checkpoint_history":log,
       "all_visible_source_proved_absent":False,
       "found_genuine_original_527D_constructor":False,
       "GOAT_independent_S80_score":None,
       "S80_promotion":False,
       "interpretation":"Git -G detects string-added/removed commits, not every meaning or unreachable local files; any hits require manual source inspection and numeric replay."
    }
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({
        "status":report["status"],
        "history_commit_count":report["commits_accessible_before_cutoff"],
        "signature_hits":{k:v["match_count"] for k,v in hits.items()},
        "first_matches":{k:v["displayed"][:12] for k,v in hits.items()},
        "named_original_s4_s5_script_candidates":direct,
        "checkpoint_source_added":diff_nondata,
        "original_527D_verified":False,
        "S80_promotion":False},indent=2))
    return report

def selftest():
    assert PIN.startswith("7165e1") and len(PIN)==40
    assert all(" " not in z for z in ("owned_S4","dimension_527_416","protocol_operations"))
    print("S132 PASS: history-bound source archaeology protocol")
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path)
    ap.add_argument("--self-test",action="store_true")
    a=ap.parse_args()
    if a.self_test:selftest()
    else:
        if a.out is None: ap.error("--out required")
        scan(a.out)
