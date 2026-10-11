"""S114 exhaustive repository provenance audit for original S4/S5 527D designs.

Examines ALL fetched local branch heads, versioned scripts/workflows, and
commit history. Reports matching file:line evidence rather than inferring
where training code ought to be.
"""
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
KEYWORDS=("designs_sha256","27407","series5/models","series5__","527","owned__poly","S5","series4")
SCOPES=("scripts",".github/workflows","test","analysis")
def call(*args,valid=(0,)):
    p=subprocess.run(["git",*args],capture_output=True,text=True)
    if p.returncode not in valid:raise ValueError(p.stderr[-1000:])
    return p.stdout

def audit(out):
    all_refs=call("for-each-ref","--format=%(refname:short)","refs/remotes/origin/","refs/tags/").splitlines()
    refs=sorted(set(r for r in all_refs if r.startswith("origin/") or
                    r.startswith("v273-open") or r.startswith("v273-series")))
    scan=[]
    pattern="designs_sha256|27407|series5/models|series5__|527|owned__poly|series4"
    for ref in refs:
        raw=call("grep","-n","-I","-E",pattern,ref,"--",*SCOPES,valid=(0,1))
        matches=[]
        for row in raw.splitlines():
            if "/evidence/" in row or "/source-archives/" in row:continue
            if len(row)>500:row=row[:500]
            matches.append(row)
        if matches:scan.append(dict(ref=ref,source_lines=matches[:60]))
    historical={}
    for term in ("designs_sha256","527","27407","series5/models","owned__poly"):
        cmd=["log","--all","--since=2026-10-06","-G",term,"--format=%H %s","--",*SCOPES]
        historical[term]=call(*cmd).splitlines()[:30]
    output={"status":"S114_GITHUB_FULL_BRANCH_WORKFLOW_PROVENANCE",
      "refs_scanned":len(refs),"refs_with_source_hits":len(scan),
      "ref_details":scan,"commit_history_matches":historical,
      "original_527D_builder_located_and_executed":False,
      "validated_S80_externally":False,"promoted":False}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(output,indent=2)+"\n")
    short=[{"ref":r["ref"],"terms":list(r["source_lines"])} for r in scan]
    print(json.dumps({"status":output["status"],"refs_scanned":len(refs),
      "matches":short,"history":historical},indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    audit(p.parse_args().out)
