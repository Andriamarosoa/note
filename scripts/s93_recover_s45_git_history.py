"""S93 locate the genuine S45 producers in git history, not surrogate models."""
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
TERMS=("S45","series45","s45","risk_scores","S40_temporal_CNN","S19_recurrent_ABA","historical_repair_coherent")
def git(*args):
    r=subprocess.run(["git",*args],capture_output=True,text=True,check=True)
    return r.stdout
def run(out):
    commits={}
    for term in TERMS:
        # -G examines changed source lines, including deleted code. All refs fetched in CI.
        lines=git("log","--all","--format=%H","-G",term,"--","scripts","analysis",".github/workflows")
        commits[term]=list(dict.fromkeys(lines.splitlines()))[:30]
    source_candidates=[]
    for h in sorted({x for group in commits.values() for x in group}):
        paths=git("show","--pretty=format:","--name-only",h).splitlines()
        matches=[p for p in paths if p.endswith(".py") and p.startswith("scripts/")]
        if matches: source_candidates.append({"commit":h,"python_files":matches[:30]})
    result={"status":"S93_HISTORY_PROVENANCE_SCAN","terms":commits,"candidate_source_commits":source_candidates,
            "s45_new_wav_inference_verified":False,"external_S80_measured":False,"promotion":False}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":result["status"],"candidate_commits":len(source_candidates),
                       "hits_by_term":{k:len(v) for k,v in commits.items()}},sort_keys=True))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);run(p.parse_args().output)
