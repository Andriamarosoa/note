"""Independently recount every S704-S709 emitted-stream artifact; read-only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile
from scripts.s704_verify_emitted_artifact import verify, METRICS

REPO = "Andriamarosoa/note"
SPECS = [
    ("s704", 39092626757, "S704_JOINT_CAUSAL_ACOUSTIC_LOOPS", "a137ae0b4f62141c2dd3cfdce46cd6e30a314f99"),
    ("s705", 39092939353, "S705_TEMPORAL_HARMONIC_CAUSAL_LOOPS", "8180a0c8847fc598d5fb859434658d3d11ab3835"),
    ("s706", 39093239181, "S706_SHARP_SPECTRAL_TEMPORAL_CAUSAL_LOOPS", "20f8536b33d3a839accc7d8cad0cba2dde6c445c"),
    ("s707-705", 39093963095, "S707_REFINED_S705_JOINT_CAUSAL_LOOPS", "2cdff0539b0fcbe7dfde9c02e2e9450635856643"),
    ("s707-706", 39093963095, "S707_REFINED_S706_JOINT_CAUSAL_LOOPS", "2cdff0539b0fcbe7dfde9c02e2e9450635856643"),
    ("s708", 39094682419, "S708_K_GUIDED_S706_JOINT_CAUSAL_LOOPS", "d19f1741de1e9cdc6e52a6f665b71a18ca36b873"),
    ("s709", 39095466721, "S709_SIX_STRING_NATIVE_CAUSAL_HEADS", "d133288208c3a5220c05b0fba0ebb76222287a35"),
]

def api(path):
    return json.loads(subprocess.check_output(["gh", "api", path], text=True))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    completed=set()
    all_evidence={}
    for tag, run_id, name, sha in SPECS:
        if run_id not in completed:
            status=api(f"repos/{REPO}/actions/runs/{run_id}")
            if status["status"] != "completed":
                subprocess.run(["gh", "run", "watch", str(run_id), "--repo", REPO,
                                "--exit-status", "--interval", "30"], check=True)
                status=api(f"repos/{REPO}/actions/runs/{run_id}")
            assert status["conclusion"]=="success", (run_id, status["conclusion"])
            assert status["head_sha"]==sha, (run_id,status["head_sha"],sha)
            completed.add(run_id)
        artifacts=api(f"repos/{REPO}/actions/runs/{run_id}/artifacts?per_page=100")["artifacts"]
        matches=[a for a in artifacts if a["name"]==name and not a["expired"]]
        assert len(matches)==1, (tag, [a["name"] for a in artifacts])
        meta=matches[0]
        assert meta["workflow_run"]["head_sha"]==sha
        archive=args.out/(tag+".zip")
        with archive.open("wb") as fh:
            subprocess.run(["gh","api",f"repos/{REPO}/actions/artifacts/{meta['id']}/zip"],
                           stdout=fh,check=True)
        assert archive.stat().st_size==meta["size_in_bytes"]
        digest=hashlib.sha256(archive.read_bytes()).hexdigest()
        assert meta["digest"]=="sha256:"+digest, (tag,meta["digest"],digest)
        result=verify(archive)
        assert not result["TEST_opened"] and not result["promoted"]
        assert all(result[p]["candidate"]["max_active"]<=6 for p in ("HELD","DEV"))
        with zipfile.ZipFile(archive) as z:
            def member(wanted):
                entries=[n for n in z.namelist() if n==wanted or n.endswith("/"+wanted)]
                assert len(entries)==1,(wanted,entries)
                return entries[0]
            report=json.loads(z.read(member("report.json")))
            (args.out/(tag+"-report.json")).write_text(json.dumps(report,indent=2)+"\n")
            exact_block_end={}
            for phase in ("HELD","DEV"):
                rows=[json.loads(l) for l in z.read(member(phase.lower()+"_events.jsonl")).splitlines()]
                exact_block_end[phase]=[r["name"] for r in rows if r["samples"]%1024==0]
        result["provenance"]={
            "repository":REPO,"run_id":run_id,"head_sha":sha,
            "artifact_id":meta["id"],"artifact_name":name,
            "artifact_size_in_bytes":meta["size_in_bytes"],"artifact_digest":meta["digest"],
            "verifier_commit":os.environ.get("GITHUB_SHA"),
        }
        result["paired"]={p:report[p].get("paired",{}) for p in ("HELD","DEV")}
        result["non_regression"]={p:report[p].get("non_regression") for p in ("HELD","DEV")}
        result["exact_block_end_sources"]=exact_block_end
        result["rounds_summary"]=[
            {k:v for k,v in r.items() if k!="scores"} | {"policy_evaluations":len(r["scores"])}
            for r in report["rounds"]
        ]
        (args.out/(tag+"-verified.json")).write_text(json.dumps(result,indent=2)+"\n")
        all_evidence[tag]=result
        print("VERIFIED_EVIDENCE "+json.dumps({"tag":tag,"verified":result},separators=(",",":")),flush=True)
        brief={"tag":tag,"policy_evaluations":result["policy_evaluations"],
               "all_targets_above_80":result["all_targets_above_80"],
               **{p:{k:result[p]["candidate"][k] for k in METRICS} for p in ("HELD","DEV")}}
        print("VERIFIED_SCORES "+json.dumps(brief,separators=(",",":")),flush=True)
    total=sum(r["policy_evaluations"] for r in all_evidence.values())
    summary={"artifacts_verified":len(all_evidence),"policy_evaluations":total,
             "passing_candidates":[tag for tag,r in all_evidence.items() if r["all_targets_above_80"]],
             "all_raw_event_counts_match":all(r["all_raw_event_counts_match_report"] for r in all_evidence.values()),
             "TEST_opened":False,"promoted":False}
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("FINAL_RECOUNT "+json.dumps(summary,separators=(",",":")),flush=True)

if __name__=="__main__":
    main()
