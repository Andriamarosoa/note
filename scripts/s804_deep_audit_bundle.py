"""S804 audited real GuitarSet FIT80 evidence and complete Python static inventory.

Full Git-tracked Python AST coverage + bounded, explicitly identified hot-path
source excerpts + verification of all eighty signed S800 FIT tracks. Only
sanitized text/numeric aggregates are sent to NVIDIA; no audio, truth tables,
raw OOF arrays, reserved CAL/HELD/TEST material, or credentials.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, re, subprocess
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from causal_note.s795_activek_experts import HEADS, FEATURES, PITCHES
from scripts.s800_oof_trace import verify_snapshot, SCHEMA as S800_SCHEMA

SCHEMA="S804_FIT80_CODE_AND_DATA_DEEP_AUDIT_BUNDLE_V1"
FOCUS=(
    "src/causal_note/s795_activek_experts.py",
    "src/causal_note/s797_fragment_router.py",
    "src/causal_note/s793_zero_added_latency.py",
    "src/causal_note/strict_live.py",
    "scripts/s704_joint_metrics.py",
    "scripts/s613_train_strict_live.py",
    "scripts/s800_oof_trace.py",
    "scripts/s801_fit_fragment_ablations.py",
    "scripts/s802_retrigger_forensics.py",
    "scripts/s803_causal_frame_replay.py",
    "scripts/s803_six_role_audits.py",
)
# Only include matching code windows; report omitted lines explicitly.
ANCHORS=("def decode(", "def route_probabilities(", "def event_score(",
         "def metric(", "def aggregate(", "def veto_promotion(",
         "def verify_snapshot(", "def evaluate(", "def gating(",
         "def classify_events(", "def audit(",
         "def schedule(", "def process_block(")
SENSITIVE=re.compile(r"(?i)(?:nvapi-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|"
                     r"github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{32,})")
def digest(data):return hashlib.sha256(data).hexdigest()
def tracked(root):
    raw=subprocess.run(["git","ls-files","-z"],cwd=root,check=True,
                       capture_output=True).stdout
    return [str(Path(x.decode())) for x in raw.split(b"\0") if x]
def code_inventory(root):
    paths=[x for x in tracked(root) if x.endswith(".py")]
    if len(paths)<100:raise AssertionError("abnormally small Python inventory")
    syntax_errors=[];lines=0;functions=0;classes=0
    by_dir=Counter();sha=hashlib.sha256()
    for path in sorted(paths):
        f=root/path
        raw=f.read_bytes()
        sha.update(path.encode()+b"\0"+hashlib.sha256(raw).digest())
        source=raw.decode("utf-8-sig")
        lines+=len(source.splitlines())
        by_dir[path.split("/")[0]]+=1
        try:
            tree=ast.parse(source,filename=path)
            functions+=sum(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))
                           for n in ast.walk(tree))
            classes+=sum(isinstance(n,ast.ClassDef) for n in ast.walk(tree))
        except (SyntaxError,UnicodeError) as err:
            syntax_errors.append({"file":path,"line":getattr(err,"lineno",None),
                                  "error_type":type(err).__name__})
    excerpts=[]
    for path in FOCUS:
        f=root/path
        if not f.exists():raise AssertionError("required source missing: "+path)
        content=f.read_text(encoding="utf-8")
        if SENSITIVE.search(content):raise AssertionError("credential-looking code in "+path)
        source=content.splitlines()
        include=set(range(min(25,len(source))))
        for index,line in enumerate(source):
            if any(token in line for token in ANCHORS):
                include.update(range(max(0,index-4),min(len(source),index+38)))
        chosen=sorted(include)
        # Explicit hard cap and gap labels, never imply complete contents.
        parts=[];last=-2
        for index in chosen:
            if sum(len(z) for z in parts)>7400:break
            if index>last+1:parts.append("... [other lines omitted]\n")
            parts.append(str(index+1)+": "+source[index][:250]+"\n")
            last=index
        excerpts.append({"path":path,"sha256":digest(content.encode()),
                         "total_lines":len(source),
                         "excerpt_lines":len([z for z in parts if re.match(r"\d+:",z)]),
                         "excerpt":"".join(parts)})
    if syntax_errors:raise AssertionError("Python AST parse failures: "+str(syntax_errors[:8]))
    return {"python_file_count":len(paths),"python_lines":lines,
            "function_count":functions,"class_count":classes,
            "source_tree_digest":sha.hexdigest(),"syntax_errors":syntax_errors,
            "by_top_folder":dict(sorted(by_dir.items())),
            "deep_code_excerpts":excerpts,
            "all_python_files_AST_parsed":True,
            "full_source_review_by_models":False}

def dataset_audit(manifest_path,traces,s803_path,s802_path):
    manifest=json.loads(manifest_path.read_text())
    s803=json.loads(s803_path.read_text())
    s802=json.loads(s802_path.read_text())
    if manifest.get("schema")!=S800_SCHEMA or not manifest.get("complete"):
        raise AssertionError("signed complete S800 manifest absent")
    if s803.get("schema")!="S803_FIT80_CAUSAL_VETO_SIX_METRIC_V1" or not s803.get("complete"):
        raise AssertionError("complete S803 real FIT report required")
    if s802.get("schema")!="S802_FIT80_NATIVE_SYNTHETIC_COLLISION_FORENSIC_V1" or not s802.get("complete"):
        raise AssertionError("S802 real post-hoc matching required")
    for data in (manifest,s802,s803):
        if any(data.get(n) for n in ("CAL_opened","HELD_opened","TEST_opened")):
            raise AssertionError("heldout split opened")
    items=manifest.get("fit_files",[])
    if len(items)!=80 or len({e["index"] for e in items})!=80:
        raise AssertionError("missing 80 distinct GuitarSet OOF traces")
    tracks=[];pitch_events=Counter();by_comp=defaultdict(lambda:dict(tracks=0,
                             frames=0,source_onsets=0,gt_notes=0))
    for entry in sorted(items,key=lambda e:e["index"]):
        info=verify_snapshot(entry,traces)
        path=traces/("fit_%03d.npz"%entry["index"])
        with np.load(path,allow_pickle=False) as z:
            X=z["features"]
            decision=z["decision_ends"]
            metrics=z["metric_ends"]
            probs=[z["head_"+h] for h in HEADS]
            if X.shape[1:]!=(PITCHES,FEATURES):raise AssertionError("feature ABI drift")
            if len(decision)!=len(X) or np.any(np.diff(decision)<=0):
                raise AssertionError("nonmonotonic causal decision clock")
            if not np.isfinite(X).all():raise AssertionError("invalid FIT features")
            if len(metrics)==0 or np.any(np.diff(metrics)<=0):
                raise AssertionError("nonmonotonic MIDI256 metric clock")
            for prob in probs:
                if prob.shape!=X.shape[:2] or not np.isfinite(prob).all() or (
                        np.any(prob<0) or np.any(prob>1)):
                    raise AssertionError("OOF head probability corrupted")
        comp=info["composition"]
        stats=by_comp[comp]
        ons=[e for e in info["source_events"] if e["type"]=="note_on"]
        pitches=[int(e["pitch"]) for e in ons]
        pitch_events.update(pitches)
        stats["tracks"]+=1;stats["frames"]+=int(info["source_counts"]["frames"])
        stats["source_onsets"]+=len(ons)
        stats["gt_notes"]+=len(info["fit_truth"])
        tracks.append({"track_index":int(entry["index"]),
                       "composition":comp,"frames":int(info["source_counts"]["frames"]),
                       "source_onsets":len(ons),"ground_truth_notes":len(info["fit_truth"]),
                       "feature_sha256":entry.get("sha256",entry.get("npz_sha256","signed_by_S800_manifest"))})
    if len(by_comp)!=5:raise AssertionError("five held-out FIT folds required")
    total_frames=sum(q["frames"] for q in tracks)
    if total_frames!=306536 or s803["source_counts"]["frames"]!=306536 or (
            s803["source_counts"]["k_correct"]!=189124):
        raise AssertionError("signed frames/count parity drift")
    if len(s802["observations"]["consensus"])!=136:
        raise AssertionError("S802 signed collision observations drift")
    policy_summaries={}
    for policy in s803["policies"]:
        row=s803["policy_results"][policy]
        policy_summaries[policy]={
            "real_MIDI_six_metrics":row["metrics"],
            "delta_vs_FIT_native":row["delta"],
            "composition_deltas":row["by_composition_delta"],
            "synthetic_onsets":row["synthetic_onsets"],
            "no_regression":row["six_metric_no_regression_fit"]}
    collisions={}
    for policy,rows in s802["observations"].items():
        composition=Counter((z["composition"],"TP" if z["synthetic_matched_truth"] else "FP")
                            for z in rows)
        by_pitch=Counter((int(z["pitch"]),"TP" if z["synthetic_matched_truth"] else "FP")
                         for z in rows)
        collisions[policy]={"total":len(rows),
            "truth_matched":sum(bool(z["synthetic_matched_truth"]) for z in rows),
            "native_retrigger":sum(bool(z["source_retrigger"]) for z in rows),
            "by_composition":{c:{"TP":composition[(c,"TP")],
                                  "FP":composition[(c,"FP")]} for c in sorted(by_comp)},
            "by_pitch":{str(p):{"TP":by_pitch[(p,"TP")],"FP":by_pitch[(p,"FP")]}
                        for p in sorted({i[0] for i in by_pitch})}}
    return {"dataset":"GuitarSet real FIT80 out-of-fold","count_tracks":len(tracks),
            "count_frames":total_frames,"compositions":dict(sorted(by_comp.items())),
            "per_track_audit":tracks,"pitch_histogram_native_onsets":
                {str(k):v for k,v in sorted(pitch_events.items())},
            "source_signed_counts":s803["source_counts"],
            "native_fit_six_scores":s803["baseline_fit"],
            "policy_results":policy_summaries,
            "retrospective_s802_collisions":collisions,
            "CAL_HELD_TEST_opened":False,
            "SHA256_snapshots_verified":True,
            "notes":"Only aggregate FIT annotations summarized; no truth in causal decoder"}

def make(root,manifest,traces,s803,s802):
    code=code_inventory(root)
    data=dataset_audit(manifest,traces,s803,s802)
    payload={"schema":SCHEMA,"repo":"Andriamarosoa/note",
             "commit":subprocess.run(["git","rev-parse","HEAD"],cwd=root,
                  capture_output=True,text=True,check=True).stdout.strip(),
             "reference":"S708","production_LIVE_modified":False,
             "science_promoted":False,
             "code_audit":code,"dataset_audit":data,
             "review_limitations":{
                "NVIDIA_receives":"named hot-path excerpts and 80-track numeric summaries",
                "NVIDIA_does_not_receive":"all repository lines, raw audio, full JAMS, or OOF",
                "whole_tree_local_AST_scan":True,
                "full_data_samples_locally_verified":80,
                "external_models_are_advisory":True}}
    raw=json.dumps(payload,ensure_ascii=False,sort_keys=True).encode()
    payload["evidence_digest"]=digest(raw)
    return payload

def main():
    p=argparse.ArgumentParser()
    for k in ("root","manifest","traces","s803","s802","out"):
        p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args()
    report=make(a.root,a.manifest,a.traces,a.s803,a.s802)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("S804_REAL_80_TRACK_DATA_AND_CODE_AST_PASS",
          json.dumps({"tracks":report["dataset_audit"]["count_tracks"],
                     "python_files":report["code_audit"]["python_file_count"],
                     "evidence_digest":report["evidence_digest"]}),flush=True)
if __name__=="__main__":main()
