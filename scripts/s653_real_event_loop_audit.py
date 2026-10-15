"""S653: actual NOTE original 17 DEV WAV event-ledger, after every loop candidate.

Pre-frozen S648 checkpoint and policies 0.85/0.15 and 0.90/0.15.
This does NOT retrain or select thresholds, has NO model promotion,
and does not use reference labels during causal inference.
Captures exact corrected AND regressed original reference note identities;
K1..K4+ is local simultaneous note-birth count, NOT original S80 Exact-K.

All source-level and corpus-level counts must reproduce published S648
S614 original and corresponding predeclared variant, else CI fails.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import json
import tempfile
import zipfile
import joblib
from causal_note.strict_live import load_heads
from causal_note.s648_event_birth import BirthPolicy,decode_birth
from causal_note.s652_event_round_audit import (
    audit_event_round,require_event_audit_before_next_round
)
from scripts.s616_live_dev_ablation import ROOT,MODEL
from scripts.s613_prepare_strict_live import download,digest
from scripts.s626_train_sharp_native_onset import get_source

S614_SHA="7ccf046e3a1870eaca3a108e6f9565298aeeadf43732a2f83559542c03f5f17f"
S648_SHA="7613dca4aa9bc5992c96a75ab55f714d8720c55b6880ee9e223e39ea880eb4e9"
LOCKED_VARIANTS=(("birth085_rise015",BirthPolicy(.85,.15)),
                 ("birth090_rise015",BirthPolicy(.90,.15)))


def compare_fixed_s648_report(result,legacy):
    if legacy["lock_from_CALIBRATION_before_DEV"]!="S614_original":
        raise AssertionError("A previously rejected trained model was silently promoted")
    if legacy["checkpoint_sha256"]!=S648_SHA:
        raise AssertionError("Wrong real S648 checkpoint")
    for policy,phase in result["policies"].items():
        for corpus in ("GuitarSet","URMP"):
            rows=[x for x in phase["by_source"] if x["corpus"]==corpus]
            sums={k:sum(x["delta"][k] for x in rows) for k in ("TP","FP","FN")}
            a=legacy["metrics"]["DEV"]["score"]["S614_original"][corpus]
            b=legacy["metrics"]["DEV"]["score"][policy][corpus]
            for key in sums:
                if sums[key]!=b[key]-a[key]:
                    raise AssertionError(f"Event ledger {policy} {corpus} {key} is not the original S648 result")
    return True


def summarize_event_policy(rows):
    out={}
    for corpus in ("GuitarSet","URMP"):
        selected=[r for r in rows if r["corpus"]==corpus]
        counts={key:sum(r["delta"][key] for r in selected) for key in ("TP","FP","FN")}
        counts["newly_correct_truths"]=sum(r["matched_reference_partition"]["corrected"] for r in selected)
        counts["lost_previous_correct_truths"]=sum(r["matched_reference_partition"]["lost"] for r in selected)
        counts["new_harmonic_like_fp_signatures"]=sum(
            r["false_note_diagnostic_counts"].get("possible_harmonic_confusion",0)
            for r in selected)
        counts["sources_regressed"]=sum(r["delta"]["F1"]<0 for r in selected)
        counts["sources_improved"]=sum(r["delta"]["F1"]>0 for r in selected)
        by_k=defaultdict(Counter)
        by_register=defaultdict(Counter)
        by_role=defaultdict(Counter)
        for record in selected:
            role="solo" if record["source"].endswith("_solo") else (
                "accompaniment" if record["source"].endswith("_comp") else "unknown")
            c=record["matched_reference_partition"]
            by_role[role].update(corrected=c["corrected"],lost=c["lost"])
            for k,counts_by_status in record["classifications"].items():
                if k.startswith("K"):
                    by_k[k].update(counts_by_status)
                else:
                    by_register[k].update(counts_by_status)
        counts["by_local_onset_K"]={key:dict(v) for key,v in by_k.items()}
        counts["by_register"]={key:dict(v) for key,v in by_register.items()}
        counts["by_role"]={key:dict(v) for key,v in by_role.items()}
        if (counts["newly_correct_truths"]-
            counts["lost_previous_correct_truths"]!=counts["TP"]):
            raise AssertionError("Aggregate corrected vs lost identities mismatch")
        out[corpus]=counts
    return out


def main(checkpoint,past,out):
    if out.exists():raise FileExistsError("No overwriting frozen S653 original-audio evidence")
    if digest(checkpoint.read_bytes())!=S648_SHA:
        raise AssertionError("Trained S648 model differs")
    legacy=json.loads(past.read_text())
    model=joblib.load(checkpoint)
    protocol=json.loads((ROOT/"protocol.json").read_text())
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    if digest(MODEL.read_bytes())!=S614_SHA or digest((ROOT/"protocol.json").read_bytes())!=frozen["protocol_sha256"]:
        raise AssertionError("Original NOTE / protocol changed")
    native=load_heads(MODEL,expected_sha256=S614_SHA)
    specs=sorted((s for s in protocol["sources"] if s["split"]=="dev" and s["corpus"] in ("GuitarSet","URMP")),
                 key=lambda s:(s["corpus"],s["name"]))
    assert len(specs)==17
    result=dict(status="S653_REAL_EVENT_LEVEL_IN_LOOP_AUDIT",
        original_S614_sha256=S614_SHA,trained_S648_sha256=S648_SHA,
        auditable_source_count=17,local_K_not_global_Exact_K=True,
        no_RETRAIN=True,no_DEVELOPMENT_policy_selection=True,
        TEST_not_opened=True,no_promotion=True,policies={})
    with tempfile.TemporaryDirectory() as temp:
        cache=Path(temp);(cache/"data").mkdir()
        for key in ("GuitarSet_audio","GuitarSet_annotations"):
            archive=protocol["guitar_archives"][key]
            download(archive["url"],cache/"data"/(key+".zip"),
                archive["size"],archive["sha256"])
        with zipfile.ZipFile(cache/"data/GuitarSet_audio.zip") as z:
            for policy_name,policy in LOCKED_VARIANTS:
                records=[]
                for s in specs:
                    row=get_source(s,cache,native,z if s["corpus"]=="GuitarSet" else None)
                    row["selection"]=dict(native.metadata["selection"])
                    baseline,_=decode_birth(row,model,BirthPolicy())
                    proposed,_=decode_birth(row,model,policy)
                    audit=audit_event_round(s["name"],s["corpus"],row["truth"],baseline,proposed)
                    audit["source_audio_sha256"]=row["source_sha256"]
                    records.append(audit)
                    print("S653 IN-LOOP",policy_name,s["name"],
                          "corrected",audit["matched_reference_partition"]["corrected"],
                          "lost",audit["matched_reference_partition"]["lost"],
                          "delta_FP",audit["delta"]["FP"],flush=True)
                require_event_audit_before_next_round(records,[s["name"] for s in specs])
                result["policies"][policy_name]=dict(
                    by_source=records,
                    audit_completed_before_next_variant=True,
                    summary=summarize_event_policy(records))
                print("S653 BOUNDED ROUND",policy_name,json.dumps(
                      result["policies"][policy_name]["summary"]),flush=True)
    compare_fixed_s648_report(result,legacy)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n")
    print("S653 FINAL EVENT VERDICT",json.dumps(
        {name:part["summary"] for name,part in result["policies"].items()},indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--s648-report",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    main(args.checkpoint,args.s648_report,args.out)
