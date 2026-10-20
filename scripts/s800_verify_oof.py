"""S800 complete audit of genuine FIT80 OOF trace artifact, no score promotion."""
import argparse,json
from pathlib import Path
from collections import Counter
from causal_note.s795_activek_experts import HEADS
from scripts.s795_train_activek_midi256 import COUNT_KEYS
from scripts.s800_oof_trace import SCHEMA,verify_snapshot
def verify(report_dir):
    root=Path(report_dir)
    m=json.loads((root/"report.json").read_text(encoding="utf-8"))
    if m["schema"]!=SCHEMA or not m.get("complete"):
        raise AssertionError("S800 OOF replay not completed")
    if not m.get("FIT_source_parity_S795") or m["fit_done"]!=80:
        raise AssertionError("signed S795 FIT source counts not reproduced")
    if len(m["fit_files"])!=80 or len(m["folds_done"])!=5 or (
            len({x["index"] for x in m["fit_files"]})!=80):
        raise AssertionError("missing or duplicate FIT80 OOF records")
    counts={k:0 for k in COUNT_KEYS}
    folds=Counter()
    sha=set()
    for entry in m["fit_files"]:
        record=verify_snapshot(entry,root/"fit_oof")
        folds[record["composition"]]+=1
        sha.add(entry["npz_sha256"])
        for k in COUNT_KEYS:counts[k]+=entry["source_counts"][k]
    if len(sha)!=80 or dict(folds)!={f:16 for f in m["folds_done"]}:
        raise AssertionError("OOF track cardinality / duplicate evidence")
    if counts!=m["S795_archived_fit_counts"] or counts!=m["FIT_source_counts"]:
        raise AssertionError("FIT counts diverge from immutable S795")
    for key in ("CAL_opened","HELD_opened","TEST_opened",
                "cal_used_for_training","cal_used_for_selection",
                "production_LIVE_modified","physical_mic_latency_measured","promoted"):
        if m[key]:raise AssertionError("restricted outcome violated: "+key)
    if m["head_order"]!=list(HEADS):raise AssertionError("S795 head ABI changed")
    return dict(schema="S800_FIT80_OOF_EVIDENCE_AUDIT_V1",pass_all=True,
        tracks=80,folds=dict(folds),frames=m["FIT_source_counts"]["frames"],
        signed_parity=True,all_five_OOF_heads_saved=True,
        no_CAL_selection=True,official_metrics_unchanged=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--report-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    v=verify(a.report_dir)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n")
    print("S800_TRACE_AUDIT_PASS",json.dumps(v),flush=True)
