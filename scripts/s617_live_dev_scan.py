"""S617 label-separated URMP dev: native causal early-onset bridge trials.

Historical S614 test and S80 paths are untouched; only the 5 known URMP
development recordings can influence this experiment. This is not a new
test F1, and no release/promotion is performed.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time

from causal_note.strict_live import SR,load_heads,Selection
from causal_note.s616_live_guard import GuardParameters, replay
from causal_note.s617_live_attack_bridge import BridgeParameters, NativeAttackBridge
from scripts.s616_live_dev_ablation import ROOT, MODEL, BASELINE_URMP, prepare_dev_source, aggregate
from scripts.s613_prepare_strict_live import digest
from scripts.s613_train_strict_live import event_score, offset_audit, decode


def variations():
    result=[("S614_baseline",None,GuardParameters(),5),
            ("S616_rescue_0.6",None,GuardParameters(attack_rescue=.6),5),
            ("S616_release_4",None,GuardParameters(),4)]
    for onset in (.60,.75,.90):
        for sustain in (.1,.3):
            for fund in (.05,.2):
                result.append((f"bridge_{onset:.2f}_{sustain:.2f}_{fund:.2f}",
                               BridgeParameters(onset,sustain,fund),
                               GuardParameters(),5))
    for onset,fund in ((.75,.05),(.75,.20),(.9,.05),(.9,.20)):
        result.append((f"bridge_rescue_{onset:.2f}_{fund:.2f}",
                       BridgeParameters(onset,.3,fund),
                       GuardParameters(attack_rescue=.4),5))
    return result


def evaluate(root,protocol,heads,policies):
    results={k:[] for k,*_ in policies}
    names=[]
    for spec in protocol:
        source=prepare_dev_source(spec,root)
        ends,x=source["ends"],source["features"]
        p,k=heads.predict(x)
        cfg=Selection(**heads.metadata["selection"])
        expected=decode(ends,p,k,cfg,source["samples"])
        actual=replay(ends,p,k,x,cfg,GuardParameters(),source["samples"])
        assert expected==actual
        for name,bridge,guard,release in policies:
            policy=replace(cfg,release_frames=release)
            if bridge is None:
                output=replay(ends,p,k,x,policy,guard,source["samples"])
            else:
                selector=NativeAttackBridge(policy,bridge,guard)
                output=[]
                for end,pitch_probs,dist,features in zip(ends,p,k,x):
                    output.extend(selector.step(pitch_probs,dist,int(end),int(end),features))
                output.extend(selector.finish(source["samples"]))
            score,pairs=event_score(output,source["truth"])
            offsets=offset_audit(output,source["truth"],pairs,source["samples"])
            results[name].append(dict(source=spec["name"],score=score,offset=offsets))
        names.append(spec["name"])
        print("S617 SOURCE",spec["name"],"notes",len(source["truth"]),flush=True)
    return results


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--out",type=Path,required=True)
    opts=a.parse_args()
    out=opts.out
    if out.exists(): raise FileExistsError("S617 results are immutable")
    out.mkdir(parents=True)
    protocol=json.loads((ROOT/"protocol.json").read_text())
    chosen=[s for s in protocol["sources"] if s["split"]=="dev" and s["corpus"]=="URMP"]
    assert len(chosen)==5 and all(s["split"]!="test" for s in chosen)
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    import hashlib
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest()==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    policies=variations()
    start=time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/"data").mkdir()
        results=evaluate(root,chosen,load_heads(MODEL,expected_sha256=frozen["model_sha256"]),policies)
    baseline=aggregate(results["S614_baseline"])
    assert all(baseline[key]==v for key,v in BASELINE_URMP.items()),baseline
    rows=[]
    for name,bridge,guard,release in policies:
        items=results[name]
        score=aggregate(items)
        eligible=sum(x["offset"]["matched_noteons_with_observed_reference_offset"] for x in items)
        correct=sum(x["offset"]["offsets_within_max_50ms_or_20pct_duration"] for x in items)
        rows.append(dict(name=name,F1=score["F1"],TP=score["TP"],FP=score["FP"],FN=score["FN"],
                         F1_delta_points=100*(score["F1"]-baseline["F1"]),
                         offset_conditional=correct/eligible if eligible else 0.,
                         per_source={r["source"]:{k:r["score"][k] for k in ("F1","TP","FP","FN")} for r in items}))
    report=dict(status="S617_URMP_DEV_ONLY_NO_PROMOTION",
                original_S80_and_S614_unchanged=True, model_sha256=frozen["model_sha256"],
                original_baseline_reproduced=True,test_not_opened=True,
                selection_tuning_on_development_only=True,
                variants=rows,best_dev=max(rows,key=lambda r:r["F1"])["name"],
                elapsed_seconds=time.perf_counter()-start)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S617 DEV SUMMARY",json.dumps(
        sorted([{k:x[k] for k in ("name","F1","TP","FP","FN","F1_delta_points")} for x in rows],
               key=lambda z:-z["F1"]),indent=2),flush=True)


if __name__=="__main__":
    main()
