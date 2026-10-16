"""Run causal harmonic novelty rescue against frozen S613 full hybrid, DEV ONLY.

Uses the same 17 original development songs and source-disjoint annotation
protocol as S618, compares true FN rescue versus FP and grouped-onset K.
No training or test labels fed to inference. Fixed candidate gate grid.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path

from causal_note.strict_live import load_heads, Selection, SR
from scripts.s613_train_strict_live import event_score, real_notes, summary
from scripts.s615_train_multihop import source_data
from scripts.s615_exactk_audit import exact_k_onset_audit, merge_k_reports
from scripts.s618_missed_attack_sweep import score_events
from scripts.s619_novelty_onset_gate import RescueGate, causal_replay


GATES=tuple(
    RescueGate(weak_onset_min=on,relative_novelty_min=novelty,
               absolute_novelty_min=.04,novelty_mode=kind)
    for on in (.45,.50,.55)
    for novelty in (.15,.35,.55)
    for kind in ("lag","recent")
)


def run(root, out, *, model=Path("model/live/strict_note/NOTE_strict_live.npz")):
    if out.exists():
        raise FileExistsError(out)
    protocol=json.loads((root/"protocol.json").read_text())
    specs=[s for s in protocol["sources"] if s["split"]=="dev"]
    if len(specs)!=17:
        raise ValueError("Require all 17 dev source groups")
    frozen=load_heads(model)
    if frozen.metadata["architecture"]!="native_causal_hybrid":
        raise ValueError("Require full S613 neural + ONNX hybrid")
    select=Selection(**frozen.metadata["selection"])
    rows=[]
    reference_scores=[]
    reference_ks=[]
    for i,s in enumerate(specs,1):
        ends,x,_,_,notes,length=source_data(root,s,256)
        p,k=frozen.predict(x)
        truth=real_notes(notes,length/SR)
        events=score_events(ends,p,k,select,length)
        score,pairs=event_score(events,truth,.08)
        reference_scores.append(score)
        reference_ks.append(exact_k_onset_audit(events,truth))
        rows.append(dict(source=s,ends=ends,x=x,p=p,k=k,length=length,
                         truth=truth,original={j for i,j in pairs}))
        print(json.dumps(dict(prepared=i,total=len(specs),source=s["name"],
                              baseline_FN=score["FN"])),flush=True)
    baseline=summary(reference_scores)
    reports=[]
    for gate in GATES:
        evaluated=[]
        kreports=[]
        gained=lost=0
        attempts=0
        by_corpus=defaultdict(list)
        for r in rows:
            events,n=causal_replay(r["ends"],r["x"],r["p"],r["k"],
                                   r["length"],gate,selection=select)
            attempts+=n
            score,pairs=event_score(events,r["truth"],.08)
            new={j for i,j in pairs}
            gained+=len(new-r["original"])
            lost+=len(r["original"]-new)
            evaluated.append(score)
            by_corpus[r["source"]["corpus"]].append(score)
            kreports.append(exact_k_onset_audit(events,r["truth"]))
        combined=summary(evaluated)
        onset_k=merge_k_reports(kreports)
        output=dict(config=asdict(gate),
                    TP=combined["TP"],FP=combined["FP"],FN=combined["FN"],
                    F1=combined["F1"],
                    recall=combined["TP"]/(combined["TP"]+combined["FN"]),
                    gain_TP=combined["TP"]-baseline["TP"],
                    gain_FP=combined["FP"]-baseline["FP"],
                    recovered_old_FN=gained,
                    lost_old_TP=lost,
                    candidate_boosted_pitch_frames=attempts,
                    grouped_exact_k=onset_k["onset_exact_k_rate"],
                    exact_pitch_sets=onset_k["pitch_set_rate"],
                    grouped_K_by_true_k=onset_k["by_true_k"],
                    by_corpus={name:summary(parts) for name,parts in by_corpus.items()})
        assert output["gain_TP"]==gained-lost
        reports.append(output)
        print(json.dumps(dict(gate=asdict(gate),FN=output["FN"],
                              recovered=gained,lost=lost,
                              FP=output["FP"],F1=round(output["F1"],5),
                              K3=output["grouped_K_by_true_k"].get("3"),
                              K4=output["grouped_K_by_true_k"].get("4"))),flush=True)
    best=max(reports,key=lambda d:(d["F1"],d["gain_TP"]))
    best_recall=max(reports,key=lambda d:(d["TP"],-d["FP"]))
    report=dict(status="DEV_ONLY_CAUSAL_WEAK_ONSET_NOVELTY_ABLATION_NO_PROMOTION",
                baseline=baseline,baseline_grouped_k=merge_k_reports(reference_ks),
                candidates=reports,best_F1=best,best_recall=best_recall,
                dev_sources=len(rows),hop=256,lookahead_samples=0,
                warning="All thresholds inspected on same dev sources. No independent generalization or LIVE quality claim; don't promote.")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--root",type=Path,required=True)
    a.add_argument("--model",type=Path,default=Path("model/live/strict_note/NOTE_strict_live.npz"))
    a.add_argument("--out",type=Path,required=True)
    x=a.parse_args()
    run(x.root,x.out,model=x.model)


if __name__=="__main__":
    main()
