"""Controlled DEV-only grid of selector thresholds on an already trained S615 head.

Replays native network predictions ONCE/source/hop, no retraining,
no input from future frames. Reports best NoteOn F1 and per-K Exact-K,
plus the number of additional FP to avoid hiding errors with recall.

Never optimizes/promotes on a held-out test set. Candidate selection
and labels are DEV-only, all reports stamped exploratory.
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path

from causal_note.strict_live import CausalSelection, NativeLiveHeads, Selection, SR
from scripts.s613_train_strict_live import event_score, real_notes, summary
from scripts.s615_exactk_audit import exact_k_onset_audit, merge_k_reports
from scripts.s615_train_multihop import source_data


def selector_grid():
    for onset, sustain, release, refractory in itertools.product(
        (.20,.30,.40,.50,.60,.70),
        (.25,.35,.50),
        (3,5,8),
        (6,10),
    ):
        yield Selection(onset=onset, active_on=sustain,
                        onset_reset=min(.18,onset/2.),
                        release_frames=release,
                        refractory_frames=refractory,
                        count_confidence=1.1)


def emit(ends, predicted, kcount, frames, config, hop):
    selector=CausalSelection(config,hop=hop)
    events=[]
    for end,p,k in zip(ends,predicted,kcount):
        events.extend(selector.step(p,k,int(end),int(end)))
    events.extend(selector.panic(int(frames)))
    return events


def evaluate(root, path, *, hops=(128,256,512), out=None):
    original=json.loads((root/"protocol.json").read_text())
    # The checkpoint was trained on 6 GuitarSet train clips and tuned
    # against exactly the FIRST THREE development clips of the pinned S613
    # protocol. This selector audit must replay that identical pilot scope.
    selected=[s for s in original["sources"] if s["split"]=="dev"
              and s["corpus"]=="GuitarSet"][:3]
    if len(selected)<2:
        raise ValueError("Need development guitar recordings")
    head=NativeLiveHeads(path)
    trained=set(int(h) for h in head.metadata.get("trained_hops",[]))
    modes=[h for h in hops if h in trained]
    if not modes:
        raise ValueError("No requested training hops")
    result={}
    for hop in modes:
        cache=[]
        for spec in selected:
            ends,x,_,_,notes,length=source_data(root,spec,hop)
            p,k=head.predict(x)
            truth=real_notes(notes,length/SR)
            cache.append((ends,p,k,length,truth))
        configs=list(selector_grid())
        rows=[]
        for idx,config in enumerate(configs):
            dev=[]
            events_list=[]
            for ends,p,k,length,truth in cache:
                ev=emit(ends,p,k,length,config,hop)
                score,_=event_score(ev,truth,.08)
                dev.append(score)
                events_list.append((ev,truth))
            point=summary(dev)
            rows.append(dict(index=idx,selection=asdict(config),
                             TP=int(point.get("TP",0)),
                             FP=int(point.get("FP",0)),
                             FN=int(point.get("FN",0)),
                             F1=point["F1"],
                             events=events_list))
        scored=sorted(rows,key=lambda a:(a["F1"],-a["FP"]),reverse=True)
        # Analyse separate of F1 top and two much-less-aggressive alternatives.
        shortlist=scored[:10]
        safe=min(scored,key=lambda a:(a["FP"],-a["TP"]))
        if safe not in shortlist:
            shortlist.append(safe)
        analyzed=[]
        for r in shortlist:
            kreport=merge_k_reports([
                exact_k_onset_audit(e,t) for e,t in r["events"]
            ])
            analyzed.append(dict(
                selection=r["selection"], F1=r["F1"], TP=r["TP"],
                FP=r["FP"], FN=r["FN"],
                onset_exact_k=kreport["onset_exact_k_rate"],
                exact_pitches=kreport["pitch_set_rate"],
                by_true_k=kreport["by_true_k"],
            ))
        # Counterfactual F1 optimum based entirely on DEV, NEVER proof
        # of independent accuracy. All K3/K4 effects visible in reports.
        result[str(hop)]=dict(
            configurations=len(configs),dev_sources=len(cache),
            baseline_selector=head.metadata.get("selection_by_hop",{}).get(
                str(hop),head.metadata["selection"]),
            top=analyzed,
            f1_best=analyzed[0],
        )
        print(json.dumps(dict(hop=hop,configurations=len(configs),
                              best=result[str(hop)]["f1_best"]),
                         sort_keys=True),flush=True)
    report=dict(status="EXPLORATORY_DEVELOPMENT_ONLY_NO_PROMOTION",
                trained_model=str(path),results=result,
                reason="Grid tuned on same dev sources: selection overfit possible. Verify on unseen compositions and original S613 before merge.")
    if out is not None:
        if out.exists():
            raise FileExistsError(out)
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--model",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    opts=ap.parse_args()
    evaluate(opts.root,opts.model,out=opts.out)


if __name__=="__main__":
    main()
