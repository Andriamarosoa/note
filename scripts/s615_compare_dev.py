"""Compare frozen S613 hop256 versus a S615 candidate ON THE SAME DEV songs.

No test source/labels used; this is NOT an independent promotion evaluation.
No cherry-picked test audio. Baseline frozen checkpoint is read-only.
Reports NOTE F1 + true onset grouped Exact-K K1..6 separately per corpus.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from causal_note.strict_live import NativeLiveHeads, CausalSelection, Selection, SR
from scripts.s613_train_strict_live import real_notes, event_score, summary
from scripts.s615_train_multihop import source_data
from scripts.s615_exactk_audit import exact_k_onset_audit, merge_k_reports


def measure(root, heads, hop, specs):
    rows = []
    k_reports = []
    for spec in specs:
        ends,x,_,_,notes,length=source_data(root,spec,hop)
        truth=real_notes(notes,length/SR)
        p,k=heads.predict(x)
        sel_kwargs=heads.metadata.get("selection_by_hop",{}).get(
            str(hop),heads.metadata["selection"])
        selector=CausalSelection(Selection(**sel_kwargs),hop=hop)
        emitted=[]
        for end,ph,kframe in zip(ends,p,k):
            emitted.extend(selector.step(ph,kframe,int(end),int(end)))
        emitted.extend(selector.panic(length))
        sc,_=event_score(emitted,truth,.08)
        rows.append(dict(source=spec["name"],corpus=spec["corpus"],**sc))
        k_reports.append(exact_k_onset_audit(emitted,truth))
    return dict(noteon_80ms=summary(rows),
                onset_k=merge_k_reports(k_reports),
                by_corpus={corpus:summary([r for r in rows if r["corpus"]==corpus])
                           for corpus in sorted({r["corpus"] for r in rows})},
                recordings=rows)


def compare(root, candidate, *, baseline=Path("model/live/strict_note/NOTE_strict_live.npz"),
            output=None):
    protocol=json.loads((root/"protocol.json").read_text(encoding="utf-8"))
    specs=[s for s in protocol["sources"] if s["split"]=="dev"]
    if not specs:
        raise ValueError("No development-only recordings")
    original=NativeLiveHeads(baseline)
    contender=NativeLiveHeads(candidate)
    trained=tuple(contender.metadata.get("trained_hops",()))
    if not trained:
        raise ValueError("Candidate has no declared trained hops")
    results={"S613_frozen_dev_hop256":measure(root,original,256,specs)}
    for hop in trained:
        results[f"S615_candidate_hop{hop}"]=measure(root,contender,hop,specs)
    comparison=dict(status="DEV_ONLY_NO_PROMOTION",
                    baseline_model=str(baseline),candidate_model=str(candidate),
                    dev_sources=len(specs),trained_hops=list(trained),
                    results=results,
                    warning="Same-source DEV relative scores only. Full independent test not run. Grouped-onset K is not historical S80 Exact-K.")
    if output is not None:
        if output.exists():
            raise FileExistsError("Refusing to overwrite comparative report")
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(comparison,indent=2)+"\n",encoding="utf-8")
    compact={key:dict(F1=round(v["noteon_80ms"]["F1"],5),
                      TP=v["noteon_80ms"]["TP"],
                      FP=v["noteon_80ms"]["FP"],
                      FN=v["noteon_80ms"]["FN"],
                      onset_exact_k_rate=round(v["onset_k"]["onset_exact_k_rate"],5),
                      by_true_k=v["onset_k"]["by_true_k"])
             for key,v in results.items()}
    print(json.dumps(dict(status=comparison["status"],dev_sources=len(specs),
                          score=compact),indent=2))
    return comparison


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--candidate",type=Path,required=True)
    parser.add_argument("--baseline",type=Path,
                        default=Path("model/live/strict_note/NOTE_strict_live.npz"))
    parser.add_argument("--out",type=Path)
    a=parser.parse_args()
    compare(a.root,a.candidate,baseline=a.baseline,output=a.out)


if __name__=="__main__":
    main()
