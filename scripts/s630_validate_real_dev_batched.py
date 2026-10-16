"""S630 verify frozen S613 batch-of-four ONNX note/event parity on real DEV.

Replays only original 17 composition-disjoint GuitarSet+URMP DEV true
256-hop feature tracks, original frozen hybrid. No new selector,
threshold tuning or model and no TEST. Exact grouped-K1..6 measured
on same original reference annotations. ONNX numerical discrepancies
reported rather than silently inventing musical equivalence.
"""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np

from causal_note.strict_live import load_heads,Selection,CausalSelection,SR
from scripts.s613_train_strict_live import event_score,real_notes,summary
from scripts.s615_train_multihop import source_data
from scripts.s615_exactk_audit import exact_k_onset_audit,merge_k_reports


def decode(p,k,ends,selection,length):
    sel=CausalSelection(selection,hop=256)
    ev=[]
    for t,one,count in zip(ends,p,k):
        ev.extend(sel.step(one,count,int(t),int(t)))
    ev.extend(sel.panic(length))
    return ev


def eval_track(root,s,model,selection):
    ends,x,_,_,truth_notes,length=source_data(root,s,256)
    truth=real_notes(truth_notes,length/SR)
    reference_p,reference_k=model.predict(x)
    cp=[];ck=[]
    for i in range(0,len(x),4):
        p,k=model.predict(x[i:i+4])
        cp.append(p);ck.append(k)
    p=np.concatenate(cp);k=np.concatenate(ck)
    pred_deviation=float(np.max(np.abs(reference_p-p)))
    count_deviation=float(np.max(np.abs(reference_k-k)))
    ref_event=decode(reference_p,reference_k,ends,selection,length)
    bat_event=decode(p,k,ends,selection,length)
    cols=("type","pitch","decision_sample")
    original=[tuple(e[key] for key in cols) for e in ref_event]
    batched=[tuple(e[key] for key in cols) for e in bat_event]
    ref_score,_=event_score(ref_event,truth,.080)
    batch_score,_=event_score(bat_event,truth,.080)
    return dict(source=s["name"],corpus=s["corpus"],reference=ref_score,
      batched=batch_score,nominal_event_identity=(original==batched),
      original_note_ons=sum(e["type"]=="note_on" for e in ref_event),
      batched_note_ons=sum(e["type"]=="note_on" for e in bat_event),
      max_pitch_probability_difference=pred_deviation,
      max_K_probability_difference=count_deviation,
      reference_grouped_K=exact_k_onset_audit(ref_event,truth),
      batched_grouped_K=exact_k_onset_audit(bat_event,truth))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--model",type=Path,default=Path("model/live/strict_note/NOTE_strict_live.npz"))
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    proto=json.loads((a.root/"protocol.json").read_text())
    songs=[s for s in proto["sources"] if s["split"]=="dev"]
    assert len(songs)==17
    model=load_heads(a.model)
    assert model.metadata["architecture"]=="native_causal_hybrid"
    sel=Selection(**model.metadata["selection"])
    rows=[]
    for i,s in enumerate(songs,1):
        result=eval_track(a.root,s,model,sel)
        rows.append(result)
        print("S630_REAL_SOURCE "+json.dumps(dict(
             index=i,source=s["name"],
             identical=result["nominal_event_identity"],
             max_p_delta=result["max_pitch_probability_difference"],
             ref=result["reference"],batch=result["batched"])),flush=True)
    old=summary([x["reference"] for x in rows])
    new=summary([x["batched"] for x in rows])
    assert (old["TP"],old["FP"],old["FN"])==(1241,734,838),old
    report=dict(status="S630_VALIDATED_FULL_ORIGINAL_17_DEV",
        original_hybrid=old,batch_four_hybrid=new,
        source_count=len(rows),test_sources_read=0,
        all_nominal_events_identical=all(r["nominal_event_identity"] for r in rows),
        max_absolute_pitch_probability_difference=max(r["max_pitch_probability_difference"] for r in rows),
        max_absolute_count_probability_difference=max(r["max_K_probability_difference"] for r in rows),
        original_grouped_K=merge_k_reports([r["reference_grouped_K"] for r in rows])["by_true_k"],
        batched_grouped_K=merge_k_reports([r["batched_grouped_K"] for r in rows])["by_true_k"],
        per_source=rows,
        note=("This only verifies batched model outputs and NOMINAL note event"
              " parity on original DEV. Real batching waits for future"
              " capture and changes actual emitted timing/latency. The"
              " user Windows Realtek CPU overload is not independently"
              " reproduced by this Linux workflow."),
    )
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2)+"\n")
    print("S630_REAL_SUMMARY "+json.dumps(dict(
        reference=old,batch=new,
        all_events_identical=report["all_nominal_events_identical"],
        max_p=report["max_absolute_pitch_probability_difference"],
        max_k=report["max_absolute_count_probability_difference"],
        original_K=report["original_grouped_K"],batched_K=report["batched_grouped_K"]
    )),flush=True)


if __name__=="__main__":
    main()
