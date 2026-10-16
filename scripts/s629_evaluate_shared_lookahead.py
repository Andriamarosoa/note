"""S629 original frozen S613 future native hop reuse vs precise S628 phase.

Same 17 source-disjoint original DEV GuitarSet/URMP, original NoteOn
±80ms and group K1..6 metrics. Uses actual future original head
probabilities already computed at later 256-hop frames (not synthetic
future). Selection remains original single K/ON/OFF state machine.

S629's 5ms requested is ACTUAL 5.805ms at 256 hop; 10ms requested is
ACTUAL 11.610ms at 512 samples. No independent TEST, no promotion.
"""
from __future__ import annotations
import argparse,json,math
from collections import defaultdict
from pathlib import Path

from causal_note.strict_live import SR,HOP,load_heads,Selection
from causal_note.lookahead import LookaheadConfig
from scripts.s628_evaluate_audio_lookahead import decode_with_lookahead
from scripts.s615_train_multihop import source_data
from scripts.s613_train_strict_live import event_score,real_notes,summary
from scripts.s615_exactk_audit import exact_k_onset_audit,merge_k_reports

MODES=("delay","average","confirm","max_onset","future")


def evaluate(root,out,model):
    if out.exists():raise FileExistsError(out)
    protocol=json.loads((root/"protocol.json").read_text())
    specs=[s for s in protocol["sources"] if s["split"]=="dev"]
    if len(specs)!=17:raise AssertionError("Only 17 original source-disjoint DEV songs")
    frozen=load_heads(model)
    assert frozen.metadata["architecture"]=="native_causal_hybrid"
    selection=Selection(**frozen.metadata["selection"])
    rows=[]
    for i,s in enumerate(specs,1):
        ends,features,_,_,notes,length=source_data(root,s,256)
        p,k=frozen.predict(features)
        truth=real_notes(notes,length/SR)
        row=dict(
            ends=ends,prob=p,count=k,truth=truth,
            length=length,spec=s)
        old=decode_with_lookahead(row,selection)
        score,pairs=event_score(old,truth,.08)
        row["baseline_score"]=score
        row["baseline_matches"]={j for _,j in pairs}
        row["original_events"]=old
        rows.append(row)
        print(json.dumps(dict(source_index=i,source=s["name"],
                              base_TP=score["TP"])),flush=True)
    base=summary([r["baseline_score"] for r in rows])
    if (base["TP"],base["FP"],base["FN"])!=(1241,734,838):
        raise AssertionError("Historical hybrid baseline regressed "+str(base))
    baseline_k=merge_k_reports([
        exact_k_onset_audit(row["original_events"],row["truth"])
        for row in rows])
    tests=[]
    for requested_ms in (5.,10.):
        conf=LookaheadConfig(milliseconds=requested_ms)
        hops=math.ceil(conf.samples/HOP)
        actual_samples=hops*HOP
        for mode in MODES:
            stats=[];onsets=[];by_corpus=defaultdict(list)
            old_missing_recovered=old_correct_lost=0
            for row in rows:
                # Feature/prob from the physically later ORIGINAL 256
                # frame. Never use a model estimate from t+L before
                # audio at that future time could have arrived.
                future=row["prob"][hops:]
                new=decode_with_lookahead(
                    row,selection,shifted_prob=future,
                    shift_samples=actual_samples,mode=mode,block_size=256)
                scores,pairs=event_score(new,row["truth"],.08)
                now={j for _,j in pairs}
                old_missing_recovered+=len(now-row["baseline_matches"])
                old_correct_lost+=len(row["baseline_matches"]-now)
                stats.append(scores)
                by_corpus[row["spec"]["corpus"]].append(scores)
                onsets.append(exact_k_onset_audit(new,row["truth"]))
            m=summary(stats);g=merge_k_reports(onsets)
            assert old_missing_recovered-old_correct_lost==m["TP"]-base["TP"]
            record=dict(
                requested_ms=requested_ms,
                effective_ms=actual_samples*1000/SR,
                effective_samples=actual_samples,
                mode=mode,
                TP=m["TP"],FP=m["FP"],FN=m["FN"],F1=m["F1"],
                recovered_old_misses=old_missing_recovered,
                lost_previously_correct=old_correct_lost,
                changed_TP=m["TP"]-base["TP"],
                changed_FP=m["FP"]-base["FP"],
                by_corpus={c:summary(sc) for c,sc in by_corpus.items()},
                grouped_onset_K_by_true=g["by_true_k"],
                grouped_onset_K_accuracy=g["onset_exact_k_rate"],
                pitch_set_accuracy=g["pitch_set_rate"],
            )
            tests.append(record)
            print("S629_TRIAL "+json.dumps(dict(
                requested=requested_ms,effective=record["effective_ms"],mode=mode,
                TP=m["TP"],FP=m["FP"],FN=m["FN"],F1=m["F1"],
                gain=old_missing_recovered,lost=old_correct_lost,
                K3=record["grouped_onset_K_by_true"].get("3"),
                K4=record["grouped_onset_K_by_true"].get("4")
            )),flush=True)
    result=dict(
        status="S629_DEV_ONLY_SHARED_LOOKAHEAD_NO_PROMOTION",
        frozen_original=base,
        frozen_original_grouped_K=baseline_k["by_true_k"],
        all_trials=tests,best_f1_dev=max(tests,key=lambda x:x["F1"]),
        source_dev_count=17,
        no_future_before_capture=True,
        test_sources_read=0,
        warning=("5ms and 10ms requested round UP to native original 256-hop"
                 " decisions; these are DIFFERENT FUTURE TIMES from S628 exact"
                 " phase. Frozen original trained only for 256 hop. No independent"
                 " test/generalization or Windows iRig+FluidSynth latency proof.")
    )
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n")
    print("S629_SUMMARY "+json.dumps(dict(
        baseline=base,best_dev=result["best_f1_dev"],
        mode_at_10ms=[r for r in tests if r["requested_ms"]==10.],
    )),flush=True)
    return result


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--root",type=Path,required=True)
    a.add_argument("--out",type=Path,required=True)
    a.add_argument("--model",type=Path,default=Path(
        "model/live/strict_note/NOTE_strict_live.npz"))
    opts=a.parse_args()
    evaluate(opts.root,opts.out,opts.model)


if __name__=="__main__":
    main()
