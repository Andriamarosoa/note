"""S628 causally real future evidence for frozen full S613 on 17 DEV songs.

This is NOT shift labels backward or read artificial "future features".
For decision t, sound through t+L MUST exist; right-aligned original
acoustic features are recomputed on a second PCM phase at t+L with
the ORIGINAL frozen neural+ONNX head. The ORIGINAL frozen K head
from t controls active polyphony. The original selector still emits
after t+L (and PortAudio callback quantization), never at t.

0ms original vs configurable 2.5/5/10ms; delay-only 5ms control
separates improvement from simply changing event timestamps.
Report TP, FP, FN, FN recovered/lost identity, K1..K6, per corpus and
timing. NO test labels or songs, no new model training or promotion.
Thresholds/mode selection on DEV is explicitly exploratory only.
"""
from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path

import numpy as np

from causal_note.strict_live import CausalSelection,Selection,SR,HOP,load_heads,feature_track
from causal_note.lookahead import LookaheadConfig,combine_reference_and_future
from scripts.s613_train_strict_live import event_score,real_notes,summary
from scripts.s615_train_multihop import source_data
from scripts.s615_exactk_audit import exact_k_onset_audit,merge_k_reports


MS_VALUES=(2.5,5.0,10.0)
MODES=("delay","average","confirm","max_onset","future")


def decode_with_lookahead(record,sel,shifted_prob=None,shift_samples=0,mode="delay",
                          block_size=256):
    if block_size<=0:
        raise ValueError("Blocksize must be positive")
    original=record["prob"]
    n=len(record["ends"])
    stream=CausalSelection(sel,hop=HOP)
    events=[]
    known_future=len(shifted_prob) if shifted_prob is not None else 0
    if mode!="delay" and shift_samples<=0:
        raise ValueError("Need real PCM shift for future evidence")
    for i,(t,p,k) in enumerate(zip(record["ends"],original,record["count"])):
        t=int(t)
        if shift_samples:
            # This is a callback delivered only after END OF PACKET, never
            # a fictional instantaneous release at t+5ms.
            physical_end=t+shift_samples
            if i<known_future:
                delivery=((physical_end+block_size-1)//block_size)*block_size
            else:
                delivery=record["length"] # EOF has NO future PCM
            delivery=min(delivery,record["length"])
            if delivery<t:raise AssertionError("Backdating emission impossible")
            if i<known_future:
                mixed=combine_reference_and_future(p,shifted_prob[i],mode)
            else:
                mixed=p
        else:
            delivery=t
            mixed=p
        emission=stream.step(mixed,k,t,delivery)
        if i>=known_future and shift_samples:
            for e in emission:
                e["reason"]="lookahead_eof_unconfirmed_"+e["reason"]
        events.extend(emission)
    events.extend(stream.panic(record["length"]))
    return events


def evaluate(root,out,model_path):
    if out.exists():raise FileExistsError(out)
    protocol=json.loads((root/"protocol.json").read_text())
    sources=[s for s in protocol["sources"] if s["split"]=="dev"]
    if len(sources)!=17:raise ValueError("Need original 17 dev compositions")
    heads=load_heads(model_path)
    if heads.metadata.get("architecture")!="native_causal_hybrid":
        raise ValueError("Only original S613 native full hybrid neural+ONNX allowed")
    sel=Selection(**heads.metadata["selection"])
    reports={}
    rows=[]
    for i,spec in enumerate(sources,1):
        ends,x,_,_,notes,length=source_data(root,spec,256)
        p,k=heads.predict(x)
        truth=real_notes(notes,length/SR)
        row=dict(spec=spec,ends=ends,prob=p,count=k,truth=truth,length=length)
        old=decode_with_lookahead(row,sel)
        scored,matches=event_score(old,truth,.080)
        row["old"]=old
        row["baseline_score"]=scored
        row["baseline_truth"]={j for _,j in matches}
        rows.append(row)
        print(json.dumps(dict(index=i,source=spec["name"],baseline_TP=scored["TP"])),flush=True)
    baseline=summary([row["baseline_score"] for row in rows])
    if (baseline["TP"],baseline["FP"],baseline["FN"])!=(1241,734,838):
        raise AssertionError("S613 full baseline changed unexpectedly: "+str(baseline))
    k_base=merge_k_reports([
        exact_k_onset_audit(row["old"],row["truth"]) for row in rows])
    trial=[]
    for ms in MS_VALUES:
        conf=LookaheadConfig(milliseconds=ms)
        for i,row in enumerate(rows):
            audio=np.load(root/"cache"/f"{row['spec']['id']}.audio.npy",allow_pickle=False)
            if len(audio)!=row["length"] or audio.dtype!=np.float32:
                raise ValueError("Source audio provenance/shape changed")
            shifted=audio[conf.samples:]
            ends_future,x_future=feature_track(shifted,block_size=2048,hop=256)
            if not np.array_equal(ends_future,row["ends"][:len(ends_future)]):
                raise AssertionError("Physical future phase/decision mismatch")
            future,_=heads.predict(x_future)
            row["future"]=future
        for mode in MODES:
            metrics=[]; grouped=[]; by_corpus=defaultdict(list)
            gained=lost=0
            for row in rows:
                ev=decode_with_lookahead(
                    row,sel,row["future"],conf.samples,mode,block_size=256)
                scored,pairs=event_score(ev,row["truth"],.080)
                recovered={j for _,j in pairs}
                gained+=len(recovered-row["baseline_truth"])
                lost+=len(row["baseline_truth"]-recovered)
                metrics.append(scored)
                grouped.append(exact_k_onset_audit(ev,row["truth"]))
                by_corpus[row["spec"]["corpus"]].append(scored)
            m=summary(metrics)
            k=merge_k_reports(grouped)
            result=dict(
                requested_lookahead_ms=ms,
                actual_lookahead_ms=conf.samples*1000/SR,
                lookahead_samples=conf.samples,mode=mode,
                TP=m["TP"],FP=m["FP"],FN=m["FN"],F1=m["F1"],
                extra_true_notes=m["TP"]-baseline["TP"],
                additional_false_notes=m["FP"]-baseline["FP"],
                recovered_missed_truth=gained,
                lost_previously_correct_truth=lost,
                grouped_onset_K=k["by_true_k"],
                grouped_K_rate=k["onset_exact_k_rate"],
                grouped_pitch_set_rate=k["pitch_set_rate"],
                by_corpus={c:summary(v) for c,v in by_corpus.items()},
                physical_min_added_latency_ms=ms,
                note=("Delivery timestamp is next 256-sample callback after "
                      "physical t+L, not arbitrary backdated original t.")
            )
            assert gained-lost==result["extra_true_notes"]
            trial.append(result)
            print("S628_TRIAL "+json.dumps(dict(
                lookahead_ms=ms,mode=mode,F1=m["F1"],TP=m["TP"],FP=m["FP"],
                FN=m["FN"],recovered=gained,lost=lost,
                K3=k["by_true_k"].get("3"),K4=k["by_true_k"].get("4")
            )),flush=True)
    best=max(trial,key=lambda r:r["F1"])
    winner=dict(
        status="S628_5MS_LOOKAHEAD_DEV_ONLY_NO_PROMOTION",
        model=str(model_path),source_split="17 source-disjoint DEV",
        tested_lookahead_ms=list(MS_VALUES),tested_modes=list(MODES),
        frozen_s613_hybrid_reference=baseline,
        frozen_reference_grouped_K=k_base["by_true_k"],
        experiments=trial,best_dev_f1=best,
        no_future_before_arrival=True,test_sources_read=0,
        warning=("Any apparent F1 improvements selected on these 17 DEV tracks "
                 "are not independent. 5ms necessarily increases real LIVE "
                 "latency; actual total depends on callback batching, CPU, "
                 "capture and FluidSynth. Mode=delay controls alignment gains. "
                 "Frozen 256-hop head was not trained with future-aligned "
                 "acoustics; no claim that lookahead improves unseen songs.")
    )
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(winner,indent=2)+"\n")
    print("S628_SUMMARY "+json.dumps(dict(
        baseline=baseline,best=best,
        five_ms=[r for r in trial if r["requested_lookahead_ms"]==5.0]
    )),flush=True)
    return winner


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--model",type=Path,default=Path("model/live/strict_note/NOTE_strict_live.npz"))
    a=ap.parse_args()
    evaluate(a.root,a.out,a.model)


if __name__=="__main__":
    main()
