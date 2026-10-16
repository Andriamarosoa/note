"""S630 CPU batch amortization: per-hop original vs B=2/4/8 vectorized.

Identical frozen original S613 full ONNX+neural, same PCM and callback
schedule. Measures real speed, not guitar F1 or iRig hardware. Exactly
the same 126-dim native 256-hop features and same original selector.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

from causal_note.strict_live import StrictLiveStream,load_heads,HOP,SR
from causal_note.batched_live import BatchedLiveStream


def bench(model,pcm,callback):
    results={}
    with threadpool_limits(limits=1):
        for batch in (1,2,4,8,16):
            stream=(StrictLiveStream(model) if batch==1 else
                    BatchedLiveStream(model,batch_hops=batch))
            milliseconds=[]
            start=time.perf_counter()
            events=[]
            for i in range(0,len(pcm),callback):
                t=time.perf_counter()
                events.extend(stream.push(pcm[i:i+callback],start_sample=i))
                milliseconds.append((time.perf_counter()-t)*1000)
            events.extend(stream.finish())
            spent=(time.perf_counter()-start)*1000
            audio_ms=len(pcm)*1000/SR
            results[str(batch)]=dict(
                batch_hops=batch,worker_total_ms=round(spent,3),
                audio_duration_ms=round(audio_ms,3),
                worker_over_audio=round(spent/audio_ms,4),
                worker_p95_ms=round(float(np.percentile(milliseconds,95)),3),
                worker_p99_ms=round(float(np.percentile(milliseconds,99)),3),
                worst_worker_ms=round(float(np.max(milliseconds)),3),
                callback_period_ms=round(callback*1000/SR,3),
                midi_notes_on=sum(e["type"]=="note_on" for e in events),
                max_intentional_wait_ms=round((batch-1)*HOP*1000/SR,3),
                infer_calls=(stream.stats["batch_inference_calls"]
                             if batch>1 else len(pcm)//HOP),
            )
        denom=results["1"]["worker_total_ms"]
        for row in results.values():
            row["cpu_ratio_vs_reference"]=round(row["worker_total_ms"]/denom,3)
    return results


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model",type=Path,default=Path("model/live/strict_note/NOTE_strict_live.npz"))
    ap.add_argument("--duration-s",type=float,default=2.0)
    ap.add_argument("--callback-size",type=int,default=256)
    a=ap.parse_args()
    if a.duration_s<=0 or a.callback_size<=0:ap.error("positive duration/callback required")
    t=np.arange(int(SR*a.duration_s))/SR
    pcm=(.04*np.sin(2*np.pi*82.4069*t)+
         .02*np.sin(2*np.pi*164.8138*t)+
         .01*np.sin(2*np.pi*247.2206*t)).astype(np.float32)
    results=bench(load_heads(a.model),pcm,a.callback_size)
    print("S630_BENCHMARK "+json.dumps(dict(
        context="Synthetic Linux CPU only, not Windows Realtek",
        callback_size=a.callback_size,
        results=results),sort_keys=True),flush=True)


if __name__=="__main__":
    main()
