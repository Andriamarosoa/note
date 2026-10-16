"""S628 actual worker CPU cost frozen S613 vs +5ms future acoustic branch.

Synthetic E2 harmonics, not guitar F1 or Windows/iRig-to-piano latency.
Identical PCM/callback sizes and frozen model in both branches.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from causal_note.strict_live import StrictLiveStream,SR,load_heads
from causal_note.lookahead import LookaheadLiveStream,LookaheadConfig


def bench(heads,audio,*,block_size=256):
    report={}
    with threadpool_limits(limits=1):
        for name,stream in (
            ("S613_zero_future",StrictLiveStream(heads)),
            ("S628_five_ms_delay_only",
             LookaheadLiveStream(heads,config=LookaheadConfig(milliseconds=5,mode="delay"))),
            ("S628_five_ms_average",
             LookaheadLiveStream(heads,config=LookaheadConfig(milliseconds=5,mode="average"))),
        ):
            cost=[];emitted=[]
            start=time.perf_counter()
            for i in range(0,len(audio),block_size):
                t0=time.perf_counter()
                emitted+=stream.push(audio[i:i+block_size],start_sample=i)
                cost.append((time.perf_counter()-t0)*1000)
            emitted+=stream.finish()
            wall=(time.perf_counter()-start)*1000
            report[name]=dict(
                audio_ms=len(audio)*1000/SR,worker_total_ms=round(wall,3),
                fraction_cpu_relative_to_audio_ms=round(wall/(len(audio)*1000/SR),4),
                worker_callback_p95_ms=float(np.percentile(cost,95)),
                worker_callback_p99_ms=float(np.percentile(cost,99)),
                block_duration_ms=block_size*1000/SR,
                note_ons=sum(e["type"]=="note_on" for e in emitted),
            )
        base=report["S613_zero_future"]
        report["five_ms_average_CPU_ratio_vs_original"]=round(
            report["S628_five_ms_average"]["worker_total_ms"]/
            base["worker_total_ms"],3)
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model",type=Path,default=Path("model/live/strict_note/NOTE_strict_live.npz"))
    ap.add_argument("--duration",type=float,default=1.0)
    ap.add_argument("--block-size",type=int,default=256)
    args=ap.parse_args()
    if args.duration<=0 or args.block_size<=0:raise ValueError("positive duration & block size")
    t=np.arange(int(args.duration*SR))/SR
    audio=(.04*np.sin(2*np.pi*82.4*t)+.02*np.sin(2*np.pi*164.8*t)
           +.014*np.sin(2*np.pi*247.2*t)).astype(np.float32)
    print(json.dumps(bench(load_heads(args.model),audio,block_size=args.block_size),
                     indent=2),flush=True)


if __name__=="__main__":
    main()
