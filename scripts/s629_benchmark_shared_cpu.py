"""S629 CPU-only comparison of original, exact phase S628, single-pass S629.

Same frozen S613 neural+ONNX model, same PCM, identical callback shape.
Not calibrated mic/FluidSynth or end-to-end latency measurements.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from causal_note.strict_live import SR,StrictLiveStream,load_heads
from causal_note.lookahead import LookaheadConfig,LookaheadLiveStream,SharedHopLookaheadLiveStream


def benchmark(heads,audio,block_size):
    stats={}
    with threadpool_limits(limits=1):
        for name,stream in (
            ("reference_0ms",StrictLiveStream(heads)),
            ("s628_exact_phase_10ms",LookaheadLiveStream(
                heads,config=LookaheadConfig(milliseconds=10,mode="average"))),
            ("s629_shared_hop_10ms",SharedHopLookaheadLiveStream(
                heads,config=LookaheadConfig(milliseconds=10,mode="average"))),
            ("s629_shared_hop_5ms",SharedHopLookaheadLiveStream(
                heads,config=LookaheadConfig(milliseconds=5,mode="average"))),
        ):
            timings=[]
            begin=time.perf_counter()
            ons=0
            for start in range(0,len(audio),block_size):
                t=time.perf_counter()
                events=stream.push(audio[start:start+block_size],start_sample=start)
                ons+=sum(e["type"]=="note_on" for e in events)
                timings.append((time.perf_counter()-t)*1000)
            ons+=sum(e["type"]=="note_on" for e in stream.finish())
            wall_ms=(time.perf_counter()-begin)*1000
            real_audio_ms=len(audio)/SR*1000
            stats[name]=dict(
                wall_ms=round(wall_ms,2),
                relative_to_audio_ms=round(wall_ms/real_audio_ms,4),
                worker_p95_ms=round(float(np.percentile(timings,95)),3),
                worker_p99_ms=round(float(np.percentile(timings,99)),3),
                callback_period_ms=round(block_size*1000/SR,3),
                emitted_note_ons=ons,
            )
            if hasattr(stream,"diagnostics"):
                stats[name]["effective_lookahead_ms"]=stream.diagnostics()["actual_lookahead_ms"]
        base=stats["reference_0ms"]["wall_ms"]
        for label in ("s628_exact_phase_10ms","s629_shared_hop_10ms","s629_shared_hop_5ms"):
            stats[label]["cost_vs_reference"]=round(stats[label]["wall_ms"]/base,2)
    return stats


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s",type=float,default=1.5)
    parser.add_argument("--block-size",type=int,default=256)
    parser.add_argument("--model",type=Path,default=Path("model/live/strict_note/NOTE_strict_live.npz"))
    args=parser.parse_args()
    if args.duration_s<=0 or args.block_size<1:
        parser.error("Use positive duration/block-size")
    t=np.arange(int(SR*args.duration_s))/SR
    signal=(.04*np.sin(2*np.pi*82.4069*t)+
            .02*np.sin(2*np.pi*164.8138*t)+
            .015*np.sin(2*np.pi*247.2206*t)).astype(np.float32)
    print(json.dumps(dict(platform="CI_SYNTHETIC_CPU_NOT_IRIG",block_size=args.block_size,
        duration_s=args.duration_s,stats=benchmark(
            load_heads(args.model),signal,args.block_size)),indent=2))


if __name__=="__main__":
    main()
