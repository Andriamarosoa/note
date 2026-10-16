"""S620 host-CPU workload budget: frozen S613 vs 128+256 parallel inference.

This benchmark is OFFLINE *execution of a strictly causal model* on
synthesized harmonic audio; it is not music accuracy, not real iRig latency.
Use the exact same 44100Hz PCM and callback packet size for all candidates.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from causal_note.strict_live import StrictLiveStream, SR
from causal_note.dual_hop import DualHopLiveStream, DualHopConfig
from causal_note.strict_live import load_heads


def benchmark(path,*,block_size=256,duration_s=2.0):
    if block_size<1 or duration_s<.1:
        raise ValueError("Block size >=1 and duration >=0.1 seconds required")
    t=np.arange(int(SR*duration_s))/SR
    mono=(.025*np.sin(2*np.pi*82.41*t)+
          .020*np.sin(2*np.pi*164.82*t)+
          .016*np.sin(2*np.pi*246.94*t)+
          .012*np.sin(2*np.pi*329.63*t)).astype(np.float32)
    heads=load_heads(path)
    stats={}
    with threadpool_limits(limits=1):
        for name,stream in (
            ("original",StrictLiveStream(heads)),
            ("dual_shadow",DualHopLiveStream(heads,config=DualHopConfig(mode="shadow"))),
            ("dual_guard",DualHopLiveStream(heads,config=DualHopConfig(mode="guard"))),
        ):
            costs=[]
            result=[]
            wall_start=time.perf_counter()
            for start in range(0,len(mono),block_size):
                t0=time.perf_counter()
                result.extend(stream.push(mono[start:start+block_size],
                                          start_sample=start))
                costs.append((time.perf_counter()-t0)*1000)
            result.extend(stream.finish())
            wall=(time.perf_counter()-wall_start)*1000
            stats[name]=dict(
                wall_ms=round(wall,3),
                p95_worker_per_callback_ms=round(float(np.percentile(costs,95)),3),
                p99_worker_per_callback_ms=round(float(np.percentile(costs,99)),3),
                callback_period_ms=round(block_size*1000/SR,3),
                realtime_fraction=round(wall/(duration_s*1000),4),
                emitted_note_on=sum(e["type"]=="note_on" for e in result),
                emitted_note_off=sum(e["type"]=="note_off" for e in result),
                event_key=[(e["type"],e["pitch"],e["decision_sample"]) for e in result],
            )
        assert stats["original"]["event_key"]==stats["dual_shadow"]["event_key"]
    for rec in stats.values():
        del rec["event_key"]
    return dict(
        checkpoint=str(path),sample_rate=SR,block_size=block_size,
        audio_duration_seconds=duration_s,
        scores_are_not_real_guitar_accuracy=True,
        cpu_only_not_end_to_end_latency=True,
        original_ref_parity=True,results=stats,
        shadow_over_original_wall_ratio=round(
            stats["dual_shadow"]["wall_ms"]/
            max(1e-5,stats["original"]["wall_ms"]),3),
    )


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",type=Path,
                   default=Path("model/live/strict_note/NOTE_strict_live.npz"))
    p.add_argument("--block-size",type=int,default=256)
    p.add_argument("--duration-s",type=float,default=2.)
    x=p.parse_args()
    print(json.dumps(benchmark(x.model,block_size=x.block_size,
                               duration_s=x.duration_s),indent=2))


if __name__=="__main__":
    main()
