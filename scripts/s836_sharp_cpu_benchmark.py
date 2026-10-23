"""S836: deterministic same-audio S706 spectral bottleneck benchmark, no music score claim."""
from __future__ import annotations
import json
import statistics
import time
import numpy as np

from causal_note.s706_sharp_spectra import sharp_track
from causal_note.s836_incremental_sharp256 import IncrementalSharp256


def execute(audio, mode, *, warmup=48):
    engine=IncrementalSharp256()
    times=[]
    hashes=[]
    for index,end in enumerate(range(256,len(audio)+1,256)):
        tail=audio[max(0,end-8448):end]
        n=len(tail)
        start=time.perf_counter()
        if mode=="reference":
            output=sharp_track(tail,[n] if end==256 else [n-256,n])[-1]
        elif mode=="incremental":
            output=engine.step(tail,end=end)
        else:
            raise ValueError("unknown benchmark engine")
        elapsed=(time.perf_counter()-start)*1000
        if index>=warmup:times.append(elapsed)
        hashes.append(output)
    return times,np.stack(hashes)


def main():
    rate=44100
    hops=320
    t=np.arange(hops*256,dtype=np.float64)/rate
    audio=(.12*np.sin(2*np.pi*82.4069*t)
          +.07*np.sin(2*np.pi*164.8138*t)
          +.03*np.sin(2*np.pi*110*t)).astype(np.float32)
    out={}
    for mode in ("reference","incremental"):
        times,features=execute(audio,mode)
        out[mode]=dict(mean_ms=round(float(np.mean(times)),5),
            p50_ms=round(float(np.percentile(times,50)),5),
            p95_ms=round(float(np.percentile(times,95)),5),
            p99_ms=round(float(np.percentile(times,99)),5),
            frames=len(times))
        if mode=="reference":
            gold=features
        else:
            if not np.array_equal(gold,features):
                where=np.argwhere(gold!=features)
                raise AssertionError("spectral regression at "+str(where[0]))
    out["strict_feature_parity"]=True
    out["ratio_p50_reference_over_incremental"]=round(
        out["reference"]["p50_ms"]/max(out["incremental"]["p50_ms"],1e-6),4)
    out["limited_claim"]="S706 component only on synthetic same audio; excludes S708 net, six decoders and Windows hardware"
    print("S836_SYNTHETIC_CPU_RESULT",json.dumps(out,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
