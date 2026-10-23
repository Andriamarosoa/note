"""Sequential CPU timing of identical current-hop reference/compiled outputs."""
import argparse
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from scipy.io import wavfile
from threadpoolctl import threadpool_limits

from onset256.event_confidence import load_event_head
from onset256.live_event_stream import LowLatencyOnsetStream
from onset256.model import load
from onset256.runtime import Stream
from scripts.onset_event_integrity256 import RecordingRisk, compact, one, original_models


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--hops", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--reverse", action="store_true")
    args = p.parse_args()
    if not 0 <= args.warmup < args.hops:
        raise ValueError("invalid warmup")
    track = "00_BN1-129-Eb_comp"
    with zipfile.ZipFile(args.root / "GuitarSet_audio.zip") as z:
        sr, pcm = wavfile.read(io.BytesIO(z.read(track + "_mic.wav")))
    if sr != 44100 or pcm.dtype != np.int16 or pcm.ndim != 1:
        raise ValueError("wrong audio ABI")
    x = pcm.astype(np.float32) / 32768.
    if args.hops * 256 > len(x):
        raise ValueError("prefix exceeds audio")
    outputs = {}; results = {}
    with threadpool_limits(limits=1):
        order = ("compiled", "reference") if args.reverse else ("reference", "compiled")
        for mode in order:
            native, s701 = load("model/live/strict_note/NOTE_strict_live.npz",
                                one(args.root / "s701", "s701_onset_shadow.joblib"))
            risk, recovery, _ = original_models(args.root, "BN1")
            cp = args.previous / "trained" / "checkpoints" / "selected_BN1.joblib"
            head = load_event_head(cp, cp.with_suffix(".json"), outer="BN1")
            if mode == "compiled":
                stream = LowLatencyOnsetStream(native, s701, risk, recovery, head)
            else:
                risk = RecordingRisk(risk)
                stream = Stream(native, s701, risk_model=risk, recovery_model=recovery)
            events = []; elapsed = []; cpu = []
            for hop in range(args.hops):
                start = hop * 256
                before = time.perf_counter_ns()
                cpu_before = time.process_time_ns()
                d = stream.push(x[start:start+256], start_sample=start)
                if mode == "reference":
                    d = d[0]
                    q = head.predict(risk.evidence, sample=d.sample)
                    intents = head.select(d.predecision_intents, q, sample=d.sample)
                    intents += d.fn_recovery_intents
                else:
                    intents = d.intents
                cpu.append((time.process_time_ns() - cpu_before) / 1e6)
                elapsed.append((time.perf_counter_ns() - before) / 1e6)
                events.extend(compact(intents))
            stream.finish()
            outputs[mode] = events
            measured = np.asarray(elapsed[args.warmup:])
            results[mode] = dict(ms_p50=float(np.quantile(measured, .50)),
                                 ms_p95=float(np.quantile(measured, .95)),
                                 ms_p99=float(np.quantile(measured, .99)),
                                 ms_max=float(measured.max()),
                                 process_cpu_ms_p50=float(np.quantile(cpu[args.warmup:], .50)),
                                 process_cpu_ms_p95=float(np.quantile(cpu[args.warmup:], .95)),
                                 over_hop_count=int(np.sum(measured > 256/44100*1000)))
    if outputs["reference"] != outputs["compiled"]:
        raise AssertionError("benchmark changed current-hop events")
    report = dict(track=track, hops=args.hops, warmup=args.warmup,
                  measured_hops=args.hops-args.warmup, events=len(outputs["compiled"]),
                  event_parity=True, selected_model=head.selected_model,
                  selected_cut=head.cut, results=results,
                  execution_order=order,
                  hop_samples=256, added_hold_samples=0,
                  hardware_latency_measured=False,
                  scope="local CPU prefix; excludes model loading and audio I/O")
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
