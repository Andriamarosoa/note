"""Verify the compiled live stream against sealed fresh-audio event outputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from scipy.io import wavfile
from threadpoolctl import threadpool_limits

from onset256.compiled_inference import accelerated
from onset256.event_confidence import load_event_head
from onset256.live_event_stream import LowLatencyOnsetStream
from onset256.model import load
from scripts.onset_event_integrity256 import (
    COMPS, counts, compact, digest, dump, one, original_models, sum_scores, verify_inputs)


def fold(task):
    root, previous, output, comp = task
    root, previous, output = Path(root), Path(previous), Path(output)
    with threadpool_limits(limits=1):
        manifest = json.loads(one(root / "manifest", "report.json").read_text())
        traces = one(root / "traces", "fit_000.json").parent
        expected = json.loads((previous / "trained" / "predictions.json").read_text())
        native, s701 = load("model/live/strict_note/NOTE_strict_live.npz",
                            one(root / "s701", "s701_onset_shadow.joblib"))
        risk, recovery, _ = original_models(root, comp)
        selected = previous / "trained" / "checkpoints" / f"selected_{comp}.joblib"
        head = load_event_head(selected, selected.with_suffix(".json"), outer=comp)
        s701 = accelerated(s701)
        risk.model = accelerated(risk.model)
        recovery.model = accelerated(recovery.model)
        head.model = accelerated(head.model)
        results = []
        with zipfile.ZipFile(root / "GuitarSet_audio.zip") as audio:
            for item in sorted((x for x in manifest["fit_files"] if x["composition"] == comp),
                               key=lambda x: x["index"]):
                path = traces / f"fit_{item['index']:03d}.json"
                if digest(path) != item["json_sha256"]:
                    raise ValueError("annotation SHA mismatch")
                meta = json.loads(path.read_text())
                sr, pcm = wavfile.read(io.BytesIO(audio.read(meta["name"] + "_mic.wav")))
                if sr != 44100 or pcm.dtype != np.int16 or pcm.ndim != 1:
                    raise ValueError("original audio ABI changed")
                wav = pcm[:meta["samples"]].astype(np.float32) / 32768.
                stream = LowLatencyOnsetStream(native, s701, risk, recovery, head)
                events = []; elapsed = []
                for start in range(0, len(wav) - 255, 256):
                    before = time.perf_counter_ns()
                    d = stream.push(wav[start:start+256], start_sample=start)
                    elapsed.append((time.perf_counter_ns() - before) / 1e6)
                    events.extend(compact(d.intents))
                stream.finish()
                keys = sorted((e["sample"], e["pitch"]) for e in events)
                if keys != [tuple(e) for e in expected[meta["name"]]]:
                    raise AssertionError(f"compiled stream changed sealed events: {meta['name']}")
                record = dict(track=meta["name"], composition=comp, frames=len(elapsed),
                              score=counts(events, meta["fit_truth"]), exact_event_parity=True,
                              ms_p50=float(np.quantile(elapsed, .5)),
                              ms_p95=float(np.quantile(elapsed, .95)),
                              ms_p99=float(np.quantile(elapsed, .99)),
                              over_hop_count=int(np.sum(np.asarray(elapsed) > 256/44100*1000)))
                results.append(record)
                print("COMPILED_ONSET_EXACT_TRACK", comp, len(results), flush=True)
        dump(output / f"{comp}.json", results)
        return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, choices=range(1, 6), default=5)
    args = p.parse_args()
    verify_inputs(args.root)
    if args.out.exists():
        raise FileExistsError("immutable compiled replay")
    args.out.mkdir(parents=True)
    tasks = [(str(args.root), str(args.previous), str(args.out), c) for c in COMPS]
    if args.workers == 1:
        groups = list(map(fold, tasks))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            groups = list(pool.map(fold, tasks))
    rows = [x for group in groups for x in group]
    total = sum_scores([r["score"] for r in rows])
    if len(rows) != 80 or sum(r["frames"] for r in rows) != 306536:
        raise AssertionError("compiled replay incomplete")
    original = json.loads((args.previous / "replayed" / "report.json").read_text())
    wanted = original["totals"]["selected_hgb_or_control69"]
    if any(total[k] != wanted[k] for k in ("tp", "fp", "fn")):
        raise AssertionError("compiled score differs from verified fresh replay")
    result = dict(complete=True, tracks=80, frames=306536, totals=total,
                  per_track=rows, hop_samples=256, added_hold_samples=0,
                  all_events_identical=True, weights_retrained=False,
                  parallel_workers=args.workers, concurrent_replay_timing_not_hardware_latency=True,
                  hardware_latency_measured=False)
    dump(args.out / "report.json", result)
    print("COMPILED_ONSET_80_WAV_EXACT_PARITY", json.dumps(total), flush=True)


if __name__ == "__main__":
    main()
