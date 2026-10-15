#!/usr/bin/env python3
"""Small real-model checks; synthetic signals are not an accuracy benchmark."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from causal_note.live_engine import HOP, SAMPLE_RATE, LiveEngine


def replay(waveform):
    engine = LiveEngine.from_bundle(prime_silence=True)
    rows = []
    for start in range(0, len(waveform) - HOP + 1, HOP):
        rows.extend(x.json() for x in engine.process(waveform[start:start + HOP], start_sample=start))
    return engine, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    silent, silent_rows = replay(np.zeros(80 * HOP, np.float32))
    assert not silent_rows
    silent.finalize()
    t = np.arange(100 * HOP, dtype=np.float32) / SAMPLE_RATE
    # A fixed plucked harmonic signal exercises nonzero inference, without claiming
    # that synthetic detections prove recognition on a guitar or other instrument.
    tone = np.asarray(.2 * np.exp(-t * 3) * (np.sin(2 * np.pi * 164.8138 * t)
                     + .3 * np.sin(4 * np.pi * 164.8138 * t)), np.float32)
    full, full_rows = replay(tone)
    prefix, prefix_rows = replay(tone[:60 * HOP])
    assert prefix_rows == [x for x in full_rows if x["sample"] <= 60 * HOP]
    ends = full.finalize(len(tone), reason="eof")
    prefix.finalize(60 * HOP)
    assert not full.active_ids and not prefix.active_ids
    assert full.events["note_on"] == full.events["note_off"]
    report = {"stage": "S610", "tensorflow": importlib.metadata.version("tensorflow"),
              "model_sha256": full.metadata["artifact"]["sha256"],
              "silence_has_no_notes": True, "prefix_invariant": True,
              "finite_model_outputs": True, "all_notes_closed": True,
              "synthetic_note_on_count": full.events["note_on"],
              "synthetic_accuracy_evaluated": False, "physical_audio_io_tested": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
