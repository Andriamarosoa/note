#!/usr/bin/env python3
"""Audit a real S610 replay against the separately checked-out MIDI reference.

Input WAV must be PCM16 mono 44.1 kHz. This audits implementation parity and
event safety, not musical accuracy or microphone/synth latency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from causal_note.live_engine import LiveEngine
from scripts.live_notes import wav_blocks


REFERENCE_CODE = r'''
import json,sys,wave
import numpy as np
from src.polyphonic.tflite_runtime import PolyphonicBundle,TFLitePolyphonicModel
from src.polyphonic.decoder import PolyphonicDecoder,PolyphonicDecoderConfig
from src.polyphonic.audio_evidence import PolyphonicAudioEvidencePolicy
from src.stream.ring_buffer import MonoRingBuffer
b=PolyphonicBundle('artifacts/guitar_midi_polyphonic_v2_2_0')
m=TFLitePolyphonicModel(b,2)
d=PolyphonicDecoder(PolyphonicDecoderConfig(**b.metadata['decoder']))
e=PolyphonicAudioEvidencePolicy.from_metadata(44100,256,b.metadata)
e.prime_silence()
r=MonoRingBuffer(4096); w=np.zeros(4096,np.float32); rows=[]; pending=None
def append(events,sample):
 for x in events:
  rows.append(dict(type=x.kind,pitch=x.pitch,velocity=x.velocity,sample=sample,reason=x.reason))
with wave.open(sys.argv[1],'rb') as f:
 assert (f.getframerate(),f.getnchannels(),f.getsampwidth())==(44100,1,2)
 total=f.getnframes()
 for start in range(0,total-256+1,256):
  hop=np.frombuffer(f.readframes(256),'<i2').astype(np.float32)/32768.
  ev=e.process(hop); r.write(hop)
  if ev.onset.is_onset: pending=ev.audio_hop_index
  if not ev.calibrated or r.available<512: continue
  r.copy_latest_into(w); p=m.infer(w,r.available)
  events=d.step(p.frame_probability,p.onset_probability,p.harmonic_amplitude,
      audio_active=ev.activity.active,audio_hop_index=ev.audio_hop_index,
      audio_onset=pending is not None,audio_onset_hop_index=pending)
  pending=None
  append(events,start+256)
 append(d.panic(reason='eof'),total)
open(sys.argv[2],'w').write(json.dumps(rows))
'''


def check_events(rows: list[dict], *, end_sample: int) -> dict:
    active = {}
    seen = set()
    previous = 0
    maximum = 0
    on = off = 0
    for row in rows:
        assert previous <= row["sample"] <= end_sample, "non-monotonic or future event"
        previous = row["sample"]
        assert row["clock"] == "decision_sample_end"
        assert 40 <= row["pitch"] <= 76
        if row["type"] == "note_on":
            assert row["note_id"] not in seen
            assert row["pitch"] not in active
            active[row["pitch"]] = row["note_id"]
            seen.add(row["note_id"])
            on += 1
        else:
            assert row["type"] == "note_off"
            assert active.pop(row["pitch"]) == row["note_id"]
            off += 1
        maximum = max(maximum, len(active))
        assert len(active) <= 6
    assert not active and on == off
    return {"note_on": on, "note_off": off, "maximum_active": maximum,
            "timestamps_monotonic_and_within_source": True,
            "balanced_unique_ids": True, "active_notes_at_end": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    upstream = args.upstream_root.resolve()
    provenance = json.loads((ROOT / "src/causal_note/live_reference/PROVENANCE.json").read_text())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip()
    assert head == provenance["commit"]
    for row in provenance["files"]:
        assert hashlib.sha256((upstream / row["upstream_path"]).read_bytes()).hexdigest() == row["upstream_sha256"]
        assert hashlib.sha256((ROOT / row["local_path"]).read_bytes()).hexdigest() == row["local_sha256"]
    stream = wav_blocks(args.wav)
    info = next(stream)
    assert info["channels"] == 1
    engine = LiveEngine.from_bundle(prime_silence=True)
    rows = []
    prefix = []
    for sample, hop in stream:
        events = engine.process(hop, start_sample=sample)
        rows.extend(x.json() for x in events)
        if sample < 44100 * 2:
            prefix.extend(x.json() for x in events)
    rows.extend(x.json() for x in engine.finalize(info["sample_count"], reason="eof"))
    integrity = check_events(rows, end_sample=info["sample_count"])
    # A separate run which never receives the suffix must make identical prefix decisions.
    early = LiveEngine.from_bundle(prime_silence=True)
    early_rows = []
    stream = wav_blocks(args.wav)
    next(stream)
    for sample, hop in stream:
        if sample >= 44100 * 2:
            break
        early_rows.extend(x.json() for x in early.process(hop, start_sample=sample))
    stream.close()
    assert early_rows == prefix
    early.finalize()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "reference.json"
        subprocess.run([sys.executable, "-c", REFERENCE_CODE, str(args.wav.resolve()), str(path)],
                       cwd=upstream, check=True)
        reference = json.loads(path.read_text())
    keys = ("type", "pitch", "velocity", "sample", "reason")
    comparison = [{key: row[key] for key in keys} for row in rows]
    assert comparison == reference, "MIDI reference replay differs"
    protected = json.loads((ROOT / "analysis/v273_live_and_offline_protected_modes.json").read_text())
    for name, expected in protected["protected_git_blob_sha1"].items():
        path = ROOT / name
        data = path.read_bytes() if path.exists() else subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=ROOT)
        actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        assert actual == expected, f"Protected legacy file changed: {name}"
    report = {"stage": "S610", "source_sha256": hashlib.sha256(args.wav.read_bytes()).hexdigest(),
              "source": info, "upstream_commit": head,
              "pinned_files_verified": len(provenance["files"]),
              "legacy_protected_files_verified": len(protected["protected_git_blob_sha1"]),
              "reference_events_exact_match": True, "matched_events": len(reference),
              "real_model_prefix_invariant": True, "integrity": integrity,
              "musical_accuracy_evaluated": False, "physical_latency_evaluated": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
