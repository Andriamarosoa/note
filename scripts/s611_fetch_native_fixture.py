"""Fetch only the pinned GuitarSet MIC member used in the earlier S434 smoke.

No annotations. HTTP ranges and member hashes are checked; the complete archive
hash is deliberately not claimed as recomputed. Raw audio is not committed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import wave
import zipfile
import numpy as np
from scripts.s183_idmt_selective_http_range_s35 import HttpRangeZIP
from scripts.s411_extract_continuous_GuitarSet_WAV_pitch_survival_features import decode_pcm

SOURCE = Path(__file__).resolve().parents[1] / "analysis/evidence/v273-s611-native-stream/fixture_provenance.json"


def fetch(out):
    if out.exists():
        raise ValueError("Fixture output already exists")
    source = json.loads(SOURCE.read_text())
    def receive(start, end):
        req = urllib.request.Request(source["archive_url"], headers={
            "Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
        with urllib.request.urlopen(req, timeout=60) as response:
            if response.status != 206 or response.headers.get("Content-Range") != f'bytes {start}-{end}/{source["archive_bytes"]}':
                raise ValueError("Exact bounded HTTP Range required")
            data = response.read(end - start + 2)
        if len(data) != end - start + 1:
            raise ValueError("Truncated or oversized range")
        return data
    reader = HttpRangeZIP(source["archive_url"], source["archive_bytes"], fetch=receive, window=1048576)
    with zipfile.ZipFile(reader) as archive:
        info = archive.getinfo(source["member"])
        if info.CRC != source["member_crc32"] or info.file_size != source["member_size"]:
            raise ValueError("GuitarSet member metadata changed")
        raw = archive.read(info)
    if hashlib.sha256(raw).hexdigest() != source["member_sha256"]:
        raise ValueError("GuitarSet member hash changed")
    audio = decode_pcm(raw)[:int(source["excerpt_seconds"] * 44100)]
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(44100)
        wav.writeframes(np.clip(np.rint(audio * 32768), -32768, 32767).astype("<i2").tobytes())
    if hashlib.sha256(out.read_bytes()).hexdigest() != source["excerpt_sha256"]:
        raise ValueError("S434 original excerpt differs")
    print(json.dumps(dict(status="S434_ORIGINAL_AUDIO_RESTORED", bytes_read=reader.network_bytes,
                          wav_sha256=source["excerpt_sha256"], annotations_read=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    fetch(parser.parse_args().out)
