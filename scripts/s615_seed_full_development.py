"""S615 full train/dev-only data seed; do not create legacy S613 256-hop cache.

Reuse exactly pinned S613 protocol and SHA-verified GuitarSet/URMP sources.
Download/decode original audio ONCE and save mono PCM + ground-truth ONLY
for train/dev sources. Test files and test labels are never opened.

This avoids writing ~gigabytes of unnecessary baseline feature .npz files
before S615's own three-hop feature extraction. Can be rerun idempotently.

Example:
 python -m scripts.s615_seed_full_development --root data/S615_full
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from causal_note.strict_live import SCHEMA, SR
from scripts.s613_prepare_strict_live import (
    digest, download, labels, urmp_blob,
)


PINNED = Path("analysis/evidence/v273-s613-strict-live/protocol.json")


def source_groups(protocol):
    groups = {part: {(s["corpus"], s["composition"]) for s in protocol["sources"]
                     if s["split"] == part}
              for part in ("train", "dev", "test")}
    if any(groups[a] & groups[b] for a,b in
           (("train","dev"),("train","test"),("dev","test"))):
        raise ValueError("Pinned source protocol has leaked compositions")
    return groups


def source_audio(spec, archives, audio_zip, root):
    if spec["corpus"] == "GuitarSet":
        data = audio_zip.read(spec["audio"])
    elif spec["corpus"] == "URMP":
        data = urmp_blob(spec["audio"], root)
    else:
        raise ValueError("Full S615 seed may only read GuitarSet and URMP train/dev")
    rate, signal = wavfile.read(io.BytesIO(data))
    if signal.dtype not in (np.int16,np.int32) or signal.ndim not in (1,2):
        raise ValueError("Unsupported PCM format: " + spec["name"])
    arr = signal.astype(np.float32) / (
        32768.0 if signal.dtype == np.int16 else 2147483648.0)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    if rate != SR:
        gcd = np.gcd(rate, SR)
        arr = resample_poly(arr, SR // gcd, rate // gcd).astype(np.float32)
    arr = np.asarray(arr[:int(spec["seconds"] * SR)], dtype=np.float32)
    return arr, data, int(rate), int(signal.shape[1] if signal.ndim == 2 else 1)


def seed_full(root, *, corpus=None, subset_sources=None, dev_only=False):
    protocol = json.loads(PINNED.read_text(encoding="utf-8"))
    if protocol["schema"] != SCHEMA or protocol["sample_rate"] != SR:
        raise ValueError("Pinned original S613 source protocol incompatible")
    source_groups(protocol)
    sources = [s for s in protocol["sources"] if s["split"] in ("train","dev")]
    if dev_only:
        sources = [s for s in sources if s["split"] == "dev"]
    if corpus is not None:
        if corpus not in ("GuitarSet","URMP"):
            raise ValueError("corpus must be GuitarSet or URMP")
        sources = [s for s in sources if s["corpus"] == corpus]
    if subset_sources is not None:
        if subset_sources < 1:
            raise ValueError("subset_sources must be positive")
        sources = sources[:subset_sources]
        print(json.dumps(dict(status="SMOKE_SUBSET_NOT_FULL_TRAINING",
                              sources=len(sources))), flush=True)
    root = Path(root)
    cache = root / "cache"
    data = root / "data"
    cache.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    target = root / "protocol.json"
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Existing protocol differs from pinned input")
    else:
        target.write_text(json.dumps(protocol,indent=2)+"\n",encoding="utf-8")

    archive = protocol["guitar_archives"]
    guitar = [s for s in sources if s["corpus"] == "GuitarSet"]
    audio_zip = None
    if guitar:
        for key,name in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            meta = archive[key]
            download(meta["url"], data/name, meta["size"], meta["sha256"])
        audio_zip = zipfile.ZipFile(data/"GuitarSet_audio.zip")
    receipts = []
    try:
        for spec in sources:
            stem = cache / spec["id"]
            wav = stem.with_suffix(".audio.npy")
            info = stem.with_suffix(".json")
            if wav.exists() and info.exists():
                previous=json.loads(info.read_text(encoding="utf-8"))
                if (previous.get("source") != spec or
                        not previous.get("annotations_read") or
                        "labels" not in previous or
                        np.load(wav,mmap_mode="r",allow_pickle=False).dtype!=np.float32):
                    raise ValueError("Existing cached train/dev source invalid")
                receipts.append(dict(name=spec["name"],split=spec["split"],
                                     status="cached"))
                continue
            if wav.exists() or info.exists():
                raise FileExistsError("Partial source cache; inspect: " + str(stem))
            signal, raw, rate, channels = source_audio(spec,archive,audio_zip,root)
            truth = labels(spec,root)
            if not truth["notes"]:
                raise ValueError("Unannotated source: " + spec["name"])
            np.save(wav,signal,allow_pickle=False)
            proof=dict(
                source=spec, audio_sha256=digest(raw),
                prepared_samples=int(len(signal)), native_rate=rate,
                native_channels=channels, annotations_read=True,
                labels=truth, source_split=spec["split"],
                s615_full_seed=True,
            )
            info.write_text(json.dumps(proof,indent=2)+"\n",encoding="utf-8")
            rec=dict(name=spec["name"],split=spec["split"],corpus=spec["corpus"],
                     seconds=round(len(signal)/SR,3),
                     notes=len(truth["notes"]),status="seeded")
            print(json.dumps(rec),flush=True)
            receipts.append(rec)
    finally:
        if audio_zip is not None:
            audio_zip.close()
    if subset_sources is None and corpus is None and not dev_only:
        n_train = sum(s["split"]=="train" for s in sources)
        n_dev = sum(s["split"]=="dev" for s in sources)
        if n_train!=53 or n_dev!=17:
            raise AssertionError(f"Expected 53 train / 17 dev, got {n_train}/{n_dev}")
    return receipts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--corpus",choices=("GuitarSet","URMP"))
    parser.add_argument("--dev-only",action="store_true",
                        help="Seed all 17 development tracks only (no training or held-out test)")
    parser.add_argument("--subset-sources",type=int,
                        help="Smoke only: source subset, NOT full training")
    opts=parser.parse_args()
    receipts=seed_full(opts.root,corpus=opts.corpus,
                       subset_sources=opts.subset_sources,dev_only=opts.dev_only)
    print(json.dumps(dict(result="S615_SEED_FINISHED",
                          count=len(receipts),
                          splits={split:sum(s["split"]==split for s in receipts)
                                  for split in ("train","dev")},
                          test_data_read=False)),flush=True)


if __name__=="__main__":
    main()
