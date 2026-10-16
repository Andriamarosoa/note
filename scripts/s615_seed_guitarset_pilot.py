"""Seed a SMALL real GuitarSet train/dev S615 pilot from pinned S613 archives.

No test audio/labels are read. This is an unpromotable exploratory pilot,
NOT a replacement for the full GuitarSet+URMP independent validation.
Downloads the SAME pinned, SHA256-verified S613 full GuitarSet ZIPs.
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
from scripts.s613_prepare_strict_live import download, labels, digest


def seed(root, *, ntrain=6, ndev=3):
    if ntrain < 1 or ndev < 1:
        raise ValueError("Need train/dev sources")
    ref=Path("analysis/evidence/v273-s613-strict-live/protocol.json")
    proto=json.loads(ref.read_text(encoding="utf-8"))
    if proto["schema"] != SCHEMA:
        raise ValueError("Mismatch with pinned S613 protocol")
    train=[s for s in proto["sources"] if s["split"]=="train" and s["corpus"]=="GuitarSet"][:ntrain]
    dev=[s for s in proto["sources"] if s["split"]=="dev" and s["corpus"]=="GuitarSet"][:ndev]
    if len(train)!=ntrain or len(dev)!=ndev:
        raise ValueError("Requested too many pinned GuitarSet sources")
    root.mkdir(parents=True,exist_ok=True)
    data=root/"data"
    cache=root/"cache"
    data.mkdir(exist_ok=True)
    cache.mkdir(exist_ok=True)
    if not (root/"protocol.json").exists():
        (root/"protocol.json").write_text(json.dumps(proto,indent=2)+"\n",encoding="utf-8")
    elif json.loads((root/"protocol.json").read_text()) != proto:
        raise ValueError("Protocol mismatch in existing pilot directory")
    archives=proto["guitar_archives"]
    for name,file in (("GuitarSet_audio","GuitarSet_audio.zip"),
                      ("GuitarSet_annotations","GuitarSet_annotations.zip")):
        desc=archives[name]
        download(desc["url"],data/file,desc["size"],desc["sha256"])
    with zipfile.ZipFile(data/"GuitarSet_audio.zip") as z:
        for spec in train+dev:
            stem=cache/spec["id"]
            wav_path=stem.with_suffix(".audio.npy")
            meta_path=stem.with_suffix(".json")
            if wav_path.exists() and meta_path.exists():
                print(json.dumps(dict(source=spec["name"],status="cached")),flush=True)
                continue
            blob=z.read(spec["audio"])
            rate,pcm=wavfile.read(io.BytesIO(blob))
            if pcm.dtype not in (np.int16,np.int32) or pcm.ndim not in (1,2):
                raise ValueError("Unsupported pinned GuitarSet WAV")
            signal=pcm.astype(np.float32)/(32768 if pcm.dtype==np.int16 else 2147483648)
            if signal.ndim==2:
                signal=signal.mean(axis=1)
            if rate!=SR:
                g=np.gcd(rate,SR)
                signal=resample_poly(signal,SR//g,rate//g).astype(np.float32)
            signal=np.asarray(signal[:int(spec["seconds"]*SR)],np.float32)
            truth=labels(spec,root)
            np.save(wav_path,signal,allow_pickle=False)
            proof=dict(source=spec, audio_sha256=digest(blob),
                       prepared_samples=len(signal),native_sample_rate=int(rate),
                       annotations_read=True,labels=truth,
                       pilot_only=True)
            meta_path.write_text(json.dumps(proof,indent=2)+"\n",encoding="utf-8")
            print(json.dumps(dict(source=spec["name"],split=spec["split"],
                                  samples=len(signal),notes=len(truth["notes"]),
                                  pilot_only=True)),flush=True)
    return train,dev


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--train-sources",type=int,default=6)
    p.add_argument("--dev-sources",type=int,default=3)
    a=p.parse_args()
    seed(a.root,ntrain=a.train_sources,ndev=a.dev_sources)


if __name__=="__main__":
    main()
