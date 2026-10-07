"""Download a pinned official inference snapshot and verify every source blob."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request

from scripts.yourmt3_exactk_common import (SPACE_REVISION, CHECKPOINT_PATH,
    CHECKPOINT_SHA256, digest, require, write_json)


def source_hash(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def main(a):
    manifest = json.loads(a.manifest.read_text())
    require(manifest["revision"] == SPACE_REVISION and manifest["repo_type"] == "space",
            "source revision drift")
    a.output.mkdir(parents=True, exist_ok=True)
    prefix = f"https://huggingface.co/spaces/mimbres/YourMT3/resolve/{SPACE_REVISION}/"

    def get_source(entry):
        path = a.output / entry["path"]
        if path.exists() and source_hash(path.read_bytes()) == entry["oid"]:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(prefix + entry["path"], timeout=90) as stream:
                    data = stream.read()
                require(len(data) == entry["size"] and source_hash(data) == entry["oid"],
                        "source checksum mismatch: " + entry["path"])
                path.write_bytes(data)
                return
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(get_source, manifest["files"]))
    checkpoint = a.output / CHECKPOINT_PATH
    if not checkpoint.exists() or digest(checkpoint) != CHECKPOINT_SHA256:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temp = checkpoint.with_suffix(".part")
        for attempt in range(3):
            try:
                with urllib.request.urlopen(prefix + CHECKPOINT_PATH, timeout=120) as stream, temp.open("wb") as out:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        out.write(block)
                require(digest(temp) == CHECKPOINT_SHA256, "checkpoint checksum mismatch")
                temp.replace(checkpoint)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
    write_json(a.output / "verified.json", {"space_revision": SPACE_REVISION,
                "checkpoint_sha256": digest(checkpoint), "source_manifest_sha256": digest(a.manifest),
                "source_files": len(manifest["files"])})
    print("Pinned official YourMT3+ source and checkpoint verified.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("analysis/yourmt3-source-manifest.json"))
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
