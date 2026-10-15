"""Pinned real audio for a new causal NOTE branch; composition-disjoint splits.

No previous BP/V88 prediction, S80 K, or annotation time is an inference input.
GuitarSet player 05 is test-only. All arrangements of each URMP title share a
split. This new experiment is retrospective with respect to earlier audits.
Held-out labels are opened only by the separate `score` stage.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from causal_note.strict_live import feature_track, SR, SCHEMA

REV = "3a62b3a1c8bb263a6a6ac37559070339ddb81080"
BASE = f"https://huggingface.co/datasets/J1mmymm/MIMuT_Data_v2/resolve/{REV}/"


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def download(url, path, size, sha=None):
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as r, temporary.open("wb") as f:
            while chunk := r.read(1024 * 1024):
                f.write(chunk)
        if temporary.stat().st_size != size:
            raise ValueError("Pinned download size changed")
        temporary.rename(path)
    if path.stat().st_size != size or (sha and digest(path.read_bytes()) != sha):
        raise ValueError("Pinned download identity changed")
    return path.read_bytes()


def manifest(root, guitar_sources, urmp_tree):
    path = root / "protocol.json"
    if path.exists():
        return json.loads(path.read_text())
    source_registry = json.loads(guitar_sources.read_text())
    archive = source_registry["archives"]
    g = archive["GuitarSet_annotations"]
    download(g["url"], root / "data/GuitarSet_annotations.zip", g["size"], g["sha256"])
    with zipfile.ZipFile(root / "data/GuitarSet_annotations.zip") as z:
        names = [n for n in z.namelist() if n.endswith(".jams")]
    arrangements = sorted({n[3:-5] for n in names})
    compositions = sorted({n.split("-", 1)[0] for n in arrangements}, key=lambda s: digest(s.encode()))
    assert len(compositions) == 15 and len(arrangements) == 60
    # Metadata-only stratification: leave at least one progression from every
    # style in training; don't accidentally hold out the entire Funk family.
    import re
    style = lambda name: re.sub(r"\d+$", "", name)
    assignment = {}
    for split in ("test", "dev"):
        used_styles = set()
        for composition in compositions:
            group = style(composition)
            remaining = sum(style(c) == group and c not in assignment for c in compositions)
            if composition not in assignment and group not in used_styles and remaining > 1:
                assignment[composition] = split
                used_styles.add(group)
                if len(used_styles) == 3:
                    break
    sources = []
    for i, composition in enumerate(compositions):
        split = assignment.get(composition, "train")
        for arrangement in [a for a in arrangements if a.split("-", 1)[0] == composition]:
            player = "05" if split == "test" else "04" if split == "dev" else f"{int(digest(arrangement.encode())[:8], 16) % 4:02d}"
            name = f"{player}_{arrangement}"
            assert name + ".jams" in names
            sources.append(dict(corpus="GuitarSet", split=split, composition=composition,
                                name=name, id=digest(name.encode()), offset=0., seconds=24.,
                                audio=name + "_mic.wav", annotation=name + ".jams"))
    listing = json.loads(urmp_tree.read_text())
    mixtures = [x for x in listing if "/AuMix_" in x["path"] and x["path"].endswith(".wav")]
    groups = sorted({x["path"].split("/")[-2].split("_")[1] for x in mixtures}, key=lambda s: digest(s.encode()))
    for i, composition in enumerate(groups):
        split = "test" if i < 5 else "dev" if i < 10 else "train"
        choices = [x for x in mixtures if x["path"].split("/")[-2].split("_")[1] == composition]
        item = min(choices, key=lambda x: digest(x["path"].encode()))
        folder = item["path"].rsplit("/", 1)[0]
        anns = [x for x in listing if x["path"].startswith(folder + "/Notes_") and x["path"].endswith(".txt")]
        sources.append(dict(corpus="URMP", split=split, composition=composition, name=folder.split("/")[-1],
                            id=digest(item["path"].encode()), audio=item, annotations=anns, offset=0., seconds=24.))
    for item in source_registry["sources"]["IDMT"]:
        sources.append(dict(corpus="IDMT", split="test", composition=item["label"], name=item["label"],
                            id=item["source_id"], audio=item["audio"], annotation=item["annotation"],
                            offset=0., seconds=24., archive=archive["IDMT_audio"]))
    result = dict(schema=SCHEMA, seed=613, sources=sources, guitar_archives=archive,
                  sample_rate=SR, duration_limit=24.,
                  partitions="SHA256(composition), three distinct styles each in GuitarSet test/dev with at least one train family/style: 3 test / 3 dev / 9 train families, all tempi/keys/roles grouped; URMP 5 / 5 / 17 titles",
                  guitar_player_05="test only, never used by a training or tuning loader",
                  annotation_gate="train/dev labels allowed; test labels only after sealed predictions",
                  historical_sources_previously_exposed=True,
                  metric="pitch-identical one-to-one NoteOn, actual emission time, tolerances 50/80ms",
                  no_foreign_models=True, original_S80_and_recent_checkpoints_unchanged=True)
    save(path, result)
    return result


def urmp_blob(item, root):
    path = root / "data" / (digest(item["path"].encode()) + Path(item["path"]).suffix)
    blob = download(BASE + urllib.parse.quote(item["path"], safe="/"), path, item["size"], item.get("lfs", {}).get("oid"))
    if not item.get("lfs"):
        assert hashlib.sha1(b"blob " + str(len(blob)).encode() + b"\0" + blob).hexdigest() == item["oid"]
    return blob


def idmt_blob(spec, item, root):
    from scripts.s183_idmt_selective_http_range_s35 import HttpRangeZIP
    path = root / "data" / (digest(item["path"].encode()) + Path(item["path"]).suffix)
    if not path.exists():
        archive = spec["archive"]
        def receive(start, end):
            for attempt in range(4):
                try:
                    request = urllib.request.Request(archive["url"], headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
                    with urllib.request.urlopen(request, timeout=120) as response:
                        if response.status != 206 or response.headers.get("Content-Range") != f'bytes {start}-{end}/{archive["size"]}':
                            raise ValueError("Exact IDMT range required")
                        blob = response.read(end - start + 2)
                    if len(blob) != end - start + 1:
                        raise ValueError("Truncated IDMT range")
                    return blob
                except Exception:
                    if attempt == 3:
                        raise
        reader = HttpRangeZIP(archive["url"], archive["size"], fetch=receive, window=1048576)
        with zipfile.ZipFile(reader) as z:
            info = z.getinfo(item["path"])
            assert info.file_size == item["size"] and info.CRC == item["crc32"]
            path.write_bytes(z.read(info))
    blob = path.read_bytes()
    import zlib
    assert len(blob) == item["size"] and zlib.crc32(blob) == item["crc32"]
    return blob


def labels(spec, root):
    notes, proof = [], []
    if spec["corpus"] == "GuitarSet":
        with zipfile.ZipFile(root / "data/GuitarSet_annotations.zip") as z:
            blob = z.read(spec["annotation"])
        proof.append(digest(blob))
        for ann in json.loads(blob)["annotations"]:
            if ann.get("namespace") == "note_midi":
                for row in ann["data"]:
                    notes.append(dict(time=float(row["time"]), duration=float(row["duration"]),
                                      pitch=int(round(float(row["value"]))), track=str(ann["annotation_metadata"]["data_source"])))
    elif spec["corpus"] == "IDMT":
        blob = idmt_blob(spec, spec["annotation"], root)
        proof.append(digest(blob))
        for event in ET.fromstring(blob).iter():
            if event.tag.split("}")[-1].lower() != "event":
                continue
            fields = {x.tag.split("}")[-1].lower(): (x.text or "").strip() for x in event.iter()}
            notes.append(dict(time=float(fields["onsetsec"]), duration=float(fields["offsetsec"]) - float(fields["onsetsec"]),
                              pitch=int(float(fields["pitch"])), track="guitar"))
    else:
        for item in spec["annotations"]:
            blob = urmp_blob(item, root)
            proof.append(digest(blob))
            for t, hz, duration in np.loadtxt(io.BytesIO(blob), ndmin=2):
                notes.append(dict(time=float(t), duration=float(duration), pitch=int(round(69 + 12 * np.log2(hz / 440))),
                                  track=item["path"].split("/")[-1]))
    return dict(notes=sorted(notes, key=lambda n: (n["time"], n["pitch"])), sha256=proof)


def prepare(root, protocol, phase):
    choices = [s for s in protocol["sources"] if (s["split"] == "test") == (phase == "test-audio")]
    archive = protocol["guitar_archives"]["GuitarSet_audio"]
    download(archive["url"], root / "data/GuitarSet_audio.zip", archive["size"], archive["sha256"])
    def one(spec):
        target = root / "cache" / spec["id"]
        if target.with_suffix(".npz").exists() and target.with_suffix(".json").exists():
            return dict(name=spec["name"], cached=True)
        if spec["corpus"] == "GuitarSet":
            with zipfile.ZipFile(root / "data/GuitarSet_audio.zip") as z:
                blob = z.read(spec["audio"])
        elif spec["corpus"] == "IDMT":
            blob = idmt_blob(spec, spec["audio"], root)
        else:
            blob = urmp_blob(spec["audio"], root)
        rate, pcm = wavfile.read(io.BytesIO(blob))
        assert pcm.dtype in (np.int16, np.int32) and pcm.ndim in (1, 2)
        channels = 1 if pcm.ndim == 1 else pcm.shape[1]
        audio = pcm.astype(np.float32) / (32768 if pcm.dtype == np.int16 else 2147483648)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        native_rate = rate
        if rate != SR:
            divisor = np.gcd(rate, SR)
            audio = resample_poly(audio, SR // divisor, rate // divisor).astype(np.float32)
        audio = audio[:int(spec["seconds"] * SR)].copy()
        ends, features = feature_track(audio)
        np.save(target.with_suffix(".audio.npy"), audio, allow_pickle=False)
        np.savez_compressed(target.with_suffix(".npz"), ends=ends, features=features)
        proof = dict(source=spec, audio_sha256=digest(blob), prepared_samples=len(audio), native_rate=int(native_rate), native_channels=int(channels),
                     feature_schema=SCHEMA, feature_shape=list(features.shape), annotations_read=phase != "test-audio")
        if phase != "test-audio":
            assert spec["split"] != "test"
            proof["labels"] = labels(spec, root)
        save(target.with_suffix(".json"), proof)
        return dict(name=spec["name"], split=spec["split"], seconds=len(audio) / SR, features=list(features.shape))
    with ThreadPoolExecutor(max_workers=3) as executor:
        for result in executor.map(one, choices):
            print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--guitar-sources", type=Path)
    parser.add_argument("--urmp-tree", type=Path)
    parser.add_argument("--phase", choices=("development", "test-audio"), required=True)
    args = parser.parse_args()
    (args.root / "cache").mkdir(parents=True, exist_ok=True)
    (args.root / "data").mkdir(parents=True, exist_ok=True)
    protocol = manifest(args.root, args.guitar_sources, args.urmp_tree)
    prepare(args.root, protocol, args.phase)


if __name__ == "__main__":
    main()
