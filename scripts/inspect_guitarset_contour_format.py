"""Print the serialized shape of one GuitarSet pitch_contour annotation."""
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
archive_path = ROOT / "data" / "GuitarSet" / "annotation.zip"

with zipfile.ZipFile(archive_path, "r") as archive:
    member = sorted(name for name in archive.namelist() if name.startswith("00_") and name.endswith(".jams"))[0]
    document = json.loads(archive.read(member).decode("utf-8"))

for annotation in document["annotations"]:
    if annotation.get("namespace") != "pitch_contour":
        continue
    data = annotation.get("data")
    print("member", member)
    print("source", annotation.get("annotation_metadata", {}).get("data_source"))
    print("data_type", type(data).__name__)
    if isinstance(data, dict):
        print("data_keys", sorted(data.keys()))
        for key, value in data.items():
            print("field", key, "type", type(value).__name__, "preview", repr(value)[:1000])
    elif isinstance(data, list):
        print("length", len(data))
        print("first", repr(data[:2])[:2000])
    else:
        print("preview", repr(data)[:2000])
    break
