"""S90: inspect archived producer bundles without opening labels or claiming S80 validation."""
from __future__ import annotations
import argparse, json, re, hashlib
from pathlib import Path
from zipfile import ZipFile, BadZipFile

ARCHIVES = (
    "analysis/evidence/v273-audit-memory/source-archives/upstream-producers.zip",
    "analysis/evidence/v273-audit-memory/source-archives/model-replay-bundles.zip",
)
PATTERN = re.compile(r"s45|s4[0-6]|series4[0-6]|ridge|risk|proposal|producer|family", re.I)

def inventory(root: Path):
    records = []
    for relative in ARCHIVES:
        path = root / relative
        entry = {"archive": relative, "exists": path.is_file(), "matches": [], "files": 0}
        if path.is_file():
            entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            try:
                with ZipFile(path) as z:
                    bad = z.testzip()
                    if bad: raise ValueError("Corrupt ZIP member: " + bad)
                    names = z.namelist()
                    entry["files"] = len(names)
                    entry["matches"] = sorted(n for n in names if PATTERN.search(n))
            except BadZipFile as exc:
                raise ValueError("Invalid archive " + relative) from exc
        records.append(entry)
    return {"status": "S90_PRODUCER_ARCHIVE_INVENTORY_ONLY", "archives": records,
            "s45_deployable": False, "external_s80_score_measured": False,
            "promotion_approved": False,
            "note": "Filename discovery alone cannot establish a runnable original S45 producer."}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = inventory(a.root)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
