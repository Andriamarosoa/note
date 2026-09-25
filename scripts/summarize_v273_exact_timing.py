"""Require complete, disjoint coverage of the single audited outer fold."""
import argparse
import json
from pathlib import Path
import subprocess

from scripts.rebuild_v273_sources import digest, write_json


def summarize(root, config):
    cfg = json.loads(config.read_text())
    allowed = {m for m, fold in cfg["member_folds"].items() if fold == 3}
    parts = [json.loads((root / f"part-{i}" / "report.json").read_text()) for i in range(5)]
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    tracks = []
    for i, part in enumerate(parts):
        if (part["status"] != "passed" or part["outer_fold"] != 3 or part["part"] != i
                or part["parts"] != 5 or len(part["tracks"]) != 10
                or part["training_performed"] or part["source_sha"] != source_sha
                or part["config_sha256"] != digest(config)):
            raise RuntimeError(f"inconsistent audit part {i}")
        for key in ("previous_metadata_sha256", "source_state_sha256", "proposal_source_kind"):
            if part[key] != parts[0][key]:
                raise RuntimeError(f"source differs between audit parts: {key}")
        tracks.extend(part["tracks"])
    if len(tracks) != 50 or {r["member"] for r in tracks} != allowed:
        raise RuntimeError("fold 3 audit coverage incomplete or duplicated")
    if not all(r["all_runtime_comparisons_equal"] for r in tracks):
        raise RuntimeError("runtime/cache disagreement")
    report = {key: value for key, value in parts[0].items() if key not in ("part", "parts", "totals", "tracks")}
    report.update(tracks=sorted(tracks, key=lambda r: r["member"]),
        totals={key: sum(p["totals"][key] for p in parts) for key in parts[0]["totals"]},
        all_historical_cache_features_identical=all(p["all_historical_cache_features_identical"] for p in parts),
        scope="all 50 frozen fold-3 tracks only; no other fold inference, training or calibration")
    report["audit_part_sha256"] = {str(i): digest(root / f"part-{i}" / "report.json") for i in range(5)}
    write_json(root / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "tracks"}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args()
    summarize(args.root, args.config)
