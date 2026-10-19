"""Require resume.md + resultat.md for every NOTE series S788 onward.

This validator guarantees reporting *structure and provenance fields*, not
scientific correctness. Six independent S787 audits and the final 256-hop MIDI
test remain separate prerequisites for any model promotion.

Usage:
  python -m scripts.verify_series_reports --series S788
  python -m scripts.verify_series_reports --series-from-branch
  python -m scripts.verify_series_reports --series S788 --mode release
  python -m scripts.verify_series_reports --all
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

MIN_SERIES = 788
STATUS = {"en_cours", "termine", "echoue", "annule"}
KINDS = {"experience", "infrastructure"}
REQUIRED = {
    "resume.md": (
        "Objectif", "Hypothèse", "Modifications", "Protocole",
        "Audits obligatoires", "Statut"
    ),
    "resultat.md": (
        "Exécution", "Métriques", "Comparaison à S708",
        "Audits", "Verdict", "Prochaine action"
    ),
}
ROLES = ("Données", "Acoustique", "Architecture",
         "Apprentissage", "Métriques", "LIVE")
METRICS = ("Onset F1", "Exact-K", "Active Exact-K",
           "Offset", "Full Note F1", "Frame Pitch F1")


def series_number(name: str) -> int:
    match = re.fullmatch(r"S([1-9]\d{2,})", name.strip())
    if not match or int(match.group(1)) < MIN_SERIES:
        raise ValueError(f"series must be S{MIN_SERIES} or later, got {name!r}")
    return int(match.group(1))


def from_branch(branch: str) -> str:
    # Example: codex/note-s788-mandatory-series-reports.
    match = re.search(r"(?:^|/)note-s([1-9]\d{2,})(?:-|$)", branch)
    if not match:
        raise ValueError(f"cannot identify NOTE series from branch {branch!r}")
    return f"S{series_number('S' + match.group(1))}"


def parse_document(path: Path, number: int) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"missing mandatory file: {path}")
    content = path.read_text(encoding="utf-8")
    front = re.match(r"\A---\s*\n(.*?)\n---\s*\n", content, flags=re.S)
    if not front:
        raise ValueError(f"{path}: YAML-like metadata header absent")
    fields: dict[str, str] = {}
    for entry in front.group(1).splitlines():
        line = entry.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"{path}: invalid metadata line {entry!r}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key in fields or not value:
            raise ValueError(f"{path}: duplicate or empty field {key!r}")
        fields[key] = value
    mandatory = ("serie", "statut", "type", "reference", "branche",
                 "run_id", "commit_sha", "promotion")
    for key in mandatory:
        if key not in fields:
            raise ValueError(f"{path}: missing metadata {key}")
    if fields["serie"] != f"S{number}":
        raise ValueError(f"{path}: wrong series identifier")
    if fields["statut"] not in STATUS:
        raise ValueError(f"{path}: unknown status {fields['statut']!r}")
    if fields["type"] not in KINDS:
        raise ValueError(f"{path}: unknown series type")
    if fields["reference"] != "S708":
        raise ValueError(f"{path}: official protected reference must be S708")
    if f"note-s{number}" not in fields["branche"].lower():
        raise ValueError(f"{path}: unexpected branch identity")
    if fields["promotion"] not in {"true", "false"}:
        raise ValueError(f"{path}: promotion must be true or false")
    for heading in REQUIRED[path.name]:
        if not re.search(r"^##\s+" + re.escape(heading) + r"\s*$",
                         content, flags=re.M):
            raise ValueError(f"{path}: missing section ## {heading}")
    if len(content.split()) < 95:
        raise ValueError(f"{path}: document is too short to be auditable")
    if fields["statut"] != "termine" and fields["promotion"] == "true":
        raise ValueError(f"{path}: unfinished series cannot be promoted")
    if fields["statut"] == "termine":
        if not fields["run_id"].isdigit():
            raise ValueError(f"{path}: completed series needs a GitHub run ID")
        if not re.fullmatch(r"[0-9a-f]{40}", fields["commit_sha"]):
            raise ValueError(f"{path}: completed series needs exact 40-char SHA")
    return fields


def validate_series(root: Path, series: str, mode: str = "draft") -> dict[str, str]:
    n = series_number(series)
    directory = root / "series" / f"S{n}"
    docs = {name: parse_document(directory / name, n) for name in REQUIRED}
    a, b = (docs[name] for name in ("resume.md", "resultat.md"))
    for key in ("serie", "statut", "type", "reference",
                "branche", "run_id", "commit_sha", "promotion"):
        if a[key] != b[key]:
            raise ValueError(f"S{n}: {key} differs between documents")
    if mode == "release" and a["statut"] != "termine":
        raise ValueError(f"S{n}: cannot release with status {a['statut']}")
    results = (directory / "resultat.md").read_text(encoding="utf-8")
    for name in METRICS:
        if not re.search(re.escape(name), results, flags=re.I):
            raise ValueError(f"S{n}: missing disclosure for {name}")
    if a["statut"] == "termine":
        run = a["run_id"]
        if (f"github.com/Andriamarosoa/note/actions/runs/{run}"
            not in results):
            raise ValueError(f"S{n}: completed report lacks actual GitHub run URL")
    if a["type"] == "experience" and a["statut"] == "termine":
        for role in ROLES:
            pattern = r"^\|\s*" + re.escape(role) + (
                r"\s*\|\s*(PASS|FAIL|UNKNOWN)\s*\|\s*\S[^|]*\|\s*$")
            match = re.search(pattern, results, flags=re.M)
            if not match:
                raise ValueError(f"S{n}: no final audit table row for {role}")
            if a["promotion"] == "true" and match.group(1) != "PASS":
                raise ValueError(f"S{n}: promotion blocked by {role} audit")
    # S793 global promotion veto: research may continue freely, but every
    # future S794+ model promotion MUST supply raw provenance and repeated
    # paired physical latency evidence. Never accept a precomputed boolean.
    if n >= 794 and a["type"] == "experience" and a["promotion"] == "true":
        from src.causal_note.s793_zero_added_latency import (
            LatencyViolation, SchedulingContract, veto_promotion
        )
        proof = directory / "latency_evidence.json"
        if not proof.is_file():
            raise ValueError(f"S{n}: missing mandatory S793 hardware latency evidence")
        try:
            evidence = json.loads(proof.read_text(encoding="utf-8"))
            veto_promotion(
                contract=SchedulingContract(**evidence["scheduling"]),
                reference=evidence["reference_events"],
                candidate=evidence["candidate_events"],
                baseline_runtime=evidence["baseline_runtime"],
                candidate_runtime=evidence["candidate_runtime"],
                baseline_metrics=evidence["baseline_metrics"],
                candidate_metrics=evidence["candidate_metrics"],
            )
        except (ValueError, KeyError, TypeError, LatencyViolation) as exc:
            raise ValueError(f"S{n}: S793 zero-added-latency promotion veto: {exc}") from exc
    return {"series": f"S{n}", "status": a["statut"],
            "type": a["type"], "promotion": a["promotion"],
            "mode": mode, "path": str(directory)}


def main() -> None:
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--series")
    group.add_argument("--series-from-branch", action="store_true")
    group.add_argument("--all", action="store_true")
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--mode", choices=("draft", "release"), default="draft")
    args = p.parse_args()
    try:
        if args.all:
            dir_ = args.root / "series"
            series = sorted(
                (p.name for p in dir_.glob("S*") if p.is_dir()
                 and re.fullmatch(r"S[1-9]\d{2,}", p.name)
                 and int(p.name[1:]) >= MIN_SERIES),
                key=lambda s: int(s[1:])
            )
            if not series:
                raise ValueError("no series S788+ found")
        else:
            series = [from_branch(os.environ.get("GITHUB_REF_NAME", ""))
                      if args.series_from_branch else args.series]
        for identifier in series:
            report = validate_series(args.root, identifier, args.mode)
            print("NOTE_SERIES_DOCS_PASS", report, flush=True)
    except (ValueError, OSError) as error:
        print("NOTE_SERIES_DOCS_FAIL", error, file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
