"""Rebuild missing upstream sources with the historical training programs.

These are newly trained sources, not restored historical bytes. This command
never claims that V27.3 or the paired native-decay experiment has been trained.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("v81", "audit", "v84", "v86", "v87", "v88")
DATA_MD5 = {
    "annotation.zip": "b39b78e63d3446f2e54ddb7a54df9b10",
    "audio_mono-pickup_mix.zip": "aecce79f425a44e2055e46f680e10f6a",
}
SOURCE_KIND = "new_training_from_historical_programs_not_restored_bytes"


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def source_args(source_dir):
    source_dir = Path(source_dir)
    result = ["--base-model", str(source_dir / "v84/control.epoch-01.keras")]
    for name, weights in (
        ("v86", "v86-state-transition-refiner.weights.h5"),
        ("v87", "v87-causal-candidate-memory.weights.h5"),
        ("v88", "v88-regime-moe.weights.h5"),
    ):
        result += [f"--{name}-weights", str(source_dir / name / weights),
                   f"--{name}-report", str(source_dir / name / "report.json")]
    return result


def stage_spec(stage, dataset_dir, output_dir):
    """Exact historical settings; stop V8.4 after the checkpoint actually used."""
    if stage not in STAGES:
        raise ValueError(stage)
    root, dataset = Path(output_dir), str(dataset_dir)
    audit = str(root / "audit/report.json")
    stream = str(root / "v81/stream.epoch-03.keras")
    base = str(root / "v84/control.epoch-01.keras")
    if stage == "v81":
        script = "train_v81.py"
        args = [dataset, "--epochs", "3", "--filters", "32", "--train-examples", "800",
                "--validation-examples", "200", "--learning-rate", "0.001",
                "--seed", "1337", "--output", str(root / "v81/stream.keras")]
        outputs = ["v81/stream.epoch-03.keras"]
    elif stage == "audit":
        script = "audit_v81_train_fp_harmonics.py"
        args = [dataset, "--model", stream, "--threshold", "0.40", "--target-fp", "4000",
                "--min-tracks", "24", "--max-tracks", "64", "--control-stride", "256",
                "--max-control-rms-delta-db", "3.0", "--output", audit]
        outputs = ["audit/report.json"]
    elif stage == "v84":
        script = "train_v84_ab.py"
        args = [dataset, "--source-model", stream, "--replay-audit", audit,
                "--epochs", "1", "--filters", "32", "--train-examples", "800",
                "--validation-examples", "200", "--learning-rate", "0.0001",
                "--seed", "1337", "--replay-fraction", "0.25",
                "--max-replay-per-track", "16", "--output-dir", str(root / "v84")]
        outputs = ["v84/control.epoch-01.keras", "v84/manifest.json"]
    else:
        script = {
            "v86": "train_v86_state_transition_proposals.py",
            "v87": "train_v87_causal_candidate_memory.py",
            "v88": "train_v88_regime_moe.py",
        }[stage]
        args = [dataset, "--base-model", base, "--train-audit", audit,
                "--train-members", "30", "--epochs", "20",
                "--output-dir", str(root / stage)]
        for previous, weights in (
            ("v86", "v86-state-transition-refiner.weights.h5"),
            ("v87", "v87-causal-candidate-memory.weights.h5"),
        ):
            if previous < stage:
                args += [f"--{previous}-weights", str(root / previous / weights),
                         f"--{previous}-report", str(root / previous / "report.json")]
        weights = {
            "v86": "v86-state-transition-refiner.weights.h5",
            "v87": "v87-causal-candidate-memory.weights.h5",
            "v88": "v88-regime-moe.weights.h5",
        }[stage]
        outputs = [f"{stage}/{weights}", f"{stage}/report.json"]
    return [sys.executable, "-B", str(ROOT / "scripts" / script), *args], outputs


def validate_sources(source_dir, required_stages=STAGES):
    root = Path(source_dir)
    state = json.loads((root / "rebuild-state.json").read_text())
    if state.get("source_kind") != SOURCE_KIND:
        raise RuntimeError("source provenance mismatch")
    for stage in required_stages:
        record = state.get("stages", {}).get(stage, {})
        if record.get("status") != "completed":
            raise RuntimeError(f"source stage incomplete: {stage}")
        if not record.get("outputs"):
            raise RuntimeError(f"missing source outputs: {stage}")
        for file in record["outputs"]:
            path = (root / file["path"]).resolve()
            if not path.is_relative_to(root.resolve()) or digest(path) != file["sha256"]:
                raise RuntimeError(f"source output changed: {stage}")
    return state


def verify_dataset(dataset_dir):
    for name, expected in DATA_MD5.items():
        if digest(Path(dataset_dir) / name, "md5") != expected:
            raise RuntimeError(f"GuitarSet checksum mismatch: {name}")


def run_stage(args):
    root = args.output_dir.resolve()
    dataset = args.dataset_dir.resolve()
    verify_dataset(dataset)
    state_path = root / "rebuild-state.json"
    previous = STAGES[:STAGES.index(args.stage)]
    if previous:
        state = validate_sources(root, previous)
    elif state_path.exists():
        raise FileExistsError("refusing to overwrite an existing rebuild")
    else:
        state = {
            "schema_version": 1, "source_kind": SOURCE_KIND,
            "created_at_utc": now(), "dataset_md5": DATA_MD5,
            "source_code_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            ).strip(),
            "native_decay_comparison_started": False,
            "historical_v273_reproduced": False, "stages": {},
        }
    if args.stage in state["stages"] or (root / args.stage).exists():
        raise FileExistsError(f"refusing to overwrite stage {args.stage}")
    command, outputs = stage_spec(args.stage, dataset, root)
    record = {"status": "running", "started_at_utc": now(), "command": command}
    state["stages"][args.stage] = record
    write_json(state_path, state)
    print(json.dumps({"stage": args.stage, "status": "real_source_training_or_audit_started",
                      "native_decay_comparison_started": False}), flush=True)
    try:
        subprocess.run(command, cwd=ROOT, check=True)
        record["outputs"] = [{"path": p, "sha256": digest(root / p)} for p in outputs]
        if args.stage == "audit":
            audit = json.loads((root / "audit/report.json").read_text())
            if len(audit["scope"]["members"]) < 30:
                raise RuntimeError("historical audit settings yielded fewer than 30 source tracks")
        record.update(status="completed", completed_at_utc=now())
    except BaseException:
        record.update(status="failed", finished_at_utc=now())
        write_json(state_path, state)
        raise
    write_json(state_path, state)
    print(json.dumps({"stage": args.stage, "status": "completed"}), flush=True)


def mine(args):
    source = args.source_dir.resolve()
    state = validate_sources(source)
    verify_dataset(args.dataset_dir)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    common = [sys.executable, "-B"]
    cluster_dir = args.output_dir / "clusters"
    spectral_dir = args.output_dir / "spectral"
    subprocess.run([
        *common, str(ROOT / "scripts/train_v91_ordinal_cardinality.py"), "mine",
        str(args.dataset_dir.resolve()), *source_args(source), "--shard-index", str(args.shard),
        "--shard-count", "8", "--output-dir", str(cluster_dir.resolve()),
    ], cwd=ROOT, check=True)
    subprocess.run([
        *common, str(ROOT / "scripts/train_v100_spectral_string_slots.py"), "mine",
        str(args.dataset_dir.resolve()), "--cache-dir", str(cluster_dir.resolve()),
        "--shard-index", str(args.shard), "--output-dir", str(spectral_dir.resolve()),
    ], cwd=ROOT, check=True)
    cache = spectral_dir / f"v100-spectral-shard-{args.shard:02d}.npz"
    write_json(args.output_dir / "provenance.json", {
        "source_kind": SOURCE_KIND, "source_code_sha": state["source_code_sha"],
        "source_manifest_sha256": digest(source / "rebuild-state.json"),
        "shard": args.shard, "sha256": digest(cache),
        "native_decay_comparison_started": False, "historical_v273_reproduced": False,
    })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    stage = sub.add_parser("stage")
    stage.add_argument("--stage", choices=STAGES, required=True)
    stage.add_argument("--dataset-dir", type=Path, required=True)
    stage.add_argument("--output-dir", type=Path, required=True)
    cache = sub.add_parser("mine")
    cache.add_argument("--source-dir", type=Path, required=True)
    cache.add_argument("--dataset-dir", type=Path, required=True)
    cache.add_argument("--output-dir", type=Path, required=True)
    cache.add_argument("--shard", type=int, choices=range(8), required=True)
    args = p.parse_args()
    run_stage(args) if args.command == "stage" else mine(args)


if __name__ == "__main__":
    main()
