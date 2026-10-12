"""S172: recover ORIGINAL archived S1/S3/S4 fitted states, audit six S45 gaps.

This is archive reconstruction/provenance, NEVER external WAV inference.
No truth labels, MIDI, or candidate outcome files may be read. S2 is a
router without a separate checkpoint. Do not call these six producers ready.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path

SOURCE = Path("analysis/evidence/v273-regression-loops")
REQUIRED = {
    "S1_acoustic": "regression_loops_s1__acoustic43__cost_1.3",
    "S2_source": "regression_loops_s2__source_count90",
    "S3_repair": "regression_loops_s3__tree_nested_source_count90",
    "S4_repair": "regression_loops_s4__guard_acoustic58__blend_1__cost_2",
    "historical_repair_coherent": "repair_coherent",
    "S35_multiroute": "series35__lambda1__threshold0",
}
SCRIPTS = {
    "S1_acoustic": "scripts/v273_regression_loops.py",
    "S2_source": "scripts/route_v273_regression_loops.py",
    "S3_repair": "scripts/nonlinear_v273_regression_loops.py",
    "S4_repair": "scripts/guard_v273_regression_loops.py",
    "S35_multiroute": "scripts/train_v273_extended_dynamic_router.py",
}
def require(ok, why):
    if not ok:
        raise ValueError(why)

def digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def run(root: Path, output: Path):
    require(not output.exists(), "refusing to overwrite immutable S172 result")
    catalog = json.loads((root / "analysis/v273_s45_original_champion_catalog.json").read_text())
    graph = json.loads((root / "analysis/v273_s80_full_dependency_contract.json").read_text())
    manifest = json.loads((root / SOURCE / "storage-manifest.json").read_text())
    original = {x["family"]: x for x in catalog["original_families"]}
    statuses = {x["id"]: x["state"] for x in graph["families"]}
    require(len(original) == 16 and len(set(original)) == 16, "S44 family count drift")
    require(set(REQUIRED) <= set(original) and set(REQUIRED) <= set(statuses),
            "one of six precise original producers absent")
    require(all(original[f]["original_variant"] == v for f, v in REQUIRED.items()),
            "historical champion variant drift")
    require(all(statuses[f] == "historical_vector_only" for f in REQUIRED),
            "dependency contract changed; manual reevaluation mandatory")
    for family, path in SCRIPTS.items():
        require((root / path).is_file(), "historical source missing for " + family)
    file_records = {x["stored"]: x for x in manifest["files"]}
    require(len(file_records) == len(manifest["files"]), "ambiguous archive entries")
    archives = {int(x["series"]): x for x in manifest["model_archives"]}
    require(set(archives) == {3, 4}, "historical compact trained model inventory drift")
    # Prevent use of truth-bearing historical OOF predictions as new-WAV sources.
    target = output / "recovered_historical_models"
    target.mkdir(parents=True)
    model_proofs = []
    packed = root / SOURCE / "series1/models.json.gz"
    record = file_records["series1/models.json.gz"]
    compressed = packed.read_bytes()
    require(digest(compressed) == record["stored_sha256"], "S1 compressed SHA drift")
    plain = gzip.decompress(compressed)
    require(digest(plain) == record["source_sha256"], "S1 original JSON SHA drift")
    (target / "S1_models.json").write_bytes(plain)
    model_proofs.append({"family": "S1_acoustic", "source_archive": str(packed.relative_to(root)),
                         "restored_file": "S1_models.json",
                         "original_file_sha256": digest(plain), "bytes": len(plain)})
    for number, family in ((3, "S3_repair"), (4, "S4_repair")):
        info = archives[number]
        chunks = sorted(info["parts"], key=lambda x: x["file"])
        require([x["file"] for x in chunks] ==
                [f"series{number}/models-xz.joblib.part{i:02d}" for i in range(len(chunks))],
                "archive part order or count drift")
        contents = []
        for part in chunks:
            path = root / SOURCE / part["file"]
            data = path.read_bytes()
            require(len(data) == part["bytes"] and digest(data) == part["sha256"],
                    "historical part corrupt: " + part["file"])
            require(file_records.get(part["file"], {}).get("stored_sha256", part["sha256"]) ==
                    part["sha256"], "inconsistent checksum inventory")
            contents.append(data)
        combined = b"".join(contents)
        require(len(combined) == info["compact_bytes"] and
                digest(combined) == info["compact_sha256"],
                "rejoined historical model archive invalid")
        name = family + "_models_xz.joblib"
        (target / name).write_bytes(combined)
        model_proofs.append({"family": family, "parts": len(chunks),
                             "restored_file": name, "restored_sha256": digest(combined),
                             "bytes": len(combined),
                             "historical_model_state_sha": info["joblib_state_hash"],
                             "serialization_differs_from_original": bool(info["serialization_bytes_changed"])})
    for n in (2, 3, 4):
        key = f"series{n}/router-proofs.json"
        record = file_records[key]
        require(digest((root / SOURCE / key).read_bytes()) == record["stored_sha256"],
                f"S{n} historical router-proof SHA drift")
    require((root / "scripts/train_v273_extended_dynamic_router.py").exists(),
            "historical S35 code lost")
    # There is NO proof of S35 original held-fold model inference on new GOAT.
    # Also S4_repair here is not S4_control111 from the separate S5 chain.
    results = {
        "status": "S172_THREE_ORIGINAL_MODEL_ARCHIVES_RECOVERED_SIX_UNSEEN_PRODUCERS_STILL_MISSING",
        "catalog_sha256": digest((root / "analysis/v273_s45_original_champion_catalog.json").read_bytes()),
        "historical_model_archives_restored": 3,
        "model_proofs": model_proofs,
        "source_logic_files_present": sorted(SCRIPTS),
        "historical_router_proofs_verified": ["S2", "S3", "S4"],
        "not_same_as_S4_control111": True,
        "historical_repair_coherent_original_execution": False,
        "S35_original_new_audio_router_execution": False,
        "all_16_S45_new_audio_sources_available": False,
        "six_missing_unseen_sources": list(REQUIRED),
        "s45_new_audio_sources_verified_after": 10,
        "new_wav_predictions_produced": False,
        "true_K_or_MIDI_opened": False,
        "independent_ExactK_measured": False,
        "S80_promoted": False,
        "next_required": "construct real 255-group S1-S4 input proposals and audio feature lineage from S5/S18, recover coherent repair and S35 original held-fold route with model/policy equivalence, then assess 16/16",
    }
    (output / "report.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: results[k] for k in (
        "status", "historical_model_archives_restored", "historical_router_proofs_verified",
        "s45_new_audio_sources_verified_after", "six_missing_unseen_sources",
        "new_wav_predictions_produced", "S80_promoted")}, indent=2))
    return results

def selftest():
    require(len(REQUIRED) == 6 and len(set(REQUIRED)) == 6, "six source gate")
    require(REQUIRED["S4_repair"].startswith("regression_loops_s4__"), "S4 identity collision")
    require(not any("H9" in x for x in REQUIRED), "prohibited family")
    print("S172 PASS original six distinct source keys; S4 repair != S4 control111")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--out", type=Path)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        selftest()
    else:
        require(args.out is not None, "--out required")
        run(args.root, args.out)
