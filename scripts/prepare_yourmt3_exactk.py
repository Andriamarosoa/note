"""Replay the untouched freeze_local_combo and verify native group assignments."""
from __future__ import annotations
import argparse
import json
import wave
import zipfile
from pathlib import Path
import numpy as np

from causal_note.guitarset import index_guitarset, load_boundary_slots
from scripts.yourmt3_exactk_common import (FOLDS, SAMPLE_RATE, cluster_ends, assign_onsets,
                                          digest, metrics, require, write_json)


def main(a):
    import tensorflow as tf
    from scripts.v273_window_experiment import load_bundle
    from scripts.audit_v273_candidate_hidden1_multifold_worker import SEED, predict
    from scripts.train_v273_group_gate_ab import build_model
    from scripts.summarize_v273_reference_hidden1_pairwise_confirmation import discover_outer

    require(tf.__version__ == "2.15.1", "baseline TensorFlow version drift")
    tf.config.experimental.enable_op_determinism()
    a.output.mkdir(parents=True, exist_ok=False)
    cache, _, bundle_report = load_bundle(a.bundle, a.config)
    cfg = json.loads(a.config.read_text())
    names = np.asarray(cache["members"]).astype(str)
    folds = np.asarray([cfg["member_folds"][s] for s in names])
    tracks = {t.annotation_member: t for t in index_guitarset(a.dataset)}
    all_y, all_base = [], []
    manifest = {"status": "verified", "folds": list(FOLDS), "fold_3_evaluated": False,
                "player_05_evaluated": False, "config_sha256": digest(a.config),
                "bundle_manifest_sha256": digest(a.bundle / "bundle.json"), "by_fold": {}}
    for fold in FOLDS:
        ids = np.flatnonzero(folds == fold)
        members = names[ids]
        require(all(s[:2] in {"00", "01", "02", "03", "04"} for s in members), "player leak")
        starts = np.asarray(cache["cluster_start_samples"][ids], np.int64)
        ends = cluster_ends(cache["sequence"][ids], cache["mask"][ids], starts, cache["stats"][ids])
        exact = np.asarray(cache["exact"][ids], np.int32)
        y = np.minimum(exact, 6)
        track_metadata = {}
        for member in sorted(set(members)):
            local = np.flatnonzero(members == member)
            t = tracks[member]
            with zipfile.ZipFile(t.audio_zip) as z, z.open(t.audio_member) as f, wave.open(f) as wav:
                require(wav.getframerate() == SAMPLE_RATE, "audio sample rate drift")
                frame_count = wav.getnframes()
            refs = sorted(n.onset_sample for slot in load_boundary_slots(t.annotation_zip, member)
                          for n in slot if n.onset_sample < frame_count)
            recounted, assignments = assign_onsets(refs, starts[local], ends[local])
            np.testing.assert_array_equal(recounted, exact[local], err_msg=f"native target mismatch: {member}")
            track_metadata[member] = {"audio_member": t.audio_member, "frames": frame_count,
                                      "reference_attacks": len(refs),
                                      "unassigned_reference_attacks": int(np.sum(assignments < 0))}
        _, saved, _, weights = discover_outer(a.fold_root, fold)
        model = build_model("learned_gate", SEED)
        model.load_weights(weights)
        _, base = predict(model, cache, ids)
        measured = metrics(y, base)
        expected = saved["reference"]["freeze_local_combo"]
        require(abs(measured["exact"] - expected["exact"]) < 1e-12, "freeze reference replay drift")
        for k in range(7):
            require(abs(measured["by_k"][str(k)]["exact"] - expected["by_k"][str(k)]["exact"]) < 1e-12,
                    f"freeze per-K replay drift: fold {fold}, K{k}")
        np.savez_compressed(a.output / f"fold-{fold}.npz", global_index=ids, member=members,
                            starts=starts, ends=ends, exact=exact, k=y, baseline=base)
        write_json(a.output / f"tracks-{fold}.json", track_metadata)
        manifest["by_fold"][str(fold)] = {"rows": len(ids), "tracks": len(track_metadata),
                                        "weights_sha256": digest(weights),
                                        "cohort_sha256": digest(a.output / f"fold-{fold}.npz"),
                                        "baseline": measured}
        all_y.extend(y.tolist()); all_base.extend(base.tolist())
        print(json.dumps({"fold": fold, "rows": len(ids), "reference_verified": True}), flush=True)
        del model
        tf.keras.backend.clear_session()
    aggregate = metrics(all_y, all_base)
    require(aggregate["rows"] == 59309 and aggregate["correct"] == 48454, "global baseline drift")
    require(aggregate["poly"]["rows"] == 7385 and aggregate["poly"]["correct"] == 2530, "poly baseline drift")
    manifest["baseline"] = aggregate
    write_json(a.output / "manifest.json", manifest)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "fold-root", "config", "dataset", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    main(p.parse_args())
