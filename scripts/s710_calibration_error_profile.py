"""Profile trusted S710 CAL probabilities; these are not emitted-stream scores.

Run only on the checksum-verified S710_CAL_DIAGNOSIS_AND_PROBABILITIES artifact.
This command reads CAL labels for diagnosis, never HELD, DEV, or TEST labels.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np


def profile(cache):
    matrices = {family: {texture: np.zeros((7, 7), np.int64)
                        for texture in ("comp", "solo")}
                for family in ("706", "709")}
    structure = dict(sources=0, frames=0, active_frames=0,
                     clipped_count_differs_from_unique_pitches=0,
                     notes=0, notes_shorter_than_35ms=0,
                     same_pitch_onsets_less_than_65ms_apart=0)
    manifest = json.loads((cache / "manifest.json").read_text())
    assert manifest["TEST_opened"] is False
    diagnosed = json.loads((cache / "diagnosis_per_source.json").read_text())
    diagnosed = {family: {row["name"]: row for row in rows}
                 for family, rows in diagnosed.items()}
    gate_keys = ("neither_onset_nor_presence_reaches_threshold", "onset_only_below_threshold",
                 "presence_only_below_threshold", "peaks_do_not_overlap",
                 "joint_gate_available_other_failure")
    gates = {family: {texture: dict.fromkeys(gate_keys, 0) for texture in ("comp", "solo")}
             for family in matrices}
    for path in sorted(cache.glob("cal_*.joblib")):
        item = joblib.load(path)
        assert item["spec"]["composition"] in manifest["cal_compositions"]
        texture = item["spec"]["name"].rsplit("_", 1)[1]
        t = item["decision_ends"] / 44100
        count = np.zeros(len(t), np.int64)
        presence = np.zeros((len(t), 49), bool)
        onsets = {}
        for note in item["truth"]:
            active = (t >= note["time"]) & (t < note["time"] + note["duration"])
            count += active
            presence[:, note["pitch"] - 40] |= active
            onsets.setdefault(note["pitch"], []).append(note["time"])
            structure["notes"] += 1
            structure["notes_shorter_than_35ms"] += int(note["duration"] < .035)
        count = np.minimum(count, 6)
        structure["sources"] += 1
        structure["frames"] += len(t)
        structure["active_frames"] += int((count > 0).sum())
        structure["clipped_count_differs_from_unique_pitches"] += int(
            (count != np.minimum(presence.sum(1), 6)).sum())
        for starts in onsets.values():
            structure["same_pitch_onsets_less_than_65ms_apart"] += int(
                (np.diff(sorted(starts)) < .065).sum())
        for family, (post, probabilities) in item["models"].items():
            assert probabilities.shape == (len(t), 7)
            np.add.at(matrices[family][texture], (count, probabilities.argmax(1)), 1)
            row = diagnosed[family][item["spec"]["name"]]
            missed = set(range(len(item["truth"]))) - set(row["emitted"]["matched_truth_ids"])
            assert len(missed) == row["missed_notes"]
            for j in missed:
                note = item["truth"][j]
                within = np.abs(t - note["time"]) <= .080
                pitch = note["pitch"] - 40
                onset = within & (post[:, pitch, 0] >= .6)
                present = within & (post[:, pitch, 1] >= .5)
                if not onset.any() and not present.any():
                    key = gate_keys[0]
                elif not onset.any():
                    key = gate_keys[1]
                elif not present.any():
                    key = gate_keys[2]
                elif not (onset & present).any():
                    key = gate_keys[3]
                else:
                    key = gate_keys[4]
                gates[family][texture][key] += 1
    assert structure["sources"] == 32
    result = dict(note="CAL shadow probability and gate diagnostics only; not emitted-stream performance.",
                  TEST_opened=False, checkpoint_sha256=manifest["checkpoint_sha256"],
                  structure=structure, families={},
                  missed_gate_thresholds=dict(onset=.6, presence=.5, tolerance_seconds=.080),
                  missed_gate_profile=gates)
    for family, textures in matrices.items():
        gates[family]["all"] = {key: sum(gates[family][texture][key] for texture in ("comp", "solo"))
                                for key in gate_keys}
        assert sum(gates[family]["all"].values()) == sum(row["missed_notes"] for row in diagnosed[family].values())
        assert sum(gates[family]["all"][key] for key in gate_keys[:4]) == sum(
            row["missed_causes"]["head_gate_closed"] for row in diagnosed[family].values())
        result["families"][family] = {}
        for texture, matrix in dict(textures, all=sum(textures.values())).items():
            rows = []
            for k in range(7):
                n = int(matrix[k].sum())
                rows.append(dict(reference_k=k, frames=n, correct=int(matrix[k, k]),
                                 accuracy=float(matrix[k, k] / n) if n else None,
                                 under=int(matrix[k, :k].sum()),
                                 over=int(matrix[k, k + 1:].sum())))
            result["families"][family][texture] = dict(
                confusion_true_rows_predicted_columns=matrix.tolist(), by_reference_k=rows)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cache", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = profile(args.cache)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"structure": result["structure"], "output": str(args.out)}))
