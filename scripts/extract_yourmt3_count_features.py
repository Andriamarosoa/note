"""Extract frozen YourMT3+ representations for an Exact-K linear-probe audit.

This does not decode notes and does not train YourMT3+. It reuses the verified
cohort and pinned checkpoint, extracts two representation levels for every
native Exact-K row, and leaves all label use to the cross-fold summarizer.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from scripts.run_yourmt3_exactk import load_model
from scripts.yourmt3_exactk_common import FOLDS, SAMPLE_RATE, digest, require, write_json


def _local_summary(tensor, frame: int, time_size: int, radius: int = 2) -> np.ndarray:
    """Summarize one model representation around one native ~93 ms group."""
    x = tensor.detach().cpu().numpy()
    axes = [i for i, size in enumerate(x.shape[:-1]) if size == time_size]
    require(len(axes) == 1, f"cannot identify time axis in shape {x.shape} for T={time_size}")
    axis = axes[0]
    lo, hi = max(0, frame - radius), min(time_size, frame + radius + 1)
    index = np.arange(lo, hi, dtype=np.int64)
    x = np.take(x, index, axis=axis).mean(axis=axis)

    # Preserve the learned vector direction via the mean vector, plus
    # channel/latent structure with simple per-group statistics.
    if x.ndim == 1:
        groups = x[None, :]
    else:
        groups = x.reshape(-1, x.shape[-1])
    mean_vector = groups.mean(axis=0)
    stats = np.stack(
        [
            groups.mean(axis=1),
            groups.std(axis=1),
            np.sqrt(np.mean(np.square(groups), axis=1)),
            groups.max(axis=1),
            groups.min(axis=1),
        ],
        axis=1,
    ).reshape(-1)
    return np.concatenate([mean_vector, stats]).astype(np.float32, copy=False)


def main(a):
    import soundfile as sf
    import torch
    import torchaudio
    from utils.audio import slice_padded_array

    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    require(a.fold in FOLDS, "excluded fold")
    require(not a.output.exists(), "refusing to overwrite output")
    a.output.mkdir(parents=True)

    manifest = json.loads((a.cohort / "manifest.json").read_text())
    require(manifest["status"] == "verified", "unverified cohort")
    cohort_file = a.cohort / f"fold-{a.fold}.npz"
    require(
        digest(cohort_file) == manifest["by_fold"][str(a.fold)]["cohort_sha256"],
        "cohort checksum drift",
    )
    with np.load(cohort_file, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    tracks = json.loads((a.cohort / f"tracks-{a.fold}.json").read_text())

    model = load_model(a.source)
    model_sr = int(model.audio_cfg["sample_rate"])
    input_frames = int(model.audio_cfg["input_frames"])
    enc_features = [None] * len(data["k"])
    proj_features = [None] * len(data["k"])
    shape_record = None

    with zipfile.ZipFile(a.dataset / "audio_mono-pickup_mix.zip") as archive:
        for track_no, member in enumerate(sorted(tracks), 1):
            require(member[:2] in {"00", "01", "02", "03", "04"}, "excluded player")
            rows = np.flatnonzero(data["member"] == member)
            audio_bytes = archive.read(tracks[member]["audio_member"])
            samples, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            require(sr == SAMPLE_RATE and samples.ndim == 1, "audio format drift")

            audio = torch.from_numpy(samples).unsqueeze(0)
            audio = torchaudio.functional.resample(audio, SAMPLE_RATE, model_sr)
            segments = slice_padded_array(audio, input_frames, input_frames)
            segments = torch.from_numpy(segments.astype("float32")).unsqueeze(1)

            centers = ((data["starts"][rows].astype(np.float64) + data["ends"][rows]) / 2.0)
            centers_model = np.rint(centers * model_sr / SAMPLE_RATE).astype(np.int64)
            seg_id = centers_model // input_frames
            local_sample = centers_model % input_frames
            require(np.all(seg_id < len(segments)), f"segment mapping overflow: {member}")

            for begin in range(0, len(segments), a.batch_size):
                end = min(len(segments), begin + a.batch_size)
                wanted = np.flatnonzero((seg_id >= begin) & (seg_id < end))
                if not len(wanted):
                    continue
                batch = segments[begin:end]
                with torch.inference_mode():
                    spec = model.spectrogram(batch)
                    pre = model.pre_encoder(spec)
                    enc = model.encoder(inputs_embeds=pre)["last_hidden_state"]
                    proj = model.pre_decoder(enc)

                t = int(spec.shape[1])
                if shape_record is None:
                    shape_record = {
                        "spectrogram": list(spec.shape[1:]),
                        "pre_encoder": list(pre.shape[1:]),
                        "encoder": list(enc.shape[1:]),
                        "pre_decoder": list(proj.shape[1:]),
                    }

                for local_idx in wanted:
                    cohort_idx = int(rows[local_idx])
                    bi = int(seg_id[local_idx] - begin)
                    frame = int(round(float(local_sample[local_idx]) * max(t - 1, 0) / max(input_frames - 1, 1)))
                    frame = min(max(frame, 0), t - 1)
                    enc_features[cohort_idx] = _local_summary(enc[bi], frame, t, a.radius)
                    proj_features[cohort_idx] = _local_summary(proj[bi], frame, t, a.radius)

            print(
                json.dumps(
                    {
                        "fold": a.fold,
                        "track": track_no,
                        "tracks": len(tracks),
                        "member": member,
                        "rows": int(len(rows)),
                    }
                ),
                flush=True,
            )

    require(all(x is not None for x in enc_features), "missing encoder features")
    require(all(x is not None for x in proj_features), "missing pre-decoder features")
    enc_matrix = np.stack(enc_features)
    proj_matrix = np.stack(proj_features)
    require(np.all(np.isfinite(enc_matrix)) and np.all(np.isfinite(proj_matrix)), "non-finite features")

    np.savez_compressed(
        a.output / f"features-fold-{a.fold}.npz",
        global_index=data["global_index"],
        k=data["k"],
        baseline=data["baseline"],
        encoder=enc_matrix,
        predecoder=proj_matrix,
    )
    report = {
        "status": "completed",
        "fold": a.fold,
        "rows": int(len(data["k"])),
        "encoder_feature_dim": int(enc_matrix.shape[1]),
        "predecoder_feature_dim": int(proj_matrix.shape[1]),
        "representation_shapes": shape_record,
        "protocol": {
            "checkpoint_sha256": json.loads((a.source / "verified.json").read_text())["checkpoint_sha256"],
            "context_samples": input_frames,
            "model_sample_rate": model_sr,
            "native_sample_rate": SAMPLE_RATE,
            "local_radius_frames": a.radius,
            "decoded_notes_used": False,
            "labels_used_during_extraction": False,
            "fold3_used": False,
            "player05_used": False,
        },
        "feature_sha256": digest(a.output / f"features-fold-{a.fold}.npz"),
    }
    write_json(a.output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "cohort", "dataset", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--fold", type=int, choices=FOLDS, required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--radius", type=int, default=2)
    main(p.parse_args())
