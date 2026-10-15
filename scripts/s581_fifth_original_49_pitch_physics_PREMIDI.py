"""S581: independently measure 49 raw audio pitch features at ALL frozen V88
piano onset candidates including K0, regardless of Basic Pitch grouping.

Unlike S515, do not restrict to S498 canonical BP births, which dropped
102 original MAESTRO positive attacks. This script only reads audio WAV,
S579's immutable 575 V88 prelabel event candidates and S539's already
frozen independent Basic Pitch notes. No MIDI labels, fit or thresholds.
At every V88 time, produce independent 49x11 real harmonic/frame/onset
features for use by a future event-conditioned pitch expert.

This is retrospectively exposed MAESTRO audio, not an independent F1
result, live benchmark, new S80 decision or trained transcription model.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.s511_new_raw_harmonic_pitch_onset_generators import (
    fetch_track, decode_external_wav, SR)
from scripts.s515_freeze_real_WAV_49_pitch_frame_onset_harmonic_experts import features


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def filesha(p):
    return sha(Path(p).read_bytes())


def source_events(s579, count):
    with np.load(s579, allow_pickle=False) as npz:
        ids = npz["row_id"].astype(str)
        rec = npz["member"].astype(str)
        times = npz["start_samples"].astype(np.int64) / SR
        kval = npz["original_V88_K"].astype(int)
    need(count>0 and len(ids) == count and len(set(ids)) == count
         and len(set(rec)) == 5 and np.all((times >= 0) & (times < 12))
         and all(0 <= k <= 6 for k in kval),
         "S581 original V88 SHA-frozen K0-K6 event times and IDs changed")
    return ids, rec, times, kval


def acoustic_inputs(wave, group, notes):
    """Original S515 feature ABI, but no BP-event existence requirement."""
    onset_rows = []
    for entry in group:
        t = float(entry["time_s"])
        births = [int(n["pitch_MIDI"]) for n in notes
                  if (t - .0002 <= float(n["predicted_onset_s"]) <= t + .0202)]
        active_time = t + .030
        active = [int(n["pitch_MIDI"]) for n in notes
                  if (float(n["predicted_onset_s"]) <= active_time
                      < float(n["predicted_offset_s"]))]
        onset_rows.append(dict(event_time_seconds=t,
                               bp_note_pitches=sorted(set(births)),
                               bp_active_pitches_at_birth_plus30=sorted(set(active))))
    X = features(wave, onset_rows, notes)
    need(len(X) == len(group),
         "S581 all frozen V88 event centers must produce raw pitch observations")
    return X, onset_rows


def run(args):
    need(not args.out.exists(), "S581 prelabel feature bank already exists")
    m = json.loads(args.manifest.read_text())
    v = json.loads(args.v88_report.read_text())
    bp = json.loads(args.bp_report.read_text())
    need(m["status"] == "S577_PREREGISTERED_FIVE_MORE_TRAIN_SPLIT_COMPOSERS_AFTER_S544_NO_WAV_MIDI"
         and m["five_next_distinct_composers"] == 5 and m["source_MIDI_files_not_read"]
         and len(m["chosen_metadata_only"]) == 5 and m["source_MIDI_files_not_read"]
         and v["status"] ==
         "S579_MAESTRO_V3_REAL_AUDIO_ORIGINAL_PRETRAINED_V88_PLUS_S466_PREMIDI"
         and v["prelabel_predictions_frozen"]
         and v["output_sha256"] == filesha(args.v88)
         and v["preregistered_S577_SHA"] == filesha(args.manifest),
         "S581 original prelabel real piano V88 WAV identity not valid")
    need(bp["status"] == "S580_PRETRAINED_BASIC_PITCH_NOTE_EVENTS_ON_THIRD_SOURCE_MAESTRO_PREMIDI"
         and bp["original_MAESTRO_MIDI_ANNOTATIONS_OPENED"] is False
         and bp["frozen_notes_SHA256"] == filesha(args.bp),
         "S581 original frozen independent Basic Pitch MIDI40..88 notes changed")
    ids, rec, times, K = source_events(args.v88, int(v["original_V88_event_count"]))
    groups = collections.defaultdict(list)
    for i in range(len(ids)):
        groups[rec[i]].append(dict(v88_id=str(ids[i]), time_s=float(times[i]),
                                   original_frozen_V88_K=int(K[i])))
    notes = collections.defaultdict(list)
    for line in args.bp.read_text().splitlines():
        x = json.loads(line)
        notes[x["WAV_filename_SHA256"]].append(x)
    proof = {x["WAV_filename_SHA256"]: x for x in
             v["exact_original_audio_source_fingerprints"]}
    chosen = {x["filename_SHA256"]: x for x in m["chosen_metadata_only"]}
    need(len(groups)==5 and set(groups) == set(notes) == set(proof) == set(chosen),
         "S581 original three piano source audio/identities do not match")
    output = []
    per = []
    for ident in sorted(groups):
        item = chosen[ident]
        data, address = fetch_track(m["source_audio_mirror"], item["audio_filename"])
        need(sha(data) == proof[ident]["original_WAV_SHA256"],
             "S581 original full MAESTRO WAV differs from S579 pre-MIDI")
        wave, fmt = decode_external_wav(data)
        offset = int(float(item["audio_excerpt_start_seconds"]) * SR)
        seconds = int(float(item["audio_excerpt_duration_seconds"]) * SR)
        need(seconds == 12*SR and len(wave) >= offset+seconds,
             "S581 original 12 s piano excerpt timing changed")
        held = wave[offset:offset+seconds]
        candidates = groups[ident]
        feats, flags = acoustic_inputs(held, candidates, notes[ident])
        need(len(feats) == len(flags) == len(candidates),
             "S581 missing candidate evidence at independent V88 centers")
        for source, x, bpheads in zip(candidates, feats, flags):
            arr = np.asarray(x, np.float32)
            need(arr.shape == (49, 11) and np.isfinite(arr).all(),
                 "S581 per-pitch original S515 feature ABI must be 49 by 11 finite")
            output.append(dict(
                V88_original_event_id=source["v88_id"],
                source_recording_sha=ident,
                original_WAV_sha256=proof[ident]["original_WAV_SHA256"],
                original_V88_candidate_start_s=source["time_s"],
                original_V88_K_soft_prior_ONLY=source["original_frozen_V88_K"],
                old_basicpitch_birth_pitches=bpheads["bp_note_pitches"],
                old_basicpitch_active_pitches=bpheads["bp_active_pitches_at_birth_plus30"],
                per_midi_features_49x11=arr.round(6).tolist(),
                original_MIDI_range=[40,88],
                no_original_MAESTRO_MIDI_labels_read=True,
                V88_K0_cannot_delete_a_note_onset_candidate=True))
        per.append(dict(source_recording_sha=ident,
            frozen_V88_event_centers=len(candidates),
            original_V88_K0_events=sum(z["original_frozen_V88_K"] == 0 for z in candidates),
            original_WAV_byte_SHA_verified=True,
            raw_acoustic_49x11_features_produced=len(feats)))
        print("S581 PRELABEL 49 PITCH AUDIO AT EVERY K0-K6 V88 ONSET",
              json.dumps(per[-1]), flush=True)
    need(len(output) == v["original_V88_event_count"] and len({z["V88_original_event_id"] for z in output}) == len(output),
         "S581 all original 575 piano V88 events must be measured, including K0")
    args.out.mkdir(parents=True)
    file = args.out / "S581_ALL_V88_K0_K6_ORIGINAL_WAV_49_PITCH_FEATURES_PREMIDI.jsonl"
    file.write_text("".join(json.dumps(x) + "\n" for x in output))
    report = dict(
        status="S581_PRELABEL_FIVE_NEW_PIANO_ALL_V88_RAW_49_PITCH_FEATURES",
        genuine_training_input_features_not_model_predictions=True,
        real_audio_piano_five_source_WAV_SHAs_verified=True,
        original_V88_onset_candidates=len(output),
        original_V88_K0_candidates_included=int(sum(K == 0)),
        all_49_pitch_features_at_every_piano_V88_event=True,
        original_S515_11_acoustic_features_ABI_reused=True,
        no_original_true_MIDI_annotation_read=True,
        no_MIDI_based_candidate_or_threshold_selection=True,
        not_full_stream_note_on_F1_or_a_trained_detector=True,
        preregistered_5_new_composers_MIDI_STILL_SEALED=True,
        original_S80_Live_and_BP_event_decoder_unchanged=True,
        no_promotion=True,
        per_recording=per,
        frozen_feature_ledger_SHA256=filesha(file),
        frozen_source_SHA256=dict(S577=filesha(args.manifest),
                                  S579=filesha(args.v88),
                                  S539=filesha(args.bp)))
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def selftest():
    # Test K0 retains an independent note candidate with 49 real
    # acoustic descriptors even when Basic Pitch supplies zero notes.
    duration = 2*SR
    t = np.arange(duration, dtype=np.float32)/SR
    wave = np.zeros(duration,np.float32)
    wave[int(.4*SR):] = (.25*np.sin(2*np.pi*164.8*t[:duration-int(.4*SR)]))
    out, flags = acoustic_inputs(wave,
        [dict(v88_id="p:1", time_s=.4, original_frozen_V88_K=0)], [])
    need(len(out) == 1 and np.asarray(out[0]).shape == (49,11)
         and np.isfinite(out[0]).all() and not flags[0]["bp_note_pitches"],
         "S581 K0 V88 original physical onset must not require BP note event")
    print("S581 SELFTEST PASS acoustic evidence emitted at K0 without Basic Pitch notes")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    for x in ("manifest","v88","v88-report","bp","bp-report","out"):
        p.add_argument("--" + x, type=Path)
    a = p.parse_args()
    if a.self_test:
        selftest()
    else:
        for x in ("manifest","v88","v88_report","bp","bp_report","out"):
            need(getattr(a,x) is not None, "S581 missing " + x)
        run(a)
