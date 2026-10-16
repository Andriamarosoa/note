# NOTE strict LIVE — harmonic expert (EXPERIMENTAL, not promoted)

Date: 16 October 2026. Base: `codex/note-live-audio-monitor-wasapi`.
Working branch: `codex/note-live-harmonic-shadow-expert`.

## Purpose

Distinguish real upper-note onsets overlapping with partials from a lower note,
without treating shared FFT-bin energy as an independent note and without
automatically deleting a real upper-note octave/unison-in-harmonics.

Examples: E2 (MIDI40) alone vs E3 (MIDI52) alone vs E2+E3 simultaneous vs
E2 held then E3 attacked. All four user WAVs are valuable diagnostic examples,
but are NOT a statistically sufficient training/validation dataset.

## Architecture: keep NOTE S613 unchanged by default

Original S613 path (unchanged when option omitted):

`PCM -> causal FFT 2048/4096/8192 -> native K-active head -> native onset/frame/release heads -> CausalSelection -> NoteOn/Off`

New optional path:

`same native causal feature-per-pitch tensor + same native pitch/K heads -> pairwise harmonic genuine-onset expert -> original CausalSelection`

- Seven harmonic parent relations: +12/+19/+24/+28/+31/+34/+36 semitones (partials 2..8).
- Only the existing spectral, onset, sustain, K and earlier-frame-difference features are read. **No new FFT, neural backbone, window, future samples, microphone buffering, MIDI driver, or post-onset waiting.**
- Pairwise specialist input: 17-dimensional vector; trainable logistic checkpoint, separately versioned. No expert checkpoint is shipped/pretrained in this experiment.
- **`off` (default):** same exact S613 code path/results, no specialist instantiated.
- **`shadow`:** specialist identifies potential harmonic-parent pairs and collects aggregate diagnostics; note events EXACTLY unchanged. With trained checkpoint it can score pairs, but still does not suppress anything. This is the initial safety evaluation.
- **`guard`:** explicit opt-in with trained checkpoint only; removes *individual candidate upper onsets* if all applicable parent-vs-upper decisions rate the upper note as non-independent. It cannot create notes, does not change K logits/heads and cannot fix the E2 fundamental already absent from the native onset head.
- A suppressed upper onset is not proof it is an harmonic. **No fixed-frequency or forced E2/E3 suppression is enabled.**

The original `model/live/strict_note/NOTE_strict_live.npz` and S80 S397 checkpoints, selectors, provenance, and locked benchmark files are unchanged. Status: **not promoted**.

## Quick Windows tests (copy into project directory)

Switch to experiment branch explicitly; your former branch is preserved:

```powershell
git fetch origin
git switch codex/note-live-harmonic-shadow-expert
git pull --ff-only

python -m unittest -v test.test_live_harmonic_expert
python -m scripts.benchmark_live_harmonic_expert --wav "test_03_E2_E3_ensemble.wav"

python -m scripts.live_strict_notes --wav "test_03_E2_E3_ensemble.wav" --out "baseline_E2_E3.jsonl"
python -m scripts.live_strict_notes --wav "test_03_E2_E3_ensemble.wav" --harmonic-mode shadow --out "shadow_E2_E3.jsonl"
Compare-Object (Get-Content baseline_E2_E3.jsonl) (Get-Content shadow_E2_E3.jsonl)
```

The last command must print **nothing** (bitwise-identical NOTE events). The
benchmark prints median/p95/p99 full computation per 256-sample block for
`off` versus `shadow`. Verify the p95 stays below the 5.805 ms hop on your
Windows machine, with other programs running. This does not estimate iRig /
FluidSynth hardware latency. If it is too slow, stop the experiment.

To observe in real-time, keeping output identical to baseline:

```powershell
python -m scripts.live_strict_notes --mic --device "MME:External Microphone" --harmonic-mode shadow --soundfont "C:\\SoundFonts\\piano.sf2" --block-size 256
```

Change the example SoundFont to your actual existing SF2 path. For benchmarking
there is no need to load any SoundFont.

## Training instructions (DO NOT train from the four test WAVs alone)

The guarded mode requires authentic supervised guitar/piano `NOTE_ON`
labels. Do not use the S613 held-out 22 tracks for training or thresholds.
Collect at least three distinct *train* source groups and two distinct
*development* groups, preferably many more; split by musicians and songs,
not by frame. Label all real child onsets and parents, including negatives
for E2 alone and genuine E2+E3 simultaneous and sequential.

Create a source manifest with keys `sources` and entries:
```json
{"sources": [
  {"wav": "train1.wav", "split": "train", "group": "player_A",
   "notes": [{"pitch": 40, "time_s": 1.20, "duration_s": 0.8},
             {"pitch": 52, "time_s": 1.20, "duration_s": 0.7}]},
  {"wav": "dev1.wav", "split": "dev", "group": "player_B",
   "notes": [{"pitch": 40, "time_s": 1.20, "duration_s": 0.9}]}
]}
```

Example format only; actual files, timings and multiple disjoint groups are
required for real training.

```powershell
python -m scripts.build_live_harmonic_pairs --manifest "annotations_harmonics.json" --output-dir "data/harmonic_pairs_v1"
python -m scripts.train_live_harmonic_expert --train-pairs "data/harmonic_pairs_v1/train_harmonic_pairs.npz" --dev-pairs "data/harmonic_pairs_v1/dev_harmonic_pairs.npz" --out "model/experimental/harmonic_expert_v1.npz"
```

The trainer selects a threshold on DEV such that real independent upper
attacks retain >=99% recall in pair classification; this **does not guarantee**
zero regressions in NOTE. Use the printed `threshold` for diagnostic
counterfactual, **only after** full replay tests on dev:

```powershell
python -m scripts.live_strict_notes --wav "dev_heldout.wav" --harmonic-mode guard --harmonic-checkpoint "model/experimental/harmonic_expert_v1.npz" --harmonic-threshold 0.05 --out "guard_dev.jsonl"
```

**That 0.05 is merely a placeholder**; replace with the calibrated threshold
printed by the training script. Never turn guard on during actual performance
until validation. No benchmark gain or Exact-K improvement is claimed yet.

## Required validation for promotion

1. Identical events, timestamp and note counts in default and `shadow`, for
   every one of the four WAVs and at least GuitarSet/URMP/IDMT dev clips.
2. Causality test with future-mutated audio; same emitted prefix and zero
   lookahead; no new FFT or delayed confirmation.
3. CPU median/p95/p99 on target iRig Windows PC with default and shadow.
   Audio capture and FluidSynth output measurements separately.
4. Evaluate optional `guard` with a frozen trained checkpoint on **new**
   compositions; measure event NOTE_ON precision/recall/F1 and **strict**
   Exact-K *new attacks*, separately from K-active frames.
5. Stratify K0,K1,K2,K3,K4,K5,K6, true octave E2+E3, true neighboring
   notes, E2 then E3, and other harmonic intervals. Count improvements and
   regressions per subgroup; reject if genuine polyphonic voices are lost.
6. Only after all this, decide whether a novel expert merits a PR/merge.
   Do not equate the historical S80 poly Exact-K 81.98% with S613 69.03%
   held-out NoteOn F1 — different tasks/protocols.

## Current status

Code and safety/benchmark scripts are implemented on a separate branch.
A new genuine-upper-onset checkpoint has **not** been trained, nor has any
new performance score been verified. This is infrastructure for a controlled,
causal ablation, not a new official NOTE model.
