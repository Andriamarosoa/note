# S620 — Two *simultaneous* native NOTE decision cadences (128 + 256)

Branch `codex/note-live-s620-dual-hop`; **no change to official S613 baseline,
S80 Exact-K references, live audio output engine or frozen model weights**.

## Goal

Preserve every original validated S613 full hybrid (neural+ONNX) 256-hop
pitch decision while adding a separate strictly-causal hop128 path for
**independent early guitar attacks**. Use both histories, not `hop=block`.

At SR 44,100Hz:
- hop128 makes decisions every **2.90 ms** on the sound *already received*.
- hop256 baseline makes decisions every **5.80 ms**.
- Both retain the original FFT 2048/4096/8192 and bounded PAST spectral
  history. They do not automatically wait an extra FFT window to predict.
- `--block-size` is independent; the microphone still delivers all samples
  only at end-of-callback. With block2048 the ~46.44ms packet dominates.
- Running both heads costs substantially MORE computation than S613 alone,
  possibly *increasing* end-to-end latency if the worker misses its budget.

## Implementation

`src/causal_note/dual_hop.py`:
- two causal frontends (128 and 256), same PCM with independent timelines;
- **hop256 uses the original pretrained S613 full hybrid and original
  `CausalSelection` unchanged**;
- fast hop128 **reuses S613's weights trained only at hop256**. This is
  deliberately marked experimental and does not imply its predictions have
  been trained/validated at 128;
- fast candidate must have true independent positive spectrum novelty at
  that pitch, not only a harmonic present with a nearby lower pitch;
- fast candidates pass per-pitch NOTE `CausalSelection` with its original
  sustain, release, refractory and K constraints; no future acoustic frames;
- `DualSourceEventMerger` tracks notes active in BOTH branches and emits
  one physical MIDI on/off, with original 256 heads prioritized on
  coincident timestamps. This prevents duplicate sound but can change
  lengths/timing; investigate on real audio.
- Normal mode `shadow`: computes dual fast candidates but emits only the
  original hop256 events, bitwise-event identical on equal PCM.
- Experimental `guard`: fuses the new candidates. No performance claim.

## Windows iRig commands

```powershell
git fetch origin
git switch codex/note-live-s620-dual-hop
git pull --ff-only
python -m unittest -v test.test_s620_dual_hop

# Safe two-branch measurement: piano hears EXACT same S613 notes;
# diagnostic for fast candidate count and extra CPU at shutdown.
python -m scripts.live_strict_notes --mic --device "MME:External Microphone" --dual-hop-mode shadow --block-size 256 --queue-blocks auto --soundfont "PATH_TO_PIANO.sf2"

# Experimental union of the fast branch; only after checking shadow CPU.
python -m scripts.live_strict_notes --mic --device "MME:External Microphone" --dual-hop-mode guard --dual-hop-fast-onset 0.50 --dual-hop-novelty-ratio 0.55 --block-size 256 --queue-blocks auto --soundfont "PATH_TO_PIANO.sf2"

# Reproducible causal WAV replay, no piano time distortion.
python -m scripts.live_strict_notes --wav test_04_E2_puis_E3.wav --dual-hop-mode shadow --out s620_shadow_test04.jsonl
python -m scripts.live_strict_notes --wav test_04_E2_puis_E3.wav --dual-hop-mode guard --out s620_guard_test04.jsonl
```

The dual prototype cannot be combined with `--harmonic-mode guard` and
requires `--hop-size 256` for the reference. No forced coupling to the
microphone callback block size. The `guard` mode adds computation and
**does not provide validated F1 or Exact-K**. For development only.

## Evaluation

`scripts/s620_dualhop_eval.py`:
- loads the *same* 17 pinned original GuitarSet+URMP DEV compositions;
- caches native right-aligned 126 feature vectors at both 128 and 256
  without reading held-out TEST audio or labels;
- runs real frozen S613 ONNX+neural heads at BOTH cadences, then
  simulates strict chronological source-first MIDI event fusion;
- verifies original S613 TP=1241, FP=734, FN=838 at ±80ms;
- explores 24 predeclared fast pitch novelty gates, counting recovered
  previously FN, lost previous TP, additional false notes and grouped
  onset Exact-K K1..K6 with full pitch-set accounting; reports GuitarSet
  and URMP separately.
- numbers found by tuning on DEV are NOT independent generalization data.
  This grouped-onset K is NOT the historical S80 Exact-K protocol.
- source-trained 128-hop model is still required before any justified
  LIVE promotion.

GitHub workflow: [S620 dual-hop causal fusion validation](https://github.com/Andriamarosoa/note/actions/workflows/note-s620-dual-hop.yml).
Do not claim its outcome until the individual run has completed.

## Promotion gate

S620 must exceed the heldout native NOTE S613 **69.03% NoteOn ±80ms**
on an independent future-free evaluation, and must not regress K2, K3,
K4 true polyphonic notes, double-pick attacks, or missing old TPs.
For an 80% target, *also* measure p95/p99 compute, microphone callback
queue and actual iRig-to-FluidSynth audible event delay on Windows.
Do not simply compare to offline S601/S605 Basic Pitch 80% (noncausal).
