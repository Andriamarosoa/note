# S634 — 3 real-audio windows for pitch rescue (x1/2, x1, x2)

## Verified predecessors

- `Andriamarosoa/midi`, [codex/dual-stream-bass](https://github.com/Andriamarosoa/midi/tree/codex/dual-stream-bass):
  `src/polyphonic/model.py` uses `AveragePooling1D(pool_size=2,strides=2)`
  on a separately trained bass branch, with the original branch unchanged;
  `configs/polyphonic_dual_stream_bass.yaml` requests a 8192-sample model
  input with normal 4096-sample history. This is **not** an interchangeable
  NOTE inference checkpoint.
- `Andriamarosoa/note`, `codex/v273-high-pitch-rescue`: experimental
  octave-up pitch-compressed view (factor 2).
- `Andriamarosoa/note`, `codex/v273-high-pitch-rescue-v2`: the later
  paired history compression deliberately preserved original frequency
  and tested K-logit residuals, distinct from literal pitch transposition.
- S632/S633 `register_rescue` changes decision thresholds and is NOT the
  requested octave-transposition mechanism; it remains an independent
  opt-in ablation.

## S634 precise meaning of "±2"

We implement multiplicative **frequency** `×2` and `÷2`, not ±2
semitones nor ±2 octaves. These are respectively +12 and -12 MIDI
semitones. The number 2 refers to the waveform time factor.

| View | Input PCM at decision t | Virtual model window | Restitution original MIDI |
| --- | --- | --- | --- |
| Native | 8192 past PCM samples | 8192 at 44100 Hz | unchanged |
| Bass rescue x2 | 16384 past PCM samples | pair-average into 8192 | predicted note -12 |
| High rescue /2 | 4096 past PCM samples | repeat every source sample to 8192 | predicted note +12 |

At each **physically received** 256-sample hop, evaluate three instances
of the S613 causal 126-feature acoustic extractor. All three pitch
views are batched through the **same frozen native NOTE hybrid MLP+ONNX
weights**. Only native K0..6 is retained; no independent rescue K
is granted authority to alter the polyphony count.

No positive/future audio lookahead: the compressed bass view merely
reads more past, up to 16384 samples = 371.5 ms. Its **extra history**
can smear brief attacks and has yet to demonstrate improved accuracy.

The x2 pair-average transform has limited antialias filtering and the
/2 repeat transform is a zero-order hold: both change instrument
timbres. This is **not** a calibrated pitch-shift neural model.
Unobservable MIDI bins beyond 40..88 are zeroed.

## Run WAV-only diagnostic

Do not enable these 3 branches in the microphone hot path before
profiling CPU. Earlier batch1 NOTE CLI hit `input_queue_full`;
three simultaneous acoustic feature extractors + one batched neural
call can cause *another* overflow on the same hardware.

```powershell
git fetch origin
git switch codex/s634-three-view-factor2-rescue-audit
python -m scripts.s634_three_view_wav --wav ".\test.wav" --out ".\three_views.json" --seconds 8 --confidence 0.6
```

The JSON reports candidate **attacks**, remapped pitches, original
native K, and offline CPU/audio ratio. These are **not validated
transcriptions** and should not be described as F1 or precision.

All changes are confined to:
- `src/causal_note/three_view_pitch.py`
- `scripts/s634_three_view_wav.py`
- `test/test_s634_three_view_pitch.py`

The original `scripts.live_strict_notes` (including optional S633
baseline switch), S614 frozen model and S80 are unchanged.

## Evidence required before promotion

1. Run blinded score on previously unexposed compositions, comparing
   native only vs x2 bass and /2 high (true TP/FP/FN by register,
   overlap, Exact-K and instrument source).
2. Analyze octave alias/harmonic false positives and pitch-class
   boundaries. Deduplicate BEFORE emitting MIDI, not after a second
   attack.
3. Verify steady-state CPU/audio ratio and PortAudio queue on actual
   Windows hardware, with batch4 + native control. If the triple view
   overruns, reduce computation or train a single multi-view model.
4. Maintain original note-on sample clock and no future-frame access.
5. Only after objective gain without regression consider opt-in LIVE
   shadow; no automatic guard or checkpoint promotion.
