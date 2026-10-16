# S629 — Windows iRig S628 microphone overflow: evidence and CPU fix

## Actual user-provided Windows failure (not hypothetical)

Command:
```powershell
python -m scripts.live_strict_notes --mic --soundfont ".\SalamanderGrandPiano-V3+20200602.sf2" --hop-size 256 --block-size 256 --lookahead-ms 10 --lookahead-mode average
```

Capture selects Realtek MME External Microphone, 17.415ms negotiated
latency, PortAudio callback 256/44100 = **5.805ms**. FluidSynth WDM-KS
headphones latency 3.379ms, 394 audio callbacks, **0 callback render
errors**, 0 notes sent (capture stream terminated within 46.44ms
processed/model time).

Original S628 exact phase at 10ms:
```
"actual_lookahead_samples":441
"reference_frames":8
"lookahead_confirmed_frames":6
"total_received_samples":2048
"p95_worker_callback_ms":19.439635006710887
"max_worker_callback_ms":19.520800095051527
"queue_blocks":13
"queue_nominal_capacity_ms":75.46
BufferError: input_queue_full(maxsize=13, block_size=256, ...)
```

The 19.44ms **worker** p95 is **3.35× the 5.805ms new-audio
period**; captured audio accumulates faster than inference consumes it.
A larger `--queue-blocks` only delays overflow and increases LIVE
latency. Git untracked `.sf2` files are unrelated; they are valid
local soundfonts and are not automatically staged. The SDL3 warning
is not itself evidence of a piano/audio failure.

## Source root cause

`src/causal_note/lookahead.py` S628 exact-phase runs:
1. `reference_front` multi-FFT + full frozen neural/ONNX hybrid
   once per 256-hop;
2. a SECOND `future_front` full multi-FFT + full
   frozen neural/ONNX hybrid on waveform shifted by exact 441
   samples, again once per hop.

Thus 2× the acoustic feature extraction and two ONNX model passes
per ~5.8ms callback, before piano output. On a slower Windows CPU
this can violate the real-time deadline. The previously successful
Linux synthetic benchmark (S628 run 38892790895) showed ~**1.94×**
baseline CPU for dual phase: the observed Windows overflow is
consistent with, but not definitively isolated to, that workload
without a matched original 0ms Windows run.

## S629 bounded native-shared-hop future

New `SharedHopLookaheadLiveStream`:
- A SINGLE existing original `CausalPitchFeatures(hop=256)`
  is run, and the exact original full frozen S613
  neural+ONNX hybrid is inferred **once**, at each original hop.
- Keep the previous K/onset/sustain/release predictions in a short
  chronological ring. Wait for the **NEXT** *physically captured*
  N original hop predictions to be produced. Fuse original attack
  onset/sustain with the future prediction via same S628 declared
  `--lookahead-mode`; original old K and old release persist.
- A real future event never gets emitted early or backdated.
  Original `decision_sample` and later actually available
  `available_sample` are preserved. No extra voice selector.
- EOF incomplete lookahead is marked
  `lookahead_eof_unconfirmed_*` and uses original frozen p.
- Original `--lookahead-ms 0` runs old original S613 exactly
  unchanged, not this experimental class.
- Keep S628 original exact PCM shift available with
  `--lookahead-engine phase` explicitly, to preserve reproducibility.
- NEW engine is `--lookahead-engine shared` and defaults
  to shared when lookahead>0 on S629 branch.

**Quantization, not hidden!**

| Requested future | S628 exact-phase | New S629 shared hop |
|---|---:|---:|
| 5ms | 220 samples = 4.989ms | 256 samples = 5.805ms |
| 10ms | 441 samples = 10.000ms | **512 samples = 11.610ms** |

The rounding is ALWAYS upward to the nearest 256-hop boundary. This
is **not the same** experimental timing as S628, so prior S628
F1 cannot be carried over to this branch without an explicit replay.

## Measured CPU improvement — original GitHub Linux CI

[Verified S629 CPU and tests success, run 38893977284](https://github.com/Andriamarosoa/note/actions/runs/38893977284).
All **11 tests passed**, including original delay-only event parity,
one frozen prediction per hop, no future-data access and small/large
audio callback equivalence. On 1.5s identical synthesized E2:
256-callback test, single BLAS thread:

| Path | Worker walltime / audio | Worker callback p95 | relative original CPU |
|---|---:|---:|---:|
| S613 original 0ms | **23.80%** | 1.470ms | 1.00× |
| S628 phase exact 10ms | **45.96%** | 2.795ms | 1.93× |
| S629 shared native 10ms | **24.46%** | 1.565ms | 1.03× |

512-callback variation also succeeded: original 23.28%, exact phase
45.42%, S629 shared 23.94%. Still **not a Windows iRig-to-piano
roundtrip measurement** and not a guarantee that Windows's CPU
can meet the 5.8ms deadline, even at single-pass.

## Windows runnable steps (experiment, not promotion)

```powershell
git fetch origin
git switch codex/note-s629-shared-hop-lowcpu-lookahead
git pull --ff-only

python -m unittest -v test.test_s629_shared_hop_lookahead

# COMPARE: no intentional lookahead, same SoundFont.
python -m scripts.live_strict_notes --mic --hop-size 256 --block-size 256 --lookahead-ms 0 --soundfont ".\SalamanderGrandPiano-V3+20200602.sf2"

# Test CPU-efficient true future, actual 11.61ms at requested 10ms.
python -m scripts.live_strict_notes --mic --hop-size 256 --block-size 256 --lookahead-ms 10 --lookahead-engine shared --lookahead-mode average --soundfont ".\SalamanderGrandPiano-V3+20200602.sf2"

# To reproduce previous overloaded exact 10.00ms S628 branch:
python -m scripts.live_strict_notes --mic --hop-size 256 --block-size 256 --lookahead-ms 10 --lookahead-engine phase --lookahead-mode average --soundfont ".\SalamanderGrandPiano-V3+20200602.sf2"
```

On Windows the `--hop-size 256` affects model decision cadence; the
`--block-size 256` is **input callback size**, whose duration is
5.805ms. Increasing it to 512 only if source/device needs fewer
callbacks will also add coarse delivery latency; don't mistake this
for increasing the acoustic context of the trained heads.

**If S629 still overflows:** compare original `--lookahead-ms 0`
on the same device and settings; record both actual
`p95_worker_callback_ms` and baseline audio worker utilization.
If original is itself >5.8ms per callback, CPU capture/ONNX/Windows
performance is an independent bottleneck not fixed by reducing
S628's second model. Do not conceal this by enlarging the queue.

## Accuracy: distinct fresh replay required

The original 17 GuitarSet/URMP DEV NoteOn comparator is
TP1241 FP734 FN838 (F1=61.22%). S628 exact-phase 5ms/10ms achieved
essentially no reliable gain. S629 shared reuses physically later
original 256-hop outputs instead of recomputing phase-shifted 441-sample
features; its F1/Exact-K must be evaluated on the same dev tracks.
[17-DEV S629 accuracy workflow](https://github.com/Andriamarosoa/note/actions/workflows/note-s629-original-data-dev-eval.yml).
Do not promote without exact results and independent heldout + Windows
measurement. This branch does not change frozen S613/S80 officially.


## Measured 17-song accuracy result — NOT a promotion (2026-10-16)

[Original 17 source-distinct DEV full ONNX+neural run **38894253766 — SUCCESS**](https://github.com/Andriamarosoa/note/actions/runs/38894253766).
All 10 S629 combinations passed canonical original frozen comparator
TP=1241 FP=734 FN=838, grouped K3=42/183 and K4=3/44.
The 5ms/10ms parameters are rounded to effective 5.805/11.610ms.

| Requested / actual | Mode | TP | FP | FN | F1 | Recovered old FN / lost prior TP | Grouped K3 exact | Grouped K4 exact |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0ms original | original | **1241** | **734** | **838** | **61.2235%** | reference | **42/183** | **3/44** |
| 5 -> 5.805ms | delay-only | 1226 | 749 | 853 | 60.4835% | 2 / 17 | 42 | 3 |
| 5 -> 5.805ms | average | 1210 | 672 | 869 | 61.0957% | 1 / 32 | 44 | 3 |
| 5 -> 5.805ms | confirm | 1192 | 624 | 887 | 61.2067% | 3 / 52 | 41 | 2 |
| 5 -> 5.805ms | max_onset | 1244 | 759 | 835 | 60.9505% | 3 / 0 | 41 | 3 |
| 5 -> 5.805ms | future | 1241 | 735 | 838 | **61.2084%** | 0 / 0 | 42 | 3 |
| 10 -> 11.610ms | delay-only | 1214 | 761 | 865 | 59.8915% | 3 / 30 | 43 | 3 |
| 10 -> 11.610ms | average | 1156 | 561 | 923 | 60.9062% | 2 / 87 | 38 | 2 |
| 10 -> 11.610ms | confirm | 1116 | 510 | 963 | 60.2429% | 3 / 128 | 35 | 2 |
| 10 -> 11.610ms | max_onset | 1250 | 767 | 829 | 61.0352% | 10 / 1 | 41 | 4 |
| 10 -> 11.610ms | future | 1241 | 735 | 838 | **61.2084%** | 0 / 0 | 42 | 3 |

**Verdict:** **all 10 tested S629 shared-hop candidate modes have
worse NoteOn F1** than frozen original 0ms. The closest `future`
mode has zero net additional true note detections and one extra FP.
The requested exact 10ms `average` also degrades substantially,
dropping TP by 85 and K3 42->38. A fast solution to user Windows
overflow does not establish an accuracy benefit: S629 is for
diagnosing and optionally testing lookahead acoustics; keep
`--lookahead-ms 0` as recommended operational configuration.

These figures are computed on the SAME 17 original DEV compositions,
not new held-out music. The original model/selection was frozen.

### Current independent Windows diagnostic

On this branch, the original reference
`--lookahead-ms 0` now additionally logs
`s629_original_worker_diagnostics` at stop:
- `worker_p95_ms`, `worker_max_ms`, `worker_total_ms`;
- `model_audio_processed_ms` and `worker_over_audio_ratio`;
- nominal callback duration ms.

Compare these to `s629_lookahead_diagnostics` from shared mode
on the SAME mic/output/block settings. **worker_over_audio_ratio>=1**
means the worker cannot keep pace with input across that sampled
interval, even if a larger queue postpones the overrun. This is
separate from soundfont rendering and negotiated input latency.
