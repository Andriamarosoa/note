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
