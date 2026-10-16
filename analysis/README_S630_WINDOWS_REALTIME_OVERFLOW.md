# S630 — diagnose/fix Realtek iRig input_queue_full with batched frozen native ONNX

## User's exact failure

```text
BufferError: Microphone input overflow:
input_queue_full(maxsize=13, block_size=256, queue_nominal_audio_ms=75.5)
```

This error comes from `scripts/live_strict_notes.py` producer callback
`inbox.put_nowait`, not from a corrupted audio file or a piano
SoundFont. One 256-sample callback is **5.805ms** at 44,100Hz. The input
thread receives microphone PCM while the inference worker consumes
queue blocks; 13 blocks = 75.46ms of buffering. When all 13 are
occupied, NOTE safely stops and releases active notes rather than
playing stale/late notes. It is not fixed by increasing the queue size;
that merely adds backpressure delay before failure.

Earlier S628 user log had **p95 19.44ms per 5.805ms incoming callback**
while exact-phase future lookahead performed 2x FFT/neural/ONNX. S629
removed the duplicate second pass, but user still reports this
input_queue_full exception. We **cannot** conclude the remaining
cause without user-specific S629 worker statistics and input-side
producer metrics: original frozen S613 might be too slow on Windows
or PortAudio callback/output-thread contention may consume CPU.
The CPU microbenchmark on Linux is only a component test.

## S630 new independent optimization

Frozen S613 has already **batched** `heads.predict(features)` support:
the same three ONNX tree sessions can run across (N,49,126) instead of
being invoked **three times per individual 256-hop**.

S630 `src/causal_note/batched_live.py` collects N consecutive
ORIGINAL NOTE causal acoustic hop256 feature rows and runs a SINGLE
original hybrid MLP+tree-predict call per group. The original
`CausalSelection` independently consumes each original 49-note
onset/sustain/release and K0..6 in original time order. There is
exactly one MIDI note owner; no feature, attack, model checkpoint,
selector threshold or Exact-K interpretation is replaced.

`--inference-batch-hops` default **1**, so unchanged official
S613/N629 output; experimental 2,4,8,16,32 possible only with
`--hop-size 256` and original/no-sidecars or shared lookahead.
No artificial model audio gaps or enlarged queue.

**Unavoidable cost:** waiting until N future native hop frames have
arrived to process the previous ones; for N=4 the first frame waits
**17.415ms** extra before batch inference, plus capture callback
delivery and computing. This is for throughput, not for any unproven
5ms lookahead accuracy gain.

## Measured verified Linux CPU results

[GitHub Actions successful 38895654147](https://github.com/Andriamarosoa/note/actions/runs/38895654147).
[Repeat with actual input-queue diagnostic code 38895991687 SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38895991687).

11? test suite **10 tests PASSED**; original frozen hybrid sample
nominal notes invariant under batching, future/callback manipulation
does not access uncaptured samples. 2s identical synthetic E2, single
thread CPU, 256-sample PortAudio-like callback:

| Batch hops | Frozen ONNX calls for 344 frames | CPU/audio time | Worker p95 callback | Added intentional max wait |
|---|---:|---:|---:|---:|
| 1: original S613 | 344 | 18.45% | 1.151ms | 0ms |
| 2 | 172 | 15.71% | 1.470ms | 5.805ms |
| 4 | **86** | **14.04%** | **2.139ms** | **17.415ms** |
| 8 | 43 | 12.87% | 3.518ms | 40.635ms |
| 16 | 22 | 12.20% | 6.224ms | 87.075ms |

Batch4 reduces CPU work to about 76% of single-hop baseline in
this environment. Still **NO WINDOWS/iRig/FluidSynth latency or queue
stability measured**, and output will be audibly later.

`--block-size 1024` (23.22ms microphone callback) with
`--inference-batch-hops 4` uses 13.72% audio duration,
worker callback p95 3.333ms on Linux. This increases input callback
delivery latency further. The larger callback on its own is a
transport change; it does NOT retrain or change original model hop256.

## Windows user's CLI test matrix

First update to experimental branch and run unit tests:

```powershell
git fetch origin
git switch codex/note-s630-batched-onnx-live-audit
git pull --ff-only
python -m unittest -v test.test_s630_batched_live
```

**A. Frozen reference, same iRig and piano, no lookahead:**
```powershell
python -m scripts.live_strict_notes --mic --device "MME:External Microphone" --soundfont ".\SalamanderGrandPiano-V3+20200602.sf2" --hop-size 256 --block-size 256 --lookahead-ms 0 --inference-batch-hops 1
```

**B. Same original NOTE decision 256 with vectorized batch4:**
```powershell
python -m scripts.live_strict_notes --mic --device "MME:External Microphone" --soundfont ".\SalamanderGrandPiano-V3+20200602.sf2" --hop-size 256 --block-size 256 --lookahead-ms 0 --inference-batch-hops 4
```

**C. If input queue still overflows: reduce callback overhead (higher output latency):**
```powershell
python -m scripts.live_strict_notes --mic --device "MME:External Microphone" --soundfont ".\SalamanderGrandPiano-V3+20200602.sf2" --hop-size 256 --block-size 1024 --lookahead-ms 0 --inference-batch-hops 4
```

Keep same `--device`, piano SF2 and speaker across all tests to avoid
confounding MME vs WASAPI vs FluidSynth. User's last iRig had selected
MME automatically; explicitly selecting "WASAPI:External Microphone"
is separately useful for input quality but changes the capture
host API/negotiated 22ms latency. Do NOT inadvertently mix up changes.

**D. Optional later, ONLY if batch4 stable: shared-hop future test:**
```powershell
python -m scripts.live_strict_notes --mic --device "MME:External Microphone" --soundfont ".\SalamanderGrandPiano-V3+20200602.sf2" --hop-size 256 --block-size 256 --inference-batch-hops 4 --lookahead-ms 5 --lookahead-engine shared --lookahead-mode average
```
This adds 5.805ms actual acoustic future *in addition* to the up-to
17.415ms batch completion delay. It has NOT shown a convincing F1
benefit; do not recommend it for performance.

## What to send back for machine-specific proof

Compare the JSON emitted **on shutdown or overflow**:
- `s630_input_queue_diagnostics`:
  `callbacks_received`, `callbacks_processed`,
  `peak_queue_blocks`, `remaining_queue_blocks`, 
  `worker_total_ms`, `model_processed_audio_ms`,
  `worker_over_audio_ratio`.
- `s630_batched_diagnostics`: batch inference count,
  `worker_p95_ms`, `worker_over_audio_ratio`,
  effective future, batching wait.
- Original `s629_original_worker_diagnostics` and piano diagnostics.

If `worker_over_audio_ratio>1` across steady recording,
no finite queue will sustain realtime. If ratio <1 but brief
producer bursts cause overrun, inspect MME/WASAPI PortAudio callback
timing, Windows power states, CPU contention and warm-up behavior.
Never discard frames silently or rewrite event `time` to nominal
decision time. Test Windows end-to-end latency separately before
any claimed LIVE promotion.

## Accuracy safety

Batch4 nominal event identity was checked on synthetic frozen test
PCM with causal invariants. [Real 17 GuitarSet+URMP DEV grouped-K
validation](https://github.com/Andriamarosoa/note/actions/workflows/note-s630-real-dev-nominal-parity.yml)
runs frozen original S613 against batch4 `heads.predict` on ALL 17
original composition-separated clips to detect numeric ONNX batch
discrepancies and confirm actual NOTEon/F1 and Exact-K.
The results must not be reported before successful completion.
Actual MIDI delivery timestamps will necessarily differ: batch
waiting intentionally adds physical latency. This is not new
independent held-out testing.

**Status: experimental research branch, not promoted or merged.**
