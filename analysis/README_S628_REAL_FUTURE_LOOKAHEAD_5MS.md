# S628 — Adjustable 5 ms of actual future audio, causal in real time

This is a **research experiment**, not a promotion or a workaround
that pretends the model can see future samples without waiting.

## User-requested knob

`--lookahead-ms` is in real milliseconds of physically captured future
audio. Default **0** executes original frozen NOTE S613 precisely as
before. At `--lookahead-ms 5`, native 44,100Hz rounds to **220 samples
= 4.9887 ms**, with an independently phase-aligned right-anchored
acoustic frontend seeing the PCM through **t+220 samples** before
deciding what original NOTE should emit for nominal frame t.

The frozen S613 model and original K0–K6 counting/head/selector stay
intact; only onset and sustain **decision evidence** can be combined
with the additional future sample window. Audio before t+5ms cannot
cause a post-lookahead event to emit.

**Real extra latency is unavoidable**, and may be larger than 5ms:
actual GUI microphone capture packets, worker compute and FluidSynth
audio output all add delay. MIDI `available_sample` and `time`
are actual physical *post-arrival* values, NEVER backdated to the
nominal t; `decision_sample` remains the nominal reference frame
so latency can be audited.

## Accurate experiment controls

Configurable `--lookahead-mode`:
- `delay`: **timing-only control**: wait for actual 5ms PCM and emit
  same frozen K/onset decisions WITHOUT using it. This catches
  apparent NoteOn F1 gains that are only timing/alignment effects.
- `average` (default): mean original and future onset & sustain.
- `confirm`: average onset and require at least as strict sustain
  as original/future min (may prevent some FP and cause FN).
- `max_onset`: max onset+sustain (may rescue FN and greatly increase FP).
- `future`: use frozen future-phase onset+sustain directly with
  original frozen K and release (may alter lots of note emissions).

These are **explicit competing hypotheses**, not learned parameters or
supported recommendations. No claim of F1/Exact-K improvement.

## Windows iRig playback

Select branch:
```powershell
git fetch origin
git switch codex/note-s628-lookahead-5ms-ablation
git pull --ff-only
python -m unittest -v test.test_s628_true_audio_lookahead

# Original NOTE reference (no intentional future delay):
python -m scripts.live_strict_notes --mic --device "WASAPI:External Microphone" --block-size 256 --hop-size 256 --lookahead-ms 0 --soundfont "C:\\path\\to\\YOUR_GM_PIANO.sf2"

# Controlled delay-only mode: preserve original model note decisions
python -m scripts.live_strict_notes --mic --device "WASAPI:External Microphone" --block-size 256 --hop-size 256 --lookahead-ms 5 --lookahead-mode delay --soundfont "C:\\path\\to\\YOUR_GM_PIANO.sf2"

# Experimental true future-acoustic phase blending:
python -m scripts.live_strict_notes --mic --device "WASAPI:External Microphone" --block-size 256 --hop-size 256 --lookahead-ms 5 --lookahead-mode average --soundfont "C:\\path\\to\\YOUR_GM_PIANO.sf2"

# Alternative gate experiment (risk of many extra false notes):
python -m scripts.live_strict_notes --mic --device "WASAPI:External Microphone" --block-size 256 --hop-size 256 --lookahead-ms 5 --lookahead-mode max_onset --soundfont "C:\\path\\to\\YOUR_GM_PIANO.sf2"
```

Change delay:
```powershell
# 2.5ms and 10ms are possible
python -m scripts.live_strict_notes --wav .\\test_guitar.wav --lookahead-ms 2.5 --lookahead-mode average --out .\\s628_2_5ms.jsonl
python -m scripts.live_strict_notes --wav .\\test_guitar.wav --lookahead-ms 10 --lookahead-mode average --out .\\s628_10ms.jsonl
```

Stop using `--lookahead-ms` if your real microphone queue/CPU
overruns or the piano has unacceptably delayed notes. A capture
`--block-size 2048` is itself roughly **46.4ms**, so reducing
it is essential to *hear* a 5ms effect.

## LIVE architecture and study

S628 is a two-context **single-selector** experiment:
1. Original past-only S613 hop256 features/models produce
   `(onset,sustain,release,K)` for reference decision frame t.
2. Experiment waits **L physical samples**, then computes a SECOND
   frozen-head 2048/4096/8192 trailing-window acoustic feature for
   frame physically ending at t+L. This consumes extra CPU and may
   cause microphone worker queue problems; input capture stays
   independent and bounded as before.
3. Only onset/sustain probabilities are combined; count-K and release
   always remain from original S613 frame t.
4. The original `CausalSelection` at nominal frame t accepts
   evidence only when future samples are ALREADY captured. The
   output `time` is when the callback audio actually became
   available (not t), ensuring latency accounting.
5. At EOF, residual frames with unavailable future are decoded
   using ORIGINAL reference probabilities and events explicitly
   marked `lookahead_eof_unconfirmed_` to prohibit fake future.
6. `--lookahead-ms=0` does not instantiate this stream at all.
   No changes to historical S80, published S613 S619 or S623.

**Not trained for future alignment**: These are experiments on the
original frozen model and should not be interpreted as a newly
calibrated noncausal note-prediction model. Any successful new lookahead
variant would ideally need fresh source-disjoint training.

## Evaluation and gates

[CI S628 real 5ms future audio workflow](https://github.com/Andriamarosoa/note/actions/workflows/note-s628-real-future-lookahead.yml):
- 17 pinned original DEV tracks from GuitarSet and URMP only;
- verify original no-lookahead TP1241 FP734 FN838;
- compare each physical L=2.5, 5, 10ms with `delay`,
  `average`, `confirm`, `max_onset`, `future`;
- baseline timing-only `delay` proves whether improvement is
  from actual additional acoustic evidence or from late matching;
- identify restored older false negatives, lost formerly good TPs
  and new FP; grouped onset K1..6 and exact pitch-set metrics;
- use timestamp physically available (callback next 256 samples);
- no held-out TEST accessed; all modes tuned/evaluated on DEV are
  **NOT independent generalization**.

Never claim that NOTE is still "strictly zero-lookahead causal" with
this flag enabled. It remains *streaming with bounded L-millisecond
lookahead*, sacrificing minimum latency for a hypothesis about attack
evidence. To restore strictly no intentional future, pass `--lookahead-ms 0`.


## Completed independent-source DEV ablation within same 17 original recordings

[GitHub full causal PCM run 38892231185 — SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38892231185).
This is **DEV tuning only**; "independent-source" means GuitarSet/URMP
training compositions excluded, NOT a fresh independent held-out TEST.

Frozen S613 original no intentional future:
**TP1241 FP734 FN838 F1=61.2235%**.

### Exact 5 ms variants (220 PCM samples)

| Mode | TP | FP | FN | F1 | Old misses recovered / previous TP lost | K3 exact /183 | K4 exact /44 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ORIGINAL 0ms | 1241 | 734 | 838 | **61.22%** | reference | **42** | 3 |
| delay only +5ms | 1226 | 749 | 853 | 60.48% | +2 / -17 | 42 | 3 |
| average +5ms | 1214 | 684 | 865 | 61.05% | +1 / -28 | 39 | 3 |
| confirm +5ms | 1198 | 640 | 881 | 61.17% | +2 / -45 | 43 | 2 |
| max_onset +5ms | 1248 | 775 | 831 | 60.85% | +9 / -2 | 37 | 5 |
| future +5ms | 1239 | 727 | 840 | **61.26%** | +7 / -9 | **37** | 5 |

The extra 5ms of actual sound produces different predictions, but no
5ms configuration demonstrates useful *joint* F1 and K3/K4 advantage
relative to S613. Best 5ms F1 is just **+0.037 percentage points**
versus 0ms and loses FIVE K3 exact-group cases. The 5ms mode
`future` has -2 TP, -7 FP and +2 FN, so even the small F1 gain
is due to fewer FP, not better recall.

`max_onset` shows the tradeoff: +7 net recovered true attacks,
**+41 FP**, and K3 falls from 42 to 37 despite K4=5.

### Wider 2.5/10ms study

Best F1 among all 15 predeclared DEV configurations:
`--lookahead-ms 10 --lookahead-mode future`: TP=1241 FP=731
FN=838 F1=**61.2688%**, K3=40/183, K4=4/44.
The change versus frozen 0ms is +0.045 percentage points, with no
net gain of correct notes and slightly worse K3. Insufficient for
promotion or a causal conclusion. Timing-only controls were also
recorded, distinguishing latency-induced evaluation differences
from new future-evidence decisions.

### CPU (Linux synthetic, NOT real Windows roundtrip)

[GitHub CPU benchmark 38892790895 — SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38892790895).
Same 1.5sec three-harmonic E2 synthetic waveform, 256-sample blocks:
- Original: wall/audio 22.09%, callback worker p95=1.36ms.
- Real +5ms branch: 42.86%, worker p95=2.60ms.
- **1.94× CPU cost**; this benchmarks frozen inference only,
  not Windows Realtek iRig capture/FluidSynth output delay.

**Verdict: S628 5ms lookahead is implemented and configurable but
NOT PROMOTED.** The 5ms best F1 is effectively unchanged within the
accuracy of this one DEV benchmark, K3 regresses and extra CPU/latency
are unavoidable. Do not assert that 5ms lookahead improves NOTE.
To test on user WAV or iRig, compare physically simultaneous
recordings at 0ms and 5ms `future`, keeping actual emitted event
timestamps, and perform independent K2–K4 note annotation checks.
