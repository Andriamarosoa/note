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
