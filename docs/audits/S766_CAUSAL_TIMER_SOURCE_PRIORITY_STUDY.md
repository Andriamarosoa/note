# S766 — Causal additional-note lifetime control and non-regression audit

## Frozen source and measured motivation

Original S708 GuitarSet CAL32 signed reference (S765 clean replay):
- Onset F1: 74.2638623327%
- Exact-K: 62.0455409172%
- Active Exact-K: 56.2762487156%
- Offset accuracy: 77.3291925466%
- Full Note F1: 57.1319311663%
- Frame Pitch F1: 76.4626353687%

S765 genuine new acoustic proposal generator, FIT80 5-composition cross-fit:
- 872 eligible FIT80 proposals, 259 onset-correct, 168 full-note-correct.
- CAL32 280 eligible proposals after source priority.
- risk .50: 64 proposed notes accepted, 28 source preemptions, 10 lost previously correct source note offsets. Onset/Exact-K/Active Exact-K/Frame Pitch improve, Offset and FullNote decline.
- risk .70: 10 accepted, 5 preemptions, 1 correct source offset lost. Still unsafe.
- risks .85/.95: 0 accepted, so baseline unchanged, no improvement or promotion.
- Signed original MIDI events remain unchanged but evaluation truth matching is affected by inserted extra NoteOns; preserving original event bytes alone does not establish metric safety.

## S766 changes

Files:
- scripts/s766_causal_extra_note_timer_cal32.py
- test/test_s766_causal_extra_note_timer_cal32.py
- .github/workflows/note-s766-extra-timer-cal32.yml

Purpose: attempt to reduce source preemption by setting a causal maximum lifetime of additional notes AT NOTE-ON, WITHOUT seeing future original source Notes, future audio or future candidate release. A source note appearing later still wins if an extra note is active.

Predeclared durations (4 native hops per block, 256 samples per hop, 44.1 kHz):

- 4 blocks = 4096 samples = ~92.9ms
- 8 blocks = 8192 samples = ~185.8ms
- 16 blocks = 16384 samples = ~371.5ms
- 32 blocks = 32768 samples = ~743.0ms

Predeclared quality gates .50, .60, .70. Twelve combinations total.

Each accepted candidate emits its genuine release whenever it arrives causally, or is closed by the earlier timer event that was fixed at onset. No future candidate NoteOff influences the NoteOn classifier or its timestamp.

All twelve final MIDI streams are constructed before opening CAL annotations. FIT80 supervision and GroupKFold composition cross-validation remain unchanged. CAL32 is diagnostic; it is NOT a held-out unseen benchmark because repeated S765/S762 experiments saw CAL32. HELD/TEST untouched.

## Measured outputs requested in the CI report

For every variant: six official S704 MIDI metrics, source parity, true positive/false positive changes, accepted notes, preemptions, natural vs timer releases, offset match losses, long note onset losses, recovered versus displaced truth matches, per-recording examples.

Promotion rejected for any of: one of six metrics regresses, a source-previously-correct offset is lost, previously correct long note onset lost, total onset TP decreases, or frozen S708 parity fails.

Even a passing CAL-only result is just a candidate for independent evaluation after freezing architecture; it is not an automatic promotion.

## Interpretation

Independent decoders are already a useful finding, but Extra-Note arbitration is still the bottleneck. S766 is not guaranteed to fix stolen onset matches. Timer control reduces the duration of collisions and may reduce source preemptions, but also risks ending real polyphonic notes too early. Results are required to distinguish those effects.

Never claim six scores >80% until all have been independently evaluated on the SAME complete serialized MIDI output.


## Executed result — GitHub Actions 39237250199

Actual run: https://github.com/Andriamarosoa/note/actions/runs/39237250199
Artifact: https://github.com/Andriamarosoa/note/actions/runs/39237250199/artifacts/12180067997

All 80 FIT recordings, all 32 CAL recordings and all twelve complete MIDI variants were measured. Unit tests passed after fixing an explicit-zero diagnostic counter. The workflow finished FAILURE because the signed S708 baseline had a reproducibility drift of exactly one 256-sample frame in Exact-K and Active Exact-K: +7.9174056246e-6 and +9.9757589059e-6 score fractions; native Frame Pitch F1 differed +2.879479805e-6. Onset/Offset/Full Note remained exactly identical. Since strict parity requires deviations under 1e-8, this is a genuine gating failure. Do not waive or silently round it.

### What the timer experiment actually revealed

- Quality risk .50: 64 accepted additional notes. A 4-block max timer reduces preemptions **28 → 16** relative to the 32-block cap, but **all four caps still lose 10 original correct source offsets**. No long source onsets or source matched truth IDs are lost; 28 new missed truth onsets are recovered. Offset and Full Note F1 still decline.
- Risk .60: 29 additional notes accepted, with 4–14 preemptions depending on cap, but **the same 3 source-correct offsets are lost at ALL four caps**. 16 new missing truth onsets recovered. Offset and Full Note F1 decline.
- Risk .70: 10 added notes accepted, 1–5 source preemptions, **the same 1 original correct offset lost at ALL caps**. 7 new missing truth onsets recovered. All four caps degrade Full Note F1 and conditional Offset accuracy.
- Across every risk and cap, **zero original truth onset IDs are displaced** according to the audit; the original correct offset losses nevertheless remain. This indicates that shortening the lifetime avoids some live source preemption without resolving later note-instance/offset matching competition in the evaluator. The diagnostic doesn't prove exactly which predicted event stole each correct offset; that requires a per-event matching trace.
- Best Full Note among risk .70 capped variants occurs at 32 blocks: Full Note declines 0.0218 percentage points; Offset declines 0.0909 percentage points. Onset improves 0.0837 points; Exact-K +0.1298; Active Exact-K +0.1636; Frame Pitch +0.1717 points. Still unsafe.
- No S766 candidate promoted, and 80% goals not achieved. CAL32 has already been exposed to multiple trials, hence even a safe version would require independent confirmation.

**Revised root cause:** Timers reduce preemption counts but do not remove correct-offset matching loss. Continuing to shorten note lifetimes alone is not an effective way to obtain six-metric non-regression. The priority is proper matching identity and whether the source-active guard rejects genuine same-pitch retriggers. This motivates the separate S767 frozen-source audit; it does not justify rewriting S704 official metrics.
