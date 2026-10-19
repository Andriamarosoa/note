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
