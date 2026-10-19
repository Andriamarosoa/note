# S768 — Independent causal all-pitch Onset rescue toward >80% F1

**Objective:** Improve S708's REAL GuitarSet CAL32 Onset F1 (74.2638623%) to >80%, without regressing five other official final-MIDI scores. S768 itself is a SHADOW ONSET-ONLY research stage, not a promoted MIDI system.

## Why not another selector sweep?

S767 proved from exact CAL32 original reference:
- 1170 baseline-missed true attack instances.
- Existing S765/S763 acoustic candidate union recovered only 97/1170 in an oracle one-to-one match.
- 1073 misses lacked any matching candidate.
- The learned arbiter cannot add correct attacks that no candidate generator offered. Timing and thresholds on the same candidate pool cannot fix that recall ceiling.

## Design: evaluate every note 40..88 at every causal 1024-sample fragment

S768 deliberately does NOT pass through S765's weak-attack gated candidate list. It computes features for all 49 MIDI pitches at every received four-hop (4×256=1024 samples, 44.1kHz) acoustic decision.

87 features:
- 39 S701 native causal onset, sustain, release, harmonic, octave/fifth interference and K posterior features (with lagged history);
- 16 S733 auditory scene residual, acoustic novelty, explained away content, harmonic partial support and K-related context;
- 8 partials of current high-resolution spectra; 8 partial-wise changes versus one prior decision; 8 partial-wise changes versus four prior decisions; and 8 normalized current partials.

No future signal in any feature. Exact-K model remains signed S708 and no previous note stream is modified. The trained complementary classifier learns only **baseline-missed** FIT true attacks (not all easily detected attacks).

Training uses 80 FIT recordings, five original GuitarSet compositions, a fixed random seed, deterministic subsampling of positive windows, hard acoustic negatives and random negatives, and HistGradientBoostingClassifier.

Each FIT composition is predicted by a model that did NOT see that composition. The 11 thresholds are scored on **OOF FIT** only. One threshold is frozen before the CAL32 results are read.

At inference the new Onset head produces stand-alone note-on hypotheses with strict per-pitch causal hysteresis, 4-block refractory and a past-only 3-block S708 source NoteOn cooldown. NoteOn decisions never see future S708 MIDI events or future NoteOff.

The output is scored with scripts.s613_train_strict_live.event_score, the SAME pitch-specific one-to-one ±80ms matching used for onset in the official S704 evaluator. But there is **no paired NoteOff and no final MIDI stream in S768**, so an improvement in this Onset shadow score is not proof of a new official six-axis MIDI baseline.

## Gates and honest reporting

- All S708 + S707 original MIDI events and weights immutable.
- Strict signed baseline six-metric reference parity 1e-8 on CAL32, no lowering the test tolerance.
- No ground truth to candidate model inference. CAL32 labels only opened AFTER all eleven independent candidate streams have been emitted.
- No tuning thresholds based on CAL32. Multiple CAL32 trials are diagnostic because this split is repeatedly explored, NOT an unexposed final test.
- No HELD/TEST and no LIVE deployment.
- Report FIT OOF results for 11 thresholds, 32-recording CAL32 grid, frozen FIT threshold and F1. Even if it exceeds 80, do not claim official six-axis promotion until paired causal MIDI and cross-corpus validation satisfy non-regression.

## Success criteria

1. S768 independent stand-alone Onset F1 >80% on GuitarSet CAL32 at threshold selected OOF from FIT, without any calibration leakage. If not reached, record the exact F1 and failed recall/precision tradeoff, don't fabricate gain.
2. Adapt the winning high-recall generator to a separate note-instance-aware MIDI arbiter, preserving correct source note pair identities, then evaluate ALL SIX official S704 metrics on the SAME serial MIDI stream without regressions.
3. Confirm on an independently held-out composition/corpus not already used for research; only after freezing the model can a candidate be proposed as the new reference.

**Current official reference remains S708** until gate 2 and 3. S768 is an important shift from selector hyperparameter sweeps to learning the entire causal pitch/onset evidence lattice.
