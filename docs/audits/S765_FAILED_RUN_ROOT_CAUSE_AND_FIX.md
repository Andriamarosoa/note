# S765 forensic failure audit and corrective experiment

Date: 2026-10-19. Branch: codex/note-s765-fit-ensemble-arbiter.

## Original failed run
Run https://github.com/Andriamarosoa/note/actions/runs/39231888681
Job tests: 29 passed. Full FIT80/CAL32 replay executed but job failed due to S708 strict reference parity.

Artifact ID 12177212063 yielded:
- FIT80 had ONLY 5 eligible candidate notes, from only 2 of 5 training compositions; positive onset 5, positive complete notes 2.
- Five rows and 2 training compositions cannot support meaningful five-fold composition-generalized calibration. The learned arbiter OOF observed zero predicted positives for all thresholds 0.50 / 0.70 / 0.85 / 0.95.
- All 32 CAL32 recordings completed. The independently decoded FullNote branch generated 3,434 paired candidates but source guard rejected 3,431 because the corresponding note was already active in S708. The remaining 3 were blocked by the six-voice budget. **0 genuinely new eligible proposal notes on CAL32.** All 4 evaluated MIDI streams were identical to native source.
- Independent CAL replay drift: S708 Exact-K by +1/126304 frame; Active Exact-K by +1/100243 active frames, Frame Pitch F1 by approximately +0.00000288 absolute. Onset, Offset and Full Note exactly matched the original frozen values. This is a strict reproducibility failure regardless of its small size; do not relax the parity gate or claim improvement.

## Root cause
The FullNote decoder in S763 reinterprets essentially the SAME upstream attack/sustain posteriors and largely reproduces S708's recognized NoteOns. Giving each axis an independent state does not automatically generate missing acoustic pitch attacks. A risk model cannot admit correct unknown notes when none were proposed.

The failed run exposed a second, independent issue: inferring that all 5 FIT compositions are represented simply because all 80 recordings were processed was incorrect. We must report actual training coverage and fold count, and never claim 5 folds when proposals appear in only 2 compositions.

## Implemented correction, still experimental
The independent **acoustic_birth_pairs** generator in scripts/s765_fit_independent_arbiter.py now observes the existing signed S733 H4 pitch/novelty evidence, sustain, spectral rise and independent partial support. It emits additional NoteOn proposals for weak attacks that S763 FullNote missed. Onsets never inspect future audio or candidate NoteOff. Proposed new notes are closed causally by received release/sustain observations, a maximum-duration safety bound or source preemption; future closure does not determine birth features. S708 source events stay intact.

The FIT dual-head model still trains only on FIT80 with grouped out-of-fold composition splits, and calibrated gates 0.50, 0.70, 0.85, 0.95 are considered for full MIDI CAL32. There is no guarantee of successful promotion. Source reference parity is still enforced.

Regression tests now cover a weak-onset missing note being born independently of FullNote, release closure at a future time, invariance of the birth decision under future release changes, and source priority. A first attempt hit a local loop variable typo and an omitted synthetic fixture field; both have been corrected. Test CI passed on the new run.

## Promotion rules
Do not consider an S765 result promotable unless all of:
1. Strict signed S708 six-score parity on GuitarSet CAL32.
2. All source MIDI originals retained unchanged.
3. At inference no ground-truth labels or future acoustic samples/events influence a current proposal.
4. A sufficient FIT-only composition-grouped independent training set and genuine new notes. No invented five-fold CV if effective coverage smaller.
5. No worsening of ANY six official S704 MIDI metrics and no lost previously correct source offsets/onsets.
6. New held-out evaluation after freeze; CAL32 has been reused heavily and cannot be presented as an untouched test.
7. LIVE remains unchanged until measured and separately verified.

If all metrics stay unchanged because the learned risk gate refuses every proposal, classify as **no improvement**, NOT an architectural success.
