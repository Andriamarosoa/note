# S642 — Acoustic weak-onset candidate + trained per-note 0–3-hop waiting

**Branch** `codex/s642-trained-weak-onset-delay`.

## Motivation and traceability

S638 matched NOTE and MIDI under a common causal exact-note metric
on the same 12 GuitarSet development audio recordings. NOTE original
S614 had **901 TP / 682 FP / 679 FN, 56.97% F1** on the common
pitch range 40–76 under late-only 80ms tolerance, while MIDI
V2.2.1 had 397 / 1150 / 1183 and 25.39% F1.

The S640 acoustic replay diagnosed the 613 reference notes neither
model matched: 379 low S614 attack probability, 53 onset/sustain
asynchrony, 93 events earlier than reference, 84 late and 4
unmatched despite a ready candidate. This is exploratory failure
classification, not a causal intervention on the model.

S641 tested counterfactual uniform hold on **already emitted** S614
NoteOn, without changing pitches: 17.41ms increased causal DEV F1
from 56.97% to 59.25%, with **59 new** and **23 lost** reference
matches. The 17ms selection was made after seeing DEV labels and
cannot be treated as a valid independently established threshold.

## S642 hypothesis

Train two genuinely new heads from original TRAIN only:
- learned weak attack detector, using the original S614 native
  49×126 acoustic representation and 49×3 probabilities, physically
  gated and emitting at most one additional candidate per frame;
- separate trained 0,1,2,3-hop (at 44.1kHz / 256 hop) action model
  applied *at the original live NoteOn decision*, using only the
  acoustic/current+previous probabilities. The matching NoteOff
  is delayed by equal hops, protecting MIDI duration.

The discrete delay is **not** chosen by maximising S641 DEV F1;
the original TRAIN truth is used to label the minimal positive delay
necessary to avoid a premature NoteOn, while all other events learn
action 0. Prediction never sees the labels or future frames.

Both heads are HistGradientBoosting classifiers on the
S618/S626-compatible current+past acoustic feature ABI. This uses
a fully causal per-event feature vector at inference, but its batch
replay alone is not yet a proof of correct Windows hardware LIVE
output: queue, pending note lifecycles, actual latency and noteoff
ordering need a separate streaming runtime evaluation before use.

## Frozen comparison

- Train: 17 URMP and 18 GuitarSet recordings with no composition
  shared with DEV, original protocol hashes verified.
- DEV: 5 URMP and 12 GuitarSet, previously exposed in S614 studies,
  never a new independent holdout.
- Original S614 checkpoint checksum held fixed, original S80 untouched.
- Predeclared variant order: original S614, candidate-only,
  delay-only, candidate+delay.
- Re-evaluate true NoteOn/NoteOff under original ±80ms onset
  protocol and separately strict causal [0,+80ms] within MIDI 40–76.
- Audit TP, FP, FN, corrections, regressions, recording-level wins,
  latency action distribution and conditional NoteOff accuracy.
- No official model promotion from a DEV-only result.

## GitHub Actions proof

[Run and artifacts](https://github.com/Andriamarosoa/note/actions/workflows/v273-s642-learn-weak-note-and-timing.yml)

Reference train script:
`python -m scripts.s642_train_weak_and_delay --out /tmp/S642`

Test:
`python -m unittest -v test.test_s642_dual_learned`

The S642 CI run number, actual SHA256 values, F1 scores and decision
will be appended **only after** the run succeeds or fails. Never
extrapolate scores from S640/641 counterfactuals.

## Acceptance rule

No promotion unless:
1. Exact original S614 can be reconstructed at zero change.
2. Both datasets improve without material regression in true notes
   or spurious events, with source-level failure audit.
3. The actual event ordering remains legal for NoteOn/NoteOff.
4. Strict live block-by-block inference on real WAV and hardware
   latency/overflow checks have been performed.
5. Independent frozen sources beat S614 on one shared protocol.

Until then, S614 **69.03% historical symmetric TEST** remains
official; S632 remains experimental.
