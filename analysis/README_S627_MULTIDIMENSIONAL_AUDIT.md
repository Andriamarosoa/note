# S627 — Forensic audit, not new selectors chosen by intuition

**Protocol:** pinned NOTE S613 full native ONNX-tree+neural hybrid; 17
GuitarSet+URMP DEV source compositions; 24 seconds each; MIDI40..88;
NoteOn matches ±80ms. No held-out TEST access. Official S613, S80,
S619/S623 checkpoints and existing LIVE MIDI production are unchanged.

This follows S624 (real K/duration census), S625 (true exact
E2+E3 / F+Em chord-presence counts), S626 (which exact pairs/chords
are missed by the original model), and addresses their limitation:
the fact that a pair is frequently missed **does not establish WHY**.

## Atomic unit

One *real note event*:
- label = original S613 NoteOn within ±80ms: matched TP versus FN.
- all notes counted exactly once, unlike K3/K4's multiple possible
  pair relations.
- observed context: corpus, true independent composition, exact
  MIDI pitch/low or high register, duration, active-K, onset-K,
  co-active pitch intervals and same/other-note attack timing.
- actual *first available* frame features: log harmonic H1/H2/H3,
  harmonic strength ratios, three FFT resolutions, spectral novelty
  at exact physical 2/4/8/16-hop delays and short/long RMS.
- actual frozen head outputs onset/sustain/release, K0..6 confidence;
  original selector active/armed/high/low state BEFORE deciding.
- *post-hoc* maximum over ±80ms kept separate from online information;
  it can identify potential head/selector disagreement but CANNOT be
  used to propose zero-latency LIVE output.
- matched successful TPs as controls for FNs, first exact same
  composition/pitch/active-K, relaxing only when insufficient.
- per-split denominators, composition support and tiny-bin warnings.

One *false predicted NoteOn*:
- real 49-pitch score/selector state at emission, not a guessed source.
- relate to currently labeled active pitches and same-pitch nearby
  truth onset; distinguish late ±80ms alignment from a likely
  harmonic-interval, close-note, stale/repeated note-on or no
  labeled activity.
- note: a 12-semitone relationship alone **does not prove a physical
  overtone**, and an ON/OFF mismatch alone does not establish wrong
  labels. Requires actual spectrum/stem checks.

## Code / immutable run

- `scripts/s627_multidimensional_failure_audit.py` writes full
  per-note and per-false-event JSON plus conditional summaries.
- `test/test_s627_multidimensional_failure_audit.py` checks exact
  feature indices, K-active vs near-onset K, matched controls,
  proportions and causal pre-step state.
- [GitHub workflow run](https://github.com/Andriamarosoa/note/actions/workflows/note-s627-multidimensional-forensic.yml).

**Critical ABI correction:** from
`src/causal_note/strict_live.py CausalPitchFeatures._frame`,
feature columns **54–59 are two-hop lag** and **60–65 are four-hop lag**.
The S619/S620 research scripts earlier called columns 60–62
"two-hop novelty"; that description is wrong. The present audit
records and distinguishes both physical lags explicitly before
any new feature selection. Do not misattribute a training gain to
a 2-hop attack feature without verifying this.

## Scientifically valid sequence after S627

1. Verify original **TP=1241, FN=838, FP=734** by exactly reproducing
   frozen full hybrid S613 before inspecting condition differences.
2. Compare FN vs TP conditional rates, including their denominators,
   corpus, composition, pitch/register, K, onset lag.
3. Inspect paired within-source/within-pitch contrasts; if no good
   matched control exists, report it instead of generalizing.
4. Screen interacting factors: e.g. low pitch + K>=3 + an octave +
   delayed other-pitch onset; do not assume any alone caused failure.
5. Inspect actual spectral time series of a curated set of TP/FN/FP
   matched triples, and cases where label/algorithm disagree;
   head probabilities can expose ambiguities without proving source.
6. Train or augment **only** for conditions with demonstrated poor
   independent-composition performance AND inadequate diverse
   training coverage (S624/S625), with physically realistic audio.
7. Confirm on new source-disjoint held-out tracks, same NoteOn and
   Exact-K K1–K6 plus actual Windows iRig+FluidSynth p95 latency.
   Do not promote a dev-tuned rule because it helped one chord.

This report is an *audit design*. Please see the actual workflow
conclusion and its archived JSON before asserting measured S627
correlations, let alone a causal diagnosis.
