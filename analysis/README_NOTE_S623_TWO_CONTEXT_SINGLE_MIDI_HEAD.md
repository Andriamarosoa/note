# S623 — fast hop128 EVIDENCE + frozen S613 hop256 as the ONLY MIDI selector

Branch: `codex/note-live-s623-dual-evidence`. All code is in a **draft
research branch**, none merged/promoted into official NOTE or historical S80.

## Precise bug motivating S623

S620 fast note-on proposals independently controlled the piano. The
earlier fast owner masked later *correct* S613 note-on, losing 174 old
correct NoteOns while recovering 90 missed truth IDs. It also added
208 false NoteOns, dropping F1 61.22% to 55.39%.
S622 reference-priority retriggers eliminated most lost old correct
notes (1 lost) and recovered up to 90, but still added hundreds of
false NoteOns. Best F1 58.85%, still below S613. Both are rejected.

S621 really retrained separate hop128 NOTE heads on all original 53
GuitarSet/URMP train recordings with 50% sustain and 8% background,
24 K and 28 pitch epochs. However hop128 alone reached only
**48.52% NoteOn F1** on the 17 original DEV compositions, and naive
128+256 MIDI event union remained poor: best **52.21%**, because
separate MIDI owners produced many false notes.

## S623 research hypothesis

Do NOT let hop128 emit MIDI. Treat its outputs as additional
evidence for weak SAME-PITCH onset at the **current** reference
hop256 decision, which keeps control of every NoteOn, NoteOff,
K0..6 and refractory. This preserves already-valid baseline
note timing and avoids duplicate MIDI owners, at the cost that
note onsets are emitted at 256 cadence, not the faster 128 cadence.

At every 256 decision:
- The original S613 full ONNX+neural 256-hop head predicts K and
  49 pitch onset/sustain/release probabilities.
- Separately trained S621 neural hop128 makes **two** causal
  predictions at 128/256 in that reference interval.
- `scripts/s623_fused_evidence.py` tests the latest, either or both
  past/present fast frames having same-pitch onset+sustain and
  positive acoustic novelty.
- Only when original slow per-pitch onset is weak but nonzero
  (0.35/0.45/0.50 to < original threshold 0.60), slow sustain
  >= original threshold, original note is not active and is
  already armed, its onset is raised to pass original onset gate.
- Every old strong onset and every existing MIDI sustain/release
  remains the reference. Extra fast branch can influence *whether*
  a weak reference note is emitted, NEVER independently emit notes.
- No future PCM, no after-onset hold, no FFT size changes.
- The two parallel neural/FFT computations still cost more CPU;
  S620 measured roughly 3x the original on synthetic Linux audio.
  No Windows/iRig/FluidSynth end-to-end latency is yet available.

The protocol is still a DEV-only diagnostic with **72 predeclared**
candidate evidence threshold combinations; all use the same
original 17 GuitarSet+URMP dev compositions and report true
identity-matched FN recoveries, lost old TP, total FP, NoteOn F1
±80 ms, and grouped-onset K1..6 and complete pitch-set accuracy.
The fast hop128 checkpoint is downloaded from previously completed
GitHub run 38874449408; there is no new hidden training.

## Source of truth

Experiment:
https://github.com/Andriamarosoa/note/actions/runs/38877618546

Until this run is completed successfully, no result is supported;
even then dev hyperparameter selection is NOT held-out validation.
If evidence helps, independently test on fresh unknown guitar pieces,
E2+E3 true octaves and adjacent picked strings. Preserve reference
S613 and do not claim the 80% LIVE target until this is established.
