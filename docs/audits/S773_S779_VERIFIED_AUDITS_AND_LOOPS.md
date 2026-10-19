# NOTE S773–S779 — verified audit trail, 19 October 2026

## Reference and scientific validity

- **S708 remains official**: GuitarSet CAL32 74.263862% onset F1,
  62.045541% Exact-K, 56.276249% Active Exact-K,
  77.329193% conditional Offset, 57.131931% Full Note F1,
  76.462635% Frame Pitch F1.
- CAL32 has been seen frequently by the research process. It is a
  diagnostic set, not a pristine holdout. Do not promote based on shadow
  or CAL-only improvement. Final promotion requires six non-regressing
  S704 metrics evaluated on **one serialized causal NoteOn/NoteOff stream**
  at the official 256-sample frame cadence.
- Preserve signed S708 and S707, LIVE causality, no HELD/TEST access.

## S773: stage fingerprint repeat tests — SUCCESS

Run: https://github.com/Andriamarosoa/note/actions/runs/39251829722

Repeated four selected tracks three times on two GitHub Actions
configurations (Torch threads 1/2). Each configuration is byte-identical
within a worker at all checked stages. Comparing across separate workers,
raw downstream acoustic representation, native predictions and scene
features differ by hash; **final MIDI is identical on the four clips**.
This does not mean S770 whole-CAL replay is stable.

## S775: locating the first differing processing stage — SUCCESS

Run: https://github.com/Andriamarosoa/note/actions/runs/39252456168

Two GuitarSet WAV files match exactly in source SHA and decoded PCM;
decision sample indices match. Separately generated native FFT features
differ across worker configurations, *before* model inference. Native
pitch/K posteriors differ too. Full sharp spectrum hashes also differed,
though selected sampled vectors in S777 matched.

## S777: quantify differences rather than claiming their cause — SUCCESS

Run: https://github.com/Andriamarosoa/note/actions/runs/39252988095

On the sampled vectors of the two audited clips:
- Native FFT max absolute difference: **2.384185791015625e-7**;
- Native K posteriors max absolute difference: **5.364418029785156e-7**;
- Native pitch posterior differences <=2.682209014892578e-7;
- PCM samples and sampled sharp harmonic values are exactly identical;
- after converting sampled native FFT float32 to float16 only 5/49392
  features on one clip and 2/49392 on the other still differ.

These differences are small and are a plausible threshold-boundary
instability. **Their causal role in S770's one-frame K divergence is
not yet proven**. Do not silently introduce quantization into S708;
it would change frozen model inputs and require full non-regression testing.

## S776: independent Exact-K K5/K6 confusion — SUCCESS

Run: https://github.com/Andriamarosoa/note/actions/runs/39252771727

Evaluated on CAL32, 1024-sample decision fragments (diagnostic, *not*
the official 256-sample Exact-K cadence):
- Raw K5 163/750 = 21.7333%; 585/750 undercount;
- Raw K6 0/189 = 0%; 189/189 undercount;
- Raw K6 class predictions: K2 2, K3 39, K4 90, K5 58, K6 0;
- MIDI active-count K5 73/750, K6 13/189, 86/939 overall K5/K6;
- Raw per-fragment Exact-K overall ~69.6053%, MIDI-active
  per-fragment K ~63.6214%.

S776 is a diagnostic, not a modification to the official MIDI stream.
A head trained for K5/K6 is a concrete next experiment.

## Active experiments, not verified as of drafting

- **S774**: https://github.com/Andriamarosoa/note/actions/runs/39252171132
  All-pitch harmonic-aware Onset birth classifier with octave and fifth
  neighbor cues, FIT-only five composition OOF calibration.
- **S778**: https://github.com/Andriamarosoa/note/actions/runs/39253393550
  Controlled ablation: same S774 features, FIT examples and architecture
  but remove 4x positive class weighting; choose stricter thresholds on
  FIT80 (.8–.998), CAL read after freeze.
- **S779**: https://github.com/Andriamarosoa/note/actions/runs/39253787743
  Separate 31D class-aware K5/K6 specialist, composition-grouped FIT OOF,
  CAL32 with frozen weight, no final MIDI promotion.

No new >80% outcome is claimed before CI artifacts are read and audited.
The immediate research objective is source stability plus *new genuine
candidate coverage with sufficient incremental precision*, then
instance-aware NoteOn/NoteOff association and full six-score evaluation.
