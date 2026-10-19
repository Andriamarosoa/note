# S745 — Independent FIT-prior-corrected specialist, 12 MIDI loops

Research branch `codex/note-s745-calibrated-activek-specialist-loops`.
Official unchanged S708 CAL32: **56.27624871562104% Active Exact-K**. Target: >80% on all six complete MIDI metrics, never merely the standalone network K0..K6 argmax. No distillation, new Student or LIVE deployment until all six benchmarks pass with no regression.

## Evidence-driven sequence

- S742 real CAL32: 36,890 under-counted and 6,940 over-counted active audio frames. True K5/K6 accuracy respectively 9.39% and 6.40%. Adding K-gated weak source attacks may help, but raw acoustic source recall is incomplete.
- S743 (12 real CAL32 loops): **62.3435%** best experimental Active Exact-K (+6.0673 points), but Onset F1 **71.5832%**, Offset **76.7302%**, complete-note F1 **54.5974%**; **all 12 rejected**. Candidate added 636 note births for this setting.
- S744 (2 FIT80 models, 8 real CAL32 loops): largest Active Exact-K **59.7827%** (+3.5065 points), yet Onset F1 **71.7955%**, Offset **76.3032%**, complete-note F1 **54.4687%**, 41 previously correct NoteOff matches lost. All 8 rejected. Its best Active-K model added 1,622 note births.

## S744 methodological issue identified and addressed

FIT80 involved **160,000 case-controlled labeled rows**, with **11,284** attack-positive examples (7.0525% sampled attack prevalence), from **29,030** original onset-positive training decisions, of which 11,284 were not already emitted by S708. Positives were multiplied by **4x sample weights** in the HGB loss. The learned raw predict_proba scores therefore do not represent the *natural per-pitch onset prevalence* among all candidate fragments.

S745 computes from FIT80 ONLY:
- `natural_positive`: truly missing onset-positive pitch-fragments among all source-inactive candidates;
- `total_eligible`: total source-inactive pitch-fragments before deliberate class-biased FIT subsampling;
- `sampled_positive`: positive examples included in training;
- `sampled_total`: all selected training examples.
- `weighted_training_prior`: model's FIT sampling prior after 4x positive weighting.
- `odds_multiplier = odds(natural_prior) / odds(weighted_training_prior)`.

Scores are corrected as `p_cal = (multiplier*p_raw)/(1 - p_raw + multiplier*p_raw)`. This is an approximate **FIT-only prior correction**, not a guarantee of perfect probability calibration because hard-negative case-control sampling also depends on acoustic features.

S745 then evaluates **12 frozen presets**: two independently trained positive-note HGB models × three corrected risk thresholds × two K-tail confidence thresholds. Per fragment, the router chooses among the same 32 H0-anchored acoustic head subsets; the decoder requires **simultaneous** calibrated classifier confidence, K probability tail, independent physical onset partial evidence, sustain, rise/novelty, source-inactive + armed state, and refractory protection. All candidate NoteOn streams are passed through the **original S707 NoteOff emitter** and scored against signed S708 on six complete MIDI metrics.

## S745 REAL completed GuitarSet FIT80/CAL32 result — 19 October 2026

[Verified CI run 39198255358](https://github.com/Andriamarosoa/note/actions/runs/39198255358): all unit tests and all **12 genuine full-MIDI CAL32 loops passed execution**, 0 models accepted.

- FIT80 natural prevalence among already-missing source pitch candidates: **0.3104%**.
- Model's **4x-weighted, deliberately case-controlled** effective positive training prior: **23.2837%**.
- Estimated FIT-only prior odds correction: **0.01025994963**, i.e. raw odds multiplied by approximately 1/97.47. This is approximate and hard-negative covariate bias remains.
- Strongest raw Active Exact-K S745 `depth3_risk4_tail25`: **60.0840%** versus official **56.2762%** (+3.8077 points), but Onset F1 **72.7824%** (down from 74.2639%), Offset accuracy **76.1598%** (down from 77.3292%), full-note F1 **55.1113%** (down from 57.1319%). **1,269 extra NoteOns and 39 previously correct Offsets lost. REJECTED**.
- Higher calibrated threshold `depth5_risk55_tail45`: Active Exact-K **58.1696%**, Onset F1 **74.3508%** (a small improvement), Offset accuracy **76.9052%**, full-note F1 **56.8594%**. **768 extra NoteOns, 20 previously correct Offsets lost. REJECTED**.
- Overall **all 12 trials failed the six-metric plus per-instance Offset preservation gate**. `eligible_CAL_candidate = null`, `six_gt80=false`, `promoted=false`, `LIVE_changed=false`.
- 32 independent K/H0 fragment masks are indeed selected dynamically (many observed mask IDs), so this is **not just H0 being chosen always**. The main bottleneck is the poor quality of event-level additional-note proposals and their effect on native S707 note-instance matching. Better calibration of the same incorrectly posed binary target is insufficient.

### Resulting architectural priority

Do not continue blind threshold grids or claim 80%. A next S746 must train an **event-instance risk selector** and preserve the original S708/S707 NoteOff intervals, while separately detecting genuine new fundamental note births. Validate event-level precision and all six metrics, especially previously correct note-instance offsets. This cannot be solved solely by improving a K0–K6 posterior.

## Promotion guard

- No cross-FIT/CAL label leakage, no use of withheld TEST, no changes to S708 or microphone LIVE.
- All six metrics `onset_f1,exact_k,active_exact_k,offset_accuracy,full_note_f1,frame_pitch_f1` must be measured; per-metric no-regression is mandatory.
- **Zero previously correct NoteOff or long Onset instances lost**, no onset TP regression, and independent validation after a safe CAL candidate.
- Neither GitHub Actions passing tests nor observing a higher isolated Active Exact-K score is promotion.

Action: https://github.com/Andriamarosoa/note/actions/runs/39198255358
