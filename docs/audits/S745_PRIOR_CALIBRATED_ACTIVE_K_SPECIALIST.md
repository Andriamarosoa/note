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

## Promotion guard

- No cross-FIT/CAL label leakage, no use of withheld TEST, no changes to S708 or microphone LIVE.
- All six metrics `onset_f1,exact_k,active_exact_k,offset_accuracy,full_note_f1,frame_pitch_f1` must be measured; per-metric no-regression is mandatory.
- **Zero previously correct NoteOff or long Onset instances lost**, no onset TP regression, and independent validation after a safe CAL candidate.
- Neither GitHub Actions passing tests nor observing a higher isolated Active Exact-K score is promotion.

Action: https://github.com/Andriamarosoa/note/actions/runs/39198255358
