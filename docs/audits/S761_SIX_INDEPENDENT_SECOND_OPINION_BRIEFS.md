# NOTE S761 — Six independent second-opinion briefs

Date: 2026-10-19. Status: proposed review package, **not** six started ChatGPT conversations and **not** six agent results.

## Shared factual context (send to each reviewer; do NOT reveal other reviewers' conclusions)

Repository: https://github.com/Andriamarosoa/note
Research branch: https://github.com/Andriamarosoa/note/tree/codex/note-s761-fit-dualquality-birth-routing
S761 run: https://github.com/Andriamarosoa/note/actions/runs/39221050543
S760 run: https://github.com/Andriamarosoa/note/actions/runs/39218925531
S753 head-vs-decoder audit: https://github.com/Andriamarosoa/note/actions/runs/39209836870

Frozen **S708/S707** GuitarSet CAL32 reference (32 files, composition-disjoint from FIT80):
- Onset F1 0.7426386233
- Exact-K 0.6204554092
- Active Exact-K 0.5627624872
- Offset accuracy 0.7732919255
- Full Note F1 0.5713193117
- Frame Pitch F1 0.7646263537

The **goal** is strictly >0.80 on all six metrics, not to show a K-only improvement by moving losses into Onset, Offsets or missing the same real notes. Avoid future information, backdating, TEST/HELD tuning, source-event modification and optimistic metric shortcuts. Research is strictly causal live-compatible. Preserve existing S708 reference and protected S707 NoteOff/NoteOn, propose an independent branch and test before any promotion. Prioritize real audio-to-note detection, not MIDI file export.

S761 finished **successfully as a workflow**, but produced **no eligible improving candidate**. The four S755 Exact-K experts generated 238 FIT80 candidate rows (49 positive Onsets, 37 correct complete notes), and 12 CAL32 full-MIDI tests found no model with six-score non-regression and no event losses. Best Active Exact-K only 0.5627924144 and this candidate regressed Onset/Full Note. S760 also failed safe promotion. Pass/fail here is a result of metric review, not infrastructure failure.

Key code:
- src/causal_note/s708_cardinality_selection.py
- src/causal_note/s746_instance_guard.py
- scripts/s704_joint_metrics.py
- scripts/s753_khead_decoder_anatomy.py
- scripts/s755_pitch_coherent_k_fit_cal32.py
- scripts/s760_causal_confirmation_delay.py
- scripts/s761_fit_dualquality_birth_routing.py
- test/test_s761_fit_dualquality_birth_routing.py

S753 on 31,568 S708 *four-hop decision frames*: among emitted undercounts, **48.48% K2, 52.31% K3, and 47.39% K4** were cases where the K head already predicted enough notes but MIDI emission undercounted. K5 and K6 are heavily underpredicted by the head itself. This is diagnostic, not proof of causal attribution. Full S704 scoring is on 256-sample frames, so beware confusing these cadences.

## Independent review 1 — Exact-K and decoder separation

Act as a skeptical polyphonic-audio architect. Read the code and S753 audit. Independently explain K0–K6 undercount sources, separate posterior K-head errors, per-pitch evidence failures, and JointSelector/GuidedSelector state constraints. Verify denominator and causality. Do not assume that maximizing head classification accuracy helps emitted MIDI.

Produce: (1) code-backed failure map with exact files/functions, (2) three testable causal alternatives, (3) one first minimal experiment targeting K2–K4 when head predicts enough but emission is short, (4) a separate K5–K6 plan, (5) metric invariants and expected failure signals. No invented gains.

## Independent review 2 — Harmonics, Onset, repeated pitch and acoustic novelty

Act as a guitar DSP / onset / polyphonic transcription researcher. Read S708, S731, S755 and S761 feature pathways. Challenge whether one note's partials, octave leakage, common attack or timbre changes cause incorrect new-note proposals and whether missing notes can be exposed with causal harmonic-source subtraction, independent fundamental evidence, note-onset novelty and fragment-specific routing. A candidate must be truly new, with onset within +/-80ms.

Produce: pitch-by-pitch candidate taxonomy (harmonic shadow/true overlap/new strike/old-note release), independent tests on GuitarSet, minimum-latency features, a learned selection plan and falsifiable ablations. Do not use future PCM or ground-truth at inference.

## Independent review 3 — Offset, note identity, matching, signed stream protection

Act as a strict MIDI event-state and statistical matching specialist. Read s746_instance_guard.py, s760_causal_confirmation_delay.py, scripts/s704_joint_metrics.py and the S758/S759/S760 evidence. Look for hidden interactions between supplemental NoteOn, signed NoteOn preemption, synthetic NoteOff, Hungarian onset match, same-pitch retriggers, clock ties and protected original notes. Why can a fully preserved native event sequence still lose a correct Offset match?

Produce: a deterministic time-axis counterexample, a formal invariant for native note pairs and matching, candidate scoring requirements, three focused unit tests, and a design that preserves existing matching while allowing truly positive new events without oracle access. Distinguish metric artifacts from true transcription quality.

## Independent review 4 — Calibration, FIT80 data and learning failure

Act as an ML generalization and model calibration auditor. Audit S761's independent heads: 238 candidate rows / 49 onset-positive / 37 full-positive, four expert proposals with shared events, nonindependent samples, logistic class_weight=balanced, min(p_on,p_complete), thresholds 0.50/0.70/0.85, five FIT compositions and the FIT/CAL boundary. Evaluate calibration, train/eval mismatch, event-level labels, duplicate proposals, one-class fallback, distribution shift and measured uncertainty. Do not peek at CAL to retrain features or hyperparameters.

Produce: proven issues vs hypotheses, composition-grouped OOF calibration protocol on FIT80 only, candidate dedup/labeling plan, expected reliability diagnostics and one feasible next training loop. No fabricated AUC or precision.

## Independent review 5 — Live causality, latency and deployment architecture

Act as a hard real-time audio/MIDI systems engineer. Audit what "strictly causal" means at each stage: spectral window, four-hop decision blocks (4*256 samples @ 44,100Hz ~23.22ms), S760 one-block confirmation, source-shadow priority and S707 Off scheduling. Distinguish algorithmic delay from soundcard driver latency and offline replay that incorrectly simulates causality. Do not claim the new S761 path is wired into microphone LIVE.

Produce: block-level causality proof obligations, latency budget without inventing hardware figures, race / queue / tie tests, and a guarded deploy plan only *after* score validation. Preserve the existing LIVE baseline.

## Independent review 6 — Evaluation adversary and experiment prioritization

Act as an independent reproducibility reviewer, trying to falsify every improvement claim. Inspect S704 metric formula and S753 cadence. Confirm FIT80/CAL32 composition separation, source SHA and S707 policy, no HELD/TEST access, the all-six >80 criterion, per-K scores, registration effects, repeated notes, false positives, matching preemption, and cross-source validation (e.g., IDMT as a separately documented transfer check, not silent tuning). Confirm no unearned gains from changing matching rules.

Produce: a pre-registered evaluation plan, minimal sanity and adversarial tests, stop/go criteria, confidence intervals with composition/recording grouping, and an evidence-backed ranked list of experiments.

## Review protocol

Send exactly the shared context plus **one review role** to each separate chat/reviewer. Request evidence links and explicit uncertainty, not simulated agreement. Collect reports without sharing conclusions until all reviews are returned. Synthesize conflicts, select one falsifiable modification, implement on a new non-production research branch, run FIT-only training and complete CAL32 MIDI evaluations. Maintain the six-score regression gate, signed original events, strict causality, and no deployment/promotion without verified evidence.

**Note:** This document facilitates genuine external second opinions. It is not a substitute for running independent models, and no separate ChatGPT sessions have been created by adding it to GitHub.
