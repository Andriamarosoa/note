# S821 — independent NVIDIA diagnosis: K4→K5/K6 false additions

Research facts ONLY. No architecture is endorsed before independent opinions; each model receives the **same** brief and cannot access peers' replies. No CAL32/HELD/TEST. The proposed experiments must preserve MIDI256 causal 256/44100 Hz per hop (5.805ms grid, not hardware latency), zero lookahead/hold, independently trained Active-K weights, unchanged other task teachers and S708/LIVE.

## Authenticated experimental evidence

Original GuitarSet mono 44.1kHz SHA verified, 80 tracks across BN1 Funk2 Jazz1 Rock3 SS1, five 48 TRAIN /16 inner-CAL /16 outer-composition folds. 306536 audio frames MIDI256. Active Exact-K denominator **249411** frames with truth K>=1. K0 is a diagnostic only. S708 historical H0 142213/249411 = 57.0195%. S815 private triple teacher 141946/249411 = 56.9125%. S817 two temporal high-polyphony experts added but actual policy changed 0 predictions. S818 six independent ordinal models 135694/249411 = 54.4058%, especially K5 9.64% and K6 0.78%.

The S819 true 80-track fixed-policy counterfactual **max_high@0.70** on frozen S815 + S817 changes 4867 frame decisions, total 141263/249411 = 56.6386%, worse by 683 correct frames. By composition:
BN1 changed 895, corrections 248, destroyed 269, net -21.
Funk2 changed 1048, corrections 354, destroyed 346, net +8.
Jazz1 changed 515, corrections 99, destroyed 233, net -134.
Rock3 changed 626, corrections 222, destroyed 200, net +22.
SS1 changed 1783, corrections 355, destroyed 913, net **-558**.

S820 run #39559635752 verified **GT-conditioned transition matrices** on all 80 original tracks with independently SHA-verified frozen checkpoints (5 S815 and 5 S817). For SS1:
- GT true K1 total 20503, correct delta -1, changed15
- GT true K2 total 10780, delta -8, changed151
- GT true K3 total 8192, delta -163, changed295
- GT true K4 total 12170, delta **-555**, changed916. Among those GT K4 frames **597 initially correct K4 become erroneous K5** and **68 initially correct K4 become erroneous K6**.
- GT true K5 total 4839, delta +152, changed384.
- GT true K6 total 865, delta +17, changed22.
- K0 8795 unchanged (excluded from accuracy). SS1 net -558.
- SS1 most affected group SS1-100-C# accounts for -487 correct; comp texture accounts for -505.

S820 acoustic summary: feature 222 per pitch, 'sharp' blocks indexed 126+ offset; averaged 'fundamental-like minus harmonic2-like' across 49 pitches gives **0.280527** for 355 corrections and **0.278595** for 913 destroyed predictions, difference only 0.001932 in feature coordinates. These are **not** validated physical F0 and harmonic amplitudes, not proof of physical harmonics. No individual note pitch added is known from cardinality predictions. Code:
- `src/causal_note/s817_highpoly_temporal.py` `proposals`, `select`, `fit_teacher`.
- `src/causal_note/s814_activek_teacher.py` `summarize`.
- `src/causal_note/s813_framepitch_teacher.py` contains `specialist_region` showing pitchwise harmonic competition but its MODEL/OUTPUT MUST NOT be reused by the Active-K teacher. Reimplement only *independent acoustics* when warranted.
- `scripts/s820_truek_spectral_forensics.py` and `scripts/s819_s817_gate_forensics.py` authenticate original report, PCM and GT-K decisions.

## Review requested (no recommended answer)

1. Diagnose competing, FALSIFIABLE reasons why the high-polyphony candidate improves GT K5 by +152 but destroys GT K4 by -555 in SS1. Why is the same rule mildly useful in Rock3 (+22) despite similarly high K>=4 density? Distinguish acoustic hypotheses, prior/selection hypotheses and annotation/classification mechanisms.
2. Critique the S820 acoustic proxy average: where does it discard pitch-specific information? Propose independently recomputable acoustic evidence and negative controls that could tell an actual extra fundamental from existing harmonic partials, strictly causal with 256-sample emission grid.
3. Give ≥3 alternative experimental ablations of increasing cost; consider **a train-only discriminative check of whether K4→K5/6 is justified** but do not assume it will improve anything. Require inner-CAL policy choice only; outer truth only for final score.
4. Explicitly quantify why a policy that reduces undercount but raises overcount could regress; discuss K5/K6 rarity, confidence calibration, domain shift (SS1 vs Rock3), and false positives.
5. Define how to reject an unhelpful correction, no 85% claim, no lookahead/hold or misuse of held-out test, no peer-teacher parameter sharing, 6-axis no-regression, and distinguish 5.805ms grid vs real hardware latency.

Write a skeptical, evidence-focused review in French with headings FAITS, HYPOTHÈSES CONCURRENTES, CONTREPREUVES, EXPÉRIENCES DE FALSIFICATION, IMPACT MIDI256, LIMITES ET INCERTITUDES. Do not make up tested metrics or claim to have run the code or listened to audio.
