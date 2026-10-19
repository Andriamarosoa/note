# S742 — Starting with NOTE's weakest metric: **Active Exact-K**

Date: 2026-10-19.
Branch: `codex/note-s742-active-exactk-causal-audit`.
Baseline: **S708 CAL32 active_exact_k = 56.27624871562104%**.
Goal: strictly above 80% for Active Exact-K **and all five other official MIDI metrics**, no correct-offset or long-onset regression.
Official live/reference: **S708 unchanged**.

## Root cause established from the actual code and earlier S714 audit

`scripts/s704_joint_metrics.py` scores active Exact-K by counting **actually emitted active MIDI notes** at every native audio hop where annotated active K>0, not by comparing `network.k.argmax` to truth.

`causal_note.s708_cardinality_selection.GuidedPolicy` uses `count_confidence=1.1`, deliberately disabling `JointSelector`'s strict K0–K6 budget (since softmax maximum <=1). The guide still adds weak Onsets if acoustic gates pass and `k.max()>=guide_confidence` and `k.argmax()>currently active source count`. This explains why S741's pure neural K-combination improvements barely affect active K or complete notes.

S714 reported on **exposed DEV, diagnostic only**:
- K1 under-count: 19.36% of true K1 frames
- K2: 53.90%; K3: 58.65%; K4: 72.31%; K5: 65.03%
- low register MIDI40–52 missing 44.42% of true pitch frames
- known same-pitch-unison annotation ambiguity is under 0.5% of frames and does not explain the main gap.

## Actual changes implemented

1. `scripts/s742_activek_forensics.py`: exact S714-compatible original emitted MIDI interval active-count reconstruction, replay assertion against signed S708 six-score metric, K0–K6 truth-specific under/over, unison diagnostic, harmonic-independent frozen K proposal coverage, pre-event weak K acoustic eligibility trace. No annotation enters note generation.
2. `scripts/s742_run_activek_audit.py`: reproduce the official CAL32 on real GuitarSet and write per-track + aggregate K1–K6 evidence, replay unchanged S708 signed six metrics.
3. `src/causal_note/s742_soft_activek_decoder.py`: **shadow-only** causal count-tail rescue. Uses P(K>currently-active-count), onset/sustain/rise/release and original armed/cooldown conditions. Cannot create a note on K alone, cannot move an event backwards, cannot exceed six notes. Strong NoteOn and S707 NoteOff chain remain frozen.
4. `scripts/s742_shadow_soft_activek_cal.py`: six preregistered soft-tail thresholds applied to real GuitarSet CAL32, one event stream each, full six-score paired evaluation, nonregression + preservation of previously correct offsets/long onsets. No HELD/DEV tuning, no TEST or LIVE change.
5. `test/test_s742_activek_forensics.py`, `test/test_s742_soft_activek_decoder.py`: unit test undercount, unison, excluded K0, independent K proposal source coverage, acoustic gate behavior, strict causality, safe reference fallback, non-oracle note births.

## GitHub Actions

- [S742 K1–K6 Real GuitarSet CAL32 forensic audit](https://github.com/Andriamarosoa/note/actions/runs/39195539087).
- [S742 real MIDI soft-tail K rescue ablation](https://github.com/Andriamarosoa/note/actions/runs/39195600077).

Only mark the experiment complete after the final `S742_FINAL` and `S742_SOFT_FINAL` lines and published JSON artifacts have been checked. Unit tests and GitHub job success alone do not constitute >80%.

## Scientific next decision

If the missing pitch has no acoustic candidate, improve independent harmonic fundamental heads before adjusting routing.
If credible weak candidates exist but K suppresses them, train a FIT-only selection/gating risk model using actual emitted note events; use CAL for model selection and independent recordings for final verification.
If false additions dominate, strengthen true-fundamental identity/persistence and S707 NoteOff coupling.
Never treat an oracle correction or a single historical per-K classifier score as live MIDI accuracy.
