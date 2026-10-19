# S743–S744 — Active Exact-K iterative causal fundamental rescue
Date: 2026-10-19. Goal: >80% on ALL SIX complete MIDI scores without regression; **no distillation before 80%**.

## S743 finished real GuitarSet CAL32 benchmark
[GitHub Actions 39196692946](https://github.com/Andriamarosoa/note/actions/runs/39196692946) **success**. The 12 physical multi-fundamental and K-tail variants all completed; all were rejected under strict six-metric and paired-offset guards.

| Candidate | Additional pitch NoteOns | Active Exact-K | Onset F1 | Offset accuracy | Full-note F1 | Promotion? |
|---|---:|---:|---:|---:|---:|---|
| S708 signed baseline | 0 | 56.2762% | 74.2639% | 77.3292% | 57.1319% | official |
| adaptive, tail .30, sustain .30 | 1,873 | 62.2717% | 69.7023% | 75.7875% | 52.5248% | NO |
| adaptive, tail .30, sustain .55 | 636 | **62.3435%** | 71.5832% | 76.7302% | 54.5974% | **NO** |
| adaptive, tail .55, sustain .30 | 1,624 | 62.1051% | 70.5273% | 76.0039% | 53.3123% | NO |
| adaptive, tail .55, sustain .55 | 536 | 61.5305% | 71.8269% | 76.6824% | 54.7837% | NO |
| physical, tail .30, sustain .30 | 1,682 | 61.0417% | 71.2307% | 75.9300% | 53.7777% | NO |
| physical, tail .30, sustain .55 | 485 | 60.6516% | 72.8615% | 76.7123% | 55.5596% | NO |
| physical, tail .55, sustain .30 | 1,478 | 60.9599% | 71.7426% | 76.0591% | 54.2707% | NO |
| physical, tail .55, sustain .55 | 414 | 60.1199% | 72.9068% | 76.6723% | 55.6002% | NO |

The four conservative variants (also rejected) span Active K 60.12–61.04%. S743 proves there is **acoustic recall capacity** (+6.07 percentage points in Active Exact-K), but naive independently boosted acoustic births introduce numerous incorrect NoteOns and regress Onset and Offset substantially. Selecting an isolated higher Active K score is NOT a valid promotion.

The problem has moved from a lack of proposals to **precision and downstream Offset preservation**. Unsupervised thresholds cannot reliably distinguish true additional fundamentals from harmonic children.

## S744 experimental supervised FIT-only learning (new)

Research branch `codex/note-s744-fit-trained-activek-birth-specialist` creates a true additional **specialist head**, not a Student or compressed teacher. Train two independent supervised HistGradientBoosting depth3/depth5 classifiers ONLY on FIT80 compositions, on frozen S708 temporal acoustic features, real harmonic rises, per-pitch auditory-scene features and previously emitted source state. Additional NoteOns require physical evidence + K tail; H0 and all six official S708 probabilities remain frozen.

A composition-disjoint CAL32 run evaluates two classifier variants × two K-tail levels × two specialist posterior thresholds = **eight independent loops** through original S707 NoteOff and all six official complete-MIDI score guards. Correct previously emitted offsets and long onsets must be preserved. Neither a trained model alone nor a successful CI job establishes target accuracy or LIVE eligibility.

The signed reference on this branch remains **S708 (Active-K 56.2762%)** until a safe CAL candidate survives further independently validated tests.
