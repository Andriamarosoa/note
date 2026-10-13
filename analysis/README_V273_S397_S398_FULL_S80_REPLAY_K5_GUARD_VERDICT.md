# S397–S398 — Full downstream replay of S70 5→4 guard (13 October 2026)

**Status: successfully executed, positive GuitarSet development replay, NOT externally validated and NOT promoted.** S80 remains the unchanged working reference.

## Problem and intervention

S395 located **202 of 224** true-K5→S80-K4 first losses at S70, and 22 at S69, in the exposed GuitarSet source. A proposed **label-blind** safeguard at the original S70 boundary cancels a S70 action when `s69_polymax_K==5 && s70_proposed_poly_K==4 && proposed_poly_vote_accepted`. It makes S70 fold vote fields internally consistent; does not alter any other original output or model weights.

## Real inference replay, not a posthoc flip

[GitHub Actions S397–S398 run #38352988801](https://github.com/Andriamarosoa/note/actions/runs/38352988801) completed successfully. 59,309 original GuitarSet rows, true K=5 on 355. The guarded original S70 source was regenerated *without truth*, then original eight trained S71/S72 checkpoints, four trained S79 K1 critics, and original S80 K0/K2 formula actually reran to produce an SHA-sealed candidate `f5a313366a161d9775ce8452df766e4178604dc8d6e002d781d24e70ee4b967f`. **Only AFTER the output and report were SHA-frozen** did S398 load original truth.

* Exactly **203** original S70 K5→K4 actions were suppressed.
* Full-chain replay changed **203** final S80 K4 predictions into K5.
* Of those, **202 were true K5**, one was a true K1 already misclassified at baseline. Thus **202 corrections / 0 regressions / 1 changed both wrong**.
* Outputs K0,K1,K2,K3,K4,K6 remain unchanged by this development replay; no trained model checkpoint is modified.

## Paired GuitarSet development results

| Metric | Frozen S80 S272 | Full-replayed S80+guard | Δ |
|---|---:|---:|---:|
| K0–K6 global correct | 53,327/59,309 (89.9138%) | 53,529/59,309 (**90.2544%**) | **+202 (+0.3406 pp)** |
| K2–K6 poly correct | 5,852/7,385 (79.2417%) | 6,054/7,385 (**81.9770%**) | **+202 (+2.7353 pp)** |
| True K5 correct | 99/355 (27.8873%) | 301/355 (**84.7887%**) | **+202 (+56.9014 pp)** |

New changes by original fold (NOT independent from the underlying S272 4-branch predictor): fold 0 **+75**, fold 1 **+33**, fold 2 **+49**, fold 4 **+45**. The original YourMT3+ scores remain 51,328/59,309 global and 4,028/7,385 polyphonic on the same GuitarSet events. **No independently demonstrated win over YourMT3+ is claimed.**

## Limits prohibiting promotion

1. **The guard was chosen after examining the exact same GuitarSet error population.** Consequently, the apparently perfect 202/202 correction precision is at high risk of selection bias; it must not be reported as an unseen-music gain.
2. The reconstruction S272/S273 itself applies three of four trained folds to audio that can overlap the original training data, so its 89.91% is not a clean OOF baseline. The guard is applied to **that same fold-neutral, training-overlap architecture**.
3. The label-free S397 replay and post-freeze S398 score demonstrate *that the code path works*, but **SHA freezing before loading labels within a reused dataset does not make the evaluation independent**.
4. Existing S366 IDMT 36-WAV benchmark has **only 1 true K5 out of 1,444 native events**, while K6 has zero. It therefore cannot establish whether a K5-focused guard generalizes or creates K4→K5 regressions on unseen chords. These compositions have previously been inspected; exact musical disjointness and K0 label exhaustiveness are unproven.

## Follow-up gate

Find genuinely independent, previously unused music with **substantial true-K4 and true-K5** examples and per-note annotations; pin cohort and S397 guard code/checkpoint SHAs **before** opening annotations. Apply original S70 and guard as separate controls, run all S71→S80 steps, and report **per-K corrections/regressions by composition**. Reject guard if K4 regressions overwhelm K5 recovery. Keep original S80 unchanged unless independent gains are robust.

Artifacts: [Run #38352988801](https://github.com/Andriamarosoa/note/actions/runs/38352988801) (artifact `s397-s398-S70-suppression-REAL-full-S80-replay-and-scoring-NOPROMOTION`). Sources `scripts/s397_replay_original_S70_K5_guard_full_S80_GuitarSet.py` and `scripts/s398_score_S397_full_S80_K5_guard_guitarset_after_SHA.py`.
