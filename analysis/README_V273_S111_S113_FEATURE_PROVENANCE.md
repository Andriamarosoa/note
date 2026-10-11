# S111–S113 — Authentic S4/S5 feature-layout evidence

**Status:** historical-model research only. Not an independent S80
evaluation, not a runtime for arbitrary new audio, not a model promotion.

## Verified checks

| Run | Compared | Observation |
|---|---|---|
| [S111](https://github.com/Andriamarosoa/note/actions/runs/38183264774) | S4 `owned` 527D vs S5 527D tree split feature indices | 463–466 common active features in pairwise comparisons; Jaccard approximately 0.9851–0.9957 against S5 fold 0 |
| [S112](https://github.com/Andriamarosoa/note/actions/runs/38183355117) | Per-feature median numeric split thresholds, six S4/S5 pairs | Spearman rho 0.98773–0.99512 |
| [S113](https://github.com/Andriamarosoa/note/actions/runs/38183526008) | First 111 coordinates of S4 `control` and S4 `owned` | 67–70 features split in both; median threshold rho 0.91451–0.98476 |

All calculations used the frozen tagged artifacts:
`854ada718ac9bf1556f00c96b8e1aa90402d6ab3`.

## Interpretation and missing producer

These results strongly support the **hypothesis** that S4 `owned`
and S5 use the same or closely compatible 527-coordinate layout.
S4 `control` uses 111 coordinates and appears to share a substantial
numeric prefix with S4 `owned`. These statistical associations **do
not prove** feature names, order, normalization, or a complete
111+416 construction.

Surviving checkpoints: fourteen original S4 models (the final two
S4 `owned` fold4 models are absent from the pinned archive) and four
S5 `HistGradientBoostingClassifier` models, all SHA-checked.
The S5 models expect 527 features and have a common frozen
`designs_sha256=fd4d4d32872e5d3856bf994db386e9f2976ce57a418a21b56b876bf6f6d349cd`.
No saved 527D input matrix was found in the S3 archive audit;
the exact source code constructing the 527D design is not preserved in
the pinned S4/S5 commit.

Independent external audio features already exist in
S97/S106 (57 GOAT events, waveform 42×49, summary58, flow245,
morphology490, physics294), without reading MIDI.
These are **not** a substitute for S5's missing 527D design.

## Promotion gate

Keep `freeze_local_combo` official until:

1. An authentic label-blind 527D input producer is reconstructed and
   independently replayed against archived source checksums and
   decisions (using labels only for final scoring, never inference).
2. All sixteen original S45 proposal families are made executable on
   unseen audio, and S46 and subsequent S80 gates are replayed without
   cached historical decisions or ground truth inputs.
3. A predefined, genuinely independent external dataset yields
   verified per-K, global and polyphonic Exact-K results comparing
   the frozen baseline and complete S80 under the same protocol.
4. Results, checksums and corrections/regressions are archived and
   the promotion decision is traceable.

**Promotion status:** BLOCKED. Historical S80 results alone are not
independent evidence of deployment performance.
