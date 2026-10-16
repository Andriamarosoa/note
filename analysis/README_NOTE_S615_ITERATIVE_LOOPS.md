# NOTE S615 — iterative full-corpus training and regression investigations

2026-10-16, branch `codex/note-live-s615-full-loops` (reference S613 unchanged).

## Completed loop 1 — FULL 53/17 GuitarSet+URMP train/dev

Run: https://github.com/Andriamarosoa/note/actions/runs/38863904669 — **SUCCESS**.
- 53 train and 17 dev composition-separated recordings, original pinned
  sources/annotations, 0 test labels used.
- 24 count and 28 pitch epochs on all available NOTE 256-hop data.
- Frozen S613 native hybrid vs new S615 neural-only on IDENTICAL 17 dev
  files: F1 NoteOn ±80ms **56.3923% vs 47.5483%**, regression **−8.8440 pp**.
- Original S613 model on 17 dev had grouped attack Exact-K
  K1 **446/834**, K2 **75/255**, K3 **52/183**, K4 **5/44**.
- Trained S615 hop256 had K1 **389/834**, K2 **44/255**,
  K3 **21/183**, K4 **8/44**. A small K4 improvement does NOT
  compensate K1–K3 regressions and increased harmonic overcount.
- New S615 F1 trial: TP1440, FP2538, FN639.
- This is not the historical S613 held-out F1 69.03% or S80 Exact-K.
- No promotion. Full source run archived trained neural checkpoint and
  report as `s615-full-hop256-candidate-and-dev-report`.

## Completed loop 2 — DEV-only selector grid for old 6/3 real pilot

Run: https://github.com/Andriamarosoa/note/actions/runs/38864452243
— **SUCCESS**. 108 selector configs at each hop, separately accounting
for NoteOn TP/FP/FN and onset grouped Exact-K.
- Shared model trained on pilot, original dev F1:
  hop128 31.64%, hop256 32.17%, hop512 31.75%.
- DEV-tuned best: hop128 **32.18%**, hop256 **33.37%**,
  hop512 **34.31%**.
- Best hop512 counted only **1/21** K4 groups correctly,
  so improvement is inadequate; dev selection is subject to overfit.
- Do NOT promote optimized thresholds or compare this pilot's F1 to
  the FULL 17-track dev score.

## Completed loop 3 — physical-time flux & short-RMS ablation

Dedicated branch: `codex/note-live-s616-hop-normalized`.
Run: https://github.com/Andriamarosoa/note/actions/runs/38864891583
— **SUCCESS**.
- Same pinned real 6 train / 3 dev as the earlier pilot, same MLP and
  4 K / 6 pitch epochs.
- Normalize positive log-spectral flux by 256/hop and compute short RMS
  over actual hop instead of fixed 256; 256 remains byte-identical.
- Old S615 -> S616 dev F1 ±80ms:
  hop128 31.64% -> **30.88%** (−0.75pp);
  hop256 32.17% -> **30.52%** (−1.66pp);
  hop512 31.75% -> **30.93%** (−0.83pp).
- This variant is **rejected as a performance reference**; do not
  automatically merge changed time normalization into official NOTE.
  The 256 differences arise from a new model fit, not change to frozen
  256-sample feature extraction.

## Loop 4 — original S613 hybrid expert restoration

Important confound: original S613 checkpoint is a **hybrid native
neural + ONNX tree**, while the newly retrained S615 candidate is
**neural-only**. Thus raw S613 vs S615 discrepancy does not by itself
isolate hop effects. `scripts/s615_hybrid_consensus_loop.py`
preserves original S613 native ONNX trees and uses S615 full pitch neural
weights, ablates geometric tree weights, original vs new K head, and
frozen vs retrained selectors on the same 17 dev recordings.

Run:
https://github.com/Andriamarosoa/note/actions/runs/38865434173
Read the actual run conclusion and artifacts before claiming improvement.
All mixture tuning is DEV-only, not an independent promotion result.

## Protocol and remaining gates

- Full train/dev claims require original pinned GuitarSet+URMP source groups,
  NOT 3-clip pilot. No leakage from held-out test annotations.
- Exact-K here refers to grouped new onsets within 30ms, **not** the
  historical S80 polyphonic K score or active-K head classification.
- Keep K1/K2/K3/K4/K5/K6 confusion and genuine octave notes, plus
  separately score correct pitch sets. Respect real microphone/FluidSynth
  latency and 256 original bitwise parity.
- A dev-only gain does not warrant promoting any candidate: validate on
  fresh unseen compositions before replacing the S613 reference.
