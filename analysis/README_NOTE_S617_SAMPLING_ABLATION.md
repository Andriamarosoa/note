# S617: Restore S613-style negative sampling (experimental, not promoted)

Date: 2026-10-16. Branch `codex/note-live-s617-negative-sampling`.
Based on experimental S615 full-loop branch. Original S613 hybrid,
S80 and live S613 references are unchanged. No new acoustic future samples.

## Hypothesis grounded in S615 real regression

The original S613 neural training sampler used:
- sample 100% of actual onset and release rows;
- 50% of labeled sustain rows;
- 8% of background/silent pitch rows.

The S615 research prototype inadvertently reduced those fractions to
30% sustain and 2.5% background. Although it inverse-weighted each
sampled row, it also imposed a 7,000 row cap per source/hop,
which could strongly underrepresent useful negative examples,
especially in high-pitched/overtone-rich guitar signals.

S617 changes ONLY the pitch training sampler:
- 100% onset/release preserved (even exceeding row cap);
- 50% sustain;
- 8% background;
- row budget now 15,000 per source/hop;
- no changes to native S613 feature schema, 49 pitch grid, MLP heads,
  FFTs, physical-time lags, causal selection or real-time behavior.

No new hand-engineered harmonic suppression, thresholds or neural
backbone inserted.

## Real same-source GuitarSet pilot (measured and complete)

GitHub Actions: https://github.com/Andriamarosoa/note/actions/runs/38866226910
Status SUCCESS. Exact same 6 train/3 dev GuitarSet recordings, original
S615 vs S617 shared MLP, same 4 K epochs and 6 pitch epochs.

| Hop | Original S615 dev F1 ±80ms | S617 dev F1 ±80ms | Difference | S615 FP -> S617 FP |
|---|---:|---:|---:|---:|
| 128 | 31.64% | **36.34%** | +4.70pp | 263 -> 259 |
| 256 | 32.17% | **36.60%** | +4.43pp | 230 -> 218 |
| 512 | 31.75% | **36.22%** | +4.47pp | 271 -> 257 |

At hop 256: true positive attacks **111 -> 127**, false positives
**230 -> 218**, grouped-onset Exact-K **13.81% -> 18.78%**.
K4 only **0/21 -> 2/21**, both old and new have zero exact
pitch sets for the 21 K4 groups. K3 **6/45 -> 8/45**.
This confirms an improvement on the limited development pilot,
NOT a validated improvement in generalization, live latency or
historical Exact-K.

## Next validation: full corpus and comparison

Workflow launched:
https://github.com/Andriamarosoa/note/actions/runs/38866726469
Uses ALL 53 train and 17 dev originally pinned GuitarSet+URMP
sources. Full 24 count + 28 pitch epochs, the historical negative
sampling fractions and up to 15,000 rows/source/hop. Produces a
new *experimental* native neural 256-hop checkpoint, and
`COMPARISON_VS_FROZEN_S613_DEV.json` with F1, K1..K6 grouped
onset Exact-K and separately K-active frame confusion.

The original S613 frozen hybrid vs previous S615 full neural-only
on the exact 17 dev compositions scored **56.39% vs 47.55%**.
S617 must beat those **on the SAME 17 dev compositions** before
independent new-track evaluation is even considered.

S613's original output was a neural + ONNX tree hybrid. S617's
single MLP still differs architecturally; a separate
`s615_hybrid_consensus_loop.py` test (run 38865434173) checks
reuse of the frozen tree component.

Do NOT merge or promote on these development-only results.
The only valid promotion requires unseen compositions, true note
K2/K3/K4 audits, realistic iRig microphone + FluidSynth
round-trip latency, and no intolerable regressions.
