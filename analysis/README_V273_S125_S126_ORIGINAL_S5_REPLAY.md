# S125–S126 — Original S5 historical replay, residual gap and next gate

**Branch:** `codex/v273-scope-rescue` (PR #17). No independent S80
Exact-K score; no promotion. All numbers below are comparisons against
historical exposed GuitarSet fold 0 (15,952 events), NOT unseen GOAT.

## Experimental provenance

- Original SHA-pinned S5 fold0 model and original 527D model contract
  from commit `854ada718ac9bf1556f00c96b8e1aa90402d6ab3`.
- Genuine saved GuitarSet 58+42×49+46D acoustic features and frozen
  K from run 37837235723; no test annotations entered the model features.
- Three alternative unverified constructions of the missing 416D suffix;
  fixed model weights and fixed original observations.
- S123's log variant predictions were reused unchanged to avoid a
  misleading duplicate training/inference call.
- Original saved per-event S5 `raw` probabilities and SHA-256
  (`models/fold0/predictions.npz`) used **only as fidelity references**;
  archive `truth` array was never accessed by S126.
- Historic fold0 labels were accessed by S125 **only after inference**
  to compare pre-existing confusion metrics; that is development-only
  forensic research and cannot validate S80 independently.

## Results: posthoc development-fold confusion

| Fixed 527D construction | Correct K | Delta vs historical |
|---|---:|---:|
| Original archived S5 output | **13,187** | — |
| Candidate log/activation-first | 13,053 | −134 |
| Candidate amplitude/lag-first | 12,641 | −546 |
| Candidate amplitude/activation-first | 12,565 | −622 |

Run: [S125](https://github.com/Andriamarosoa/note/actions/runs/38187467369).

## Results: sample-by-sample agreement with the original S5 outputs

| Fixed 527D construction | Matching original K | Disagreements | Agreement |
|---|---:|---:|---:|
| Log/activation-first | **14,070** | 1,882 | **88.20%** |
| Amplitude/lag-first | 13,202 | 2,750 | 82.76% |
| Amplitude/activation-first | 13,158 | 2,794 | 82.48% |

For log/activation-first, original vs reconstructed posterior
probabilities have average absolute error **0.029045**, 95th-percentile
absolute error **0.178868**, and maximum absolute error **0.892753**.
**Zero of 15,952 probability vectors match within 1e−6**, so the
original numeric feature transformation is not reproduced.

For log/activation-first, among the 1,882 K disagreements:
**1,154** are on events that the original S5 predicted K2 or higher,
versus **23** on original K0 events. These are *original predictions*,
not observed ground-truth K classes.

Run: [S126](https://github.com/Andriamarosoa/note/actions/runs/38187693573).

**Interpretation:** the log/activation-first variant is a stronger
*historical-fidelity hypothesis* than the two amplitude orders. Neither
variant is proven to have the original S4-owned 527D feature semantics.
Do not select, tune or claim accuracy on new data based solely on these
historical results.

## CI deduplication and next experiment

- The 40 redundant `pull_request` Actions were rewritten in
  [771fd29](https://github.com/Andriamarosoa/note/commit/771fd297e7fa5d08e9e88956f45a9fdce8cbf610)
  to branch-scoped `push` with original file-path filters,
  `workflow_dispatch` retained, and stale-run cancellation.
- [S124](https://github.com/Andriamarosoa/note/actions/runs/38187307388)
  verified all 40; S125 and S126 code pushes each triggered only their
  matching experiment.
- The invalid S81 YAML step name was fixed and S81's preflight made
  manual-only; original preregistration is retained.
- Next: use the saved S5 original K0–K6 posteriors and acoustic inputs
  to **locate the feature-family mismatch by high-K disagreement slices**,
  then reconstruct missing statistical operations and compare replay
  on more historical folds *without using the labels as fitting targets*.
  Fully independent external WAV → original producers → S80 output
  remains the non-negotiable promotion gate.

**Official reference unchanged:** `freeze_local_combo`.

## S127 — Direction of the reconstructed count errors

[S127](https://github.com/Andriamarosoa/note/actions/runs/38187843029) completed
SUCCESS, comparing *the original S5's predicted K* with S123's log-layout
predicted K on the **same 15,952 historical events**, without reading
their true K labels.

- **1,882** original-vs-replay K disagreements.
- **1,665 undercounts** (replay K smaller) and **217 overcounts**
  (replay K greater).
- Most common transitions: original K1→replay K0 (**667**);
  K2→K1 (**335**); K3→K2 (**202**); K2→K0 (**189**);
  K3→K1 (**119**).
- **1,154** disagreements where the original S5 predicted K2+.
  Only **23** where the original S5 predicted K0.
- Mean original confidence on mismatched rows was **0.4817**;
  just **28** of the 1,882 mismatches had original confidence ≥0.8
  and **5** had confidence ≥0.9.

**Implication:** the present 416D log-layout reconstruction systematically
under-counts relative to the original frozen S5, especially on
ambiguous note-count boundaries. This is a reproducibility diagnosis,
*not* a proof that the model underestimates true K. The missing original
S4-owned feature statistics, normalizations and exact column order
still need original-source provenance or rigorous replay before any
external S80 promotion.
