# V17.7 post-audit — candidate-centric decoder

Branch: `v177-post-audit`
Source training run: `33893583081`
Audit run: `33955089270`
Outer-clean rows: 76,768
Locked12: untouched

## Headline

- V17.7 direct F1: **71.2580023481%**
- Precision: **69.8302435636%**
- Recall: **72.7453640886%**
- pred/ref: **1.0417458164**
- V17.7 count-only F1: **71.2447112509%**
- direct - count-only: **+0.0132910972 pp**
- V17.7 - V17.3: **-7.9491257215 pp**
- V17.7 - V10.4: **-9.1262313113 pp**

Direct candidate realization therefore contributes essentially nothing over the same learned K with the frozen historical ranking. The failure is upstream in learned objectness/cardinality.

## Cardinality failure

Truth mean K is **0.5490699250** while mean soft candidate-object mass is **1.2932756281**, an excess of about **+135.5%**.

Per true K:

| K | rows | valid candidates mean | soft objects | hard objects | exact K | corr(valid, soft) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 51,956 | 5.727 | 0.164 | 0.045 | 97.14% | 0.383 |
| 1 | 15,411 | 19.732 | 1.483 | 0.985 | 54.83% | 0.329 |
| 2 | 4,279 | 16.238 | 3.850 | 3.084 | 17.90% | 0.533 |
| 3 | 2,952 | 20.676 | 8.337 | 8.256 | 9.86% | 0.724 |
| 4 | 1,628 | 23.588 | 11.573 | 12.383 | 6.94% | 0.788 |
| 5 | 438 | 23.932 | 14.357 | 16.333 | 8.68% | 0.859 |
| 6 | 104 | 23.731 | 16.373 | 18.875 | 77.88% | 0.925 |

The K6 exact score is misleading because runtime caps the number of active objects at six. Before that cap, a true-K6 row activates **18.875 candidates on average**.

The number of predicted objects becomes increasingly determined by how many candidate hypotheses happen to exist, rather than by the number of births. This dependence becomes very strong for K>=3.

Using the exact Poisson-binomial MAP instead of threshold counting does not rescue cardinality: its global exact-K accuracy is lower than the current threshold decode, and poly exact-K remains similarly poor. The dominant problem is therefore the learned Bernoulli field, not merely the 0.5 decoding rule.

## Dominant confound: six-slot mass preservation is not optimization-equivalent

V17.7 transplanted the total positive/negative coefficient mass from the six-slot V17.3 formulation into a variable field of up to 48 candidate Bernoullis. The old negative mass is divided across all unmatched valid candidates.

Median per-candidate negative weights versus positive weights:

| K | positive weight mean | negative weight median | median positive/negative ratio |
|---:|---:|---:|---:|
| 1 | 1.265 | 0.2484 | 4.86x |
| 2 | 1.744 | 0.3044 | 5.21x |
| 3 | 2.217 | 0.1642 | 12.75x |
| 4 | 2.959 | 0.1008 | 28.95x |
| 5 | 4.472 | 0.0579 | 76.66x |
| 6 | 5.590 | **0.0000** | infinite |

At K6 the inherited null mass is exactly zero because the original six-slot system has no negative slot when all six slots are occupied. In the 48-candidate system, however, a K6 row has about 23.7 valid candidate hypotheses and many of them are unmatched. Those unmatched candidates therefore receive **no BCE negative presence penalty at all**; only the 0.35 Poisson-binomial term can suppress them.

The same failure grows progressively at K3-K5: total null mass may be algebraically conserved, but it is diluted over many more hypotheses, causing negative-gradient starvation per candidate.

There is also a second control-boundary issue: **522 positive rows have C==K<6**. They are candidate-set feasible, but there is no unmatched candidate token available to carry the legacy V17.3 null mass. Therefore the claim of exact coefficient-mass preservation cannot hold on those rows. The original workflow protocol assertion correctly exposed this discrepancy.

The 180 C<K rows remain the genuine candidate-representation ceiling and were correctly masked only from the impossible one-to-one set loss while remaining in outer evaluation.

## Realization is not the bottleneck

Across all five folds:

- candidate top-1 realization diagnostic: **1.0 on every fold**
- time MAE: approximately **0.25-0.34 ms**, mean below 0.35 ms

Once an object is selected, candidate identity and birth time are effectively solved in this architecture. The main failure is deciding how many candidate hypotheses should survive.

## Architectural interpretation

V17.7 as trained is rejected, but it does **not** cleanly falsify candidate-centric decoding itself.

What it cleanly falsifies is:

> independent Bernoulli objectness over raw candidates + V17.3 six-slot null-mass transplant + weak global Poisson-binomial correction.

Raw candidates are hypotheses inside the same 40 ms causal group, not guaranteed distinct birth objects. Treating every raw candidate as an independent Bernoulli also creates a multiple-hypothesis opportunity: as the candidate pool grows, each individual unmatched hypothesis receives less negative pressure while any good hypothesis can become the DP winner for a truth object.

## Recommended next architecture

Do not tune the 0.5 threshold and do not do a V17.7.x weight sweep.

The next major experiment should introduce an explicit **candidate competition / event-proposal layer** before object counting:

`raw candidates -> contextual competition / grouping -> event proposals -> set/cardinality decision`

rather than:

`raw candidates -> independent object Bernoullis`.

A strong V18 design would use candidate graph or attention edges to model mutually redundant hypotheses, form a small evidence-derived proposal set without fixed anonymous slot identities, and apply objectness/count supervision only after that competition. This preserves the useful candidate-centric identity/time path while preventing candidate multiplicity from directly becoming predicted event multiplicity.
