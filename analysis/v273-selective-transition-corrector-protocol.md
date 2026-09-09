# V27.3 — selective polyphonic transition corrector

## Frozen reference and negative result

V27.1 `poly_rescue` remains the reference with **42.5806% polyphonic exact-K**. V27.2 trained a new conditional K=2..6 head but its wholesale replacement policy fell to **41.0488%** and is not promoted.

The frozen V27.2 specialist is nevertheless complementary: an outer-set oracle choosing independently between V27.1 and that specialist reaches 65.3122% polyphonic exact-K. This oracle is descriptive, not deployable, and its outer labels cannot be used by the V27.3 rule.

## Hypothesis

V27.2 failed because it replaced every V27.1 prediction K>=2. V27.3 keeps V27.1 by default and permits only the four dominant adjacent transition proposals observed in the published V27.1 error audit:

- `2 -> 3`;
- `3 -> 2`;
- `3 -> 4`;
- `4 -> 3`.

The proposal is the argmax of the frozen V27.2 uniform conditional distribution. Its evidence against the V27.1 count is the pairwise margin:

`P_v272(proposed K) - P_v272(V27.1 K)`.

No other transition can change, regardless of confidence.

## Strict inner calibration

One threshold is selected independently for each directed transition and each outer fold. The calibration reconstructs, inside the outer-training partition:

1. the V27.1 base from strict V10.4 inner shards plus replayed V26 uniform/weighted probes;
2. the V27.2 uniform conditional probe on the same `meta_fit -> meta_val` partition;
3. all observed margin breakpoints for each eligible direction, plus an explicit `no override` option.

For each direction the fixed selection order is:

1. maximize the net number of exact rows (`corrections - regressions`);
2. minimize the number of activated overrides;
3. prefer the higher threshold.

The four row sets are disjoint, so optimizing their exact-row deltas separately also maximizes their combined inner exact-K. Event F1 is never used for threshold selection.

The V26 and V27.2 probe epoch budgets are copied from their archived, inner-only source reports and each probe is refit from scratch for exactly that many epochs. This fixed-budget replay avoids making the protocol depend on hardware-sensitive early-stopping trajectories between CPU runners. The newly observed validation histories and exact counts are recorded as diagnostics; non-finite histories or an invalid source budget still fail the fold before outer evaluation.

## Execution correction before outer results

Initial run `34293193376` was superseded after fold 1 stopped before threshold calibration and outer evaluation: a strict early-stopping replay guard observed V26 weighted epoch 9 instead of the archived epoch 12. No V27.3 outer score from that run was inspected. The correction above freezes the already archived inner-only epoch budgets and does not change the transition set, confidence score, threshold objective, outer data, primary metric, or promotion rule. Outputs from the superseded run are not eligible for the final result.

## Untouched outer evaluation

After thresholds are frozen, V27.1 and the V27.2 probability vector are loaded from the immutable V27.2 fold artifact. Outer labels are used only for the final report.

Because only predictions 2, 3 or 4 may be exchanged for another polyphonic class, correct K=0/1 rows are immutable. Consequently, the change in global exact rows must equal the change in polyphonic exact rows.

Primary metric: aggregate polyphonic exact-K. Secondary metrics: global exact-K, F1 at 50 ms, TP/FP/FN, per-fold and per-direction corrections/regressions.

Promote V27.3 only if aggregate polyphonic exact-K is strictly greater than the V27.1 reference, 42.5806%. A fold-level regression is reported but is not an additional veto. Historical validation and Locked12 remain unindexed and unevaluated.

V27.1, V27.2 and their outer error analyses were already observed before this protocol. V27.3 is therefore a nested development experiment, not a new independent validation claim.
