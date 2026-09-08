# V27.2 — conditional polyphonic count protocol

## Motivation

V27.1 is the promoted reference at **42.5806% polyphonic exact-K**. Its row-level error audit shows that the remaining failures are no longer dominated only by `K<=1` misses: the largest transitions are `3->2`, `2->3`, `3->4`, `4->3`, plus residual `2->0/1` misses.

An oracle over the existing V10.4, V26-uniform and V26-weighted count predictions reaches only **56.0047%** poly exact-K, so another router/threshold over those experts cannot approach the 80% target.

V27.2 therefore tests a genuinely new count signal: train a fresh head **only on true polyphonic rows** to discriminate `K in {2,3,4,5,6}` without spending capacity on K=0 or K=1.

## Frozen reference

Reference: V27.1 `poly_rescue`, run `34270082405`, source commit `aed78122c5d85079c84b597e78331ba080f8b4b3`.

Reference aggregate metrics:

- poly exact-K: **42.5805765%** (4,003 / 9,401)
- global exact-K: **81.4870779%**
- event F1 @ 50 ms: **79.7481733%**
- TP / FP / FN: **34,708 / 8,116 / 9,512**

## Model treatment

Use the same acoustic/candidate inputs and fresh count encoder family as V25/V26, but replace the seven-way `K=0..6` output with a five-way softmax for conditional polyphonic count:

`P(K | K>=2), K in {2,3,4,5,6}`.

Only true-polyphonic rows (`K>=2`) participate in this head's training or validation loss.

No pretrained V24/V26 weights are reused for fitting. The V27.1 outer count is frozen and used only as the reference/runtime router.

## Inner selection

For each untouched outer fold, use the existing nested partition:

- `meta_fit_idx`: inner fit
- `meta_val_idx`: inner validation
- `final_fit_idx`: all outer-training rows
- `outer_idx`: untouched outer fold

Filter each fit/validation partition to true `K>=2` rows.

Two training policies are evaluated **inside the outer-training partition only**:

1. `uniform`: one sample weight per poly row;
2. `weighted`: inverse-square-root weights over conditional classes K=2..6, computed from the inner fit partition only, clipped and normalized analogously to V26.

For each arm:

- select epoch by its own inner validation NLL with early stopping;
- record inner conditional exact-K.

Choose the arm per outer fold by the following frozen order:

1. maximize inner conditional exact-K;
2. minimize inner validation NLL at the selected epoch;
3. prefer `uniform` on an exact tie.

Then refit the selected arm from scratch on all true-poly rows in `final_fit_idx` for the selected number of epochs.

Outer labels are never used for arm or epoch selection.

## Runtime fusion

Let `base` be the frozen V27.1 prediction and `poly_head` the V27.2 conditional argmax in K=2..6.

Apply exactly:

- if `base <= 1`: keep V27.1 unchanged;
- if `base >= 2`: replace with `poly_head`.

There is **no confidence threshold** and no outer tuning.

This treatment directly attacks polyphonic misclassification such as `2<->3`, `3<->4`, `4->3`, while leaving the already-established low-K rescue behavior untouched.

A useful structural property follows: rows whose true K is 0 or 1 and whose V27.1 prediction is already >=2 are wrong before treatment and remain wrong after treatment; rows with V27.1 <=1 never change. Therefore any change in global exact-row count comes only from true-polyphonic rows.

## Metrics

Primary:

- aggregate outer **polyphonic exact-K**.

Secondary:

- global exact-K;
- per-true-K exact/under/over rates;
- full K confusion matrix;
- event F1 @ 50 ms and TP/FP/FN using the same frozen candidate realization path;
- per-fold selected arm/epochs and transition counts from V27.1.

## Promotion rule

Promote V27.2 only if aggregate untouched-outer polyphonic exact-K is **strictly greater than 42.5805765%**.

Global exact-K and event F1 are reported as secondary trade-offs and are not vetoes, consistent with the current project priority.

## Interpretation

A strong gain would establish that the seven-way all-row objective was suppressing fine polyphonic count discrimination and would justify a later extension that also learns a dedicated poly/non-poly gate for the residual V27.1 `K<=1` misses.

A weak/null gain would show that simply removing K=0/1 from the objective is insufficient, pushing the next hypothesis toward new temporal/acoustic features rather than further loss/routing changes.

Historical validation and Locked12 remain out of scope.