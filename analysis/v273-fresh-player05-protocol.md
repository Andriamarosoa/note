# Fresh independent validation — GuitarSet player 05

## Frozen question

Does the already-selected residual corrector `base_geom_attack @ p(K2)>=0.59`
improve Exact-K on a performer that was never available to any development,
training, internal-fold, or exposed-fold experiment?

This is a one-shot unseen-performer validation. Player `05` is intentionally
excluded by the historical GuitarSet loader (`ALLOWED_PLAYERS=00..04`) both in
the current branch and in the source commit that produced the frozen V8.4/V8.6/
V8.7/V8.8 proposal stack.

## Frozen model and policy

No parameter may be selected from player 05.

- robust count base: `freeze_local_combo + candidate_hidden1 [42,52,61,64]`
- B_low reconstruction: same pooled-internal algorithm and seeds used for the
  prior outer evaluation
- residual classifier: StandardScaler + balanced LogisticRegression(C=1,
  random_state=28431)
- features:
  1. best_pair_residual_ratio
  2. best_triplet_residual_ratio
  3. geom_harmonic_relation_min_error
  4. geom_span_cents
  5. amp3_over_amp2
  6. nov_onset_contrast_norm_median
  7. exc_unique_attack_fraction_min
  8. joint_unique_x_post_median
  9. weak_unique_fraction
  10. weak_unique_post1_norm
- action: only B_low rows whose robust-base prediction is K=3
- correction: 3 -> 2 when enriched p(K2) >= 0.59

These choices are frozen from internal run 37547761255, before any player-05
result is computed.

## Holdout scope

All GuitarSet tracks belonging to performer `05` are used exactly once.
Expected inventory: 48 tracks (24 compositions × comp/solo).

The holdout shares the GuitarSet recording protocol and compositions with the
development corpus, so this is **performer-independent**, not
composition-independent. It is nevertheless unseen audio from a performer
hard-excluded from the complete historical source/training pipeline.

## Leakage guard

Runtime features and all predictions are produced before player-05 annotations
are read for scoring. A prediction SHA-256 is frozen first.

After predictions are frozen, annotations are opened only to assign true note
onsets to the already-fixed candidate clusters and compute true K. The script
then reruns the labelled cluster builder only as an audit and requires its
runtime input arrays to equal the label-free arrays.

No result from player 05 may be used to change:
- features,
- threshold,
- B_low clustering,
- residual training,
- count-network weights,
- proposal-stack weights.

If anything is changed after seeing this result, player 05 becomes development
data and cannot be called independent again.

## Primary metrics and pass rule

Report:
- Exact-K global,
- Exact-K polyphonic,
- per-K Exact-K for K=0..6,
- corrections, regressions, and wrong-to-wrong changes,
- B_low/base-K3 candidate coverage.

Primary comparison: enriched corrector versus the frozen robust base.

Predeclared pass:
1. paired Exact-K corrections > regressions, and
2. polyphonic Exact-K is strictly higher than the robust base.

The historical fold-3 result is shown only as context and is not used for any
selection.
