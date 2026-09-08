# V24 injective candidate targets and count diagnostics

Base: `f8524b0d434bdfbc5129bf7a60fa75deda5debf6` on
`v240-categorical-k-candidate-subset`.

The inherited V13 event supervision independently assigned the nearest candidate
to each event. Simultaneous events could share one candidate even when a distinct
assignment existed. V24 collapsed those assignments into one subset positive,
contradicting its unique top-K runtime and the injective V23 preaudit.

V24 now uses deterministic maximum-cardinality, minimum-total-time-error matching
within the preaudit's fixed 50 ms tolerance. Candidate IDs are used at most once.
Unmatched events retain cardinality/presence supervision but have no identity/time
detail loss. A positive row contributes to the listwise subset loss only when all
K events have matched candidates; partial targets must not label unknown true
candidates as negative. Match completeness and unmatched-event counts are recorded
in the inherited `supervision.anonymous_event_targets` report block. The V13 helper and
other versioned experiments are unchanged.

Runtime diagnostics now distinguish:

- `runtime_realized_exact_categorical_k_rate`: emitted count equals predicted K.
- `runtime_realized_exact_true_k_rate`: emitted count equals annotated K.
- `rows_candidate_count_below_predicted_k`: available valid candidates are fewer
  than predicted K.
- `rows_emitted_count_below_predicted_k`: emitted count is smaller than predicted K.

Protocol fields identify the new matching and complete-subset supervision policy.
The 14 CPU-only regression tests include exhaustive small-instance matching checks,
simultaneous events, missing/out-of-tolerance candidates, incomplete supervision,
V24 hook restoration, and the four runtime count cases. Run:

```sh
python -m unittest test.test_v240_candidate_subset -v
```

The existing V24 preflight runs these tests before training. No training or dataset
evaluation was performed for this correction. Existing checkpoints learned the
old targets; this patch does not retroactively correct them or establish an F1
gain. Corrected target feasibility and model performance require a separate run.
Old and new target-based subset diagnostics must not be pooled as identical metrics.
