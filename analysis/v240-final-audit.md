# V24 corrected: final selection and cardinality audit

Audit: https://github.com/Andriamarosoa/note/actions/runs/34182431118
Audit code: `f7dee52b07ccbb74e5257598feb5f8ea6a74463d`.
Training source: run `34168832668`, commit `980a052e6cd9eee7896962ceb3a458e0ae258334`.

## Scope and verification

All five folds cover exactly 76,768 distinct outer rows and 240 tracks.
The saved selected IDs reproduce each fold's published TP, FP, FN and prediction/reference
counts exactly. The frozen-ranking replay also matches. There are zero candidate-ID
ordering violations and zero differences in selected sets when replaying the saved
ranking with the emitted count. No fitting, new model inference, threshold optimization,
or historical validation/Locked12 evaluation was performed. Nineteen unit tests passed.

The full JSON alongside this document is recovered from the successful job's printed
report; it includes per-fold results and SHA-256 fingerprints for source reports and
prediction archives.

## Official 50 ms aggregate, computed from counts

| Variant | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|
| direct | 0.79768501 | 33975 | 6989 | 10245 |
| frozen_same_count | 0.79768501 | 33975 | 6989 | 10245 |
| oracle_true_k_frozen_ranking | 0.97298866 | 41911 | 18 | 2309 |
| oracle_true_k_learned_ranking | 0.97302129 | 41909 | 13 | 2311 |
| v104 | 0.80384234 | 34017 | 6399 | 10203 |

V24 minus V10.4 is -0.00615733 F1,
with 590 additional false positives and 42 additional misses.
At identical emitted counts, learned and frozen ranking have exactly the same aggregate
TP/FP/FN. Per-fold differences cancel; this is not a claim that their predictions
are identical. Timestamps change on all 240 tracks.

## Fixed tolerance diagnostics

| Tolerance (ms) | Learned | Frozen, same count | Reversed, same count |
|---|---:|---:|---:|
| 5 | 0.52911345 | 0.49645473 | 0.14446375 |
| 10 | 0.67444591 | 0.64094196 | 0.25422615 |
| 20 | 0.76314801 | 0.74690083 | 0.52157682 |
| 50 | 0.79768501 | 0.79768501 | 0.79362322 |

The candidate cluster width is at most 40 ms (median 32.4036 ms).
The learned ranking improves temporal localization at 5, 10 and 20 ms, while the
official 50 ms tolerance largely hides that benefit. Reversing the ranking strongly
hurts tight-tolerance scores but only modestly hurts the 50 ms score.
These tolerances were fixed diagnostics; the official metric is not changed.

## Counting versus selection

Polyphonic categorical K is exact on 36.2302% of rows.
Providing true K diagnostically raises F1 to 0.97302129
with learned ranking and 0.97298866 with frozen ranking.
Those oracle scores use labels and are not deployable results or independent validation.
The oracle variants emit 41,922 versus 41,929 candidates: compare their small difference
cautiously because candidate availability differs between the direct and frozen paths.

## Decision and next hypothesis

Close corrected V24 as no promotion over V10.4 on official 50 ms F1.
The replay gives no evidence of an ignored selection path or an ID/timestamp mapping bug
on these artifacts. Preserve the tighter-tolerance ranking benefit as a separate finding.

Prioritize a controlled cardinality experiment: freeze candidate features, grouping,
ranking and timestamp realization, and change only the K estimator. Compare a proposed
count treatment with the existing categorical head under identical outer splits,
inner epoch selection and seeds. Report aggregate TP/FP/FN, confusion by true K,
K=0 false births, under/overcount, and polyphonic exact K; keep 50 ms as the official
metric and 5/10/20 ms as secondary diagnostics. Choose and document the one concrete
count treatment before launching compute. No new training was launched by this audit.
