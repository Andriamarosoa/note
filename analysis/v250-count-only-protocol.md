# V25: controlled count-only categorical versus conditional ordinal

The V24 audit (run 34182431118, code f7dee52b) reproduced published metrics and
found no candidate ID mapping problem. Learned ranking improves localization at
5/10/20 ms but matches frozen ranking at the official 50 ms tolerance. Polyphonic
K accuracy is 36.23%; true-K diagnostic F1 is about 0.973.

## Fixed hypothesis

A conditional ordinal count head may improve K estimation because positive rare
counts share supervision across successive continuation stages. This mechanism
was explored historically in V9.1; V25 is a controlled paired test on the newer
V24 dense count encoder and frozen V24 realization, not a claim of novelty.

- Control: seven-way categorical softmax P(K=0..6).
- Treatment: six conditional probabilities q_j=P(K>=j | K>=j-1). Survival is the
  cumulative product; differences give a valid seven-class distribution.
- Both optimize the same sparse categorical negative log likelihood, use argmax
  decoding, and share the V24 count-encoder architecture and initial trunk weights.
- Both are fresh count-only models. V24's auxiliary event/ranking objectives are
  removed from BOTH arms. Compare the paired arms to isolate the head treatment;
  comparison with historical joint V24 also includes this shared training change.

## Fixed realization and clean selection

Cached inputs, grouping, candidate masks, saved V24 ranking and timestamp
reconstruction are unchanged. V24 artifacts are checked against the SHA-256
fingerprints archived in analysis/v240-final-audit.json. The original V24 selected
sets and official TP/FP/FN must replay exactly before fitting.

No pretrained V24 latent representation is used for inner selection. Both count
encoders are trained from scratch: reusing a final V24 encoder could expose the
inner validation fold through that encoder's earlier training. The frozen V24
ranking is used only for outer event realization and is identical for both arms.

Five composition-group outer folds remain fixed. The lowest non-outer fold is
inner validation; the other three are inner fit. The selected epoch minimizes
inner weighted NLL (20 epochs maximum; patience 4). Refit from scratch on all four
non-outer folds for that many epochs. No outer labels select epochs or thresholds.

Class weights: inverse-square-root frequency, clipped to [0.35,4], normalized to
mean one on the fitting labels. The table is derived only from the relevant fit
partition, including when it weights inner validation. Both arms use the same
table. Adam 2e-4, batch 128, seed 16061 with identical fold/phase offsets.

## Reporting and execution

Report official F1 from summed TP/FP/FN over five folds; keep 5/10/20 ms secondary.
Report confusion by true K, zero-count false births, under/overcount, polyphonic
exact K, and candidate shortages. Archive probability arrays, selected IDs,
weights, histories, partition hashes and frozen-input fingerprints.

Five paired jobs run after a real TensorFlow preflight verifies identical initial
trunks, normalized distributions, finite gradients and a training batch for each
arm. Each job runs the two arms sequentially, with a 300-minute job limit. No
historical validation or Locked12 evaluation. No threshold search. Do not promote
an arm from isolated folds; assess the completed paired aggregate and regressions.
