# S651 — Conditional NOTE head/selection router (experiment protocol)

## Motivation: why a fixed selection is insufficient

S650 evaluated four separately fitted harmonic-evidence heads (B, T, E,
J) and eighteen fixed selectors (veto, consensus, OR, voting, weighted
scores). Some reached 63.47% DEV GuitarSet F1, exceeding exploratory
S641's 63.40%. But the **selector chosen on TRAIN**, B_AND_E,
regressed to 63.26% on DEV. Choosing the DEV winner would be leakage.

A selection must instead learn when consensus can reject harmonics and
when a faint grave fundamental needs a rescue.

## S651: trainable conditional routing, no frozen NOTE change

**Input at present causal hop only:** four independent head
probabilities, pairwise disagreements, confidence and vote count.
An enriched alternative receives selected NOTE-native onset/sustain
statistics, spectral fundamental evidence, pitch register,
harmonic competition and original native Exact-K posterior.

The API is intentionally a *router*, not a new main pitch/K model.
The resulting action is always passed through a separate,
unchanged `CausalSelection` instance; no NoteOn can be backdated.

Two predeclared learned meta-selectors:
1. `opinions_only`: confidence scores/disagreement and consensus;
2. `conditional_context`: same inputs + causal morphology,
   register, native K and physical harmonic evidence.

They are trained on actual **correction-versus-regression NoteOn**
labels from ungated Bass Rescue, rather than just a duet-present label.

## Nested composition split safeguards

The evaluation uses pinned original GuitarSet:
- TRAIN: 36 microphone recordings, nine compositions, 761 rescue
  events including 50 true new corrections (historical control);
- DEV: 12 real recordings, previously examined (exploratory only);
- GuitarSet TEST, URMP and IDMT TEST remain closed.

For each outer 5-fold **whole-composition** TRAIN split:
1. Exclude held compositions and their annotations completely;
2. build three inner whole-composition folds in the outer training set;
3. create out-of-fold predictions of the **four independently
   trained base heads** for meta-training;
4. fit the learned routing head only on these inner-OOF predictions;
5. evaluate the route on outer-held compositions using base heads
   trained only on outer TRAIN. No head ever sees its scored
   composition's labels;
6. freeze the routing family and decision threshold from outer OOF
   outcomes, independently of DEV.

An explicit predeclared improvement margin of +2 in utility
(`TP - 0.5*FP`) is required over the original S641 head; if the
learned router does not achieve this, the route falls back to S641.

The final base heads train on all TRAIN. Final router meta-inputs
come from TRAIN group-OOF base predictions only; the final DEV
model is evaluated unchanged. The original S614 heads/K stay frozen.

## Accuracy and scope limitations

- The router is a proof-of-concept, not a full integration into
  historical S80 exact-K correction head/selection paths.
- Validation on DEV already analyzed earlier is exploratory, not
  a sealed generalization claim.
- There is **zero additional future audio lookahead** in the
  experiments; this does not establish production microphone
  throughput/physical latency.
- Learned classifiers may be overfit because only 50 positive
  rescue examples exist among the 761 TRAIN candidates.
- The GitHub Actions run's `report.json` is the source of numerical
  scores. Do not state superiority unless that report is complete
  and S614, bass_raw, and S641 replication controls all pass.
