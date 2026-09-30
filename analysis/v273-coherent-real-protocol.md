# Coherent prediction on real audio: frozen diagnostic protocol

Protocol fixed before inspecting real-audio scores, 2026-09-30.

## Scope and source

Use every row of the existing internal validation partition for outer fold 3:
composition fold 0, 15,952 rows, with the full original candidate timestamps.
Do not evaluate outer fold 3, another external fold, or a newly selected subset.
This is a reused development partition, not an independent final benchmark.
Reference errors are the saved epoch-12 predictions of both ownership arms from
run 36519613364. Both are freshly trained native models; neither is the official
V27.3 ensemble. The local-only arm is the primary error reference.

Verify GuitarSet archive MD5s from `rebuild_v273_sources.DATA_MD5`, release ZIP
SHA256s, extracted inventories, partition membership, row identity and the
annotation-to-group assignment. Recompute K using all candidate timestamps and
the existing nearest-group rule (882 samples, earlier group wins ties).
Annotations are evaluation-only and never enter acoustic predictors.

## Fixed acoustic measurements

Reuse the three configurations and recurrence fitting implementation of
`probe_v273_coherent_decay.py` as published at 8f77db1. Do not tune on this audit.
For each group, forecast the 1,764 samples starting at its first candidate.
Past history is either 1,308 or 8,820 samples. Fit from past PCM only; forecast
autonomously. Normalize residual RMS with the last 1,308 past samples, floor
power 1e-12, exactly as in the synthetic probe.

Compare all three waveform residuals with the old log-power decay maximum,
the existing positive spectral-flux maximum, and a zero-wave forecast control
(post RMS / pre RMS). The zero-wave control has identical support and scale;
it tests whether phase prediction contributes beyond relative signal energy.
The old spectral features use the original 23-frame, float16 map. Historical
ownership models used 31 frames, so this audit is not an input-identical model
A/B. No network is retrained and no prediction is corrected.

Keep start/end zero-padding rows in the primary population; also report
unpadded rows. Record pre/post energy, forecast energy, numerical failures,
forecast norm above ten times pre RMS, rank, and coefficient norm. Never silently
discard failed rows. All predictors share one future sample budget of 40 ms.
This budget does not cover every onset assigned to a group: quantify owned
onsets before the origin, within the forecast, and after the forecast.

## Analyses fixed before execution

1. Novelty: AUC for any annotated onset in [origin, origin + 1,764) versus none,
   irrespective of group ownership. Compare methods and report per-track AUC.
2. Count relevance: AUC for owned K > 0 versus K = 0; K >= 2 versus K < 2;
   and adjacent K = 1/0, 2/1, 3/2, 4/3. Higher score has the same direction
   everywhere. These are diagnostic AUCs, never Exact K scores.
3. Low-K errors: for each true K = 0..3, describe scores for overcounts and
   correct predictions. Match each overcount to the temporally closest correct
   row on the same track and true K; allow reuse and disclose unique controls.
   Also compare correct versus low-K-overcount rows at the same predicted K.
   Report both saved model arms, without searching a corrective threshold.
4. Ownership: distinguish locally eligible attacks assigned to neighbors;
   tabulate owned attacks outside the forecast and no-future-onset cases.
5. Use 2,000 paired composition-block bootstrap draws (seed 9302026) for the
   primary novelty AUC difference, wide recurrence minus zero-wave control.
   Resample compositions, not individual correlated rows. This describes
   development uncertainty; it does not erase repeated use of the partition.

## Decision

Do not launch native training merely because a waveform score beats the old
decay index. Require the wide recurrence to improve novelty over the zero-wave
control with the paired interval above zero, and positive point differences on
both adjacent K 2/1 and K 3/2. Failing this gate rejects this unchanged scalar
prototype as a justified training input. Passing only justifies a subsequent
native A/B; it does not establish a solution or promote a model.

Associations with existing errors do not prove their cause. Residual energy can
also come from pitch changes, noise, or unpredictable sustained sound. No scalar
residual identifies distinct note sources or resolves ownership on its own.
