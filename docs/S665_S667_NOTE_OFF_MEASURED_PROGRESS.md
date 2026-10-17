# S665–S667 — NOTE Offset research, 17 October 2026

## Objective and immutable state

Goal: >=80% correct NoteOff under strict +/-50 ms tolerance, evaluated
conditional on already matched MIDI pitch and NoteOn. Historical GuitarSet
190 songs: N=25,395 correctly matched NoteOn/pitch, unchanged reference
S425 16,543 correct (65.1427%). Required 20,316 correct, i.e. +3,773
over S425.

S665 and S666 did not update production NOTE Live, original S80 Exact-K
59,309 events, or the 33,475 original NOTE pitch/onset predictions.

## Measured benchmarks

| Model | Correct NoteOff | Conditional accuracy | Offset MAE |
|---|---:|---:|---:|
| S425 baseline | 16,543 | 65.1427% | 87.60 ms |
| S664 acoustic candidate | 16,777 | 66.0642% | 86.48 ms |
| S665 history-only penalty .025 | 16,784 | 66.0918% | 86.61 ms |
| **S665 duration specialists .025** | **16,891** | **66.5131%** | **86.51 ms** |
| S666 direct regression best +/-50ms | 16,541 | 65.1349% | 85.63 ms |

S665 gains +348 correct offsets over S425 (fold gains +121, +62,
+63, +102). These are real fold-held GuitarSet results, but NOT
an independently prospective production certification. The penalty
configurations were compared on exposed folds. S425/S411 upstream
cross-fitting isn't fully nested. Strict Live is unchanged.

S666 direct acoustic lifespan regression was rejected for 50 ms
accuracy. It improved MAE as low as 84.22 ms, while substantially
reducing the number of offsets inside the strict 50 ms window.
Optimizing MAE is NOT a sufficient proxy for precision of NOTE closure.

At the S665 best, still 20,316-16,891 = **3,425** net correctly placed
NoteOff events are needed to reach 80%.

## Architectures

S665 uses 10 S411 genuine MIC waveform-derived horizon summaries
and 3 expert heads for candidate durations <300ms, 300-1000ms, >=1s.
It evaluates candidate success probabilities and selects without held
annotations. However S411 STFT is centered, and native S425 offset
proposals were generated from up to 5s future waveform. Hence S665
is offline experiment, not strictly causal MIDI performance.

S666 uses direct log-duration or residual-ms HistGradientBoosting
regressors and then blends an estimate with the fixed S425 baseline.
It is also offline and is rejected by the primary correctness metric.

S667 (separate branch): independently evaluates S614 strict causal
release hysteresis without replacing the upstream heads. It checks
offset tolerance, NoteOn F1 and train-held composition generalization.
S667's GuitarSet DEV strict-Live test has a **different evaluation
population** from historical S425; its scores cannot be compared by
percentage to this table.

## Reproduction

- S665 GitHub Actions: https://github.com/Andriamarosoa/note/actions/runs/39024199483
- S666 GitHub Actions: https://github.com/Andriamarosoa/note/actions/runs/39024610484
- S667 GitHub Actions: https://github.com/Andriamarosoa/note/actions/runs/39024866675

No official promotion, held future-only prospective TEST, or real
microphone-to-MIDI hardware latency evidence is claimed.

## S667 — vérification strictement causale terminée

Workflow #39024866675: **success**, 36 GuitarSet TRAIN + 12 DEV, 19 règles de
relâchement; seul l'original est retenu par la sélection entraînée sans DEV.
Sur 991 offsets de notes dont l'attaque est correctement appariée à +/-80 ms
et la fin est observable sur DEV GuitarSet:

| Politique | NoteOn F1 | Offsets exacts +/-50 ms | Offsets tolérance max(50ms,20% durée) |
|---|---:|---:|---:|
| original, release 5 / active_off 0.25 | 63.105% | 600/991 = 60.5449% | 61.4531% |
| release 5 / active_off 0.15 | 63.105% | 605/991 = 61.0494% | 61.9576% |

La précision flexible exacte du baseline est **61.4531%**,
la variante 5/0.15 **61.9576%**. L'écart strict est seulement **+5
notes**, une régression possible selon composition. La sélection TRAIN a
choisi l'original; aucune promotion. Les notes longues détectées exactement
à +/-50ms ne sont que **5** dans ces résultats; les chiffres DEV ne sont
pas à comparer directement aux 25 395 notes historiques S425-S666.

Les paramètres de release agissent sur la sortie du sélecteur causal, mais
ne réapprennent pas la preuve de présence physique indépendante. Les essais
avec release 1 ou 2 frames diminuent fortement les offsets exacts et les
fenêtres 12 frames retardent excessivement les fermetures. Pour cibler 80%,
il faut une **tête de survie/NoteOff entraînée sur les WAV continus avec
état actif par hauteur**, en conservant les NoteOn et Exact-K originaux.
