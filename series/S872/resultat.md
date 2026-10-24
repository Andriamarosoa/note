# S872 — Résultats

Cible -12 demi-tons. Données authentiques GuitarSet. PCM256 strict.

| Mesure | Standard grain 4096 | GRU hybride 2048/4096 |
|---|---:|---:|
| CAL justes ±0.5 demi-ton | 5/6 | **5/6** |
| Déjà examinées justes | 6/6 | **6/6** |
| CAL log-STFT erreur médiane (plus bas mieux) | 0.130453 | **0.126272** |
| Déjà examinées log-STFT erreur | 0.151035 | **0.138222** |
| Nombre de clips CAL spectralement améliorés | — | **5/6** |
| CPU p99 local, 220 hops | — | **1.48 ms** |

Audit de phase : retard acoustique moyen approximatif pour grain 4096 ~24.7 ms, variable selon phase. Ni les pourcentages de distance spectrale ni les 6 notes déjà observées ne constituent une précision musicale globale.

All numbers from local research on real GuitarSet S847 note excerpts; 27 TRAIN / 6 CAL / 6 previously examined notes (NOT blind). Goal: -12 semitones, source audio highs to lows, reference offline only. Exact 256 new PCM at 44.1kHz, causal, but acoustic delay >0. Six NOTE heads and MIDI unchanged; no promoted model and no official NOTE F1 retest. Trained checkpoints and WAVs are in the conversation archive.
