# S873 — Résultats

Cible -12 demi-tons. Vraie guitare, référence standard offline, aucune note/synthèse générée de novo en inférence.

| Mesure | Standard grain 8192 | GRU hybride 4096/8192 |
|---|---:|---:|
| CAL justes ±0.5 demi-ton | 6/6 | **6/6** |
| Déjà examinées justes | 6/6 | **6/6** |
| CAL log-STFT erreur médiane (↓ mieux) | 0.165482 | **0.142585** |
| Déjà examinées log-STFT erreur | 0.206923 | **0.176575** |
| Nombre de clips CAL spectralement améliorés | — | **6/6** |
| CPU p99 local, 220 hops | — | **2.13 ms** |

Note critique : le retard acoustique du procédé est positif, variable, non nul. Estimation grain 8192 : moyenne ~47.9 ms ; impulsion test à phase spécifique : 2252 samples. Comparaison Windows/iRig non réalisée. **Non promu.** Le changement n'affecte ni Onset ni les cinq autres têtes NOTE ni MIDI.

All numbers from local research on real GuitarSet S847 note excerpts; 27 TRAIN / 6 CAL / 6 previously examined notes (NOT blind). Goal: -12 semitones, source audio highs to lows, reference offline only. Exact 256 new PCM at 44.1kHz, causal, but acoustic delay >0. Six NOTE heads and MIDI unchanged; no promoted model and no official NOTE F1 retest. Trained checkpoints and WAVs are in the conversation archive.
