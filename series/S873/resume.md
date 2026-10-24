# S873 — Résumé

Deuxième boucle hybride avec deux DSP causal grains 4096 et 8192 et GRU pour corriger leur combinaison. Sur CAL : 6/6 notes exactes (idem DSP), distance spectrale 0.165482 → 0.142585, soit **13.8 % de moins**. Sur les six anciennes AUDITED : 6/6, distance 0.206923 → 0.176575. Qualité spectrale améliorée mais délai intrinsèque plus élevé, ~47.9 ms en moyenne estimée pour le grain 8192. **Non promu.**

All numbers from local research on real GuitarSet S847 note excerpts; 27 TRAIN / 6 CAL / 6 previously examined notes (NOT blind). Goal: -12 semitones, source audio highs to lows, reference offline only. Exact 256 new PCM at 44.1kHz, causal, but acoustic delay >0. Six NOTE heads and MIDI unchanged; no promoted model and no official NOTE F1 retest. Trained checkpoints and WAVs are in the conversation archive.
