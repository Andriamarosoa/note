# S872 — Résumé

Standard causal -12 plus correcteur GRU à 256 PCM, choisissant entre grains 2048 et 4096. Le réseau ne fabrique aucune fréquence ; il mélange deux versions standard en fonction du signal, avec possibilité de bypass exact.

CAL : 5/6 notes correctes avant et après IA, distance spectrale médiane 0.130453 → 0.126272 (−3.2 %). Anciennes AUDITED : 6/6 correctes avant et après IA, distance 0.151035 → 0.138222. **Non promu.**

All numbers from local research on real GuitarSet S847 note excerpts; 27 TRAIN / 6 CAL / 6 previously examined notes (NOT blind). Goal: -12 semitones, source audio highs to lows, reference offline only. Exact 256 new PCM at 44.1kHz, causal, but acoustic delay >0. Six NOTE heads and MIDI unchanged; no promoted model and no official NOTE F1 retest. Trained checkpoints and WAVs are in the conversation archive.
