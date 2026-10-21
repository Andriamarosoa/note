# Branche de référence — Frame Pitch

**SEUL AXE À AMÉLIORER DANS CETTE BRANCHE : Frame Pitch.**

- Origine immuable : `codex/note-s834-s809-six-axes-native256`, série scientifique **S834**.
- Meilleur score retenu selon la métrique décrite : **84,998850 %**.
- Métrique : **Frame Pitch F1** ; cohorte : FIT80, 80 pistes.
- Cadence : 256 échantillons @ 44,1 kHz.
- Chemins d'implémentation source à préserver :
  - `src/causal_note/s834_s809_native256_audio.py`
  - `src/causal_note/s834_s809_six_axis_pcm256.py`
  - `src/causal_note/s834_native256_decoders.py`
- Limites : Record historique S790 86,7532 % FIT80 était SHADOW1024, pas réellement natif256.

## Contrat de non-régression
Cette branche est indépendante des cinq autres branches de référence : entraîner ou corriger **uniquement Frame Pitch**. Ne modifier aucune tête, aucun sélecteur, décodeur, état, checkpoint ni métrique des cinq autres caractéristiques. Toute future proposition doit préserver la même cohorte, la même définition de métrique et la provenance de source. Aucun résultat d'une grille 1024 ne doit être attribué à une inférence native 256. Refuser une correction si une des cinq autres sorties change. Avant une fusion MIDI éventuelle, réaliser des tests appariés des six métriques sur les mêmes pistes, prouver l'absence de régression et mesurer la latence physique iRig. Aucun MIDI public modifié et aucune promotion dans cette branche.
