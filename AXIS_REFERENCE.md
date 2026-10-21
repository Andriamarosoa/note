# Branche de référence — Onset

**SEUL AXE À AMÉLIORER DANS CETTE BRANCHE : Onset.**

- Origine immuable : `codex/note-s812-protect-six-native-h0-axes`, série scientifique **S708/H0 conservé via S812**.
- Meilleur score retenu selon la métrique décrite : **73,252019 % FIT80 (74,2639 % CAL32)**.
- Métrique : **Onset event F1** ; cohorte : FIT80 pour comparaison; CAL32 officiel historique distinct.
- Cadence : Horloge source historique 1024, pas une nouvelle tête PCM256 validée.
- Chemins d'implémentation source à préserver :
  - `src/causal_note/s812_historical_axis_h0.py`
  - `src/causal_note/s712_s708_live.py`
- Limites : Ne pas remplacer par S834 71,3823 % (régression de 1,87 point sur FIT80).

## Contrat de non-régression
Cette branche est indépendante des cinq autres branches de référence : entraîner ou corriger **uniquement Onset**. Ne modifier aucune tête, aucun sélecteur, décodeur, état, checkpoint ni métrique des cinq autres caractéristiques. Toute future proposition doit préserver la même cohorte, la même définition de métrique et la provenance de source. Aucun résultat d'une grille 1024 ne doit être attribué à une inférence native 256. Refuser une correction si une des cinq autres sorties change. Avant une fusion MIDI éventuelle, réaliser des tests appariés des six métriques sur les mêmes pistes, prouver l'absence de régression et mesurer la latence physique iRig. Aucun MIDI public modifié et aucune promotion dans cette branche.
