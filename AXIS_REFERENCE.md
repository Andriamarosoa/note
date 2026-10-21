# Branche de référence — Active Exact-K

**SEUL AXE À AMÉLIORER DANS CETTE BRANCHE : Active Exact-K.**

- Origine immuable : `codex/note-s828-activek-s708-private-readout-256`, série scientifique **S828**.
- Meilleur score retenu selon la métrique décrite : **73,831146 %**.
- Métrique : **Exactitude du nombre de notes uniquement sur trames actives** ; cohorte : FIT80 80 pistes, 249 411 trames actives.
- Cadence : 256 échantillons @ 44,1 kHz.
- Chemins d'implémentation source à préserver :
  - `src/causal_note/s828_activek_s708_private_readout.py`
  - `src/causal_note/s827_s708_activek_only.py`
- Limites : SHADOW, lecture S708 gelée et readout K privé entraîné à 256; les poids du readout doivent être sérialisés avant déploiement.

## Contrat de non-régression
Cette branche est indépendante des cinq autres branches de référence : entraîner ou corriger **uniquement Active Exact-K**. Ne modifier aucune tête, aucun sélecteur, décodeur, état, checkpoint ni métrique des cinq autres caractéristiques. Toute future proposition doit préserver la même cohorte, la même définition de métrique et la provenance de source. Aucun résultat d'une grille 1024 ne doit être attribué à une inférence native 256. Refuser une correction si une des cinq autres sorties change. Avant une fusion MIDI éventuelle, réaliser des tests appariés des six métriques sur les mêmes pistes, prouver l'absence de régression et mesurer la latence physique iRig. Aucun MIDI public modifié et aucune promotion dans cette branche.
