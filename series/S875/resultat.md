---
serie: S875
date: 2026-10-24
statut: termine_local_non_promu
source: S847_GuitarSet_reel
reference: S873_hybride_recherche
promotion: false
---

# S875 — GRU correcteur multibande limité à ±20%

## Méthode et audit
Réseau GRU causal entraîné sur TRAIN, filtre FIR de 4 bandes qui se somment exactement au standard granulaire 8192 quand les gains valent 1, sans mélange de deux standards déphasés.

## Résultats observés
6/6 hauteurs correctes CAL; distance log-STFT médiane CAL 0,16548 (DSP) → 0,15381; AUDITED 0,20692 → 0,19086. Les six notes CAL et six anciennes AUDITED bénéficient toutes de cette mesure, mais S873 est meilleur sur CAL (0,14258).

## Décision de sélection
**Refus de promotion, pas assez bon face à S873.** Ne pas transformer ces scores relatifs en « pourcentage de bonne transposition ».

## Contraintes, provenance, limites
**Contraintes** : transposition de guitare aiguë vers grave −12 demi-tons, 256 nouveaux PCM à 44,1 kHz par appel, audio passé autorisé mais aucun futur. La latence intrinsèque du DSP standard n'est PAS nulle ni auditée ici; la performance CPU a été volontairement ignorée. Six têtes NOTE et MIDI inchangés, aucune promotion. 39 extraits réels GuitarSet S847 : 27 TRAIN, 6 CAL, 6 anciennes AUDITED déjà consultées, donc pas de généralisation indépendante. L'enseignant offline librosa −12 n'est jamais présent à l'inférence.
