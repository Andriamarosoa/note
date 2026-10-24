---
serie: S878
date: 2026-10-24
statut: termine_local_non_promu
source: S847_GuitarSet_reel
reference: S873_hybride_recherche
promotion: false
---

# S878 — Correction IA en cascade sur S873

## Méthode et audit
Appliquer la tête multibande entraînée S876 à la sortie du meilleur hybride antérieur S873. Le standard −12 demeure source de toute hauteur; correction acoustique seulement.

## Résultats observés
CAL 6/6 notes justes; distance médiane 0,14258 → 0,13530 (−5,11 %). Anciennes AUDITED 6/6 et distance 0,17657 → 0,15855 (−10,21 %). MAIS phrase réelle : 0,40664 → 0,41977 (pire) et dernière seconde 0,52626 → 0,56701 (pire).

## Décision de sélection
**Refus de promotion : régression mesurée sur la performance réelle malgré le gain notes isolées.** Ne pas transformer ces scores relatifs en « pourcentage de bonne transposition ».

## Contraintes, provenance, limites
**Contraintes** : transposition de guitare aiguë vers grave −12 demi-tons, 256 nouveaux PCM à 44,1 kHz par appel, audio passé autorisé mais aucun futur. La latence intrinsèque du DSP standard n'est PAS nulle ni auditée ici; la performance CPU a été volontairement ignorée. Six têtes NOTE et MIDI inchangés, aucune promotion. 39 extraits réels GuitarSet S847 : 27 TRAIN, 6 CAL, 6 anciennes AUDITED déjà consultées, donc pas de généralisation indépendante. L'enseignant offline librosa −12 n'est jamais présent à l'inférence.
