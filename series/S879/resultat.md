---
serie: S879
date: 2026-10-24
statut: termine_local_non_promu
source: S847_GuitarSet_reel
reference: S873_hybride_recherche
promotion: false
---

# S879 — Correction du faux positif du test de continuité streaming

## Méthode et audit
L'ancien script d'audit paddait des filtres STFT/FIR déjà calculés sur une forme d'onde tronquée ; ce chemin ne correspond pas à une exécution PCM256 réellement paddée. Recalculer le même standard sur le signal entièrement paddé avant de tester streaming contre bulk.

## Résultats observés
Écart absolu maximal sur 154 350 échantillons de la phrase : exactement 0,0 dans l'essai corrigé. Les 4 tests Python passent localement : bloc PCM256, absence de futur, silence/reconstruction neutre et égalité streaming/bulk.

## Décision de sélection
**L'écart précédent relevait du banc de test, pas d'une régression du modèle.** Ne pas transformer ces scores relatifs en « pourcentage de bonne transposition ».

## Contraintes, provenance, limites
**Contraintes** : transposition de guitare aiguë vers grave −12 demi-tons, 256 nouveaux PCM à 44,1 kHz par appel, audio passé autorisé mais aucun futur. La latence intrinsèque du DSP standard n'est PAS nulle ni auditée ici; la performance CPU a été volontairement ignorée. Six têtes NOTE et MIDI inchangés, aucune promotion. 39 extraits réels GuitarSet S847 : 27 TRAIN, 6 CAL, 6 anciennes AUDITED déjà consultées, donc pas de généralisation indépendante. L'enseignant offline librosa −12 n'est jamais présent à l'inférence.
