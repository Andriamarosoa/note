---
serie: S880
date: 2026-10-24
statut: termine_local_non_promu
source: S847_GuitarSet_reel
reference: S873_hybride_recherche
promotion: false
---

# S880 — Entraînement enrichi d'accords simulés

## Méthode et audit
19 mélanges de deux notes réelles, composés uniquement à partir des 27 notes TRAIN et réentraînement de l'égaliseur à gain borné, depuis S876 (fine-tune) et depuis zéro, sélection sur CAL, phrase externe examinée seulement à la fin.

## Résultats observés
CAL, distance EQ fine-tune 0,14728 ; EQ from scratch 0,14570. En cascade avec S873 sur vraie phrase : 0,41641 (fine-tune) et 0,41557 (scratch), contre S873 0,40664 ; dernière seconde pire également.

## Décision de sélection
**Les deux variantes refusées, aucune correction de la régression sur vraie phrase.** Ne pas transformer ces scores relatifs en « pourcentage de bonne transposition ».

## Contraintes, provenance, limites
**Contraintes** : transposition de guitare aiguë vers grave −12 demi-tons, 256 nouveaux PCM à 44,1 kHz par appel, audio passé autorisé mais aucun futur. La latence intrinsèque du DSP standard n'est PAS nulle ni auditée ici; la performance CPU a été volontairement ignorée. Six têtes NOTE et MIDI inchangés, aucune promotion. 39 extraits réels GuitarSet S847 : 27 TRAIN, 6 CAL, 6 anciennes AUDITED déjà consultées, donc pas de généralisation indépendante. L'enseignant offline librosa −12 n'est jamais présent à l'inférence.
