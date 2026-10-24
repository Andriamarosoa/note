---
serie: S874
date: 2026-10-24
statut: termine_local_non_promu
source: S847_GuitarSet_reel
reference: S873_hybride_recherche
promotion: false
---

# S874 — Audit des erreurs et de la méthodologie S873

## Méthode et audit
Audit du standard granulaire seul et de la porte GRU qui mélange deux lignes de retard de phases potentiellement différentes. Identification des notes AUDITED déjà réutilisées; construction de tests de mélanges numériques et séparation de leur statut des vrais accords.

## Résultats observés
Aucun nouveau score d'accuracy musicale revendiqué. Les contrôles incluent des notes isolées, une phrase réelle de 3,5 s et des mélanges synthétiques de prises réelles.

## Décision de sélection
**Audit méthodologique seulement ; éviter de promouvoir un modèle sur un score de six notes.** Ne pas transformer ces scores relatifs en « pourcentage de bonne transposition ».

## Contraintes, provenance, limites
**Contraintes** : transposition de guitare aiguë vers grave −12 demi-tons, 256 nouveaux PCM à 44,1 kHz par appel, audio passé autorisé mais aucun futur. La latence intrinsèque du DSP standard n'est PAS nulle ni auditée ici; la performance CPU a été volontairement ignorée. Six têtes NOTE et MIDI inchangés, aucune promotion. 39 extraits réels GuitarSet S847 : 27 TRAIN, 6 CAL, 6 anciennes AUDITED déjà consultées, donc pas de généralisation indépendante. L'enseignant offline librosa −12 n'est jamais présent à l'inférence.
