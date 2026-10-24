---
serie: S881
date: 2026-10-24
statut: termine_local_non_promu
source: S847_GuitarSet_reel
reference: S873_hybride_recherche
promotion: false
---

# S881 — Audit d'un autre standard via pedalboard/Rubber Band

## Méthode et audit
Tester le wrapper Python Spotify pedalboard.PitchShift(−12) appelé répétitivement sur 256 samples, puis comparer au même wrapper sur bloc entier. Ce n'est PAS l'API native streaming de Rubber Band.

## Résultats observés
Sur la vraie phrase 3,5 s, la voie wrapper par appels PCM256 ne donne du son non nul que sur 1,316 % des échantillons ; les 0,5 s de fin sont silencieuses. Appel de la phrase entière : sortie non silencieuse.

## Décision de sélection
**Rejeter cette voie d'INTÉGRATION 256, sans conclure que Rubber Band natif est défectueux.** Ne pas transformer ces scores relatifs en « pourcentage de bonne transposition ».

## Contraintes, provenance, limites
**Contraintes** : transposition de guitare aiguë vers grave −12 demi-tons, 256 nouveaux PCM à 44,1 kHz par appel, audio passé autorisé mais aucun futur. La latence intrinsèque du DSP standard n'est PAS nulle ni auditée ici; la performance CPU a été volontairement ignorée. Six têtes NOTE et MIDI inchangés, aucune promotion. 39 extraits réels GuitarSet S847 : 27 TRAIN, 6 CAL, 6 anciennes AUDITED déjà consultées, donc pas de généralisation indépendante. L'enseignant offline librosa −12 n'est jamais présent à l'inférence.
